"""
X (Twitter) and Reddit, via their official APIs only.

X:      API v2 with a bearer token (X_BEARER_TOKEN). Reading posts needs a
        paid API tier; the user-timeline endpoint returns up to the ~3,200
        most recent posts.
Reddit: OAuth "application-only" access (REDDIT_CLIENT_ID / _SECRET from
        https://www.reddit.com/prefs/apps, type "script").

Unofficial scraping (login-wall bypass, private endpoints) isn't supported:
it breaks the platforms' terms and tends to break without warning.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from kickvoice.research.net import NetError
from kickvoice.research.sources.base import CollectStats, SourceContext
from kickvoice.research.store import SELF, Account, Chunk, Document


class MissingCredentials(RuntimeError):
    pass


def _env(name: str, what: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingCredentials(f"{what}: set the {name} environment variable.")
    return value


# ── X ────────────────────────────────────────────────────────────

def collect_x(ctx: SourceContext, account: Account, limit: int | None = None) -> CollectStats:
    stats = CollectStats()
    rcfg = ctx.config.research
    try:
        token = _env(rcfg.x_bearer_token_env, "X API access")
    except MissingCredentials as e:
        stats.errors.append(str(e))
        return stats
    headers = {"Authorization": f"Bearer {token}"}
    username = account.handle.lstrip("@")
    limit = limit or rcfg.max_posts

    user = ctx.fetcher.get_json(f"{rcfg.x_api}/users/by/username/{username}",
                                headers=headers, cache=False)
    if "data" not in user:
        stats.errors.append(f"X user not found: @{username} ({user.get('errors')})")
        return stats
    user_id = user["data"]["id"]

    next_token, fetched = None, 0
    while fetched < limit:
        params = {"max_results": min(100, max(5, limit - fetched)), "exclude": "retweets",
                  "tweet.fields": "created_at,conversation_id,referenced_tweets,public_metrics,lang"}
        if next_token:
            params["pagination_token"] = next_token
        try:
            page = ctx.fetcher.get_json(f"{rcfg.x_api}/users/{user_id}/tweets",
                                        params=params, headers=headers, cache=False)
        except NetError as e:
            stats.errors.append(f"X timeline: {e}")
            break
        for tweet in page.get("data", []):
            fetched += 1
            is_reply = any(r.get("type") == "replied_to" for r in tweet.get("referenced_tweets", []))
            quoted = any(r.get("type") == "quoted" for r in tweet.get("referenced_tweets", []))
            ctx.save(Document(
                person_id=ctx.person.id, source="x", kind="reply" if is_reply else "post",
                external_id=tweet["id"], text=tweet["text"], authorship=SELF,
                url=f"https://x.com/{username}/status/{tweet['id']}",
                published_at=tweet.get("created_at"),
                meta={"metrics": tweet.get("public_metrics", {}), "quote_post": quoted,
                      "lang": tweet.get("lang")},
            ), [Chunk(tweet["text"])], stats)
        next_token = page.get("meta", {}).get("next_token")
        if not next_token:
            break
    ctx.progress(f"x: @{username} — {fetched} post(s)")
    return stats


# ── Reddit ───────────────────────────────────────────────────────

def _reddit_token(ctx: SourceContext) -> str:
    rcfg = ctx.config.research
    client_id = _env(rcfg.reddit_client_id_env, "Reddit API access")
    secret = _env(rcfg.reddit_client_secret_env, "Reddit API access")
    response = ctx.fetcher.post(rcfg.reddit_auth, data={"grant_type": "client_credentials"},
                                auth=(client_id, secret))
    if response.status != 200:
        raise NetError(f"Reddit auth failed: HTTP {response.status} {response.text[:200]}")
    return response.json()["access_token"]


def collect_reddit(ctx: SourceContext, account: Account, limit: int | None = None) -> CollectStats:
    stats = CollectStats()
    rcfg = ctx.config.research
    try:
        token = _reddit_token(ctx)
    except (MissingCredentials, NetError) as e:
        stats.errors.append(str(e))
        return stats
    headers = {"Authorization": f"bearer {token}"}
    username = account.handle.removeprefix("u/").removeprefix("/u/")
    limit = limit or rcfg.max_posts

    for listing in ("comments", "submitted"):
        after, fetched = None, 0
        while fetched < limit:
            params = {"limit": 100, "sort": "new", "raw_json": 1}
            if after:
                params["after"] = after
            try:
                page = ctx.fetcher.get_json(f"{rcfg.reddit_api}/user/{username}/{listing}",
                                            params=params, headers=headers, cache=False)
            except NetError as e:
                stats.errors.append(f"reddit {listing}: {e}")
                break
            children = page.get("data", {}).get("children", [])
            for child in children:
                fetched += 1
                d = child.get("data", {})
                if child.get("kind") == "t1":                     # comment
                    text, title, kind = d.get("body", ""), d.get("link_title", ""), "comment"
                else:                                             # t3 = post
                    title, kind = d.get("title", ""), "post"
                    text = "\n\n".join(x for x in (title, d.get("selftext", "")) if x)
                if not text or text in ("[deleted]", "[removed]"):
                    continue
                ctx.save(Document(
                    person_id=ctx.person.id, source="reddit", kind=kind,
                    external_id=d.get("name") or d.get("id", ""), text=text, authorship=SELF,
                    url="https://www.reddit.com" + d.get("permalink", ""), title=title,
                    published_at=_utc(d.get("created_utc")),
                    meta={"subreddit": d.get("subreddit"), "score": d.get("score")},
                ), None, stats)
            after = page.get("data", {}).get("after")
            if not after or not children:
                break
        ctx.progress(f"reddit: u/{username} {listing} — {fetched}")
    return stats


def _utc(ts) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None
