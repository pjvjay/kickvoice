"""
Profile packs: portable, shareable bundles of speaker voiceprints.

A pack is a single JSON file (diff-friendly, easy to keep in git) holding the
embeddings and provenance of one or more profiles. Packs are tied to the
embedding model that produced them — vectors from different models are not
comparable, so mismatched packs are refused on import.

    {
      "format": "kickvoice-profile-pack",
      "version": 1,
      "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
      "embedding_dim": 192,
      "created_at": "...",
      "profiles": [
        {"name": "...", "embedding": [192 floats], "speech_seconds": 812.4,
         "n_segments": 530, "source_urls": [...], "notes": "...",
         "enrolled_at": "...", "updated_at": "..."}
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from kickvoice.config import Config
from kickvoice.storage.profiles import ProfileStore

PACK_FORMAT = "kickvoice-profile-pack"
PACK_VERSION = 1
CONFLICT_MODES = ("skip", "overwrite", "merge")


class PackError(ValueError):
    pass


@dataclass
class ImportResult:
    added: list[str] = field(default_factory=list)
    overwritten: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def export_pack(
    store: ProfileStore,
    path: str | Path,
    names: list[str] | None = None,
    config: Config | None = None,
) -> int:
    """Write profiles (all, or `names`) to a pack file. Returns count."""
    config = config or store.config
    model = config.embedding.model_source
    profiles = store.list_profiles()
    if names:
        wanted = {n.casefold() for n in names}
        profiles = [p for p in profiles if p.name.casefold() in wanted]
        missing = wanted - {p.name.casefold() for p in profiles}
        if missing:
            raise PackError(f"Profile(s) not found: {', '.join(sorted(missing))}")

    entries = []
    for p in profiles:
        if p.embedding_model and p.embedding_model != model:
            raise PackError(
                f"'{p.name}' was made with {p.embedding_model}, not {model}"
            )
        emb = store.get_embedding(p.name)
        if emb is None:
            continue
        entries.append({
            "name": p.name,
            "embedding": [round(float(v), 7) for v in np.asarray(emb).reshape(-1)],
            "speech_seconds": p.speech_seconds,
            "n_segments": p.n_segments,
            "source_urls": p.source_urls,
            "notes": p.notes,
            "enrolled_at": p.enrolled_at,
            "updated_at": p.updated_at,
        })

    pack = {
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "embedding_model": model,
        "embedding_dim": config.embedding.embedding_dim,
        "created_at": datetime.now().isoformat(),
        "profiles": entries,
    }
    Path(path).write_text(json.dumps(pack, indent=1))
    return len(entries)


def import_pack(
    store: ProfileStore,
    path: str | Path,
    on_conflict: str = "skip",
    config: Config | None = None,
) -> ImportResult:
    """
    Load a pack into the store.

    on_conflict — what to do when a profile name already exists locally:
        skip       keep the local profile (default)
        overwrite  replace it with the pack's version
        merge      speech-weighted average of both voiceprints
    """
    if on_conflict not in CONFLICT_MODES:
        raise PackError(f"on_conflict must be one of {CONFLICT_MODES}")
    config = config or store.config
    try:
        pack = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PackError(f"Can't read pack {path}: {e}") from e

    if not isinstance(pack, dict) or pack.get("format") != PACK_FORMAT:
        raise PackError(f"{path} is not a kickvoice profile pack")
    if pack.get("version", 0) > PACK_VERSION:
        raise PackError(f"Pack version {pack.get('version')} is newer than supported")
    model, dim = config.embedding.model_source, config.embedding.embedding_dim
    if pack.get("embedding_model") != model:
        raise PackError(
            f"Pack uses embedding model '{pack.get('embedding_model')}', but this "
            f"install uses '{model}'. Voiceprints from different models can't be mixed."
        )

    # Validate everything first so a bad entry can't leave a half-import.
    entries = []
    for entry in pack.get("profiles", []):
        name = str(entry.get("name", "")).strip() if isinstance(entry, dict) else ""
        try:
            emb = np.asarray(entry.get("embedding", []), dtype=np.float64)
        except (TypeError, ValueError, AttributeError):
            emb = np.zeros(0)
        if not name or emb.shape != (dim,) or not np.all(np.isfinite(emb)) \
                or not np.linalg.norm(emb):
            raise PackError(f"Invalid profile entry in pack: {name or '<unnamed>'}")
        entries.append((name, emb, entry))

    result = ImportResult()
    for name, emb, entry in entries:
        existing = store.find_profile(name)
        if existing and on_conflict == "skip":
            result.skipped.append(existing.name)
            continue
        if existing and on_conflict == "merge":
            store.merge_into_profile(
                existing.name, emb,
                speech_seconds=float(entry.get("speech_seconds") or 0.0),
                n_segments=int(entry.get("n_segments") or 0),
                source_urls=list(entry.get("source_urls") or []),
                note=f"Merged from pack {Path(path).name}",
            )
            result.merged.append(existing.name)
            continue
        if existing:   # overwrite: drop the local one (name may differ in case)
            store.remove_profile(existing.name)
        store.add_profile(
            name=name,
            embedding=emb,
            source_urls=list(entry.get("source_urls") or []),
            n_segments=int(entry.get("n_segments") or 0),
            notes=str(entry.get("notes") or ""),
            speech_seconds=float(entry.get("speech_seconds") or 0.0),
            embedding_model=model,
        )
        (result.overwritten if existing else result.added).append(name)
    return result
