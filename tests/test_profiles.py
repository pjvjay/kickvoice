import json

import numpy as np
import pytest

from kickvoice.storage.packs import PackError, export_pack, import_pack
from kickvoice.storage.profiles import ProfileStore


def unit(seed, dim=192):
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)


def test_filenames_are_safe_and_case_distinct(config):
    store = ProfileStore(config)
    a = store.add_profile("xQc", unit(1))
    b = store.add_profile("XQC", unit(2))
    c = store.add_profile("../evil/name", unit(3))
    assert a.embedding_file != b.embedding_file
    assert "/" not in c.embedding_file and not c.embedding_file.startswith(".")
    reloaded = ProfileStore(config)
    assert np.allclose(reloaded.get_embedding("xQc"), unit(1))
    assert np.allclose(reloaded.get_embedding("XQC"), unit(2))


def test_re_adding_reuses_file(config):
    store = ProfileStore(config)
    f1 = store.add_profile("Amy", unit(1)).embedding_file
    f2 = store.add_profile("Amy", unit(2)).embedding_file
    assert f1 == f2


def test_legacy_index_loads(config):
    """Index written by the previous version (no new fields) + unknown keys."""
    pdir = config.profiles_dir
    np.save(pdir / "drdisrespect.npy", unit(5))
    (pdir / "index.json").write_text(json.dumps({"DrDisrespect": {
        "name": "DrDisrespect", "embedding_file": "drdisrespect.npy",
        "enrolled_at": "2024-01-01T00:00:00", "source_urls": ["x"], "n_segments": 12,
        "notes": "", "some_future_field": 1,
    }}))
    store = ProfileStore(config)
    p = store.get_profile("DrDisrespect")
    assert p.speech_seconds == 0.0 and p.embedding_model == ""
    assert store.find_profile("drdisrespect") is p


def test_merge_is_speech_weighted(config):
    store = ProfileStore(config)
    a, b = unit(1), unit(2)
    store.add_profile("Amy", a, speech_seconds=300)
    store.merge_into_profile("Amy", b, speech_seconds=100, n_segments=5,
                             source_urls=["https://kick.com/amy"], note="from stream")
    expected = (a * 300 + b * 100) / 400
    expected /= np.linalg.norm(expected)
    got = ProfileStore(config)
    assert np.allclose(got.get_embedding("Amy"), expected)
    p = got.get_profile("Amy")
    assert p.speech_seconds == 400 and p.n_segments == 5
    assert "https://kick.com/amy" in p.source_urls and "from stream" in p.notes
    assert p.updated_at


def test_merge_legacy_profile_equal_weight(config):
    store = ProfileStore(config)
    a, b = unit(1), unit(2)
    store.add_profile("Old", a)          # speech_seconds unknown (0)
    store.merge_into_profile("Old", b, speech_seconds=50)
    expected = (a + b) / np.linalg.norm(a + b)
    assert np.allclose(store.get_embedding("Old"), expected)


def test_merge_unknown_profile_raises(config):
    with pytest.raises(KeyError):
        ProfileStore(config).merge_into_profile("nobody", unit(1))


# ── packs ────────────────────────────────────────────────────────

def test_pack_roundtrip(config, tmp_path):
    src = ProfileStore(config)
    src.add_profile("Amy", unit(1), source_urls=["u1"], speech_seconds=120, notes="n")
    src.add_profile("Bob", unit(2), speech_seconds=60)
    pack = tmp_path / "pack.json"
    assert export_pack(src, pack) == 2

    config.profiles_dir = tmp_path / "other"
    dst = ProfileStore(config)
    r = import_pack(dst, pack)
    assert sorted(r.added) == ["Amy", "Bob"]
    assert np.allclose(dst.get_embedding("Amy"), unit(1), atol=1e-6)
    assert dst.get_profile("Amy").speech_seconds == 120
    assert dst.get_profile("Amy").source_urls == ["u1"]


def test_pack_export_selected_and_missing(config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("Amy", unit(1))
    store.add_profile("Bob", unit(2))
    assert export_pack(store, tmp_path / "p.json", names=["amy"]) == 1
    with pytest.raises(PackError):
        export_pack(store, tmp_path / "p.json", names=["Zed"])


def test_pack_model_mismatch_refused(config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("Amy", unit(1))
    export_pack(store, tmp_path / "p.json")
    config.embedding.model_source = "some/other-model"
    with pytest.raises(PackError, match="embedding model"):
        import_pack(ProfileStore(config), tmp_path / "p.json")


def test_pack_invalid_entry_imports_nothing(config, tmp_path):
    store = ProfileStore(config)
    store.add_profile("Amy", unit(1))
    export_pack(store, tmp_path / "p.json")
    data = json.loads((tmp_path / "p.json").read_text())
    data["profiles"].append({"name": "Broken", "embedding": [1.0, 2.0]})
    (tmp_path / "p.json").write_text(json.dumps(data))
    config.profiles_dir = tmp_path / "fresh"
    fresh = ProfileStore(config)
    with pytest.raises(PackError, match="Broken"):
        import_pack(fresh, tmp_path / "p.json")
    assert fresh.count == 0


def test_pack_not_a_pack(config, tmp_path):
    (tmp_path / "x.json").write_text("[1, 2]")
    with pytest.raises(PackError):
        import_pack(ProfileStore(config), tmp_path / "x.json")


@pytest.mark.parametrize("mode", ["skip", "overwrite", "merge"])
def test_pack_conflicts(config, tmp_path, mode):
    theirs = ProfileStore(config)
    theirs.add_profile("Amy", unit(2), speech_seconds=100)
    export_pack(theirs, tmp_path / "p.json")

    config.profiles_dir = tmp_path / "mine"
    mine = ProfileStore(config)
    mine.add_profile("amy", unit(1), speech_seconds=100)   # differs only by case
    r = import_pack(mine, tmp_path / "p.json", on_conflict=mode)

    if mode == "skip":
        assert r.skipped == ["amy"]
        assert np.allclose(mine.get_embedding("amy"), unit(1))
    elif mode == "overwrite":
        assert r.overwritten == ["Amy"] and mine.count == 1
        assert np.allclose(mine.get_embedding("Amy"), unit(2), atol=1e-6)
    else:
        assert r.merged == ["amy"]
        expected = (unit(1) + unit(2)) / np.linalg.norm(unit(1) + unit(2))
        assert np.allclose(mine.get_embedding("amy"), expected, atol=1e-6)
