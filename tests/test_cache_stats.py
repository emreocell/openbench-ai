"""`ResponseCache.stats` and `openbench cache stats`: what is in the cache and how old it is."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.cache import CacheRecord, ResponseCache
from openbench.cli import main
from openbench.exceptions import OpenBenchError

OLDEST = "2026-01-15T08:00:00+00:00"
NEWEST = "2026-06-20T17:30:00+00:00"
OLD_KEYS = {"enabled", "cache_dir", "stored_entries", "session_hits", "session_misses"}
NEW_KEYS = {"by_model", "oldest_cached_at", "newest_cached_at", "unknown_age_entries", "file_size_bytes"}


def _entry(key: str, provider: str, model: str, cached_at: str) -> dict:
    return CacheRecord(
        cache_key=key, provider_name=provider, model_name=model, model_version="1", dataset_name="d",
        dataset_version="1", record_id=key, prompt_sha256="a" * 64, generation_config_sha256="b" * 64,
        response="x", latency_ms=None, token_usage=None, cached_at=cached_at,
    ).to_dict()


def _write(cache_dir: Path, entries: list[dict], raw_lines: tuple[str, ...] = ()) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "cache_entries.jsonl"
    path.write_text("\n".join([json.dumps(e) for e in entries] + list(raw_lines)) + "\n", encoding="utf-8")
    return path


def _two_providers(cache_dir: Path) -> Path:
    return _write(cache_dir, [
        _entry("1", "openai", "gpt-b", "2026-03-01T00:00:00+00:00"),
        _entry("2", "mock", "mock-v1", NEWEST),
        _entry("3", "openai", "gpt-a", OLDEST),
        _entry("4", "openai", "gpt-b", "2026-04-01T00:00:00+00:00"),
        _entry("5", "mock", "mock-v1", "2026-02-01T00:00:00+00:00"),
    ])


def test_stats_of_an_empty_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stats = ResponseCache(cache_dir).stats()
    assert stats["stored_entries"] == 0 and stats["by_model"] == []
    assert stats["oldest_cached_at"] is None and stats["newest_cached_at"] is None
    assert stats["unknown_age_entries"] == 0 and stats["file_size_bytes"] == 0


def test_stats_with_two_providers_and_models(tmp_path):
    path = _two_providers(tmp_path)
    stats = ResponseCache(tmp_path).stats()
    assert OLD_KEYS | NEW_KEYS == set(stats)
    assert stats["stored_entries"] == 5 and stats["cache_dir"] == str(tmp_path)
    assert stats["enabled"] is True and stats["session_hits"] == 0 and stats["session_misses"] == 0
    assert stats["by_model"] == [
        {"provider": "mock", "model": "mock-v1", "entries": 2},
        {"provider": "openai", "model": "gpt-a", "entries": 1},
        {"provider": "openai", "model": "gpt-b", "entries": 2},
    ]
    assert stats["oldest_cached_at"] == OLDEST and stats["newest_cached_at"] == NEWEST
    assert stats["unknown_age_entries"] == 0
    assert stats["file_size_bytes"] == path.stat().st_size > 0


def test_stats_oldest_and_newest_compare_instants_not_text(tmp_path):
    _write(tmp_path, [
        _entry("1", "p", "m", "2026-06-01T10:00:00+02:00"),  # 08:00 UTC: the oldest instant
        _entry("2", "p", "m", "2026-06-01T09:00:00+00:00"),
        _entry("3", "p", "m", "2026-06-01T08:30:00Z"),
    ])
    stats = ResponseCache(tmp_path).stats()
    assert stats["oldest_cached_at"] == "2026-06-01T10:00:00+02:00"
    assert stats["newest_cached_at"] == "2026-06-01T09:00:00+00:00"


def test_stats_counts_an_unreadable_cached_at_separately(tmp_path):
    _write(tmp_path, [
        _entry("1", "p", "m", "last tuesday"),
        _entry("2", "p", "m", ""),
        _entry("3", "p", "m", OLDEST),
    ])
    stats = ResponseCache(tmp_path).stats()
    assert stats["stored_entries"] == 3 and stats["unknown_age_entries"] == 2
    assert stats["by_model"] == [{"provider": "p", "model": "m", "entries": 3}]
    assert stats["oldest_cached_at"] == OLDEST == stats["newest_cached_at"]


def test_stats_when_no_entry_has_a_readable_cached_at(tmp_path):
    _write(tmp_path, [_entry("1", "p", "m", "never")])
    stats = ResponseCache(tmp_path).stats()
    assert stats["oldest_cached_at"] is None and stats["newest_cached_at"] is None
    assert stats["unknown_age_entries"] == 1


def test_stats_counts_each_key_once(tmp_path):
    _write(tmp_path, [_entry("a", "p", "m", OLDEST), _entry("a", "p", "m", NEWEST)])
    stats = ResponseCache(tmp_path).stats()
    assert stats["stored_entries"] == 1 and stats["by_model"][0]["entries"] == 1
    assert stats["oldest_cached_at"] == NEWEST


def test_stats_never_creates_the_cache_folder(tmp_path):
    missing = tmp_path / "missing" / "cache"
    stats = ResponseCache(missing).stats()
    assert stats["stored_entries"] == 0 and stats["file_size_bytes"] == 0 and stats["by_model"] == []
    assert main(["cache", "stats", "--cache-dir", str(missing)]) == 0
    assert main(["cache", "stats", "--format", "text", "--cache-dir", str(missing)]) == 0
    assert not (tmp_path / "missing").exists()


def test_stats_reports_a_corrupted_cache_file(tmp_path):
    _write(tmp_path, [_entry("1", "p", "m", OLDEST)], raw_lines=("{broken",))
    with pytest.raises(OpenBenchError, match="Corrupted cache file"):
        ResponseCache(tmp_path).stats()


def test_cli_cache_stats_json_is_the_default_and_keeps_the_old_keys(tmp_path, capsys):
    _two_providers(tmp_path)
    assert main(["cache", "stats", "--cache-dir", str(tmp_path)]) == 0
    default_out = capsys.readouterr().out
    assert main(["cache", "stats", "--format", "json", "--cache-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == default_out
    payload = json.loads(default_out)
    assert OLD_KEYS | NEW_KEYS == set(payload)
    assert payload["stored_entries"] == 5
    assert payload["by_model"][0] == {"provider": "mock", "model": "mock-v1", "entries": 2}


def test_cli_cache_stats_text_has_one_line_per_provider_and_model(tmp_path, capsys):
    path = _two_providers(tmp_path)
    assert main(["cache", "stats", "--format", "text", "--cache-dir", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        f"Cache: {tmp_path}",
        f"Entries: 5 ({path.stat().st_size} bytes on disk)",
        f"Oldest entry: {OLDEST}",
        f"Newest entry: {NEWEST}",
        "By provider and model:",
        "  mock / mock-v1: 2",
        "  openai / gpt-a: 1",
        "  openai / gpt-b: 2",
        "This session: 0 hit(s), 0 miss(es)",
    ]


def test_cli_cache_stats_text_for_empty_and_unreadable_entries(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["cache", "stats", "--format", "text", "--cache-dir", str(empty)]) == 0
    out = capsys.readouterr().out
    assert "Entries: 0 (0 bytes on disk)" in out and "Oldest entry: unknown" in out and "  (none)" in out
    assert "unreadable" not in out
    _write(tmp_path / "odd", [_entry("1", "p", "m", "soon")])
    assert main(["cache", "stats", "--format", "text", "--cache-dir", str(tmp_path / "odd")]) == 0
    out = capsys.readouterr().out
    assert "Entries with an unreadable cached_at: 1" in out and "  p / m: 1" in out


def test_cli_cache_stats_reports_a_corrupted_cache(tmp_path, capsys):
    _write(tmp_path, [], raw_lines=("{broken",))
    assert main(["cache", "stats", "--cache-dir", str(tmp_path)]) == 1
    assert "Corrupted cache file" in capsys.readouterr().err


def test_cli_cache_stats_rejects_an_unknown_format(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["cache", "stats", "--format", "yaml", "--cache-dir", str(tmp_path)])
    assert excinfo.value.code == 2
