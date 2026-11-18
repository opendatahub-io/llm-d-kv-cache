"""Unit tests for crawler path discovery (post-#611 layout walker)."""

import os
import queue
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processes.crawler import (  # noqa: E402
    RESCAN_DELAY_MAX_SECONDS,
    RESCAN_DELAY_MIN_SECONDS,
    crawler_process,
    get_hex_modulo_ranges,
    hex_to_int,
    next_rescan_delay,
    put_until_accepted,
    stream_cache_files_with_mapper,
    wait_for_queue_slot,
)


class TestHexHelpers:
    def test_hex_to_int_valid(self):
        assert hex_to_int("abc") == 0xABC
        assert hex_to_int("000") == 0

    def test_hex_to_int_invalid(self):
        assert hex_to_int("not_hex") is None


class TestStreamCacheFilesWithMapper:
    def test_stream_new_layout_yields_bins(self, new_layout_cache: Path):
        paths = sorted(stream_cache_files_with_mapper(new_layout_cache))
        names = {p.name for p in paths}
        assert names == {"block1.bin", "block2.bin", "other.bin", "rank1.bin"}
        assert "skip.bin" not in names

    def test_skips_base_dir_without_rank_suffix(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        assert all("config.json" not in str(p) for p in paths)
        assert all(p.suffix == ".bin" for p in paths)

    def test_hex_modulo_filter(self, new_layout_cache: Path):
        # abc: 0xabc % 16 == 12; def: 0xdef % 16 == 15; 000: % 16 == 0
        paths_mod_12 = list(stream_cache_files_with_mapper(new_layout_cache, (12, 12)))
        assert {p.name for p in paths_mod_12} == {"block1.bin", "block2.bin"}

        paths_mod_0 = list(stream_cache_files_with_mapper(new_layout_cache, (0, 0)))
        assert {p.name for p in paths_mod_0} == {"rank1.bin"}

    def test_multiple_ranks(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        rank_dirs = {p.parts[-4] for p in paths}
        assert "model_abc123def456_r0" in rank_dirs
        assert "model_abc123def456_r1" in rank_dirs

    def test_missing_cache_path(self, tmp_path: Path):
        missing = tmp_path / "missing"
        assert list(stream_cache_files_with_mapper(missing)) == []

    def test_nested_rank_dir_is_discovered(self, tmp_path: Path):
        """Rank dirs may sit under extra prefixes; _iter_rank_dirs walks recursively."""
        cache = tmp_path / "cache"
        rank = cache / "extra" / "nested" / "model_abc123def456_r0"
        (rank / "abc" / "de_g0").mkdir(parents=True)
        (rank / "abc" / "de_g0" / "nested.bin").write_bytes(b"n")

        paths = list(stream_cache_files_with_mapper(cache))
        assert {p.name for p in paths} == {"nested.bin"}

    def test_malformed_first_level_bucket_skipped(self, new_layout_cache: Path):
        paths = list(stream_cache_files_with_mapper(new_layout_cache))
        assert all(p.name != "skip.bin" for p in paths)


class TestHexModuloRanges:
    def test_eight_processes(self):
        ranges = get_hex_modulo_ranges(8)
        assert len(ranges) == 8
        assert ranges[0] == (0, 1)
        assert ranges[7] == (14, 15)

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError):
            get_hex_modulo_ranges(3)


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


class TestEmptyFolderDiscovery:
    def test_empty_hex3_offered_only_by_owning_crawler(self, tmp_path: Path):
        cache = tmp_path / "cache"
        rank = cache / "model_abc123def456_r0"
        (rank / "abc" / "de_g0").mkdir(parents=True)
        (rank / "abc" / "de_g0" / "keep.bin").write_bytes(b"k")
        (rank / "00f").mkdir()  # 0x00f % 16 == 15, empty

        owner: queue.Queue[str] = queue.Queue()
        other: queue.Queue[str] = queue.Queue()
        list(stream_cache_files_with_mapper(cache, (15, 15), folder_queue=owner))
        list(stream_cache_files_with_mapper(cache, (12, 12), folder_queue=other))

        assert list(owner.queue) == [str(rank / "00f")]
        assert list(other.queue) == []

    def test_empty_leaf_dir_offered(self, tmp_path: Path):
        cache = tmp_path / "cache"
        rank = cache / "model_abc123def456_r0"
        (rank / "abc" / "de_g0").mkdir(parents=True)
        (rank / "abc" / "de_g0" / "keep.bin").write_bytes(b"k")
        (rank / "abc" / "ff_g0").mkdir()

        folders: queue.Queue[str] = queue.Queue()
        names = {p.name for p in stream_cache_files_with_mapper(cache, folder_queue=folders)}

        assert names == {"keep.bin"}
        assert list(folders.queue) == [str(rank / "abc" / "ff_g0")]

    def test_ttl_keeps_fresh_empty_dirs_out(self, tmp_path: Path):
        cache = tmp_path / "cache"
        rank = cache / "model_abc123def456_r0"
        (rank / "abc" / "de_g0").mkdir(parents=True)
        (rank / "abc" / "de_g0" / "keep.bin").write_bytes(b"k")
        (rank / "abc" / "ff_g0").mkdir()

        folders: queue.Queue[str] = queue.Queue()
        list(stream_cache_files_with_mapper(cache, folder_queue=folders, dir_cleanup_ttl_seconds=3600))
        assert list(folders.queue) == []


class TestWaitForQueueSlot:
    def test_returns_immediately_below_target(self):
        q: queue.Queue[str] = queue.Queue()
        assert wait_for_queue_slot(q, threading.Event(), threading.Event(), 1, 10)

    def test_off_blocks_at_min_until_deletion_turns_on(self):
        q: queue.Queue[str] = queue.Queue()
        q.put("a")
        deletion, shutdown = threading.Event(), threading.Event()
        result: list[bool] = []

        t = threading.Thread(target=lambda: result.append(wait_for_queue_slot(q, deletion, shutdown, 1, 10)))
        t.start()
        t.join(timeout=0.3)
        assert t.is_alive(), "should block while OFF and queue is at min size"

        deletion.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert result == [True]

    def test_on_blocks_at_max_until_drained(self):
        q: queue.Queue[str] = queue.Queue()
        q.put("a")
        q.put("b")
        deletion, shutdown = threading.Event(), threading.Event()
        deletion.set()
        result: list[bool] = []

        t = threading.Thread(target=lambda: result.append(wait_for_queue_slot(q, deletion, shutdown, 1, 2)))
        t.start()
        t.join(timeout=0.3)
        assert t.is_alive(), "should block while ON and queue is at max size"

        q.get()
        t.join(timeout=2.0)
        assert result == [True]

    def test_shutdown_returns_false(self):
        q: queue.Queue[str] = queue.Queue()
        q.put("a")
        deletion, shutdown = threading.Event(), threading.Event()
        shutdown.set()
        assert not wait_for_queue_slot(q, deletion, shutdown, 1, 10)

    def test_on_wait_called_while_blocked(self):
        q: queue.Queue[str] = queue.Queue()
        q.put("a")
        deletion, shutdown = threading.Event(), threading.Event()
        calls: list[int] = []

        def on_wait() -> None:
            calls.append(1)
            if len(calls) == 2:
                shutdown.set()

        assert not wait_for_queue_slot(q, deletion, shutdown, 1, 10, on_wait=on_wait)
        assert len(calls) == 2


class TestPutUntilAccepted:
    def test_puts_when_room(self):
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        assert put_until_accepted(q, "a", threading.Event())
        assert q.get_nowait() == "a"

    def test_full_queue_retries_until_room(self):
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        q.put("a")
        threading.Timer(0.2, q.get).start()
        assert put_until_accepted(q, "b", threading.Event())
        assert list(q.queue) == ["b"]

    def test_full_queue_returns_false_on_shutdown(self):
        q: queue.Queue[str] = queue.Queue(maxsize=1)
        q.put("a")
        shutdown = threading.Event()
        shutdown.set()
        assert not put_until_accepted(q, "b", shutdown)


class TestNextRescanDelay:
    def test_resets_after_productive_sweep(self):
        assert next_rescan_delay(RESCAN_DELAY_MAX_SECONDS, 5) == RESCAN_DELAY_MIN_SECONDS

    def test_doubles_after_idle_sweep(self):
        assert next_rescan_delay(1.0, 0) == 2.0

    def test_capped(self):
        assert next_rescan_delay(RESCAN_DELAY_MAX_SECONDS, 0) == RESCAN_DELAY_MAX_SECONDS


class TestCrawlerProcess:
    def _run(self, cache: Path, file_queue, deletion, shutdown, min_q: int, max_q: int) -> threading.Thread:
        config = {
            "log_level": "WARNING",
            "file_queue_min_size": min_q,
            "file_queue_maxsize": max_q,
            "file_access_time_threshold_minutes": 1.0,
            "hex_bucket_len": 3,
            "dir_cleanup_ttl_seconds": 0.0,
        }
        t = threading.Thread(
            target=crawler_process,
            args=(0, (0, 15), cache, config, deletion, file_queue, queue.Queue(), shutdown),
            daemon=True,
        )
        t.start()
        return t

    def _cold_cache(self, tmp_path: Path, count: int) -> Path:
        cache = tmp_path / "cache"
        leaf = cache / "model_abc123def456_r0" / "abc" / "de_g0"
        leaf.mkdir(parents=True)
        for i in range(count):
            f = leaf / f"{i:016x}.bin"
            f.write_bytes(b"x")
            _age(f, 3600)
        return cache

    def test_prefills_to_min_then_holds_while_off(self, tmp_path: Path):
        cache = self._cold_cache(tmp_path, 10)
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        deletion, shutdown = threading.Event(), threading.Event()

        t = self._run(cache, q, deletion, shutdown, min_q=3, max_q=100)
        try:
            deadline = time.time() + 5
            while q.qsize() < 3 and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.5)
            assert q.qsize() == 3
        finally:
            shutdown.set()
            t.join(timeout=5)
        assert not t.is_alive()

    def test_resumes_walk_when_deletion_turns_on(self, tmp_path: Path):
        cache = self._cold_cache(tmp_path, 10)
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        deletion, shutdown = threading.Event(), threading.Event()

        t = self._run(cache, q, deletion, shutdown, min_q=3, max_q=100)
        try:
            deadline = time.time() + 5
            while q.qsize() < 3 and time.time() < deadline:
                time.sleep(0.05)
            deletion.set()
            deadline = time.time() + 5
            while q.qsize() < 10 and time.time() < deadline:
                time.sleep(0.05)
            assert len(set(q.queue)) == 10
        finally:
            shutdown.set()
            t.join(timeout=5)

    def test_hot_files_not_queued(self, tmp_path: Path):
        cache = self._cold_cache(tmp_path, 2)
        leaf = cache / "model_abc123def456_r0" / "abc" / "de_g0"
        (leaf / "ffffffffffffffff.bin").write_bytes(b"h")
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        deletion, shutdown = threading.Event(), threading.Event()
        deletion.set()

        t = self._run(cache, q, deletion, shutdown, min_q=100, max_q=100)
        try:
            deadline = time.time() + 5
            while q.qsize() < 2 and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(0.3)
            assert {Path(p).name for p in q.queue} == {f"{0:016x}.bin", f"{1:016x}.bin"}
        finally:
            shutdown.set()
            t.join(timeout=5)
