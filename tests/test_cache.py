"""Tests for the content-addressed perception cache."""

from deepsight.cache import PerceptionCache


def test_put_get_roundtrip():
    cache = PerceptionCache(ttl_seconds=3600)
    cache.put("hash1", None, "sketch", "hello")
    assert cache.get("hash1", None, "sketch") == "hello"


def test_region_is_part_of_key():
    cache = PerceptionCache()
    cache.put("hash1", None, "look", "full")
    cache.put("hash1", (0, 0, 10, 10), "look", "region")
    assert cache.get("hash1", None, "look") == "full"
    assert cache.get("hash1", (0, 0, 10, 10), "look") == "region"


def test_kind_is_part_of_key():
    cache = PerceptionCache()
    cache.put("hash1", None, "sketch", "sketch answer")
    cache.put("hash1", None, "ocr", "ocr answer")
    assert cache.get("hash1", None, "sketch") == "sketch answer"
    assert cache.get("hash1", None, "ocr") == "ocr answer"


def test_missing_key_returns_none():
    cache = PerceptionCache()
    assert cache.get("nope", None, "sketch") is None


def test_expired_entry_dropped():
    cache = PerceptionCache(ttl_seconds=-1)
    cache.put("hash1", None, "sketch", "stale")
    assert cache.get("hash1", None, "sketch") is None
    assert len(cache) == 0


def test_clear():
    cache = PerceptionCache()
    cache.put("a", None, "sketch", "1")
    cache.put("b", None, "sketch", "2")
    cache.clear()
    assert len(cache) == 0


def test_image_hash_deterministic():
    a = PerceptionCache.image_hash(b"data")
    b = PerceptionCache.image_hash(b"data")
    c = PerceptionCache.image_hash(b"other")
    assert a == b
    assert a != c
    assert len(a) == 64
