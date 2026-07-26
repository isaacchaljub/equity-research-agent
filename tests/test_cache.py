from pipeline.config import MAX_CACHE_SIZE
from pipeline.retrieval import check_cache
from pipeline.retrieval import query_cache
from pipeline.retrieval import update_cache


def setup_function():
    query_cache.clear()


def teardown_function():
    query_cache.clear()


def test_empty_cache_misses():
    assert check_cache("What is Apple's P/E?") is None


def test_identical_query_hits():
    update_cache("What is Apple's trailing P/E?", "about 31.5")
    assert check_cache("What is Apple's trailing P/E?") == "about 31.5"


def test_unrelated_query_misses():
    update_cache("What is Apple's trailing P/E?", "about 31.5")
    assert check_cache("banana bread baking temperature") is None


def test_cache_evicts_oldest_over_capacity():
    for i in range(MAX_CACHE_SIZE + 5):
        update_cache(f"unique question number {i}", f"answer {i}")
    assert len(query_cache) == MAX_CACHE_SIZE
