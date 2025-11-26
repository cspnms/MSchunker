from smartchunk import chunk_text, analyze_chunks, explain_chunk
from smartchunk.core import _default_token_counter


def test_basic_chunking_and_metadata():
    text = """
    # Intro

    First paragraph. It has two sentences.

    Second paragraph here. It also has two sentences.
    """

    chunks = chunk_text(text, max_tokens=40, overlap_tokens=0, task="rag")

    # At least one chunk must be produced
    assert len(chunks) >= 1

    for ch in chunks:
        # Non-empty text
        assert ch.text.strip() != ""
        # Basic metadata must exist
        assert "split_reason" in ch.meta
        assert "strategy" in ch.meta
        # explain_chunk must return a non-empty string
        explanation = explain_chunk(ch)
        assert isinstance(explanation, str)
        assert explanation != ""


def test_empty_text_returns_empty_list():
    assert chunk_text("", max_tokens=10) == []
    assert chunk_text("   \n   ", max_tokens=10) == []


def test_hard_split_respects_max_tokens_before_overlap():
    # Flat sequence of 300 tokens, no structure.
    text = " ".join(f"word{i}" for i in range(300))
    max_tokens = 50

    chunks = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_tokens=0,  # no overlap so we test pure pre-overlap behavior
        task="rag",
    )

    counter = _default_token_counter

    # There should be more than one chunk
    assert len(chunks) > 1

    # Every chunk must be at or below the max token limit
    for ch in chunks:
        assert counter(ch.text) <= max_tokens


def test_overlap_applied_and_marked():
    # Same long sequence, but now with overlap enabled
    text = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_text(
        text,
        max_tokens=50,
        overlap_tokens=10,
        task="rag",
    )

    # We expect multiple chunks here
    assert len(chunks) > 1

    # First chunk should have no overlap-from-prev
    first = chunks[0]
    assert first.meta.get("overlap_from_prev") is False
    assert first.meta.get("overlap_tokens") == 0

    # Subsequent chunks should show overlap
    for ch in chunks[1:]:
        assert ch.meta.get("overlap_from_prev") is True
        assert ch.meta.get("overlap_tokens") > 0

    # Additionally, check that the overlap tokens from the previous chunk
    # appear at the beginning region of the current chunk.
    for prev, current in zip(chunks[:-1], chunks[1:]):
        overlap_n = current.meta["overlap_tokens"]
        if overlap_n <= 0:
            continue
        prev_tail = " ".join(prev.text.split()[-overlap_n:])
        # We don't require an exact equality of whole string, but first token should match
        assert prev_tail.split()[0] in current.text.split()[0:20]


def test_analyze_chunks_stats_is_consistent():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."

    chunks = chunk_text(text, max_tokens=8, overlap_tokens=0, task="rag")
    stats = analyze_chunks(chunks)

    assert stats["num_chunks"] == len(chunks)
    assert stats["min_tokens"] > 0
    assert stats["max_tokens"] >= stats["min_tokens"]
    assert stats["avg_tokens"] > 0
