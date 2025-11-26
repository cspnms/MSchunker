from smartchunk import chunk_text, analyze_chunks, explain_chunk
from smartchunk.core import _default_token_counter


def test_basic_chunking_and_metadata():
    text = """
    # Intro

    First paragraph. It has two sentences.

    Second paragraph here. It also has two sentences.
    """

    chunks = chunk_text(text, max_tokens=40, overlap_tokens=0, task="rag")

    assert len(chunks) >= 1

    for ch in chunks:
        assert ch.text.strip() != ""
        assert "split_reason" in ch.meta
        assert "strategy" in ch.meta
        explanation = explain_chunk(ch)
        assert isinstance(explanation, str)
        assert explanation != ""


def test_empty_text_returns_empty_list():
    assert chunk_text("", max_tokens=10) == []
    assert chunk_text("   \n   ", max_tokens=10) == []


def test_hard_split_respects_max_tokens_before_overlap():
    # Force multiple chunks by making text extremely long and setting max_tokens very low
    text = " ".join(f"word{i}" for i in range(300))
    max_tokens = 10  # <-- MAX TOKENS VERY SMALL TO GUARANTEE MULTIPLE CHUNKS

    chunks = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_tokens=0,
        task="rag",
    )

    counter = _default_token_counter

    assert len(chunks) > 1  # NOW CORRECT BECAUSE max_tokens IS VERY SMALL

    for ch in chunks:
        assert counter(ch.text) <= max_tokens


def test_overlap_applied_and_marked():
    # Again, force multiple chunks by using very small max_tokens.
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(
        text,
        max_tokens=10,   # VERY SMALL so multiple chunks are guaranteed
        overlap_tokens=3,
        task="rag",
    )

    assert len(chunks) > 1

    first = chunks[0]
    assert first.meta.get("overlap_from_prev") is False
    assert first.meta.get("overlap_tokens") == 0

    # Overlap should appear in subsequent chunks
    for prev, current in zip(chunks[:-1], chunks[1:]):
        assert current.meta.get("overlap_from_prev") is True
        assert current.meta.get("overlap_tokens") > 0

        overlap_n = current.meta["overlap_tokens"]
        prev_tail = " ".join(prev.text.split()[-overlap_n:])
        assert prev_tail.split()[0] in current.text.split()[0:20]


def test_analyze_chunks_stats_is_consistent():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."

    chunks = chunk_text(text, max_tokens=8, overlap_tokens=0, task="rag")
    stats = analyze_chunks(chunks)

    assert stats["num_chunks"] == len(chunks)
    assert stats["min_tokens"] > 0
    assert stats["max_tokens"] >= stats["min_tokens"]
    assert stats["avg_tokens"] > 0
