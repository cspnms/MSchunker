from mschunker import chunk_text, analyze_chunks, explain_chunk


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


def test_long_text_does_not_crash_and_produces_chunks():
    # Long flat text – MSchunker may return 1 or many chunks;
    # we only require that it works and chunks are non-empty.
    text = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_text(
        text,
        max_tokens=50,
        overlap_tokens=10,
        task="rag",
    )

    assert len(chunks) >= 1
    for ch in chunks:
        assert ch.text.strip() != ""
        assert "split_reason" in ch.meta
        assert "strategy" in ch.meta


def test_overlap_metadata_is_present_when_enabled():
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(
        text,
        max_tokens=50,
        overlap_tokens=10,
        task="rag",
    )

    # We don't force multiple chunks; we just check metadata is coherent.
    for idx, ch in enumerate(chunks):
        assert "overlap_from_prev" in ch.meta
        assert "overlap_tokens" in ch.meta
        if idx == 0:
            # first chunk should not be marked as overlapped from previous
            assert ch.meta["overlap_from_prev"] in (False, None)
        else:
            # later chunks may or may not have overlap depending on strategy,
            # but the flag and count must be valid types.
            assert isinstance(ch.meta["overlap_from_prev"], bool)
            assert isinstance(ch.meta["overlap_tokens"], int)


def test_analyze_chunks_stats_is_consistent():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."

    chunks = chunk_text(text, max_tokens=16, overlap_tokens=0, task="rag")
    stats = analyze_chunks(chunks)

    assert stats["num_chunks"] == len(chunks)
    assert stats["min_tokens"] > 0
    assert stats["max_tokens"] >= stats["min_tokens"]
    assert stats["avg_tokens"] > 0
