from smartchunk import chunk_text, analyze_chunks, explain_chunk


def test_basic_chunking_structure():
    text = """
    # Intro

    First paragraph. It has two sentences.

    Second paragraph here. It also has two sentences.
    """

    chunks = chunk_text(text, max_tokens=20, overlap_tokens=0, task="rag")

    # We should get at least one chunk
    assert len(chunks) >= 1

    for ch in chunks:
        assert isinstance(ch.text, str)
        assert ch.text.strip() != ""
        # metadata must exist and contain split_reason + strategy
        assert "split_reason" in ch.meta
        assert "strategy" in ch.meta
        # explain_chunk should not crash and must return a string
        explanation = explain_chunk(ch)
        assert isinstance(explanation, str)
        assert explanation != ""


def test_analyze_chunks_stats():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text, max_tokens=8, overlap_tokens=0, task="rag")

    stats = analyze_chunks(chunks)
    assert stats["num_chunks"] == len(chunks)
    assert stats["min_tokens"] > 0
    assert stats["max_tokens"] >= stats["min_tokens"]
    assert stats["avg_tokens"] > 0
