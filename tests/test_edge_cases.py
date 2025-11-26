from smartchunk import chunk_text


def test_empty_text_returns_empty_list():
    assert chunk_text("", max_tokens=10) == []
    assert chunk_text("   \n   ", max_tokens=10) == []


def test_single_long_sentence_hard_split():
    # one very long sentence, no punctuation
    text = " ".join(["token"] * 150)

    chunks = chunk_text(
        text,
        max_tokens=40,
        overlap_tokens=0,
        task="rag",
    )

    assert len(chunks) > 1

    for ch in chunks:
        assert ch.text.strip() != ""
        assert ch.meta["split_reason"] in (
            "hard_limit",
            "sentence_limit",
            "paragraph_boundary",
        )


def test_fixed_strategy_uses_hard_limit():
    text = " ".join(f"word{i}" for i in range(120))

    chunks = chunk_text(
        text,
        max_tokens=30,
        overlap_tokens=0,
        strategy="fixed",
    )

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.meta["split_reason"] == "hard_limit"
        assert ch.meta["strategy"] == "fixed"
