import random
import sys
import types
import warnings

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


def test_enforce_overlap_limits_trims_prefix_tokens():
    text = " ".join(f"word{i}" for i in range(12))

    with warnings.catch_warnings(record=True) as caught:
        chunks_untrimmed = chunk_text(
            text,
            max_tokens=3,
            overlap_tokens=2,
            task="rag",
            strategy="fixed",
            enforce_overlap_limits=False,
        )
    assert any("exceeds limits" in str(w.message) for w in caught)
    chunks_trimmed = chunk_text(
        text,
        max_tokens=3,
        overlap_tokens=2,
        task="rag",
        strategy="fixed",
        enforce_overlap_limits=True,
    )

    assert len(chunks_untrimmed) == len(chunks_trimmed)
    if len(chunks_untrimmed) > 1:
        assert len(chunks_untrimmed[1].text.split()) > 3
        assert len(chunks_trimmed[1].text.split()) <= 3
        assert chunks_trimmed[1].meta["overlap_tokens"] < chunks_untrimmed[1].meta["overlap_tokens"]


def test_custom_sentence_splitter_is_honored():
    text = "First part||Second part||Third"

    def splitter(paragraph: str):
        return [s.strip() for s in paragraph.split("||") if s.strip()]

    chunks = chunk_text(
        text,
        max_tokens=50,
        overlap_tokens=0,
        sentence_splitter=splitter,
        task="rag",
    )

    assert len(chunks) == 1
    assert "Second part" in chunks[0].text
    assert chunks[0].meta["split_reason"] == "paragraph_boundary"


def test_default_token_counter_prefers_tiktoken_when_available():
    import mschunker.core as core

    core._TIKTOKEN_ENCODER = None
    fake_encoder_called = {"called": False}

    class FakeEncoder:
        def encode(self, text: str):
            fake_encoder_called["called"] = True
            return list(text)

    fake_module = types.SimpleNamespace(get_encoding=lambda name: FakeEncoder())
    original_module = sys.modules.get("tiktoken")
    sys.modules["tiktoken"] = fake_module

    try:
        count = core._default_token_counter("abc")
        assert fake_encoder_called["called"]
        assert count == 3
    finally:
        core._TIKTOKEN_ENCODER = None
        if original_module is None:
            sys.modules.pop("tiktoken", None)
        else:
            sys.modules["tiktoken"] = original_module


def test_heading_detection_handles_markdown_and_underline():
    text = """
    # Intro

    First paragraph under intro.

    Underline Heading
    -----------------
    Second section body text.
    """

    chunks = chunk_text(text, max_tokens=30, overlap_tokens=0, task="rag")
    headings = {ch.meta.get("section_heading") for ch in chunks}

    assert "Intro" in headings
    assert "Underline Heading" in headings


def test_sentence_overflow_splits_with_sentence_reason():
    text = "One short. Second sentence is intentionally quite a bit longer than the first. Third sentence follows."

    chunks = chunk_text(text, max_tokens=8, overlap_tokens=0, task="rag")

    assert any(ch.meta.get("split_reason") == "sentence_limit" for ch in chunks)


def test_metadata_completeness_across_strategies():
    random.seed(0)
    base_text = "# Heading\n\n" + " ".join(f"token{i}" for i in range(25))
    required_keys = {
        "split_reason",
        "strategy",
        "chunk_index",
        "overlap_from_prev",
        "overlap_tokens",
    }

    for strategy in ("auto", "fixed"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunks = chunk_text(
                base_text,
                max_tokens=10,
                overlap_tokens=1,
                task="rag",
                strategy=strategy,
            )
        assert chunks, strategy
        for ch in chunks:
            assert required_keys.issubset(ch.meta.keys())
            assert ch.meta["strategy"] == strategy
            assert isinstance(ch.meta["overlap_from_prev"], bool)
            assert isinstance(ch.meta["overlap_tokens"], int)

