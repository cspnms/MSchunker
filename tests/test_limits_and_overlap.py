from smartchunk import chunk_text
from smartchunk.core import _default_token_counter


def test_max_tokens_respected_before_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    max_tokens = 30

    chunks = chunk_text(
        text,
        max_tokens=max_tokens,
        overlap_tokens=0,  # no overlap so we test pure limit
        task="rag",
    )

    counter = _default_token_counter
    assert len(chunks) >= 1
    for ch in chunks:
        assert counter(ch.text) <= max_tokens


def test_overlap_metadata_and_content():
    text = """
    First paragraph with some words that will be overlapped.

    Second paragraph continues the idea and should receive overlap.
    """

    chunks = chunk_text(
        text,
        max_tokens=25,
        overlap_tokens=8,
        task="rag",
    )

    assert len(chunks) >= 2

    first = chunks[0]
    second = chunks[1]

    # First chunk: no overlap-from-prev
    assert first.meta.get("overlap_from_prev") is False
    assert first.meta.get("overlap_tokens") == 0

    # Second chunk: has overlap
    assert second.meta.get("overlap_from_prev") is True
    assert second.meta.get("overlap_tokens") > 0

    # Overlap tokens at end of first appear at beginning of second
    first_tail = " ".join(first.text.split()[-second.meta["overlap_tokens"] :])
    assert first_tail.split()[0] in second.text.split()[0:20]
