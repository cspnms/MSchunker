from mschunker import chunk_text

text = """Sentence one. Sentence two. Sentence three."""

chunks = chunk_text(
    text,
    max_tokens=10,
    overlap_tokens=0,
    task="rag",
)

for ch in chunks:
    print("---- CHUNK ----")
    print(ch.text)
    print("META:", ch.meta)
    print()
