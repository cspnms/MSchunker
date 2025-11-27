from mschunker import chunk_text

text = """
# Intro

This is the first paragraph. It contains multiple sentences.

This is the second paragraph. Also multiple sentences.
"""

chunks = chunk_text(
    text,
    max_tokens=30,
    overlap_tokens=5,
    strategy="auto",
)

for ch in chunks:
    print("---- CHUNK ----")
    print(ch.text)
    print("META:", ch.meta)
    print()
