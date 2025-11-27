from mschunker import chunk_text, analyze_chunks, explain_chunk

text = "Sentence one. Sentence two. Sentence three. Sentence four."

chunks = chunk_text(text, max_tokens=10)

# Show statistics
stats = analyze_chunks(chunks)
print("Chunk statistics:", stats, "\n")

# Show human explanation
for ch in chunks:
    print("---- CHUNK ----")
    print(ch.text)
    print(explain_chunk(ch))
    print()
