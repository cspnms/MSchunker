Of course — here is the entire README.md as ONE single copy-paste code block, already complete, already formatted, nothing else needed.

You can paste this exactly into your GitHub README.md file.

⸻


#  SmartChunk – Intelligent Text Chunking for LLMs

**SmartChunk** is a lightweight, structure-aware, and deterministic text chunker designed for modern LLM pipelines.

It transforms long documents into **LLM-ready chunks** that maintain semantic integrity and are optimized for:

- Retrieval-Augmented Generation (**RAG**)
- Question Answering (**QA**)
- Summarization
- Memory systems
- Any workflow requiring precise text segmentation

SmartChunk respects natural document structure (sections → paragraphs → sentences) and provides rich metadata, task-aware defaults, and optional overlap for cross-chunk context.

---

##  Features

- **Structure-aware splitting**
  - Detects headings, sections, paragraphs, and sentences
- **Token / character limits**
  - Enforces `max_tokens` and/or `max_chars`
- **Hierarchical strategy**
  - Paragraphs → sentences → hard splits (fallback)
- **Optional token overlap**
  - Adds continuity across consecutive chunks
- **Rich metadata**
  - Section index, paragraph indices, sentence indices, split reasons, offsets
- **Deterministic output**
  - Same input + same settings → identical chunks
- **Lightweight**
  - Zero heavy NLP / ML dependencies
- **Clean, simple API**
  - `chunk_text(...)` handles everything
  - `Chunker` for stateful usage

---

##  Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/cspnms/MSchunker.git

(Once published to PyPI:)

pip install smartchunk


⸻

 Quickstart

from smartchunk import chunk_text

text = "... your long document ..."

chunks = chunk_text(
    text,
    max_tokens=512,
    overlap_tokens=64,
    strategy="auto",
    task="rag",
)

for ch in chunks:
    print("---- CHUNK ----")
    print(ch.text[:200], "...")
    print(ch.meta)


⸻

 API Reference

chunk_text(...) — Main function

chunks = chunk_text(
    text: str,
    max_tokens: int | None = 512,
    max_chars: int | None = None,
    overlap_tokens: int = 64,
    strategy: str = "auto",      # or "fixed"
    token_counter: callable | None = None,
    source_id: str | None = None,
    task: str | None = None,     # "rag" | "qa" | "summarization" | "memory"
)

Returns: List[Chunk]

⸻

Chunker — Stateful wrapper

from smartchunk import Chunker

c = Chunker(
    max_tokens=512,
    overlap_tokens=64,
    strategy="auto",
    task="rag",
)

chunks = c.chunk(text, source_id="doc-1")


⸻

Chunk — Data Model

Each chunk contains:
	•	.text – the chunk’s content
	•	.meta – dictionary with:
	•	section_index
	•	section_heading
	•	paragraph_indices
	•	sentence_indices
	•	split_reason
	•	strategy
	•	chunk_index
	•	overlap_from_prev
	•	overlap_tokens
	•	source_id

⸻

analyze_chunks(chunks) — Chunk statistics

from smartchunk import analyze_chunks

stats = analyze_chunks(chunks)
print(stats)

Example:

{
  "num_chunks": 12,
  "min_tokens": 118,
  "max_tokens": 482,
  "avg_tokens": 311.9
}


⸻

explain_chunk(chunk) — Human-readable explanation

from smartchunk import explain_chunk

print(explain_chunk(chunks[0]))

Possible output:

Strategy: auto | Split reason: paragraph_boundary |
Section #0 heading='Introduction' |
Paragraphs: (0, 1) | Chunk index: 0


⸻

 How SmartChunk Works

SmartChunk uses a hierarchical, structure-preserving algorithm:
	1.	Sections / Headings
	2.	Paragraphs
	3.	Sentences
	4.	Hard splits (when paragraphs or sentences exceed limits)

This design mirrors how humans write and ensures chunks are semantically coherent.

Optional overlap (overlap_tokens) adds context continuity across chunks—ideal for RAG retrieval and QA workflows.

⸻

 Design Principles
	•	Semantic integrity first
Meaning is preserved whenever possible.
	•	Deterministic and transparent
Output and split reasoning are reproducible and explainable.
	•	Lightweight
No dependencies on NLP or transformer libraries.
	•	Extensible foundation
Future roadmap:
	•	Semantic chunking (embedding-aware)
	•	Multi-granularity chunk outputs
	•	Benchmark-driven tuning
	•	Integration helpers for RAG frameworks

⸻

 License

MIT License © 2025 MS

⸻

 Contributing

Issues and pull requests are welcome.
SmartChunk is designed to evolve into a fully intelligent, future-proof chunking engine.

