<p align="center">
  <img src="logo.PNG" width="200" alt="MSchunker logo">
</p>

# MSchunker – Intelligent Text Chunking for LLMs

[![PyPI version](https://badge.fury.io/py/mschunker.svg)](https://pypi.org/project/mschunker/)
[![Python versions](https://img.shields.io/pypi/pyversions/mschunker.svg)](https://pypi.org/project/mschunker/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

**MSchunker** is a lightweight, structure-aware, deterministic text chunker designed for modern LLM pipelines.

It transforms long documents into **LLM-ready chunks** while preserving semantic boundaries and natural writing structure.  
Optimized for:

- Retrieval-Augmented Generation (**RAG**)
- Question Answering (**QA**)
- Summarization
- Memory systems
- Any workflow requiring precise text segmentation

MSchunker respects document structure (sections → paragraphs → sentences) and provides rich metadata, task-aware defaults, and optional token overlap for cross-chunk continuity.

> 🔗 Links  
> • PyPI: https://pypi.org/project/mschunker/  
> • GitHub: https://github.com/cspnms/MSchunker

---

##  Features

- **Structure-aware splitting**
  - Detects headings, sections, paragraphs, and sentences
- **Token / character limits**
  - Enforces `max_tokens` and/or `max_chars`
- **Hierarchical strategy**
  - Paragraphs → sentences → hard-split fallback
- **Optional token overlap**
  - Adds context continuity across chunks
- **Rich metadata**
  - Section index, paragraph indices, sentence indices, split reasons
- **Deterministic output**
  - Same input + same settings → identical chunks
- **Lightweight**
  - No heavy NLP / ML dependencies
- **Clean API**
  - `chunk_text()` function
  - `Chunker` class for stateful use

---

## Installation

From PyPI:

```bash
pip install mschunker

Or latest version from GitHub:

pip install git+https://github.com/cspnms/MSchunker.git


⸻

##  QuickStart

```python
from mschunker import chunk_text

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
```


⸻

##  API Reference

### chunk_text(...)

Main function:

chunks = chunk_text(
    text: str,
    max_tokens: int | None = 512,
    max_chars: int | None = None,
    overlap_tokens: int = 64,
    strategy: str = "auto",          # or "fixed"
    token_counter: callable | None = None,
    sentence_splitter: callable | None = None,
    sentence_regex: Pattern[str] | None = None,
    source_id: str | None = None,
    task: str | None = None,         # rag | qa | summarization | memory
    enforce_overlap_limits: bool = False,
)

Returns: List[Chunk]

#### Advanced configuration

- **Tokenizer-aware counting:** If `tiktoken` is installed, `chunk_text` automatically counts tokens with the `cl100k_base` encoding. Pass your own `token_counter` callable to match a different model.
- **Sentence splitting:** Provide a custom `sentence_splitter` callable (or a `sentence_regex` pattern) to handle domain-specific punctuation or multilingual text.
- **Overlap enforcement:** Set `enforce_overlap_limits=True` to trim overlapped prefixes that would otherwise exceed `max_tokens`/`max_chars`.

```python
from mschunker import chunk_text

custom_regex = r"(?<=[.!?…])\s+"  # accept unicode ellipses

chunks = chunk_text(
    text,
    max_tokens=200,
    overlap_tokens=32,
    sentence_regex=custom_regex,
    enforce_overlap_limits=True,
)
```

> ℹ️ Without `tiktoken`, token counts fall back to a whitespace split. For strict model parity, install `tiktoken` or supply a custom `token_counter` aligned with your tokenizer.

⸻

### Chunker — Stateful Wrapper

from mschunker import Chunker

c = Chunker(
    max_tokens=512,
    overlap_tokens=64,
    strategy="auto",
    task="rag",
)

chunks = c.chunk(text, source_id="doc-1")


⸻

##  Chunk Data Model

Each Chunk contains:
	•	.text — the chunk content
	•	.meta — metadata including:
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

##  Utilities

### analyze_chunks(chunks)

from mschunker import analyze_chunks

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

### explain_chunk(chunk)

from mschunker import explain_chunk

print(explain_chunk(chunks[0]))

Example result:

Strategy: auto | Split reason: paragraph_boundary |
Section #0 heading='Introduction' |
Paragraphs: (0, 1) | Chunk index: 0


⸻

##  How MSchunker Works

MSchunker uses a hierarchical, structure-preserving algorithm:
	1.	Sections / Headings
	2.	Paragraphs
	3.	Sentences
	4.	Hard splits (fallback)

This ensures chunks remain coherent and optimized for LLM input.

overlap_tokens adds cross-chunk continuity—ideal for RAG or QA systems.

⸻

## Changelog

- **0.2.0**
  - Added optional `enforce_overlap_limits` trimming for overlapped chunks.
  - Introduced tokenizer-aware default counter (uses `tiktoken` when available).
  - Allow custom sentence splitters or regex patterns for language-specific needs.
  - Defined typed chunk metadata and expanded tests to cover headings and overlaps.

⸻

##  License

MIT License © 2025 MS

⸻

##  Contributing

Issues and pull requests are welcome.
MSchunker is designed to evolve into a fully intelligent, future-proof chunking engine.
