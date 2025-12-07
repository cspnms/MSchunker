from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    List,
    Optional,
    Dict,
    Any,
    Tuple,
    Iterable,
    Pattern,
    TypedDict,
    Literal,
    cast,
)
import re
import warnings

# -----------------------------
# Types
# -----------------------------

TokenCounter = Callable[[str], int]
SentenceSplitter = Callable[[str], List[str]]
SemanticSimilarity = Callable[[str, str], float]


class ChunkMeta(TypedDict, total=False):
    """Typed metadata container for chunks."""

    section_index: Optional[int]
    section_heading: Optional[str]
    paragraph_indices: Optional[Tuple[int, int]]
    sentence_indices: Optional[Tuple[int, int]]
    split_reason: Literal["paragraph_boundary", "sentence_limit", "hard_limit"]
    strategy: Literal["auto", "fixed"]
    overlap_from_prev: bool
    overlap_tokens: int
    chunk_index: int
    source_id: Optional[str]


@dataclass
class Chunk:
    """
    Represents a single text chunk plus metadata describing how/why it was created.

    Attributes
    ----------
    text:
        The chunk text itself.
    meta:
        Metadata such as:
        - source_id: optional document id
        - section_index, section_heading
        - paragraph_indices: (start, end) within section
        - sentence_indices: (start, end) within paragraph scope
        - split_reason: "paragraph_boundary" | "sentence_limit" | "hard_limit"
        - strategy: "auto" | "fixed" | "paragraph" | "sentence"
        - overlap_from_prev: bool
        - overlap_tokens: int
        - chunk_index: int (sequential id)
    """
    text: str
    meta: ChunkMeta = field(default_factory=lambda: cast(ChunkMeta, {}))


# -----------------------------
# Basic utilities
# -----------------------------

_DEFAULT_SENTENCE_REGEX = re.compile(r"(?<=[.!?。！？])\s+")
_HEADING_REGEX = re.compile(r"^\s*#{1,6}\s+(.+)$")
_TIKTOKEN_ENCODER: Any = None


def _default_token_counter(text: str) -> int:
    """Token counter that prefers a tokenizer-aware fallback when available.

    If `tiktoken` is installed, this uses the "cl100k_base" encoding; otherwise
    it falls back to a simple whitespace split.
    """

    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken

            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODER = False

    if _TIKTOKEN_ENCODER:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            # Fallback gracefully if encoding fails
            return len(text.split())
    return len(text.split())


def _ensure_token_counter(counter: Optional[TokenCounter]) -> TokenCounter:
    return counter or _default_token_counter


def _ensure_sentence_splitter(
    sentence_splitter: Optional[SentenceSplitter],
    sentence_regex: Optional[Pattern[str]],
) -> SentenceSplitter:
    regex = sentence_regex or _DEFAULT_SENTENCE_REGEX

    def _split(paragraph: str) -> List[str]:
        paragraph = paragraph.strip()
        if not paragraph:
            return []
        parts = regex.split(paragraph)
        return [s.strip() for s in parts if s.strip()]

    return sentence_splitter or _split


def _build_meta(
    *,
    section_index: Optional[int],
    section_heading: Optional[str],
    paragraph_indices: Optional[Tuple[int, int]],
    sentence_indices: Optional[Tuple[int, int]],
    split_reason: Literal["paragraph_boundary", "sentence_limit", "hard_limit"],
    strategy: Literal["auto", "fixed"],
    source_id: Optional[str],
) -> ChunkMeta:
    meta: ChunkMeta = {
        "section_index": section_index if section_index is not None else None,
        "section_heading": section_heading,
        "paragraph_indices": paragraph_indices,
        "sentence_indices": sentence_indices,
        "split_reason": split_reason,
        "strategy": strategy,
    }
    if source_id is not None:
        meta["source_id"] = source_id
    return meta


def _length_ok(
    text: str,
    max_tokens: Optional[int],
    max_chars: Optional[int],
    token_counter: TokenCounter,
) -> bool:
    if max_chars is not None and len(text) > max_chars:
        return False
    if max_tokens is not None and token_counter(text) > max_tokens:
        return False
    return True


# -----------------------------
# Structural splitting
# -----------------------------


def _split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Split text into (heading, body) sections.

    Supports:
    - Markdown-style headings starting with '#'
    - Underline-style headings: a line followed by === or --- line
    """
    lines = text.splitlines()
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        # Underline-style heading
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line.strip() and set(next_line.strip()) in ({"="}, {"-"}):
                # Flush previous section
                if current_heading is not None or current_body:
                    sections.append((current_heading, current_body))
                current_heading = line.strip()
                current_body = []
                i += 2
                continue

        # Markdown-style heading
        m = _HEADING_REGEX.match(line)
        if m:
            if current_heading is not None or current_body:
                sections.append((current_heading, current_body))
            current_heading = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
        i += 1

    # Final section
    if current_heading is not None or current_body:
        sections.append((current_heading, current_body))

    result: List[Tuple[Optional[str], str]] = []
    for heading, body_lines in sections:
        body_text = "\n".join(body_lines).strip("\n")
        result.append((heading, body_text))
    return result or [(None, text)]


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs by blank lines."""
    text = text.strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(
    paragraph: str,
    splitter: SentenceSplitter,
) -> List[str]:
    """Split a paragraph into sentences using a provided splitter."""

    return splitter(paragraph)


# -----------------------------
# Hard splitting by tokens / chars
# -----------------------------


def _hard_split(
    text: str,
    max_tokens: Optional[int],
    max_chars: Optional[int],
    token_counter: TokenCounter,
) -> List[str]:
    """Split text greedily by tokens/characters when higher-level structure fails."""
    if max_tokens is None and max_chars is None:
        return [text]

    tokens = text.split()
    chunks: List[str] = []
    current_tokens: List[str] = []

    def current_text() -> str:
        return " ".join(current_tokens)

    for tok in tokens:
        current_tokens.append(tok)
        ct = current_text()
        if not _length_ok(ct, max_tokens, max_chars, token_counter):
            # Remove last token and flush previous
            current_tokens.pop()
            if current_tokens:
                chunks.append(" ".join(current_tokens))
            # Start new with current token
            current_tokens = [tok]
    if current_tokens:
        chunks.append(" ".join(current_tokens))
    return chunks


# -----------------------------
# Core hierarchical chunking
# -----------------------------


def _chunk_section(
    section_text: str,
    section_index: int,
    section_heading: Optional[str],
    max_tokens: Optional[int],
    max_chars: Optional[int],
    token_counter: TokenCounter,
    sentence_splitter: SentenceSplitter,
    strategy: str,
) -> List[Chunk]:
    """Chunk a single section using paragraphs/sentences and fallbacks."""
    chunks: List[Chunk] = []

    if strategy == "fixed":
        for piece in _hard_split(section_text, max_tokens, max_chars, token_counter):
            chunks.append(
                Chunk(
                    text=piece,
                    meta=_build_meta(
                        section_index=section_index,
                        section_heading=section_heading,
                        paragraph_indices=None,
                        sentence_indices=None,
                        split_reason="hard_limit",
                        strategy=cast(Literal["auto", "fixed"], strategy),
                        source_id=None,
                    ),
                )
            )
        return chunks

    paragraphs = _split_into_paragraphs(section_text)
    if not paragraphs:
        return []

    paragraph_start_idx = 0
    current_buffer: List[Tuple[int, str]] = []  # (paragraph_index, text)

    def flush_buffer(reason: str) -> None:
        nonlocal current_buffer
        if not current_buffer:
            return
        text = "\n\n".join(p for _, p in current_buffer)
        para_indices = (current_buffer[0][0], current_buffer[-1][0])
        chunks.append(
            Chunk(
                text=text,
                meta=_build_meta(
                    section_index=section_index,
                    section_heading=section_heading,
                    paragraph_indices=para_indices,
                    sentence_indices=None,
                    split_reason=cast(
                        Literal["paragraph_boundary", "sentence_limit", "hard_limit"],
                        reason,
                    ),
                    strategy=cast(Literal["auto", "fixed"], strategy),
                    source_id=None,
                ),
            )
        )
        current_buffer = []

    for local_idx, para in enumerate(paragraphs):
        para_index = paragraph_start_idx + local_idx
        if not para.strip():
            continue

        # If paragraph alone is bigger than allowed, split it further.
        if not _length_ok(para, max_tokens, max_chars, token_counter):
            # Flush current paragraphs buffer first.
            flush_buffer("paragraph_boundary")

            sentences = _split_into_sentences(para, sentence_splitter)
            if not sentences:
                # Last resort: hard split paragraph
                for piece in _hard_split(para, max_tokens, max_chars, token_counter):
                    chunks.append(
                        Chunk(
                            text=piece,
                            meta=_build_meta(
                                section_index=section_index,
                                section_heading=section_heading,
                                paragraph_indices=(para_index, para_index),
                                sentence_indices=None,
                                split_reason="hard_limit",
                                strategy=cast(Literal["auto", "fixed"], strategy),
                                source_id=None,
                            ),
                        )
                    )
                continue

            # Accumulate sentences into chunks
            sentence_start_idx = 0
            buf_sentences: List[Tuple[int, str]] = []
            for s_local_idx, sentence in enumerate(sentences):
                sent_index = sentence_start_idx + s_local_idx
                candidate_parts = [s for _, s in buf_sentences] + [sentence]
                candidate_text = " ".join(candidate_parts)
                if _length_ok(candidate_text, max_tokens, max_chars, token_counter):
                    buf_sentences.append((sent_index, sentence))
                else:
                    if buf_sentences:
                        text = " ".join(s for _, s in buf_sentences)
                        sent_indices = (buf_sentences[0][0], buf_sentences[-1][0])
                        chunks.append(
                            Chunk(
                                text=text,
                                meta=_build_meta(
                                    section_index=section_index,
                                    section_heading=section_heading,
                                    paragraph_indices=(para_index, para_index),
                                    sentence_indices=sent_indices,
                                    split_reason="sentence_limit",
                                    strategy=cast(Literal["auto", "fixed"], strategy),
                                    source_id=None,
                                ),
                            )
                        )
                    # Start new buffer with current sentence
                    buf_sentences = [(sent_index, sentence)]

            if buf_sentences:
                text = " ".join(s for _, s in buf_sentences)
                sent_indices = (buf_sentences[0][0], buf_sentences[-1][0])
                chunks.append(
                    Chunk(
                        text=text,
                        meta=_build_meta(
                            section_index=section_index,
                            section_heading=section_heading,
                            paragraph_indices=(para_index, para_index),
                            sentence_indices=sent_indices,
                            split_reason="sentence_limit",
                            strategy=cast(Literal["auto", "fixed"], strategy),
                            source_id=None,
                        ),
                    )
                )
        else:
            # Paragraph fits, try to add to current buffer
            candidate_paragraphs = [p for _, p in current_buffer] + [para]
            candidate_text = "\n\n".join(candidate_paragraphs)
            if _length_ok(candidate_text, max_tokens, max_chars, token_counter):
                current_buffer.append((para_index, para))
            else:
                # Flush and start a new chunk with this paragraph
                flush_buffer("paragraph_boundary")
                current_buffer.append((para_index, para))

    # Flush remaining buffered paragraphs
    flush_buffer("paragraph_boundary")
    return chunks


# -----------------------------
# Overlap
# -----------------------------


def _apply_overlap(
    chunks: List[Chunk],
    overlap_tokens: int,
    token_counter: TokenCounter,
    max_tokens: Optional[int],
    max_chars: Optional[int],
    enforce_overlap_limits: bool,
) -> List[Chunk]:
    """Apply token-based overlap between consecutive chunks.
    """
    if overlap_tokens <= 0 or len(chunks) <= 1:
        # Still annotate chunk_index for convenience
        for idx, ch in enumerate(chunks):
            ch.meta.setdefault("chunk_index", idx)
            ch.meta.setdefault("overlap_from_prev", False)
            ch.meta.setdefault("overlap_tokens", 0)
        return chunks

    overlapped: List[Chunk] = []
    prev_tokens: List[str] = []

    for idx, ch in enumerate(chunks):
        text = ch.text
        tokens = text.split()
        if idx == 0:
            # First chunk, no overlap
            prev_tokens = tokens
            ch.meta["chunk_index"] = idx
            ch.meta["overlap_from_prev"] = False
            ch.meta["overlap_tokens"] = 0
            overlapped.append(ch)
            continue

        # Determine overlap slice from previous chunk
        overlap_slice = prev_tokens[-overlap_tokens:] if prev_tokens else []
        overlap_used = overlap_slice
        prefix = " ".join(overlap_used)
        new_text = prefix + "\n\n" + text if prefix else text

        if overlap_used and enforce_overlap_limits:
            while overlap_used and not _length_ok(
                new_text, max_tokens, max_chars, token_counter
            ):
                overlap_used = overlap_used[1:]
                prefix = " ".join(overlap_used)
                new_text = prefix + "\n\n" + text if prefix else text
        elif overlap_used and not _length_ok(
            new_text, max_tokens, max_chars, token_counter
        ):
            warnings.warn(
                "Chunk exceeds limits after applying overlap; set"
                " enforce_overlap_limits=True to trim.",
                RuntimeWarning,
            )

        ch.text = new_text
        ch.meta["chunk_index"] = idx
        ch.meta["overlap_from_prev"] = bool(overlap_used)
        ch.meta["overlap_tokens"] = len(overlap_used)

        overlapped.append(ch)
        prev_tokens = tokens

    return overlapped


# -----------------------------
# Public API
# -----------------------------


def chunk_text(
    text: str,
    max_tokens: Optional[int] = 512,
    max_chars: Optional[int] = None,
    overlap_tokens: int = 64,
    strategy: str = "auto",
    token_counter: Optional[TokenCounter] = None,
    sentence_splitter: Optional[SentenceSplitter] = None,
    sentence_regex: Optional[Pattern[str]] = None,
    source_id: Optional[str] = None,
    task: Optional[str] = None,
    enforce_overlap_limits: bool = False,
) -> List[Chunk]:
    """Chunk text into LLM-ready pieces.

    Parameters
    ----------
    text:
        Input text to be chunked.
    max_tokens:
        Soft limit on tokens per chunk. If None, only max_chars is used.
    max_chars:
        Soft limit on characters per chunk. If None, only max_tokens is used.
    overlap_tokens:
        Number of tokens from the end of chunk N to prepend to chunk N+1.
    strategy:
        "auto"  -> sections -> paragraphs -> sentences -> hard limit
        "fixed" -> ignore structure, just hard-split
    token_counter:
        Optional custom token counter.
    sentence_splitter:
        Optional callable to split a paragraph into sentences.
    sentence_regex:
        Optional regex used by the default sentence splitter.
    source_id:
        Optional identifier for the source document, stored in metadata.
    task:
        Optional hint: "rag", "qa", "summarization", "memory".
        Used only to adjust sane defaults if max_tokens is None.
    enforce_overlap_limits:
        When True, trims overlap to keep chunks within max_tokens/max_chars.

    Returns
    -------
    List[Chunk]
        A list of Chunk objects with text and rich metadata.
    """
    token_counter = _ensure_token_counter(token_counter)
    sentence_splitter = _ensure_sentence_splitter(sentence_splitter, sentence_regex)

    if not text or not text.strip():
        return []

    # Task-adaptive defaults (only if user didn't pass explicit max_tokens)
    if max_tokens is None:
        if task in ("rag", "qa"):
            max_tokens = 256
        elif task == "summarization":
            max_tokens = 1024
        elif task == "memory":
            max_tokens = 512
        else:
            max_tokens = 512

    if strategy not in ("auto", "fixed"):
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    sections = _split_into_sections(text)
    all_chunks: List[Chunk] = []

    for section_index, (heading, body) in enumerate(sections):
        if not body.strip():
            continue
        section_chunks = _chunk_section(
            section_text=body,
            section_index=section_index,
            section_heading=heading,
            max_tokens=max_tokens,
            max_chars=max_chars,
            token_counter=token_counter,
            sentence_splitter=sentence_splitter,
            strategy=strategy,
        )
        for ch in section_chunks:
            ch.meta.setdefault("source_id", source_id)
        all_chunks.extend(section_chunks)

    # Apply overlap (and fill chunk_index)
    all_chunks = _apply_overlap(
        all_chunks,
        overlap_tokens=overlap_tokens,
        token_counter=token_counter,
        max_tokens=max_tokens,
        max_chars=max_chars,
        enforce_overlap_limits=enforce_overlap_limits,
    )

    return all_chunks


class Chunker:
    """Stateful convenience wrapper around `chunk_text`.

    Example
    -------
    >>> c = Chunker(max_tokens=512, overlap_tokens=64)
    >>> chunks = c.chunk(my_long_text, source_id="doc-1")
    """

    def __init__(
        self,
        max_tokens: Optional[int] = 512,
        max_chars: Optional[int] = None,
        overlap_tokens: int = 64,
        strategy: str = "auto",
        token_counter: Optional[TokenCounter] = None,
        sentence_splitter: Optional[SentenceSplitter] = None,
        sentence_regex: Optional[Pattern[str]] = None,
        task: Optional[str] = None,
        enforce_overlap_limits: bool = False,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.overlap_tokens = overlap_tokens
        self.strategy = strategy
        self.token_counter = token_counter
        self.sentence_splitter = sentence_splitter
        self.sentence_regex = sentence_regex
        self.task = task
        self.enforce_overlap_limits = enforce_overlap_limits

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        return chunk_text(
            text=text,
            max_tokens=self.max_tokens,
            max_chars=self.max_chars,
            overlap_tokens=self.overlap_tokens,
            strategy=self.strategy,
            token_counter=self.token_counter,
            sentence_splitter=self.sentence_splitter,
            sentence_regex=self.sentence_regex,
            source_id=source_id,
            task=self.task,
            enforce_overlap_limits=self.enforce_overlap_limits,
        )


# -----------------------------
# Introspection / evaluation helpers
# -----------------------------


def analyze_chunks(
    chunks: Iterable[Chunk],
    token_counter: Optional[TokenCounter] = None,
) -> Dict[str, Any]:
    """Return simple statistics about a list of chunks.

    Useful for debugging chunk sizes and tuning parameters.
    """
    token_counter = _ensure_token_counter(token_counter)
    chunks = list(chunks)
    if not chunks:
        return {
            "num_chunks": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "avg_tokens": 0.0,
        }

    token_counts = [token_counter(c.text) for c in chunks]
    num_chunks = len(chunks)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / num_chunks

    return {
        "num_chunks": num_chunks,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "avg_tokens": avg_tokens,
    }


def explain_chunk(chunk: Chunk) -> str:
    """Return a human-readable explanation of how a chunk was produced."""
    meta = chunk.meta or {}
    parts = [
        f"Strategy: {meta.get('strategy', 'unknown')}",
        f"Split reason: {meta.get('split_reason', 'unknown')}",
    ]
    if "section_index" in meta:
        parts.append(
            f"Section #{meta['section_index']} heading={meta.get('section_heading')!r}"
        )
    if "paragraph_indices" in meta and meta["paragraph_indices"] is not None:
        parts.append(f"Paragraphs: {meta['paragraph_indices']}")
    if "sentence_indices" in meta and meta["sentence_indices"] is not None:
        parts.append(f"Sentences: {meta['sentence_indices']}")
    if "chunk_index" in meta:
        parts.append(f"Chunk index: {meta['chunk_index']}")
    if meta.get("overlap_from_prev"):
        parts.append(f"Overlap from previous: {meta.get('overlap_tokens', 0)} tokens")
    if meta.get("source_id"):
        parts.append(f"Source: {meta['source_id']}")
    return " | ".join(parts)
