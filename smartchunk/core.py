from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import re


TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


# Simple sentence splitter
DEFAULT_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+")


def default_token_counter(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def _ensure_token_counter(token_counter: Optional[TokenCounter]) -> TokenCounter:
    return token_counter or default_token_counter


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


def _split_into_sections(text: str) -> List[Tuple[Optional[str], str]]:
    lines = text.splitlines()
    sections = []
    current_heading = None
    current_body = []

    def flush():
        nonlocal current_heading, current_body
        if current_body or current_heading is not None:
            sections.append((current_heading, "\n".join(current_body).strip()))
        current_heading = None
        current_body = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        next_line = lines[i + 1].rstrip() if i + 1 < len(lines) else ""

        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip() or None
            i += 1
            continue

        if set(next_line) <= {"=", "-"} and len(next_line) >= 3 and line.strip():
            flush()
            current_heading = line.strip()
            i += 2
            continue

        current_body.append(line)
        i += 1

    flush()
    return sections


def _split_into_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras if paras else ([text.strip()] if text.strip() else [])


def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    return [p.strip() for p in DEFAULT_SENTENCE_REGEX.split(text) if p.strip()]


def _hard_split(
    text: str,
    max_tokens: Optional[int],
    max_chars: Optional[int],
    token_counter: TokenCounter,
) -> List[str]:
    if max_chars is None and max_tokens is None:
        return [text]

    chunks = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_chars, n) if max_chars else n
        candidate = text[start:end]

        if max_tokens:
            while token_counter(candidate) > max_tokens and end > start + 1:
                end -= 1
                candidate = text[start:end]

        if not candidate:
            break

        chunks.append(candidate)
        start = end

    return chunks


def _apply_overlap(
    chunks: List[Chunk],
    overlap_tokens: int,
    token_counter: TokenCounter,
) -> List[Chunk]:
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = []
    prev_tail = None

    for idx, ch in enumerate(chunks):
        base_text = ch.text
        combined = f"{prev_tail} {base_text}".strip() if prev_tail else base_text

        meta = dict(ch.meta)
        meta["chunk_index"] = idx
        if prev_tail:
            meta["overlap_from_prev"] = True

        overlapped.append(Chunk(combined, meta))

        tokens = re.findall(r"\w+|[^\w\s]", base_text)
        prev_tail = " ".join(tokens[-overlap_tokens:]) if len(tokens) > overlap_tokens else base_text

    return overlapped


def chunk_text(
    text: str,
    max_tokens: Optional[int] = 512,
    max_chars: Optional[int] = None,
    overlap_tokens: int = 0,
    strategy: str = "auto",
    token_counter: Optional[TokenCounter] = None,
    source_id: Optional[str] = None,
) -> List[Chunk]:
    token_counter = _ensure_token_counter(token_counter)
    if not text.strip():
        return []

    if strategy not in {"auto", "fixed"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "fixed":
        parts = _hard_split(text, max_tokens, max_chars, token_counter)
        out = []
        offset = 0
        for i, part in enumerate(parts):
            start = text.find(part, offset)
            end = start + len(part)
            out.append(Chunk(part, {
                "source_id": source_id,
                "split_reason": "hard_token_limit",
                "offset_start": start,
                "offset_end": end,
                "chunk_index": i,
            }))
            offset = end
        return _apply_overlap(out, overlap_tokens, token_counter)

    sections = _split_into_sections(text)
    chunks = []
    global_offset = 0

    for sec_i, (heading, body) in enumerate(sections):
        paragraphs = _split_into_paragraphs(body)

        cursor = 0
        para_offsets = []
        for p in paragraphs:
            start = body.find(p, cursor)
            end = start + len(p)
            para_offsets.append((start, end))
            cursor = end

        for para_i, (para, (p_start, p_end)) in enumerate(zip(paragraphs, para_offsets)):
            para_global_start = global_offset + p_start
            para_global_end = global_offset + p_end

            if _length_ok(para, max_tokens, max_chars, token_counter):
                chunks.append(Chunk(para, {
                    "source_id": source_id,
                    "section_index": sec_i,
                    "section_heading": heading,
                    "paragraph_index": para_i,
                    "split_reason": "paragraph_boundary",
                    "offset_start": para_global_start,
                    "offset_end": para_global_end,
                }))
                continue

            sentences = _split_into_sentences(para)
            buf = ""
            buf_start = 0
            search_pos = 0

            def flush(reason):
                nonlocal buf, buf_start
                if not buf:
                    return
                end = buf_start + len(buf)
                chunks.append(Chunk(buf, {
                    "source_id": source_id,
                    "section_index": sec_i,
                    "section_heading": heading,
                    "paragraph_index": para_i,
                    "split_reason": reason,
                    "offset_start": para_global_start + buf_start,
                    "offset_end": para_global_start + end,
                }))
                buf = ""

            for sent in sentences:
                pos = para.find(sent, search_pos)
                candidate = f"{buf} {sent}".strip() if buf else sent

                if _length_ok(candidate, max_tokens, max_chars, token_counter):
                    if not buf:
                        buf_start = pos
                    buf = candidate
                    search_pos = pos + len(sent)
                else:
                    flush("sentence_limit")

                    if not _length_ok(sent, max_tokens, max_chars, token_counter):
                        for part in _hard_split(sent, max_tokens, max_chars, token_counter):
                            local_start = para.find(part)
                            chunks.append(Chunk(part, {
                                "source_id": source_id,
                                "section_index": sec_i,
                                "section_heading": heading,
                                "paragraph_index": para_i,
                                "split_reason": "hard_token_limit",
                                "offset_start": para_global_start + local_start,
                                "offset_end": para_global_start + local_start + len(part),
                            }))
                        continue

                    buf = sent
                    buf_start = pos
                    search_pos = pos + len(sent)

            flush("sentence_limit")

        global_offset += len(body) + 1

    return _apply_overlap(chunks, overlap_tokens, token_counter)


class Chunker:
    def __init__(
        self,
        max_tokens: Optional[int] = 512,
        max_chars: Optional[int] = None,
        overlap_tokens: int = 0,
        strategy: str = "auto",
        token_counter: Optional[TokenCounter] = None,
    ):
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.overlap_tokens = overlap_tokens
        self.strategy = strategy
        self.token_counter = token_counter

    def chunk(self, text: str, source_id: Optional[str] = None) -> List[Chunk]:
        return chunk_text(
            text,
            max_tokens=self.max_tokens,
            max_chars=self.max_chars,
            overlap_tokens=self.overlap_tokens,
            strategy=self.strategy,
            token_counter=self.token_counter,
            source_id=source_id,
        )
