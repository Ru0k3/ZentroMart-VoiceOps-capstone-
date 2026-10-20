"""Section-aware parsing and lexical retrieval for ZentroMart KB documents.

The index keeps policy evidence and metadata together. It intentionally does
not ingest support logs, which are customer-language examples rather than
policy authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
import re
from typing import Iterable


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_SECTION = re.compile(r"^## \[([^\]]+)\]\s*(.*)$")
_H1 = re.compile(r"^#\s+(.+?)\s*$")
_TOKEN = re.compile(r"[^\W_]+|₹", re.UNICODE)

# This is a retrieval tie-breaker, not a replacement for the source hierarchy
# in the KB guide. A stronger lexical match still ranks ahead of a weak match.
_SOURCE_PRIORITY = {
    "policy": 5,
    "agent_guide": 4,
    "meta": 3,
    "help": 2,
    "faq": 1,
    "marketing": 0,
}


@dataclass(frozen=True)
class KnowledgeChunk:
    section_id: str
    section_title: str
    content: str
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    version: str
    language: str
    source_path: str

    def evidence(self) -> dict[str, str]:
        """Return a compact, serializable record for answers and evaluations."""
        return asdict(self)

    def prompt_text(self) -> str:
        """Render evidence with explicit provenance for an LLM prompt."""
        return (
            f"[{self.section_id}] {self.section_title}\n"
            f"Source: {self.doc_title} ({self.doc_type}, {self.status}, "
            f"effective {self.effective_from})\n{self.content}"
        )


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        raise ValueError("Knowledge document is missing YAML frontmatter")

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text[match.end():]


def parse_document(path: Path, kb_root: Path) -> list[KnowledgeChunk]:
    """Split a Markdown file into its ``## [SECTION-ID]`` evidence chunks."""
    raw_text = path.read_text(encoding="utf-8")
    metadata, markdown = _parse_frontmatter(raw_text)
    doc_title = metadata.get("title", "")

    sections: list[KnowledgeChunk] = []
    current_id: str | None = None
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        content = "\n".join(current_lines).strip()
        sections.append(KnowledgeChunk(
            section_id=current_id,
            section_title=current_title,
            content=content,
            doc_id=metadata.get("doc_id", path.stem),
            doc_title=doc_title,
            doc_type=metadata.get("doc_type", ""),
            status=metadata.get("status", ""),
            effective_from=metadata.get("effective_from", ""),
            version=metadata.get("version", ""),
            language=metadata.get("language", ""),
            source_path=path.relative_to(kb_root).as_posix(),
        ))

    for line in markdown.splitlines():
        section_match = _SECTION.match(line)
        if section_match:
            flush()
            current_id, current_title = section_match.groups()
            current_lines = []
            continue
        if current_id is None:
            title_match = _H1.match(line)
            if title_match and not doc_title:
                doc_title = title_match.group(1)
            continue
        current_lines.append(line)

    flush()
    if not sections:
        raise ValueError(f"No section IDs found in {path}")
    return sections


def load_knowledge_base(kb_root: Path) -> list[KnowledgeChunk]:
    """Parse all KB Markdown documents, sorted for stable index ordering."""
    kb_root = kb_root.resolve()
    files = sorted(kb_root.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown documents found under {kb_root}")
    chunks: list[KnowledgeChunk] = []
    for path in files:
        chunks.extend(parse_document(path, kb_root))
    return chunks


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


class KnowledgeRetriever:
    """Small dependency-free BM25 retriever over section-level evidence."""

    def __init__(
        self,
        chunks: Iterable[KnowledgeChunk],
        *,
        include_superseded: bool = False,
    ):
        self.chunks = [
            chunk for chunk in chunks
            if include_superseded or chunk.status.casefold() != "superseded"
        ]
        self._document_tokens = [
            _tokens(f"{chunk.doc_title} {chunk.section_title} {chunk.content}")
            for chunk in self.chunks
        ]
        self._lengths = [len(tokens) for tokens in self._document_tokens]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._term_frequencies = [Counter(tokens) for tokens in self._document_tokens]
        self._document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            self._document_frequency.update(frequencies.keys())

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the best matching chunks with IDs and source metadata intact."""
        query_tokens = _tokens(query)
        if not query_tokens or top_k <= 0 or not self.chunks:
            return []

        total_documents = len(self.chunks)
        k1, b = 1.5, 0.75
        ranked: list[tuple[float, int, SearchResult]] = []

        for chunk, frequencies, length in zip(
            self.chunks, self._term_frequencies, self._lengths
        ):
            score = 0.0
            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = log(
                    1.0 + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = term_frequency + k1 * (
                    1.0 - b + b * length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    term_frequency * (k1 + 1.0) / denominator
                )

            if score > 0.0:
                priority = _SOURCE_PRIORITY.get(chunk.doc_type.casefold(), -1)
                ranked.append((score, priority, SearchResult(chunk, score)))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].chunk.section_id))
        return [result for _, _, result in ranked[:top_k]]

