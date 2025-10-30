"""Utilities for parsing resumes from various file formats."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from ._loaders import load_text_from_file


@dataclass
class ResumeSection:
    """Represents a logical section of a resume."""

    title: str
    content: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class ResumeDocument:
    """A parsed resume containing the raw text and structured sections."""

    text: str
    sections: List[ResumeSection]


_HEADING_PATTERN = re.compile(r"^\s*([A-Z][A-Z\s/&-]{2,})\s*$")


def _split_into_sections(lines: Iterable[str]) -> List[ResumeSection]:
    line_list = list(lines)
    sections: List[ResumeSection] = []
    current_title = "General"
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_lines, current_title
        if not current_lines:
            return
        content = "\n".join(line.strip() for line in current_lines if line.strip())
        if not content:
            current_lines = []
            return
        keywords = _extract_keywords(content)
        sections.append(ResumeSection(title=current_title.strip(), content=content, keywords=keywords))
        current_lines = []

    for line in line_list:
        heading_match = _HEADING_PATTERN.match(line)
        if heading_match:
            flush_section()
            current_title = heading_match.group(1).title()
        else:
            current_lines.append(line)
    flush_section()
    return sections or [ResumeSection(title="General", content="\n".join(line_list), keywords=[])]


def _extract_keywords(text: str, max_keywords: int = 12) -> List[str]:
    """Extract naive keyword candidates from a block of text.

    The implementation intentionally keeps the logic simple so that it does not
    require any heavyweight NLP dependencies. Keywords are derived by selecting
    the most frequent alpha-numeric tokens longer than three characters.
    """

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+/-]{3,}", text)
    frequency: dict[str, int] = {}
    for token in tokens:
        key = token.lower()
        frequency[key] = frequency.get(key, 0) + 1
    sorted_tokens = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _count in sorted_tokens[:max_keywords]]


def parse_resume(path: str | Path) -> ResumeDocument:
    """Parse a resume file (PDF, DOCX, Markdown, or TXT) into structured text.

    Parameters
    ----------
    path:
        The path to the resume file.

    Returns
    -------
    ResumeDocument
        The parsed resume with the original text and best-effort sections.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file does not exist: {file_path}")

    raw_text = load_text_from_file(file_path)
    lines = raw_text.splitlines()
    sections = _split_into_sections(lines)
    return ResumeDocument(text=raw_text, sections=sections)


__all__ = ["ResumeDocument", "ResumeSection", "parse_resume"]
