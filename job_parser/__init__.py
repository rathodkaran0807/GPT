"""Utilities for parsing job descriptions using lightweight NLP."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import spacy
    from spacy.language import Language
    _NLP: Language | None = None
except ImportError:  # pragma: no cover - optional dependency
    spacy = None
    _NLP = None


@dataclass
class JobDescription:
    text: str
    keywords: List[str]
    required_skills: List[str]
    responsibilities: List[str]


def _get_nlp() -> "Language | None":
    global _NLP
    if _NLP is not None:
        return _NLP
    if spacy is None:  # pragma: no cover - dependency missing
        return None
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:  # pragma: no cover - model missing
        _NLP = spacy.blank("en")
    return _NLP


def _extract_phrases(text: str) -> List[str]:
    nlp = _get_nlp()
    if nlp is None:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+/&-]{2,}", text)
        unique = sorted({token.lower() for token in tokens})
        return unique

    doc = nlp(text)
    candidates = set()
    for chunk in getattr(doc, "noun_chunks", []):
        cleaned = chunk.text.strip().lower()
        if len(cleaned) > 3:
            candidates.add(cleaned)
    for token in doc:
        if token.pos_ in {"PROPN", "NOUN"} and len(token.text) > 3:
            candidates.add(token.text.lower())
    return sorted(candidates)


def _segment_responsibilities(lines: Iterable[str]) -> List[str]:
    responsibilities: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            text = " ".join(part.strip() for part in buffer if part.strip())
            if text:
                responsibilities.append(text)
            buffer = []

    bullet_pattern = re.compile(r"^\s*(?:[-*•]+|\d+\.)\s+")
    for line in lines:
        if bullet_pattern.match(line):
            flush()
            buffer.append(bullet_pattern.sub("", line))
            flush()
        elif line.strip():
            buffer.append(line)
        else:
            flush()
    flush()
    return responsibilities


def parse_job_description(data: str | Path) -> JobDescription:
    """Parse a job description from a string or file path."""

    text: str
    if isinstance(data, Path):
        text = data.read_text(encoding="utf-8")
    elif isinstance(data, str):
        possible_path = Path(data)
        if possible_path.exists():
            text = possible_path.read_text(encoding="utf-8")
        else:
            text = data
    else:
        text = str(data)

    phrases = _extract_phrases(text)
    responsibilities = _segment_responsibilities(text.splitlines())

    # Heuristically separate required skills from general keywords.
    skill_keywords = [phrase for phrase in phrases if any(keyword in phrase for keyword in ["experience", "knowledge", "skills", "proficiency"])]
    other_keywords = [phrase for phrase in phrases if phrase not in skill_keywords]

    return JobDescription(
        text=text,
        keywords=other_keywords,
        required_skills=skill_keywords,
        responsibilities=responsibilities,
    )


__all__ = ["JobDescription", "parse_job_description"]
