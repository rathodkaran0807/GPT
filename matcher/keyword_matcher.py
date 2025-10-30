"""Keyword-based resume to job description matching engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from resume_parser import ResumeDocument
from job_parser import JobDescription


@dataclass
class MatchResult:
    overall_score: float
    section_scores: Dict[str, float]
    matched_keywords: Dict[str, List[str]]
    missing_keywords: List[str]


class KeywordMatcher:
    """Scores resume sections against job requirements."""

    def __init__(self, required_weight: float = 0.6, keyword_weight: float = 0.4) -> None:
        if required_weight < 0 or keyword_weight < 0:
            raise ValueError("Weights must be non-negative")
        total = required_weight + keyword_weight
        if total == 0:
            raise ValueError("At least one weight must be positive")
        self.required_weight = required_weight / total
        self.keyword_weight = keyword_weight / total

    def score(self, resume: ResumeDocument, job: JobDescription) -> MatchResult:
        job_keywords = self._normalize(job.keywords)
        job_required = self._normalize(job.required_skills)
        combined_job_keywords = sorted(job_keywords.union(job_required))

        section_scores: Dict[str, float] = {}
        section_matches: Dict[str, List[str]] = {}
        seen_keywords: set[str] = set()

        for section in resume.sections:
            section_tokens = self._normalize(section.keywords or self._split_words(section.content))
            matched_required = sorted(section_tokens.intersection(job_required))
            matched_optional = sorted(section_tokens.intersection(job_keywords))
            seen_keywords.update(matched_required)
            seen_keywords.update(matched_optional)

            required_score = len(job_required) and len(matched_required) / len(job_required) or 0.0
            optional_score = len(job_keywords) and len(matched_optional) / len(job_keywords) or 0.0
            score = self.required_weight * required_score + self.keyword_weight * optional_score
            section_scores[section.title] = round(score, 4)
            section_matches[section.title] = sorted(set(matched_required + matched_optional))

        missing = sorted(set(combined_job_keywords) - seen_keywords)
        overall = 0.0
        if section_scores:
            overall = round(sum(section_scores.values()) / len(section_scores), 4)

        return MatchResult(
            overall_score=overall,
            section_scores=section_scores,
            matched_keywords=section_matches,
            missing_keywords=missing,
        )

    @staticmethod
    def _normalize(words: Iterable[str]) -> set[str]:
        return {word.strip().lower() for word in words if word and word.strip()}

    @staticmethod
    def _split_words(text: str) -> List[str]:
        import re

        return re.findall(r"[A-Za-z][A-Za-z0-9_+/-]{3,}", text)


__all__ = ["KeywordMatcher", "MatchResult"]
