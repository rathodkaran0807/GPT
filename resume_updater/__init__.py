"""Generate resume updates to align with a job description."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from job_parser import JobDescription
from matcher.keyword_matcher import KeywordMatcher
from resume_parser import ResumeDocument, ResumeSection


@dataclass
class SectionUpdate:
    title: str
    suggested_bullets: List[str] = field(default_factory=list)
    integrated_keywords: List[str] = field(default_factory=list)


@dataclass
class ResumeUpdate:
    reordered_sections: List[str]
    updates: Dict[str, SectionUpdate]
    missing_keywords: List[str]


def generate_resume_update(
    resume: ResumeDocument,
    job: JobDescription,
    matcher: Optional[KeywordMatcher] = None,
) -> ResumeUpdate:
    """Produce update suggestions to tailor a resume to a job description."""

    matcher = matcher or KeywordMatcher()
    result = matcher.score(resume, job)

    section_order = sorted(
        result.section_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    reordered_titles = [title for title, _score in section_order]

    updates: Dict[str, SectionUpdate] = {}
    combined_keywords = list(dict.fromkeys(job.required_skills + job.keywords))

    for section in resume.sections:
        matched = result.matched_keywords.get(section.title, [])
        normalized_matched = {keyword.lower() for keyword in matched}
        missing = [keyword for keyword in combined_keywords if keyword.lower() not in normalized_matched]
        suggestions = _suggest_bullets(section, missing)
        content_lower = section.content.lower()
        integrated = [keyword for keyword in combined_keywords if keyword.lower() in content_lower]
        updates[section.title] = SectionUpdate(
            title=section.title,
            suggested_bullets=suggestions,
            integrated_keywords=integrated,
        )

    return ResumeUpdate(
        reordered_sections=reordered_titles,
        updates=updates,
        missing_keywords=result.missing_keywords,
    )


def _suggest_bullets(section: ResumeSection, missing_keywords: List[str]) -> List[str]:
    suggestions: List[str] = []
    content_lower = section.content.lower()
    for keyword in missing_keywords:
        template = f"Demonstrated {keyword} through measurable achievements (add specifics)."
        if keyword.lower() not in content_lower:
            suggestions.append(template)
    return suggestions


__all__ = ["ResumeUpdate", "SectionUpdate", "generate_resume_update"]
