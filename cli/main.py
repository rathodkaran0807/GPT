"""Command-line interface for tailoring resumes to job descriptions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from job_parser import parse_job_description
from matcher.keyword_matcher import KeywordMatcher
from resume_parser import parse_resume
from resume_updater import generate_resume_update


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tailor a resume to a job description")
    parser.add_argument("resume", type=Path, help="Path to the resume file (PDF/DOCX/Markdown/TXT)")
    parser.add_argument("job", type=Path, help="Path to the job description (text/Markdown)")
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to export tailored resume recommendations as Markdown",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to save the analysis as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resume = parse_resume(args.resume)
    job = parse_job_description(args.job)

    matcher = KeywordMatcher()
    match_result = matcher.score(resume, job)
    update = generate_resume_update(resume, job, matcher=matcher)

    _print_summary(match_result, update)

    if args.export:
        _export_markdown(args.export, resume, match_result, update)
        print(f"\nExported tailored recommendations to {args.export}")

    if args.json:
        _export_json(args.json, match_result, update)
        print(f"Saved analysis data to {args.json}")

    return 0


def _print_summary(match_result, update) -> None:
    print("Resume vs Job Description Match Summary")
    print("=" * 40)
    print(f"Overall score: {match_result.overall_score:.2f}")
    print("\nSection scores:")
    for title, score in match_result.section_scores.items():
        print(f"  - {title}: {score:.2f}")
    print("\nMissing keywords:")
    if match_result.missing_keywords:
        for keyword in match_result.missing_keywords:
            print(f"  - {keyword}")
    else:
        print("  None 🎉")

    print("\nRecommended section order:")
    for idx, title in enumerate(update.reordered_sections, start=1):
        print(f"  {idx}. {title}")

    print("\nSuggested bullet points:")
    for title, section_update in update.updates.items():
        if not section_update.suggested_bullets:
            continue
        print(f"  {title}:")
        for bullet in section_update.suggested_bullets:
            print(f"    • {bullet}")


def _export_markdown(path: Path, resume, match_result, update) -> None:
    lines = ["# Tailored Resume Recommendations", ""]
    lines.append(f"**Overall Match Score:** {match_result.overall_score:.2f}")
    lines.append("")
    lines.append("## Section Scores")
    for title, score in match_result.section_scores.items():
        lines.append(f"- **{title}:** {score:.2f}")
    lines.append("")
    lines.append("## Missing Keywords")
    if match_result.missing_keywords:
        for keyword in match_result.missing_keywords:
            lines.append(f"- {keyword}")
    else:
        lines.append("- None 🎉")
    lines.append("")
    lines.append("## Recommended Section Order")
    for idx, title in enumerate(update.reordered_sections, start=1):
        lines.append(f"{idx}. {title}")
    lines.append("")
    lines.append("## Suggested Bullet Points")
    for title, section_update in update.updates.items():
        if not section_update.suggested_bullets:
            continue
        lines.append(f"### {title}")
        for bullet in section_update.suggested_bullets:
            lines.append(f"- {bullet}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _export_json(path: Path, match_result, update) -> None:
    payload: Dict[str, Any] = {
        "overall_score": match_result.overall_score,
        "section_scores": match_result.section_scores,
        "matched_keywords": match_result.matched_keywords,
        "missing_keywords": match_result.missing_keywords,
        "reordered_sections": update.reordered_sections,
        "suggested_bullets": {
            title: section_update.suggested_bullets for title, section_update in update.updates.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
