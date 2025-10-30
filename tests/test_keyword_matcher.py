from resume_parser import ResumeDocument, ResumeSection
from job_parser import JobDescription
from matcher.keyword_matcher import KeywordMatcher


def test_keyword_matcher_scores_sections() -> None:
    resume = ResumeDocument(
        text="",
        sections=[
            ResumeSection(title="Experience", content="Built python data pipelines", keywords=["python", "pipelines"]),
            ResumeSection(title="Projects", content="Created cloud dashboards", keywords=["cloud", "dashboards"]),
        ],
    )
    job = JobDescription(
        text="",
        keywords=["python", "dashboards"],
        required_skills=["cloud experience"],
        responsibilities=[],
    )

    matcher = KeywordMatcher(required_weight=0.5, keyword_weight=0.5)
    result = matcher.score(resume, job)

    assert result.overall_score > 0
    assert "Experience" in result.section_scores
    assert "cloud experience" in result.missing_keywords or result.section_scores["Experience"] < 1
