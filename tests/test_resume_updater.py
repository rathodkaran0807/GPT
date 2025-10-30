from resume_parser import ResumeDocument, ResumeSection
from job_parser import JobDescription
from resume_updater import generate_resume_update


def test_generate_resume_update_produces_suggestions() -> None:
    resume = ResumeDocument(
        text="",
        sections=[
            ResumeSection(title="Experience", content="Led analytics initiatives", keywords=["analytics"]),
            ResumeSection(title="Skills", content="Python, SQL", keywords=["python", "sql"]),
        ],
    )
    job = JobDescription(
        text="",
        keywords=["analytics", "machine learning"],
        required_skills=["python experience"],
        responsibilities=[],
    )

    update = generate_resume_update(resume, job)

    assert update.reordered_sections
    assert "Experience" in update.updates
    assert any("machine learning" in bullet for bullet in update.updates["Experience"].suggested_bullets)
