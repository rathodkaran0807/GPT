from job_parser import parse_job_description


def test_parse_job_description_extracts_keywords() -> None:
    text = """Responsibilities:\n- Develop Python services\n- Collaborate with data science team\n\nRequired skills: Strong Python experience, cloud knowledge."""
    job = parse_job_description(text)

    assert any("python" in keyword for keyword in job.keywords)
    assert any("experience" in skill for skill in job.required_skills)
    assert "Develop Python services" in job.responsibilities
