from pathlib import Path

from resume_parser import parse_resume, ResumeDocument


def test_parse_markdown_resume(tmp_path: Path) -> None:
    content = """EXPERIENCE\nLed engineering teams.\n\nEDUCATION\nStudied computer science."""
    path = tmp_path / "resume.md"
    path.write_text(content)

    document = parse_resume(path)
    assert isinstance(document, ResumeDocument)
    assert document.sections[0].title == "Experience"
    assert document.sections[1].title == "Education"
    assert "engineering" in document.sections[0].keywords
