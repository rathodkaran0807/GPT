"""File loading helpers for resume parsing."""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def _load_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "pdfminer.six is required to parse PDF resumes. Install it with 'pip install pdfminer.six'."
        ) from exc

    return extract_text(str(path))


def _load_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "python-docx is required to parse DOCX resumes. Install it with 'pip install python-docx'."
        ) from exc

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_LOADERS: dict[str, Callable[[Path], str]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".txt": _load_text,
}


def load_text_from_file(path: Path) -> str:
    """Load textual content from supported resume formats."""

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported resume format '{suffix}'.")
    text = loader(path)
    if not isinstance(text, str):  # pragma: no cover - defensive
        raise TypeError("Loader did not return text")
    return text


__all__ = ["load_text_from_file"]
