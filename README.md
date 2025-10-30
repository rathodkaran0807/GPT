# GPT Resume Tailoring Toolkit

This repository provides a CLI workflow for parsing resumes, analyzing job descriptions,
matching candidate experience to employer requirements, and generating update
recommendations.

## Quick Start

1. **Install dependencies** (spaCy is optional but recommended for higher-quality phrase
   extraction):

   ```bash
   pip install -r requirements.txt  # or install the libraries listed below manually
   python -m spacy download en_core_web_sm
   ```

   If you are not using the provided `requirements.txt`, install the runtime dependencies
   individually:

   ```bash
   pip install pdfminer.six python-docx spacy
   ```

2. **Inspect the CLI options** to understand the available flags:

   ```bash
   python -m cli.main --help
   ```

3. **Run the CLI** to analyze a resume against a job description:

   ```bash
   python -m cli.main path/to/resume.md path/to/job_description.txt \
       --export recommendations.md \
       --json analysis.json
   ```

   This command prints a detailed match summary to the terminal, exports Markdown
   recommendations, and saves the raw analysis as JSON for further processing.

4. **Open the generated artifacts**:

   * `recommendations.md` – Tailored section order, missing keywords, and suggested bullet
     points ready for editing.
   * `analysis.json` – Structured scores and keyword matches suitable for automation or
     additional tooling.

## Components

### Resume Parser (`resume_parser/`)
* Loads PDF, DOCX, Markdown, and plain-text resumes.
* Splits documents into sections based on headings.
* Generates lightweight keyword candidates per section.

### Job Parser (`job_parser/`)
* Uses spaCy (or a regex-based fallback) to extract key phrases.
* Identifies responsibility bullet points and required skills.

### Matching Engine (`matcher/keyword_matcher.py`)
* Scores each resume section against job keywords and required skills.
* Reports matched keywords and remaining gaps.

### Resume Updater (`resume_updater/`)
* Suggests improved bullet points targeting missing keywords.
* Recommends an ideal section order based on match strength.

### CLI (`cli/main.py`)
* Accepts resume and job description files.
* Displays scores, missing keywords, and tailored bullet suggestions.
* Exports Markdown and JSON summaries for further editing.

## Testing

Run the unit test suite with `pytest` to confirm everything works end-to-end:

```bash
pytest
```

## Legacy Market Analysis Utilities

This repository also contains the original financial analysis helpers. Refer to the
following commands if you still need them:

```bash
python analyze_bats_csco.py
python renko_brick_optimizer.py --data 'BATS_CSCO, 5.csv'
python supertrend_backtest.py --data 'BATS_CSCO, 5.csv'
python machine_learning_patterns.py --data 'BATS_CSCO, 5.csv'
```
