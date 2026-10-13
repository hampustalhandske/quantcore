"""Tests for conformance.findings' pure Markdown-section parser.

`parse_finding_source` turns a finding source file's body into a `FindingSource`, one field per
required `## ` section. It has no file I/O, so every case here is exercised on literal text.
"""

import re

import pytest
from conformance.findings import REQUIRED_SECTIONS, FindingSource, parse_finding_source

_VALID_TEXT = """\
## Title

Example defect title.

## Severity

high: a user would get the wrong number.

## Location

quantcore/module.py:10, function `_thing`

## Root cause

Because of X.

## Standard definition and citation

Standard says Y (Author, Year).

## Proposed patch

Change X to Y.

## Promotion test

Must pass Z.
"""


def test_parse_finding_source_returns_one_field_per_required_section() -> None:
    """Every required section's body lands in the matching `FindingSource` field, trimmed."""
    parsed = parse_finding_source(_VALID_TEXT, source_name="example.md")
    assert parsed == FindingSource(
        title="Example defect title.",
        severity="high: a user would get the wrong number.",
        location="quantcore/module.py:10, function `_thing`",
        root_cause="Because of X.",
        standard_definition="Standard says Y (Author, Year).",
        proposed_patch="Change X to Y.",
        promotion_test="Must pass Z.",
    )


def _without_section(text: str, section: str) -> str:
    """Return ``text`` with the given `## <section>` heading and its body removed."""
    pattern = re.compile(rf"^## {re.escape(section)}\s*\n.*?(?=^## |\Z)", re.MULTILINE | re.DOTALL)
    return pattern.sub("", text)


@pytest.mark.parametrize("missing_section", REQUIRED_SECTIONS, ids=REQUIRED_SECTIONS)
def test_parse_finding_source_raises_on_missing_section(missing_section: str) -> None:
    """Dropping any one required section fails with an error naming the file and that section."""
    text_without_section = _without_section(_VALID_TEXT, missing_section)
    with pytest.raises(ValueError, match=re.escape("broken.md: missing required section")):
        parse_finding_source(text_without_section, source_name="broken.md")
    with pytest.raises(ValueError, match=re.escape(f"## {missing_section}")):
        parse_finding_source(text_without_section, source_name="broken.md")


def test_parse_finding_source_error_names_every_missing_section() -> None:
    """An error naming multiple sections lists each one, not just the first."""
    with pytest.raises(ValueError, match=re.escape("thin.md")) as exc_info:
        parse_finding_source("## Title\n\nOnly a title.\n", source_name="thin.md")
    message = str(exc_info.value)
    assert "thin.md" in message
    for section in REQUIRED_SECTIONS[1:]:
        assert f"## {section}" in message


def test_parse_finding_source_ignores_extra_sections() -> None:
    """A section beyond the required seven is simply not surfaced; it does not break parsing."""
    text = _VALID_TEXT + "\n## Notes\n\nExtra commentary that is not a required section.\n"
    parsed = parse_finding_source(text, source_name="example.md")
    assert parsed.promotion_test == "Must pass Z."
