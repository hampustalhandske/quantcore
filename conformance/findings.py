"""Parse hand-written finding prose out of `conformance/findings_src/<function>.md`.

Each source file holds the parts of a finding that do not change between conformance runs: what
is wrong, why, how the standard defines it, and how to fix and verify it. `build_report.py` reads
one of these per `defect`/`non-standard` function and merges it with this run's generated evidence
(case counts, worst error, an executed minimal repro) to produce
`conformance/findings/NNN-<function>.md`.

Parsing is pure and I/O-free (`parse_finding_source` takes text, not a path) so it is directly
unit-testable without touching disk.
"""

import dataclasses
import re
from pathlib import Path

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Title",
    "Severity",
    "Location",
    "Root cause",
    "Standard definition and citation",
    "Proposed patch",
    "Promotion test",
)


@dataclasses.dataclass(frozen=True)
class FindingSource:
    """One function's hand-written finding prose, split into its required sections.

    Attributes:
        title: One line naming the defect.
        severity: One line, e.g. `"high"`, each followed by a one-sentence justification of what a
            user of quantcore would get wrong.
        location: quantcore `file:line` and function name(s) the defect lives in.
        root_cause: Why the defect happens, in quantcore's own code.
        standard_definition: The standard definition of the quantity, with a citation.
        proposed_patch: A concrete, correctly-named code change.
        promotion_test: What must pass for this function to be reclassified `verified`.
    """

    title: str
    severity: str
    location: str
    root_cause: str
    standard_definition: str
    proposed_patch: str
    promotion_test: str


_SECTION_TO_FIELD: dict[str, str] = {
    "Title": "title",
    "Severity": "severity",
    "Location": "location",
    "Root cause": "root_cause",
    "Standard definition and citation": "standard_definition",
    "Proposed patch": "proposed_patch",
    "Promotion test": "promotion_test",
}

_SECTION_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_finding_source(text: str, *, source_name: str) -> FindingSource:
    """Parse one finding source file's Markdown body into a `FindingSource`.

    Args:
        text: The full Markdown content of a `conformance/findings_src/<function>.md` file.
        source_name: How to name the file in an error message (e.g. its path), so a missing
            section is diagnosable without re-deriving which file was being parsed.

    Returns:
        The parsed `FindingSource`, one field per required `## ` section, each stripped of
        leading/trailing whitespace.

    Raises:
        ValueError: A required `## <section>` heading is missing, naming both ``source_name`` and
            the missing section.
    """
    headings = list(_SECTION_HEADING_RE.finditer(text))
    bodies: dict[str, str] = {}
    for index, match in enumerate(headings):
        heading = match.group(1).strip()
        start = match.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        bodies[heading] = text[start:end].strip()

    missing = [section for section in REQUIRED_SECTIONS if section not in bodies]
    if missing:
        raise ValueError(
            f"{source_name}: missing required section(s) {', '.join(f'## {s}' for s in missing)}"
        )

    fields = {_SECTION_TO_FIELD[section]: bodies[section] for section in REQUIRED_SECTIONS}
    return FindingSource(**fields)


def load_finding_source(path: Path) -> FindingSource:
    """Read and parse a finding source file from disk.

    Args:
        path: Path to a `conformance/findings_src/<function>.md` file.

    Returns:
        The parsed `FindingSource`.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file is missing a required section (see `parse_finding_source`).
    """
    text = path.read_text(encoding="utf-8")
    return parse_finding_source(text, source_name=str(path))
