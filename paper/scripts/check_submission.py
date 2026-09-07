#!/usr/bin/env python3
"""Fail closed when planning or preliminary material can reach submission."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission.tex"
SCAN_ROOTS = (
    ROOT / "main.tex",
    ROOT / "paper.tex",
    ROOT / "supplement.tex",
    ROOT / "sections",
    ROOT / "appendix",
    ROOT / "preamble",
)
FORBIDDEN = (
    "TODO-PAPER",
    "TBD",
    "PLACEHOLDER",
    "Lorem",
    r"\outlineblock{",
    r"\claimboundary{",
    "tables/preliminary",
    "/Users/",
)


def source_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.tex")))
    return files


def main() -> int:
    failures: list[str] = []
    submission_text = SUBMISSION.read_text(encoding="utf-8")
    if r"\def\paperfinalmode{1}" not in submission_text:
        failures.append("submission.tex does not explicitly disable outline mode")
    if r"\def\paperwithoutappendix{1}" not in submission_text:
        failures.append("submission.tex does not explicitly request main-only mode")

    for path in source_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for token in FORBIDDEN:
                if path.name == "macros.tex" and token in (
                    r"\outlineblock{",
                    r"\claimboundary{",
                ):
                    continue
                if token in line:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: forbidden {token!r}"
                    )

    if failures:
        print("submission check refused:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("submission check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
