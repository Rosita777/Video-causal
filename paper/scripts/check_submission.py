#!/usr/bin/env python3
"""Fail closed when planning or preliminary material can reach submission."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission.tex"
MAIN_ONLY = ROOT / "main_only.tex"
BUILD_DIR = ROOT / "build"
MAIN_TEXT_PAGE_LIMIT = 9
OFFICIAL_FILES = {
    "iclr2027_conference.sty": (
        "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6"
    ),
    "iclr2027_conference.bst": (
        "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5"
    ),
    "natbib.sty": (
        "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d"
    ),
    "fancyhdr.sty": (
        "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea"
    ),
}
FORBIDDEN = (
    "TODO-PAPER",
    "TBD",
    "PLACEHOLDER",
    "Lorem",
    r"\outlineblock{",
    r"\claimboundary{",
    r"\iclrfinalcopy",
    "tables/preliminary",
    "/Users/",
)

LOG_FAILURES = (
    (r"(?m)^! ", "TeX error"),
    (r"(?:LaTeX|Package\s+\S+)\s+Error:", "LaTeX or package error"),
    (r"Undefined control sequence", "undefined control sequence"),
    (r"(?:Reference|Citation).*undefined", "undefined reference or citation"),
    (r"There were undefined (?:references|citations)", "undefined references"),
    (
        r"has\s+been\s+referenced\s+but\s+does\s+not\s+exist",
        "missing PDF destination",
    ),
    (r"Empty `thebibliography' environment", "empty bibliography"),
    (r"(?m)^Overfull \\[hv]box", "overfull box"),
    (r"Emergency stop", "emergency stop"),
    (r"Fatal error", "fatal TeX error"),
    (r"No pages of output", "no PDF output"),
)


def source_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.tex")
        if path.is_file() and BUILD_DIR not in path.parents
    )


def compile_driver(driver: str, failures: list[str]) -> Path | None:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk is None or pdflatex is None:
        missing = ", ".join(
            name
            for name, executable in (("latexmk", latexmk), ("pdflatex", pdflatex))
            if executable is None
        )
        failures.append(f"submission build tools unavailable: {missing}")
        return None

    BUILD_DIR.mkdir(exist_ok=True)
    command = (
        latexmk,
        "-gg",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir={BUILD_DIR}",
        driver,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        failures.append(f"{driver} build exceeded 600 seconds")
        return None

    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-12:])
        failures.append(f"{driver} failed to compile:\n{tail}")
        return None

    log_path = BUILD_DIR / f"{Path(driver).stem}.log"
    pdf_path = BUILD_DIR / f"{Path(driver).stem}.pdf"
    if not log_path.is_file() or not pdf_path.is_file():
        failures.append(f"{driver} build did not produce both a log and PDF")
        return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern, description in LOG_FAILURES:
        if re.search(pattern, log_text, flags=re.IGNORECASE):
            failures.append(f"{driver} log contains {description}")
    return log_path


def compile_checks(failures: list[str]) -> None:
    main_log = compile_driver("main_only.tex", failures)
    compile_driver("submission.tex", failures)
    if main_log is None:
        return

    log_text = main_log.read_text(encoding="utf-8", errors="replace")
    page_markers = re.findall(
        r"(?m)^ICLR-MAIN-TEXT-END-PAGE=([0-9]+)[ \t]*$", log_text
    )
    if len(page_markers) != 1:
        failures.append(
            "main_only.tex did not emit exactly one ICLR main-text page marker"
        )
        return
    main_pages = int(page_markers[0])
    if main_pages < 1 or main_pages > MAIN_TEXT_PAGE_LIMIT:
        failures.append(
            f"main text occupies {main_pages} pages; initial ICLR limit is "
            f"{MAIN_TEXT_PAGE_LIMIT}"
        )


def main() -> int:
    failures: list[str] = []
    submission_text = SUBMISSION.read_text(encoding="utf-8")
    main_only_text = MAIN_ONLY.read_text(encoding="utf-8")
    main_text = (ROOT / "main.tex").read_text(encoding="utf-8")
    if r"\def\paperprosemode{1}" not in submission_text:
        failures.append("submission.tex does not explicitly disable outline mode")
    if r"\def\paperwithoutappendix{1}" in submission_text:
        failures.append("submission.tex unexpectedly omits the inline appendix")
    if r"\iclrfinalcopy" in submission_text:
        failures.append("anonymous submission enables ICLR camera-ready mode")
    if r"\input{main.tex}" not in submission_text:
        failures.append("submission.tex does not load main.tex")
    if r"\def\paperprosemode{1}" not in main_only_text:
        failures.append("main_only.tex does not explicitly disable outline mode")
    if r"\def\paperwithoutappendix{1}" not in main_only_text:
        failures.append("main_only.tex does not explicitly omit the appendix")
    if r"\input{main.tex}" not in main_only_text:
        failures.append("main_only.tex does not load main.tex")
    if r"\usepackage{iclr2027_conference,times}" not in main_text:
        failures.append("main.tex does not load the official ICLR 2027 style")
    if re.search(
        r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\s*"
        r"\{[^}]*\bgeometry\b[^}]*\}",
        main_text,
    ):
        failures.append("main.tex still overrides official ICLR geometry")
    if r"\bibliographystyle{iclr2027_conference}" not in main_text:
        failures.append("main.tex does not use the official ICLR 2027 bibliography style")
    if r"\input{sections/08_ai_use_statement}" not in main_text:
        failures.append("main.tex omits the required ICLR 2027 AI-use statement")
    if "ICLR-MAIN-TEXT-END-PAGE=" not in main_text:
        failures.append("main.tex omits the main-text page-count marker")
    ordered_tokens = (
        r"\input{paper}",
        "ICLR-MAIN-TEXT-END-PAGE=",
        r"\input{sections/08_ai_use_statement}",
        r"\input{sections/09_ethics_statement}",
        r"\input{sections/10_reproducibility_statement}",
        r"\bibliography{references}",
        r"\input{supplement}",
    )
    positions = [main_text.find(token) for token in ordered_tokens]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        failures.append(
            "ICLR statements, references, and appendix are not in the required order"
        )

    for name, expected_sha in OFFICIAL_FILES.items():
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            failures.append(f"official template file missing or symlinked: {name}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected_sha:
            failures.append(f"official template file was modified: {name}")

    for path in source_files():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for token in FORBIDDEN:
                if path == ROOT / "preamble" / "macros.tex" and token in (
                    r"\outlineblock{",
                    r"\claimboundary{",
                ):
                    continue
                if token in line:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: forbidden {token!r}"
                    )

    if not failures:
        compile_checks(failures)

    if failures:
        print("submission check refused:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("submission check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
