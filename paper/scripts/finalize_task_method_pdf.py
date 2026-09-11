"""Scale a vector schematic to manuscript width and record its provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]


def record(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_pdf", type=Path)
    args = ap.parse_args()
    target = ROOT / "paper/figures/main/fig_task_method_overview.pdf"
    reader = PdfReader(args.source_pdf)
    if len(reader.pages) != 1:
        raise ValueError("Expected a one-page schematic")
    page = reader.pages[0]
    page.scale_to(396, float(page.mediabox.height) * 396 / float(page.mediabox.width))
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({"/Title": "Causal-role erasure and the source-slot intervention",
                         "/Subject": "Schematic task and training design"})
    with target.open("wb") as stream:
        writer.write(stream)
    with pdfplumber.open(target) as pdf:
        p = pdf.pages[0]
        chars = [c for c in p.chars if c["text"].strip()]
        minimum_font = min(c["size"] for c in chars)
        vectors = len(p.lines) + len(p.rects) + len(p.curves)
        if p.images or vectors < 10 or minimum_font < 7:
            raise ValueError("Schematic must retain vector content and readable type")
        if any(c["x0"] < -0.5 or c["x1"] > p.width + 0.5 or c["top"] < -0.5
               or c["bottom"] > p.height + 0.5 for c in chars):
            raise ValueError("Text extends beyond the figure page")
        qa = {"page_count": 1, "width_points": p.width, "height_points": p.height,
              "minimum_type_points": minimum_font, "raster_image_count": len(p.images),
              "vector_path_count": vectors, "text_character_count": len(chars),
              "font_families": sorted({c["fontname"].split("+", 1)[-1] for c in chars})}

    inputs = [
        "paper/figures/main/source/fig_task_method_approved.pptx",
        "paper/figures/main/source/fig_task_method_overview.pptx",
        "paper/figures/main/source/LUCIDE_LICENSE.txt",
        "paper/scripts/prepare_task_method_figure.mjs",
        "paper/scripts/finalize_task_method_pdf.py",
        "paper/figures/main/fig_task_method_overview.tex",
        "paper/figures/main/fig_task_method_overview_spec.md",
        "paper/sections/03_problem_formulation.tex",
        "paper/sections/04_method.tex",
    ]
    manifest = {
        "schema_version": 1, "artifact_kind": "schematic_task_method_figure",
        "source_revision": "approved reference-style slide with publication-size label edits",
        "selection": {"applicable": False, "reason": "Schematic only; no result videos or scores used"},
        "examples": {"source": "ball", "replacement": "pebble", "receiver": "water",
                     "prompt_status": "illustrative; not a literal registered case"},
        "claim_scope": "Task and training design only; no empirical success or performance claim",
        "rendering": {"backend": "bundled LibreOffice Impress PDF export",
                      "typeface": "Arial", "icons": "Lucide 1.8.0 original SVG paths",
                      "raw_export_sha256": hashlib.sha256(args.source_pdf.read_bytes()).hexdigest(),
                      "normalization": "pypdf vector-preserving page scaling to 396 points"},
        "inputs": [record(p) for p in inputs],
        "output": record("paper/figures/main/fig_task_method_overview.pdf"),
        "checks": qa,
    }
    path = target.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
