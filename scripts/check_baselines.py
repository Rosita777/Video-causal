#!/usr/bin/env python3
"""Check local baseline source readiness without importing heavy ML packages."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


BASELINES = {
    "safree_cogvideox": {
        "root": "baselines/external/SAFREE",
        "required_files": [
            "cogvideox/README.md",
            "cogvideox/cogvideox_pipeline.py",
        ],
        "packages": ["torch", "diffusers", "transformers", "accelerate"],
    },
    "videoeraser_modelscope": {
        "root": "baselines/external/VideoEraser/ModelScope",
        "required_files": [
            "README.md",
            "requirements.txt",
            "inference.py",
            "train.py",
            "models/unet_3d_condition.py",
        ],
        "packages": ["torch", "diffusers", "transformers", "accelerate", "decord", "compel"],
    },
    "t2vunlearning": {
        "root": "baselines/external/T2VUnlearning",
        "required_files": [
            "README.md",
            "test_cogvideo.py",
            "test_hunyuan.py",
            "test_hunyuan_negprompt.py",
            "receler/concept_reg_cogvideo.py",
            "diffusers/setup.py",
        ],
        "packages": ["torch", "diffusers", "transformers", "accelerate", "omegaconf", "imageio"],
    },
}


def check_file(path: Path) -> dict:
    return {
        "path": str(path),
        "status": "ok" if path.is_file() else "missing",
    }


def check_package(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"package": name, "status": "ok" if spec is not None else "missing"}


def check_repo_readiness(root: Path, required_files: Iterable[str]) -> dict:
    files = []
    for rel in required_files:
        item = check_file(root / rel)
        item["relative_path"] = rel
        files.append(item)
    status = "ok" if all(item["status"] == "ok" for item in files) else "partial"
    return {"root": str(root), "status": status, "files": files}


def check_baseline(name: str) -> dict:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline: {name}")
    config = BASELINES[name]
    root = PROJECT_ROOT / config["root"]
    repo = check_repo_readiness(root, config["required_files"])
    packages = [check_package(pkg) for pkg in config["packages"]]
    runtime_status = "ok" if all(pkg["status"] == "ok" for pkg in packages) else "missing_packages"
    return {
        "baseline": name,
        "source": repo,
        "packages": packages,
        "runtime_status": runtime_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=sorted(BASELINES),
        help="Baseline to check. Can be passed multiple times. Defaults to all.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    names = args.baseline or sorted(BASELINES)
    report = {"baselines": [check_baseline(name) for name in names]}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
