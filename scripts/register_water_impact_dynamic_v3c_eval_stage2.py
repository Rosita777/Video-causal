#!/usr/bin/env python3
"""Freeze completed v3c training hashes before any eligible v3c generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import water_impact_dynamic_v3c_eval_protocol as protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    project_root = Path.cwd()
    path = protocol.resolve_path(project_root, protocol.STAGE2_REGISTRATION)
    if args.validate:
        registered_path, payload = protocol.load_stage2_registration(project_root)
        print(f"Validated stage-2 registration: {registered_path}")
    else:
        if path.exists():
            parser.error(f"refusing to overwrite stage-2 registration: {path}")
        payload = protocol.build_stage2_registration(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        protocol.load_stage2_registration(project_root)
        print(f"Frozen stage-2 registration: {path} SHA-256={protocol.file_sha256(path)}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
