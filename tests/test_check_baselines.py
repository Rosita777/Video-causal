from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_baselines import BASELINES, check_file, check_repo_readiness


def test_check_file_reports_present_and_missing(tmp_path):
    present = tmp_path / "present.py"
    present.write_text("print('ok')\n", encoding="utf-8")

    ok = check_file(present)
    missing = check_file(tmp_path / "missing.py")

    assert ok["status"] == "ok"
    assert ok["path"] == str(present)
    assert missing["status"] == "missing"


def test_check_repo_readiness_uses_relative_paths(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "baseline.py").write_text("print('baseline')\n", encoding="utf-8")
    (root / "README.md").write_text("# baseline\n", encoding="utf-8")

    result = check_repo_readiness(
        root,
        required_files=["README.md", "baseline.py", "missing.txt"],
    )

    assert result["root"] == str(root)
    assert result["status"] == "partial"
    statuses = {item["relative_path"]: item["status"] for item in result["files"]}
    assert statuses == {
        "README.md": "ok",
        "baseline.py": "ok",
        "missing.txt": "missing",
    }


def test_safree_cogvideox_readiness_tracks_official_pipeline():
    assert BASELINES["safree_cogvideox"]["root"] == "baselines/external/SAFREE"
    assert "cogvideox/cogvideox_pipeline.py" in BASELINES["safree_cogvideox"]["required_files"]
