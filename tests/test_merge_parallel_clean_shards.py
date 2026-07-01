from pathlib import Path
import json
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_merge_parallel_clean_shards_writes_review_ready_manifest(tmp_path):
    output_root = tmp_path / "parallel"
    shard0 = output_root / "clean_shards" / "prompt_000"
    shard2 = output_root / "clean_shards" / "prompt_002"
    shard0.mkdir(parents=True)
    shard2.mkdir(parents=True)
    (shard0 / "generation_manifest.json").write_text(
        json.dumps(
            {
                "baseline": "clean",
                "model": "models/CogVideoX-2b",
                "dry_run": False,
                "generation": {"seed": 9000},
                "items": [
                    {
                        "index": 0,
                        "prompt": "Prompt zero.",
                        "target_concept": "pebble",
                        "expected_effect": "ripples",
                        "seed": 9000,
                        "video_path": "outputs/run/clean_shards/prompt_000/videos/zero.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (shard2 / "generation_manifest.json").write_text(
        json.dumps(
            {
                "baseline": "clean",
                "model": "models/CogVideoX-2b",
                "dry_run": False,
                "generation": {"seed": 9000},
                "items": [
                    {
                        "index": 0,
                        "prompt": "Prompt two.",
                        "target_concept": "ball",
                        "expected_effect": "net bulges",
                        "seed": 9002,
                        "video_path": "outputs/run/clean_shards/prompt_002/videos/two.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_root / "parallel_job_manifest.json").write_text(
        json.dumps(
            {
                "dry_run": False,
                "prompts": "prompts/controls.txt",
                "model": "models/CogVideoX-2b",
                "baselines": ["clean"],
                "generation": {"seed": 9000},
                "jobs": [
                    {
                        "baseline": "clean",
                        "source_prompt_index": 2,
                        "output_dir": str(shard2),
                        "status": "finished",
                    },
                    {
                        "baseline": "clean",
                        "source_prompt_index": 0,
                        "output_dir": str(shard0),
                        "status": "finished",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "merge_parallel_clean_shards.py"),
            "--parallel-manifest",
            str(output_root / "parallel_job_manifest.json"),
            "--output",
            str(output_root / "clean" / "generation_manifest.json"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Merged 2 clean shard manifests" in result.stdout
    merged = json.loads((output_root / "clean" / "generation_manifest.json").read_text(encoding="utf-8"))
    assert merged["baseline"] == "clean"
    assert [item["source_prompt_index"] for item in merged["items"]] == [0, 2]
    assert [item["index"] for item in merged["items"]] == [0, 1]
    assert merged["items"][1]["seed"] == 9002
