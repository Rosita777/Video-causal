import json

from scripts.build_five_mechanism_eval_candidates import build_rows as build_candidates, smoke_rows
from scripts.build_five_mechanism_smoke_review import build_rows


def test_builds_joint_backbone_rows(tmp_path):
    candidates = {row["prompt"]: row for row in build_candidates()}
    selected = smoke_rows(build_candidates())
    manifests = []
    for backbone in ["wan", "cogvideox"]:
        path = tmp_path / f"{backbone}.json"
        path.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "prompt": row["prompt"],
                            "seed": 14000 + index,
                            "video_path": f"outputs/{backbone}/{index:03d}.mp4",
                        }
                        for index, row in enumerate(selected)
                    ]
                }
            ),
            encoding="utf-8",
        )
        manifests.append((backbone, path))

    rows = build_rows(candidates, manifests)

    assert len(rows) == 20
    assert {row["backbone"] for row in rows} == {"wan", "cogvideox"}
    assert len({row["candidate_id"] for row in rows}) == 10
    assert all(row["clean_source_valid"] == "" for row in rows)
