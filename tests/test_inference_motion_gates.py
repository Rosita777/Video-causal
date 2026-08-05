#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_inference_motion_gates.py"
    spec = importlib.util.spec_from_file_location("build_inference_motion_gates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InferenceMotionGateTest(unittest.TestCase):
    def test_strict_red_mask_rejects_pink_background(self) -> None:
        module = load_module()
        video = torch.zeros((5, 32, 48, 3), dtype=torch.float32)
        video[:] = torch.tensor([220.0, 140.0, 140.0])
        video[2:4, 12:20, 20:28] = torch.tensor([240.0, 20.0, 20.0])

        gate, target, motion = module.build_gate(video, (3, 16, 24), 0.97)

        self.assertEqual(tuple(gate.shape), (3, 16, 24))
        self.assertGreater(float(target[:, 8, 12].max()), 0.5)
        self.assertEqual(float(target[:, 0, 0].max()), 0.0)
        self.assertLess(float(gate.mean()), 0.25)
        self.assertTrue(torch.isfinite(motion).all())


if __name__ == "__main__":
    unittest.main()
