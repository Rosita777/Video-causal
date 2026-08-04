#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def load_module():
    path = Path(__file__).parents[1] / "scripts" / "build_component_supervision_gates.py"
    spec = importlib.util.spec_from_file_location("build_component_supervision_gates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ComponentSupervisionGateTest(unittest.TestCase):
    def test_red_mask_selects_red_target(self) -> None:
        module = load_module()
        array = np.zeros((32, 48, 3), dtype=np.uint8)
        array[10:18, 20:28] = (220, 30, 20)
        array[2:8, 2:8] = (180, 170, 40)

        mask = module.red_object_mask([Image.fromarray(array)], (16, 24))

        self.assertGreater(float(mask[0, 7, 12]), 0.5)
        self.assertEqual(float(mask[0, 2, 2]), 0.0)

    def test_difference_mask_finds_changed_receiver(self) -> None:
        module = load_module()
        factual = np.zeros((32, 48, 3), dtype=np.uint8)
        target = factual.copy()
        factual[12:24, 8:36] = 220

        mask = module.video_difference_mask(
            [Image.fromarray(factual)] * 5,
            [Image.fromarray(target)] * 5,
            (3, 16, 24),
        )

        self.assertEqual(tuple(mask.shape), (3, 16, 24))
        self.assertGreater(float(mask[:, 8, 10].mean()), 0.5)
        self.assertEqual(float(mask[:, 0, 23].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
