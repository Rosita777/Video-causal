#!/usr/bin/env python3
"""Public-only capacity validation for the frozen v4_dev72_v3 causal graph.

The exact commands deliberately require an explicit ``--exact`` flag and a
safe project root, then publish only to the registered public artifact paths.
Running the program without arguments executes only a small, non-authorizing
smoke simulation; it can never start the one-million-iteration confirmation
accidentally.

This module contains no v2 candidate identity, prompt, score, seed, or media
input.  Its empirical inputs are the six public aggregate counts x/8 recorded
by the v3 preregistration amendment.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import hashlib
import itertools
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


PROTOCOL = "water_impact_dynamic_v4_dev72_v3_capacity_planning_v1"
DATASET_VERSION = "v4_dev72_v3"
REQUIRED_NUMPY_VERSION = "2.4.6"
BIT_GENERATOR = "PCG64"
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722

CELL_NAMES = (
    "holdout_source_new_receiver:direct",
    "holdout_source_new_receiver:natural",
    "holdout_source_seen_receiver:direct",
    "holdout_source_seen_receiver:natural",
    "seen_source_new_receiver:direct",
    "seen_source_new_receiver:natural",
)
PUBLIC_V2_ELIGIBLE = (4, 1, 6, 8, 4, 1)
V2_TRIALS_PER_CELL = 8
V3_CELL_SIZES = (48, 168, 24, 24, 96, 216)
ANCHOR_COUNTS = (24, 24, 8, 8, 8, 8)
EDGES_PER_ANCHOR = (2, 7, 3, 3, 12, 27)
GROUP_NAMES = ("G1", "G2", "G3", "global")

R1_DIRECT_OFFSETS = (0, 11)
R1_NATURAL_OFFSETS = (0, 3, 7, 11, 15, 19, 22)

SEARCH_DOMAIN = "water-impact-dynamic-v4-dev72-v3-capacity-search-v1\n"
CONFIRM_RHO010_DOMAIN = (
    "water-impact-dynamic-v4-dev72-v3-anchor-risk-mc-confirm-rho010-v1\n"
)
CONFIRM_RHO020_DOMAIN = (
    "water-impact-dynamic-v4-dev72-v3-anchor-risk-mc-confirm-rho020-v1\n"
)
CONFIRM_SHARED_FRAILTY_DOMAIN = (
    "water-impact-dynamic-v4-dev72-v3-anchor-risk-mc-confirm-shared-frailty-v1\n"
)
SMOKE_DOMAIN = "water-impact-dynamic-v4-dev72-v3-capacity-smoke-v1\n"
CONFIRMATION_PROFILE_ORDER = ("rho010", "rho020", "shared-frailty")

PUBLIC_DATA_ROOT = Path("data/water_impact_dynamic_v4")
STANDARD_FORMAL_OUTPUTS = {
    "search": PUBLIC_DATA_ROOT / "v4_causal_capacity_search_v3.json",
    "confirm": PUBLIC_DATA_ROOT / "v4_causal_capacity_confirm_v3.json",
}
FORBIDDEN_OUTPUT_TOKENS = ("sealed", "final36")

SEARCH_ITERATIONS = 200_000
CONFIRM_ITERATIONS = 1_000_000
FROZEN_BATCH_SIZE = 5_000
SEARCH_WILSON_CEILING = 0.145
CONFIRM_WILSON_CEILING = 0.15
DEFAULT_SMOKE_ITERATIONS = 500
MAX_SMOKE_ITERATIONS = 50_000

REFERENCE_RESULTS: dict[str, dict[str, Any]] = {
    "search": {
        "domain": SEARCH_DOMAIN,
        "rho": 0.10,
        "shared_frailty": False,
        "iterations": SEARCH_ITERATIONS,
        "global_failures": 28_527,
        "global_rate": 0.142635,
        "global_wilson_upper95": 0.1439260337,
    },
    "rho010": {
        "domain": CONFIRM_RHO010_DOMAIN,
        "rho": 0.10,
        "shared_frailty": False,
        "iterations": CONFIRM_ITERATIONS,
        "cell_shortage_failures": (1_566, 25_794, 1_150, 9, 197, 25_641),
        "group_failures": (42_316, 8_292, 98_317, 143_547),
        "global_failures": 143_547,
        "global_rate": 0.143547,
        "global_wilson_upper95": 0.1441246991,
    },
    "rho020": {
        "domain": CONFIRM_RHO020_DOMAIN,
        "rho": 0.20,
        "shared_frailty": False,
        "iterations": CONFIRM_ITERATIONS,
        "global_failures": 264_002,
        "global_rate": 0.264002,
        "global_wilson_upper95": 0.2647276898,
    },
    "shared-frailty": {
        "domain": CONFIRM_SHARED_FRAILTY_DOMAIN,
        "rho": 0.10,
        "shared_frailty": True,
        "iterations": CONFIRM_ITERATIONS,
        "group_failures": (42_565, 13_251, 99_458, 149_245),
        "global_failures": 149_245,
        "global_rate": 0.149245,
        "global_wilson_upper95": 0.1498320593,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_record(domain: str) -> dict[str, Any]:
    payload = domain.encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return {
        "domain": domain,
        "domain_sha256": digest.hex(),
        "uint64_first_8_bytes_big_endian": int.from_bytes(digest[:8], "big"),
    }


def beta_binomial_shortage_probability(
    n: int, x: int, *, prior_alpha: float, prior_beta: float
) -> float:
    """Return Pr(K < 4) under the registered beta-binomial posterior predictive."""

    if n < 0 or not 0 <= x <= V2_TRIALS_PER_CELL:
        raise ValueError("invalid beta-binomial count")
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("beta prior parameters must be positive")
    alpha = x + prior_alpha
    beta = V2_TRIALS_PER_CELL - x + prior_beta
    log_beta_denominator = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(
        alpha + beta
    )
    total = 0.0
    for eligible in range(min(3, n) + 1):
        log_choose = (
            math.lgamma(n + 1)
            - math.lgamma(eligible + 1)
            - math.lgamma(n - eligible + 1)
        )
        log_beta_numerator = (
            math.lgamma(eligible + alpha)
            + math.lgamma(n - eligible + beta)
            - math.lgamma(n + alpha + beta)
        )
        total += math.exp(log_choose + log_beta_numerator - log_beta_denominator)
    return total


def analytic_capacity_report() -> dict[str, Any]:
    models = {
        "M0_uniform_Beta_1_1": (1.0, 1.0, 0.05),
        "M1_Jeffreys_Beta_0p5_0p5": (0.5, 0.5, 0.15),
    }
    output: dict[str, Any] = {}
    for name, (prior_alpha, prior_beta, ceiling) in models.items():
        shortages = tuple(
            beta_binomial_shortage_probability(
                n,
                x,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
            )
            for x, n in zip(PUBLIC_V2_ELIGIBLE, V3_CELL_SIZES)
        )
        familywise = 1.0 - math.prod(1.0 - value for value in shortages)
        output[name] = {
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "cell_shortage_probabilities": dict(zip(CELL_NAMES, shortages)),
            "familywise_shortage_probability": familywise,
            "ceiling": ceiling,
            "passes": familywise <= ceiling,
        }
    return output


def wilson_upper_one_sided_95(failures: int, total: int) -> float:
    if total <= 0 or not 0 <= failures <= total:
        raise ValueError("Wilson counts are invalid")
    proportion = failures / total
    z = WILSON_Z_ONE_SIDED_95
    denominator = 1.0 + z * z / total
    return (
        proportion
        + z * z / (2.0 * total)
        + z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
    ) / denominator


def build_graphs() -> dict[str, tuple[tuple[int, ...], ...]]:
    g1_direct = tuple(
        tuple((head + offset) % 24 for offset in R1_DIRECT_OFFSETS)
        for head in range(24)
    )
    g1_natural = tuple(
        tuple((head + offset) % 24 for offset in R1_NATURAL_OFFSETS)
        for head in range(24)
    )
    g2_direct = tuple(tuple(anchor for _ in range(3)) for anchor in range(8))
    g2_natural = tuple(tuple(anchor for _ in range(3)) for anchor in range(8))
    g3_direct: list[tuple[int, ...]] = []
    g3_natural: list[tuple[int, ...]] = []
    for anchor in range(8):
        direct = tuple(
            sorted(
                {
                    4 * ((anchor + block_offset) % 8) + offset
                    for block_offset in (1, 2, 3)
                    for offset in range(4)
                }
            )
        )
        own = {4 * anchor + offset for offset in range(4)}
        extra = {4 * ((anchor + 4) % 8) + (anchor % 4)}
        natural = tuple(sorted(set(range(32)) - own - extra))
        g3_direct.append(direct)
        g3_natural.append(natural)
    return {
        "G1-D": g1_direct,
        "G1-N": g1_natural,
        "G2-D": g2_direct,
        "G2-N": g2_natural,
        "G3-D": tuple(g3_direct),
        "G3-N": tuple(g3_natural),
    }


GRAPHS = build_graphs()


def receiver_masks(graph: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(sum(1 << receiver for receiver in row) for row in graph)


def _g1_complete_cached(
    direct: tuple[int, ...], natural: tuple[int, ...]
) -> bool:
    if len(direct) != 24 or len(natural) != 24:
        raise ValueError("G1 requires exactly 24 physical heads")

    @functools.lru_cache(maxsize=None)
    def search(
        head: int, selected: int, direct_left: int, used_receivers: int
    ) -> bool:
        if selected == 8:
            return direct_left == 0
        if head == 24:
            return False
        needed = 8 - selected
        remaining = 24 - head
        if remaining < needed or direct_left < 0 or direct_left > needed:
            return False
        if direct_left > 0:
            available = direct[head] & ~used_receivers
            while available:
                bit = available & -available
                available ^= bit
                if search(
                    head + 1,
                    selected + 1,
                    direct_left - 1,
                    used_receivers | bit,
                ):
                    return True
        if needed > direct_left:
            available = natural[head] & ~used_receivers
            while available:
                bit = available & -available
                available ^= bit
                if search(
                    head + 1,
                    selected + 1,
                    direct_left,
                    used_receivers | bit,
                ):
                    return True
        return search(head + 1, selected, direct_left, used_receivers)

    return search(0, 0, 4, 0)


def g1_complete(direct: Sequence[int], natural: Sequence[int]) -> bool:
    """Exact G1 8-edge, 4/4, distinct-head, distinct-receiver completion."""

    return _g1_complete_cached(tuple(map(int, direct)), tuple(map(int, natural)))


def fixed_anchor_group_complete(
    direct: Sequence[int], natural: Sequence[int]
) -> bool:
    """Exact 8-anchor, 4/4 completion with a receiver perfect matching."""

    direct_tuple = tuple(map(int, direct))
    natural_tuple = tuple(map(int, natural))
    if len(direct_tuple) != 8 or len(natural_tuple) != 8:
        raise ValueError("fixed-anchor groups require exactly eight anchors")

    @functools.lru_cache(maxsize=None)
    def search(anchor: int, direct_left: int, used_receivers: int) -> bool:
        if anchor == 8:
            return direct_left == 0
        remaining = 8 - anchor
        if direct_left < 0 or direct_left > remaining:
            return False
        if direct_left > 0:
            available = direct_tuple[anchor] & ~used_receivers
            while available:
                bit = available & -available
                available ^= bit
                if search(anchor + 1, direct_left - 1, used_receivers | bit):
                    return True
        if remaining > direct_left:
            available = natural_tuple[anchor] & ~used_receivers
            while available:
                bit = available & -available
                available ^= bit
                if search(anchor + 1, direct_left, used_receivers | bit):
                    return True
        return False

    return search(0, 4, 0)


def g2_complete(direct_available: Sequence[bool], natural_available: Sequence[bool]) -> bool:
    """Exact G2 completion over all 70 four-direct anchor assignments."""

    direct = tuple(bool(value) for value in direct_available)
    natural = tuple(bool(value) for value in natural_available)
    if len(direct) != 8 or len(natural) != 8:
        raise ValueError("G2 requires exactly eight historical-receiver anchors")
    for direct_anchors in itertools.combinations(range(8), 4):
        chosen = set(direct_anchors)
        if all(direct[a] if a in chosen else natural[a] for a in range(8)):
            return True
    return False


def perfect_matching_exists(masks: Sequence[int]) -> bool:
    rows = tuple(map(int, masks))
    if not rows:
        return True
    order = tuple(
        sorted(range(len(rows)), key=lambda index: bin(rows[index]).count("1"))
    )

    @functools.lru_cache(maxsize=None)
    def search(position: int, used: int) -> bool:
        if position == len(order):
            return True
        available = rows[order[position]] & ~used
        while available:
            bit = available & -available
            available ^= bit
            if search(position + 1, used | bit):
                return True
        return False

    return search(0, 0)


def graph_specification(
    graphs: Mapping[str, Sequence[Sequence[int]]] = GRAPHS,
) -> dict[str, Any]:
    graph = {
        "cell_order": list(CELL_NAMES),
        "anchor_counts": list(ANCHOR_COUNTS),
        "edges_per_anchor": list(EDGES_PER_ANCHOR),
        "cell_sizes": list(V3_CELL_SIZES),
        "G1-D": [list(row) for row in graphs["G1-D"]],
        "G1-N": [list(row) for row in graphs["G1-N"]],
        "G2-D": [list(row) for row in graphs["G2-D"]],
        "G2-N": [list(row) for row in graphs["G2-N"]],
        "G3-D": [list(row) for row in graphs["G3-D"]],
        "G3-N": [list(row) for row in graphs["G3-N"]],
    }
    return {
        **graph,
        "candidate_count": sum(V3_CELL_SIZES),
        "graph_sha256": sha256_bytes(canonical_json_bytes(graph)),
    }


def _degree_summary(graph: Sequence[Sequence[int]], receiver_count: int) -> dict[str, Any]:
    source_degrees = tuple(len(row) for row in graph)
    receiver_degrees = Counter(receiver for row in graph for receiver in row)
    if set(receiver_degrees) != set(range(receiver_count)):
        raise ValueError("graph leaves a receiver isolated")
    return {
        "source_degree_min": min(source_degrees),
        "source_degree_max": max(source_degrees),
        "receiver_degree_min": min(receiver_degrees.values()),
        "receiver_degree_max": max(receiver_degrees.values()),
        "receiver_degree_histogram": dict(
            sorted(Counter(receiver_degrees.values()).items())
        ),
    }


def graph_robustness_report(
    graphs: Mapping[str, Sequence[Sequence[int]]] = GRAPHS,
) -> dict[str, Any]:
    expected_shapes = {
        "G1-D": (24, 2),
        "G1-N": (24, 7),
        "G2-D": (8, 3),
        "G2-N": (8, 3),
        "G3-D": (8, 12),
        "G3-N": (8, 27),
    }
    if set(graphs) != set(expected_shapes):
        raise ValueError("capacity graph inventory is not exact")
    for name, (rows, degree) in expected_shapes.items():
        graph = graphs[name]
        if len(graph) != rows or any(
            len(row) != degree for row in graph
        ):
            raise ValueError(f"{name}: graph shape/edge uniqueness changed")
        if name not in {"G2-D", "G2-N"} and any(
            len(set(row)) != degree for row in graph
        ):
            raise ValueError(f"{name}: graph shape/edge uniqueness changed")
    if any(
        tuple(row) != (anchor, anchor, anchor)
        for name in ("G2-D", "G2-N")
        for anchor, row in enumerate(graphs[name])
    ):
        raise ValueError("G2 parallel edge topology changed")
    if any(
        not set(direct) <= set(natural)
        for direct, natural in zip(graphs["G1-D"], graphs["G1-N"])
    ) or any(
        not set(direct) <= set(natural)
        for direct, natural in zip(graphs["G3-D"], graphs["G3-N"])
    ):
        raise ValueError("direct graph is no longer a subset of natural")
    if any(
        not 0 <= receiver < 24
        for name in ("G1-D", "G1-N")
        for row in graphs[name]
        for receiver in row
    ) or any(
        not 0 <= receiver < 32
        for name in ("G3-D", "G3-N")
        for row in graphs[name]
        for receiver in row
    ):
        raise ValueError("capacity graph receiver index escaped its pool")
    actual_specification = graph_specification(graphs)
    expected_specification = graph_specification(GRAPHS)
    if (
        actual_specification["graph_sha256"]
        != expected_specification["graph_sha256"]
        or any(
            tuple(tuple(int(value) for value in row) for row in graphs[name])
            != GRAPHS[name]
            for name in expected_shapes
        )
    ):
        raise ValueError(
            "capacity graph adjacency differs from the frozen exact graph: "
            f"actual_sha256={actual_specification['graph_sha256']}"
        )

    g1_direct = receiver_masks(graphs["G1-D"])
    g1_natural = receiver_masks(graphs["G1-N"])
    r1_checked = 0
    r1_failures = 0
    for deleted_n in range(3):
        for deleted in itertools.combinations(range(24), deleted_n):
            deleted_mask = sum(1 << receiver for receiver in deleted)
            r1_checked += 1
            if not g1_complete(
                tuple(mask & ~deleted_mask for mask in g1_direct),
                tuple(mask & ~deleted_mask for mask in g1_natural),
            ):
                r1_failures += 1

    g3_direct = receiver_masks(graphs["G3-D"])
    g3_natural = receiver_masks(graphs["G3-N"])
    r3_checked = 0
    r3_failures = 0
    for direct_anchors in itertools.combinations(range(8), 4):
        direct_set = set(direct_anchors)
        assigned = tuple(
            g3_direct[anchor] if anchor in direct_set else g3_natural[anchor]
            for anchor in range(8)
        )
        for deleted_n in range(3):
            for deleted in itertools.combinations(range(32), deleted_n):
                deleted_mask = sum(1 << receiver for receiver in deleted)
                r3_checked += 1
                if not perfect_matching_exists(
                    tuple(mask & ~deleted_mask for mask in assigned)
                ):
                    r3_failures += 1

    report = {
        "protocol": PROTOCOL,
        "dataset_version": DATASET_VERSION,
        "candidate_count": sum(V3_CELL_SIZES),
        "graph_sha256": actual_specification["graph_sha256"],
        "degrees": {
            "G1-D": _degree_summary(graphs["G1-D"], 24),
            "G1-N": _degree_summary(graphs["G1-N"], 24),
            "G3-D": _degree_summary(graphs["G3-D"], 32),
            "G3-N": _degree_summary(graphs["G3-N"], 32),
        },
        "R1_delete_up_to_2": {
            "checked": r1_checked,
            "expected": 301,
            "failures": r1_failures,
        },
        "R3_assignments_x_delete_up_to_2": {
            "variant_assignments": math.comb(8, 4),
            "deletion_sets": sum(math.comb(32, value) for value in range(3)),
            "checked": r3_checked,
            "expected": 37_030,
            "failures": r3_failures,
        },
    }
    if (
        r1_checked != 301
        or r1_failures
        or r3_checked != 37_030
        or r3_failures
    ):
        raise ValueError("frozen capacity graph failed deletion robustness")
    return report


ORACLE_C_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>

static int dfs_group(const uint64_t *d, const uint64_t *n, int anchor,
                     int direct_left, uint64_t used) {
    if (anchor == 8) return direct_left == 0;
    int remaining = 8 - anchor;
    if (direct_left < 0 || direct_left > remaining) return 0;
    if (direct_left > 0) {
        uint64_t available = d[anchor] & ~used;
        while (available) {
            uint64_t bit = available & (~available + 1);
            available ^= bit;
            if (dfs_group(d, n, anchor + 1, direct_left - 1, used | bit)) return 1;
        }
    }
    if (remaining > direct_left) {
        uint64_t available = n[anchor] & ~used;
        while (available) {
            uint64_t bit = available & (~available + 1);
            available ^= bit;
            if (dfs_group(d, n, anchor + 1, direct_left, used | bit)) return 1;
        }
    }
    return 0;
}

static int dfs_g1(const uint64_t *d, const uint64_t *n, int head,
                  int selected, int direct_left, uint64_t used) {
    if (selected == 8) return direct_left == 0;
    if (head == 24) return 0;
    int needed = 8 - selected;
    int remaining = 24 - head;
    if (remaining < needed || direct_left < 0 || direct_left > needed) return 0;
    if (direct_left > 0) {
        uint64_t available = d[head] & ~used;
        while (available) {
            uint64_t bit = available & (~available + 1);
            available ^= bit;
            if (dfs_g1(d, n, head + 1, selected + 1,
                       direct_left - 1, used | bit)) return 1;
        }
    }
    if (needed > direct_left) {
        uint64_t available = n[head] & ~used;
        while (available) {
            uint64_t bit = available & (~available + 1);
            available ^= bit;
            if (dfs_g1(d, n, head + 1, selected + 1,
                       direct_left, used | bit)) return 1;
        }
    }
    return dfs_g1(d, n, head + 1, selected, direct_left, used);
}

void count_failures(const uint64_t *g1d, const uint64_t *g1n,
                    const uint8_t *g2d, const uint8_t *g2n,
                    const uint64_t *g3d, const uint64_t *g3n,
                    size_t count, uint64_t *out) {
    uint64_t g1_fail = 0, g2_fail = 0, g3_fail = 0, global_fail = 0;
    for (size_t i = 0; i < count; ++i) {
        int ok1 = dfs_g1(g1d + 24*i, g1n + 24*i, 0, 0, 4, 0);
        int ok3 = dfs_group(g3d + 8*i, g3n + 8*i, 0, 4, 0);
        int ok2 = 0;
        for (unsigned mask = 0; mask < 256 && !ok2; ++mask) {
            if (__builtin_popcount(mask) != 4) continue;
            int valid = 1;
            for (int a = 0; a < 8; ++a) {
                valid &= (mask & (1u << a)) ? g2d[8*i+a] : g2n[8*i+a];
            }
            ok2 = valid;
        }
        g1_fail += !ok1;
        g2_fail += !ok2;
        g3_fail += !ok3;
        global_fail += !(ok1 && ok2 && ok3);
    }
    out[0] = g1_fail;
    out[1] = g2_fail;
    out[2] = g3_fail;
    out[3] = global_fail;
}
""".lstrip()
ORACLE_C_SOURCE_SHA256 = sha256_bytes(ORACLE_C_SOURCE.encode("ascii"))


class PythonOracle:
    name = "python_exact_reference"

    def count_failures(
        self,
        g1_direct: np.ndarray,
        g1_natural: np.ndarray,
        g2_direct: np.ndarray,
        g2_natural: np.ndarray,
        g3_direct: np.ndarray,
        g3_natural: np.ndarray,
    ) -> np.ndarray:
        counts = np.zeros(4, dtype=np.uint64)
        for index in range(len(g1_direct)):
            ok1 = g1_complete(g1_direct[index], g1_natural[index])
            ok2 = g2_complete(g2_direct[index], g2_natural[index])
            ok3 = fixed_anchor_group_complete(g3_direct[index], g3_natural[index])
            counts += np.asarray(
                (not ok1, not ok2, not ok3, not (ok1 and ok2 and ok3)),
                dtype=np.uint64,
            )
        return counts


class CompiledOracle:
    name = "embedded_c_exact"

    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        self.library.count_failures.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.library.count_failures.restype = None

    def count_failures(
        self,
        g1_direct: np.ndarray,
        g1_natural: np.ndarray,
        g2_direct: np.ndarray,
        g2_natural: np.ndarray,
        g3_direct: np.ndarray,
        g3_natural: np.ndarray,
    ) -> np.ndarray:
        arrays = (
            np.ascontiguousarray(g1_direct, dtype=np.uint64),
            np.ascontiguousarray(g1_natural, dtype=np.uint64),
            np.ascontiguousarray(g2_direct, dtype=np.uint8),
            np.ascontiguousarray(g2_natural, dtype=np.uint8),
            np.ascontiguousarray(g3_direct, dtype=np.uint64),
            np.ascontiguousarray(g3_natural, dtype=np.uint64),
        )
        count = len(arrays[0])
        expected_shapes = ((count, 24), (count, 24), (count, 8), (count, 8), (count, 8), (count, 8))
        if tuple(array.shape for array in arrays) != expected_shapes:
            raise ValueError("compiled oracle input shapes are not exact")
        output = np.zeros(4, dtype=np.uint64)
        self.library.count_failures(
            arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arrays[1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arrays[2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            arrays[3].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            arrays[4].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arrays[5].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            count,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        )
        return output


@contextlib.contextmanager
def compiled_oracle() -> Iterator[CompiledOracle]:
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("exact compiled oracle requires a C compiler named cc")
    with tempfile.TemporaryDirectory(prefix="v4_dev72_v3_capacity_oracle_") as directory:
        root = Path(directory)
        source = root / "oracle.c"
        library = root / "oracle.so"
        source.write_text(ORACLE_C_SOURCE, encoding="ascii")
        subprocess.run(
            [compiler, "-O3", "-std=c99", "-shared", "-fPIC", str(source), "-o", str(library)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield CompiledOracle(ctypes.CDLL(str(library)))


def compiled_oracle_self_test(oracle: CompiledOracle) -> bool:
    rng = np.random.Generator(np.random.PCG64(0x5633434F5241434C))
    count = 48
    arrays = (
        rng.integers(0, 1 << 24, size=(count, 24), dtype=np.uint64),
        rng.integers(0, 1 << 24, size=(count, 24), dtype=np.uint64),
        rng.integers(0, 2, size=(count, 8), dtype=np.uint8),
        rng.integers(0, 2, size=(count, 8), dtype=np.uint8),
        rng.integers(0, 1 << 32, size=(count, 8), dtype=np.uint64),
        rng.integers(0, 1 << 32, size=(count, 8), dtype=np.uint64),
    )
    expected = PythonOracle().count_failures(*arrays)
    actual = oracle.count_failures(*arrays)
    if not np.array_equal(actual, expected):
        raise RuntimeError("compiled completion oracle differs from Python reference")
    return True


def _masks_for(
    theta: np.ndarray,
    graph: Sequence[Sequence[int]],
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw in frozen anchor-major call order, not one (B,A,E) call."""

    count, nodes = theta.shape
    if nodes != len(graph):
        raise ValueError("theta/graph anchor count mismatch")
    masks = np.zeros((count, nodes), dtype=np.uint64)
    eligible_count = np.zeros(count, dtype=np.int32)
    for node, receivers in enumerate(graph):
        draws = rng.random((count, len(receivers))) < theta[:, node, None]
        eligible_count += draws.sum(axis=1)
        for edge, receiver in enumerate(receivers):
            masks[:, node] |= draws[:, edge].astype(np.uint64) << np.uint64(receiver)
    return np.ascontiguousarray(masks), eligible_count


def draw_eligibility_batch(
    rng: Any,
    *,
    count: int,
    rho: float,
    shared_frailty: bool,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if count <= 0 or not 0.0 < rho < 1.0:
        raise ValueError("invalid Monte Carlo batch/rho")
    x = np.asarray(PUBLIC_V2_ELIGIBLE, dtype=np.float64)
    # Frozen call 1: all six cell posteriors in iteration x cell C order.
    p = rng.beta(x + 1.0, 9.0 - x, size=(count, 6))
    concentration = (1.0 - rho) / rho
    theta: list[np.ndarray] = []
    # Frozen calls 2..7: one cell at a time, iteration x anchor C order.
    for cell, anchors in enumerate(ANCHOR_COUNTS):
        theta.append(
            rng.beta(
                p[:, cell, None] * concentration,
                (1.0 - p[:, cell, None]) * concentration,
                size=(count, anchors),
            )
        )
    if shared_frailty:
        # Exactly three group-ordered calls after all theta and before uniforms.
        for direct_cell, natural_cell in ((0, 1), (2, 3), (4, 5)):
            frailty = (
                rng.beta(9.0, 1.0, size=(count, ANCHOR_COUNTS[direct_cell])) / 0.9
            )
            theta[direct_cell] = np.minimum(
                1.0, theta[direct_cell] * frailty
            )
            theta[natural_cell] = np.minimum(
                1.0, theta[natural_cell] * frailty
            )
    masks: list[np.ndarray] = []
    edge_counts: list[np.ndarray] = []
    # Frozen order: cell, then a separate RNG call per anchor, then iteration/edge.
    for cell, name in enumerate(("G1-D", "G1-N", "G2-D", "G2-N", "G3-D", "G3-N")):
        cell_masks, cell_counts = _masks_for(theta[cell], GRAPHS[name], rng)
        masks.append(cell_masks)
        edge_counts.append(cell_counts)
    return tuple(masks), tuple(edge_counts)


def _readiness_g1(direct: np.ndarray, natural: np.ndarray) -> np.ndarray:
    d_ready = direct != 0
    n_ready = natural != 0
    return (
        (d_ready.sum(axis=1) >= 4)
        & (n_ready.sum(axis=1) >= 4)
        & ((d_ready | n_ready).sum(axis=1) >= 8)
    )


def _readiness_fixed(direct: np.ndarray, natural: np.ndarray) -> np.ndarray:
    d_ready = direct != 0
    n_ready = natural != 0
    missing_natural = ~n_ready
    missing_count = missing_natural.sum(axis=1)
    return (
        (missing_count <= 4)
        & np.all((~missing_natural) | d_ready, axis=1)
        & ((d_ready & n_ready).sum(axis=1) >= 4 - missing_count)
    )


def run_monte_carlo(
    *,
    iterations: int,
    batch_size: int,
    rho: float,
    seed: int,
    shared_frailty: bool,
    oracle: PythonOracle | CompiledOracle,
) -> dict[str, Any]:
    if iterations <= 0 or batch_size <= 0:
        raise ValueError("iterations and batch_size must be positive")
    if not 0 <= seed < 1 << 64:
        raise ValueError("PCG64 seed must be uint64")
    rng = np.random.Generator(np.random.PCG64(seed))
    exact_failures = np.zeros(4, dtype=np.uint64)
    readiness_failures = np.zeros(4, dtype=np.uint64)
    cell_shortages = np.zeros(6, dtype=np.uint64)
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        masks, edge_counts = draw_eligibility_batch(
            rng, count=count, rho=rho, shared_frailty=shared_frailty
        )
        for cell, counts in enumerate(edge_counts):
            cell_shortages[cell] += np.count_nonzero(counts < 4)
        g2_direct = np.ascontiguousarray(masks[2] != 0, dtype=np.uint8)
        g2_natural = np.ascontiguousarray(masks[3] != 0, dtype=np.uint8)
        ready1 = _readiness_g1(masks[0], masks[1])
        ready2 = _readiness_fixed(g2_direct, g2_natural)
        ready3 = _readiness_fixed(masks[4], masks[5])
        readiness_failures += np.asarray(
            (
                np.count_nonzero(~ready1),
                np.count_nonzero(~ready2),
                np.count_nonzero(~ready3),
                np.count_nonzero(~(ready1 & ready2 & ready3)),
            ),
            dtype=np.uint64,
        )
        exact_failures += oracle.count_failures(
            masks[0],
            masks[1],
            g2_direct,
            g2_natural,
            masks[4],
            masks[5],
        )
    global_failures = int(exact_failures[3])
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "rho": rho,
        "shared_frailty": shared_frailty,
        "cell_shortage_failures": {
            name: int(value) for name, value in zip(CELL_NAMES, cell_shortages)
        },
        "readiness_failure_counts": {
            name: int(value) for name, value in zip(GROUP_NAMES, readiness_failures)
        },
        "exact_failure_counts": {
            name: int(value) for name, value in zip(GROUP_NAMES, exact_failures)
        },
        "global_failure_rate": global_failures / iterations,
        "global_wilson_upper_one_sided_95": wilson_upper_one_sided_95(
            global_failures, iterations
        ),
    }


def validate_reference_result(profile: str, result: Mapping[str, Any]) -> None:
    reference = REFERENCE_RESULTS[profile]
    if result["iterations"] != reference["iterations"]:
        raise ValueError("reference iteration count differs")
    if result["rho"] != reference["rho"] or result["shared_frailty"] is not reference["shared_frailty"]:
        raise ValueError("reference scenario differs")
    exact = result["exact_failure_counts"]
    if "group_failures" in reference and tuple(
        exact[name] for name in GROUP_NAMES
    ) != reference["group_failures"]:
        raise RuntimeError("exact group failure counts differ from frozen reference")
    if exact["global"] != reference["global_failures"]:
        raise RuntimeError("global failure count differs from frozen reference")
    if "cell_shortage_failures" in reference and tuple(
        result["cell_shortage_failures"][name] for name in CELL_NAMES
    ) != reference["cell_shortage_failures"]:
        raise RuntimeError("cell shortage counts differ from frozen reference")
    for key, result_key in (
        ("global_rate", "global_failure_rate"),
        ("global_wilson_upper95", "global_wilson_upper_one_sided_95"),
    ):
        if not math.isclose(
            float(result[result_key]), float(reference[key]), rel_tol=0.0, abs_tol=5e-11
        ):
            raise RuntimeError(f"{result_key} differs from frozen reference")


def _ensure_no_symlink_components(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output path contains a symlink component: {current}")
    return absolute


def _reject_forbidden_output_path(*paths: Path) -> None:
    for path in paths:
        lowered = str(path).casefold()
        if any(token in lowered for token in FORBIDDEN_OUTPUT_TOKENS):
            raise ValueError("capacity output path may not reference sealed/final36")


def standard_formal_output(project_root: Path, profile: str) -> Path:
    if profile not in STANDARD_FORMAL_OUTPUTS:
        raise ValueError("unknown formal capacity output profile")
    lexical_root = Path(os.path.abspath(project_root))
    _reject_forbidden_output_path(lexical_root)
    real_root = _ensure_no_symlink_components(lexical_root)
    if not real_root.is_dir():
        raise ValueError("project root must be an existing real directory")
    resolved_root = real_root.resolve(strict=True)
    _reject_forbidden_output_path(resolved_root)
    output = resolved_root / STANDARD_FORMAL_OUTPUTS[profile]
    _reject_forbidden_output_path(output, output.parent.resolve(strict=True))
    _ensure_no_symlink_components(output.parent)
    return output


def write_json_exclusive_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    post_link_action: Callable[[str], None] | None = None,
) -> str:
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ValueError("output filename is empty")
    _reject_forbidden_output_path(absolute)
    parent = _ensure_no_symlink_components(absolute.parent)
    _reject_forbidden_output_path(parent.resolve(strict=True))
    if not parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    temporary = parent / f".{absolute.name}.tmp.{os.getpid()}"
    if absolute.exists() or absolute.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite capacity artifact: {absolute}")
    data = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=True
    ).encode("ascii") + b"\n"
    temporary_ownership: list[tuple[Path, tuple[int, int]]] = []
    namespace_changed = False

    def unlink_if_owned(
        candidate: Path, expected_inode: tuple[int, int]
    ) -> bool:
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != expected_inode
        ):
            return False
        candidate.unlink()
        return True

    def fsync_parent() -> None:
        directory_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def create_tracked_temporary() -> None:
        """Create/write the temp after exporting its descriptor identity."""

        nonlocal namespace_changed
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            namespace_changed = True
            try:
                opened = os.fstat(descriptor)
            except BaseException:
                # Recover only from the still-open descriptor; trusting the
                # pathname here could delete a foreign replacement.
                try:
                    opened = os.stat(descriptor)
                    temporary_ownership.append(
                        (temporary, (opened.st_dev, opened.st_ino))
                    )
                except BaseException:
                    pass
                raise
            temporary_ownership.append(
                (temporary, (opened.st_dev, opened.st_ino))
            )
            handle = os.fdopen(descriptor, "wb")
            descriptor = -1
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass

    try:
        create_tracked_temporary()
        if len(temporary_ownership) != 1:
            raise RuntimeError("capacity temporary ownership was not recorded")
        _, temporary_inode = temporary_ownership[0]
        temporary_info = os.lstat(temporary)
        if (
            not stat.S_ISREG(temporary_info.st_mode)
            or (temporary_info.st_dev, temporary_info.st_ino)
            != temporary_inode
            or temporary_info.st_nlink != 1
        ):
            raise RuntimeError(
                "capacity temporary inode changed before publication"
            )
        os.link(temporary, absolute, follow_symlinks=False)
        namespace_changed = True
        target_info = os.lstat(absolute)
        if (
            not stat.S_ISREG(target_info.st_mode)
            or (target_info.st_dev, target_info.st_ino) != temporary_inode
        ):
            raise RuntimeError("capacity output inode changed during publication")
        if not unlink_if_owned(temporary, temporary_inode):
            raise RuntimeError("capacity temporary inode changed during publication")
        fsync_parent()

        read_descriptor = os.open(
            absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            read_info = os.fstat(read_descriptor)
            if (
                not stat.S_ISREG(read_info.st_mode)
                or (read_info.st_dev, read_info.st_ino) != temporary_inode
                or read_info.st_nlink != 1
            ):
                raise RuntimeError("capacity output inode changed before readback")
            with os.fdopen(read_descriptor, "rb") as handle:
                read_descriptor = -1
                observed = handle.read()
        finally:
            if read_descriptor >= 0:
                os.close(read_descriptor)
        if observed != data:
            raise RuntimeError("capacity output readback mismatch")
        current_info = os.lstat(absolute)
        if (
            not stat.S_ISREG(current_info.st_mode)
            or (current_info.st_dev, current_info.st_ino) != temporary_inode
            or current_info.st_nlink != 1
        ):
            raise RuntimeError("capacity output inode changed after readback")
        digest = sha256_bytes(observed)
        if post_link_action is not None:
            post_link_action(digest)
    except BaseException:
        for owned_temporary, temporary_inode in temporary_ownership:
            try:
                namespace_changed = (
                    unlink_if_owned(absolute, temporary_inode)
                    or namespace_changed
                )
            except BaseException:
                pass
            try:
                namespace_changed = (
                    unlink_if_owned(owned_temporary, temporary_inode)
                    or namespace_changed
                )
            except BaseException:
                pass
        if namespace_changed:
            try:
                fsync_parent()
            except BaseException:
                pass
        raise
    finally:
        removed = False
        for owned_temporary, temporary_inode in temporary_ownership:
            try:
                removed = (
                    unlink_if_owned(owned_temporary, temporary_inode)
                    or removed
                )
            except BaseException:
                pass
        if removed:
            try:
                fsync_parent()
            except BaseException:
                pass
    return digest


def _require_exact_environment() -> None:
    if np.__version__ != REQUIRED_NUMPY_VERSION:
        raise RuntimeError(
            f"exact frozen runs require NumPy {REQUIRED_NUMPY_VERSION}; found {np.__version__}"
        )
    if np.random.PCG64.__name__ != BIT_GENERATOR:
        raise RuntimeError("exact frozen runs require NumPy PCG64")


def _run_exact_profile(
    profile: str, oracle: CompiledOracle
) -> dict[str, Any]:
    reference = REFERENCE_RESULTS[profile]
    seed = seed_record(reference["domain"])
    result = run_monte_carlo(
        iterations=int(reference["iterations"]),
        batch_size=FROZEN_BATCH_SIZE,
        rho=float(reference["rho"]),
        seed=int(seed["uint64_first_8_bytes_big_endian"]),
        shared_frailty=bool(reference["shared_frailty"]),
        oracle=oracle,
    )
    validate_reference_result(profile, result)
    return {"seed": seed, "result": result, "reference_match": True}


def _exact_artifact_base(
    *,
    status: str,
    self_test: bool,
    graph_robustness: Mapping[str, Any],
    analytic_models: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "dataset_version": DATASET_VERSION,
        "status": status,
        "numpy_version": np.__version__,
        "bit_generator": BIT_GENERATOR,
        "posterior": "M0 Beta(x+1,9-x)",
        "anchor_model": "Beta(p*kappa,(1-p)*kappa), kappa=(1-rho)/rho",
        "draw_order": (
            "per 5000-row batch: one (B,6) posterior-p call; six cell-ordered "
            "(B,A_c) theta calls; optional G1/G2/G3 frailty calls; then separate "
            "uniform calls in cell, anchor, iteration, edge order"
        ),
        "graph": graph_specification(),
        "graph_robustness": dict(graph_robustness),
        "analytic_models": dict(analytic_models),
        "oracle": {
            "engine": CompiledOracle.name,
            "embedded_c_source_sha256": ORACLE_C_SOURCE_SHA256,
            "self_test_against_python_reference": self_test,
        },
    }


def build_exact_search_artifact() -> dict[str, Any]:
    _require_exact_environment()
    graph_robustness = graph_robustness_report()
    analytic_models = analytic_capacity_report()
    with compiled_oracle() as oracle:
        self_test = compiled_oracle_self_test(oracle)
        scenario = _run_exact_profile("search", oracle)
    upper = scenario["result"]["global_wilson_upper_one_sided_95"]
    return {
        **_exact_artifact_base(
            status="exact_frozen_capacity_search_result",
            self_test=self_test,
            graph_robustness=graph_robustness,
            analytic_models=analytic_models,
        ),
        "profile": "search",
        **scenario,
        "decision": {
            "search_ceiling": SEARCH_WILSON_CEILING,
            "passes": upper <= SEARCH_WILSON_CEILING,
            "first_lattice_point": True,
            "larger_lattice_points_inspected": 0,
        },
    }


def build_combined_confirmation_artifact() -> dict[str, Any]:
    """Run all registered confirmation streams once and publish one artifact."""

    _require_exact_environment()
    graph_robustness = graph_robustness_report()
    analytic_models = analytic_capacity_report()
    scenarios: dict[str, Any] = {}
    with compiled_oracle() as oracle:
        self_test = compiled_oracle_self_test(oracle)
        for profile in CONFIRMATION_PROFILE_ORDER:
            scenarios[profile] = _run_exact_profile(profile, oracle)
    gate_upper = scenarios["rho010"]["result"][
        "global_wilson_upper_one_sided_95"
    ]
    return {
        **_exact_artifact_base(
            status="exact_frozen_capacity_confirmation_result",
            self_test=self_test,
            graph_robustness=graph_robustness,
            analytic_models=analytic_models,
        ),
        "profile": "combined_confirmation",
        "scenario_order": list(CONFIRMATION_PROFILE_ORDER),
        "scenarios": scenarios,
        "reference_match": all(
            scenario["reference_match"] for scenario in scenarios.values()
        ),
        "decision": {
            "authorization_scenario": "rho010",
            "confirmation_ceiling": CONFIRM_WILSON_CEILING,
            "passes": gate_upper <= CONFIRM_WILSON_CEILING,
            "rho020_report_only": True,
            "shared_frailty_report_only": True,
        },
    }


def smoke_report(
    *, iterations: int, batch_size: int, rho: float, shared_frailty: bool, engine: str
) -> dict[str, Any]:
    if not 1 <= iterations <= MAX_SMOKE_ITERATIONS:
        raise ValueError(
            f"smoke iterations must be in [1,{MAX_SMOKE_ITERATIONS}]"
        )
    seed = seed_record(SMOKE_DOMAIN)
    if engine == "python":
        result = run_monte_carlo(
            iterations=iterations,
            batch_size=batch_size,
            rho=rho,
            seed=int(seed["uint64_first_8_bytes_big_endian"]),
            shared_frailty=shared_frailty,
            oracle=PythonOracle(),
        )
        self_test: bool | None = None
        oracle_name = PythonOracle.name
    elif engine == "compiled":
        with compiled_oracle() as oracle:
            self_test = compiled_oracle_self_test(oracle)
            result = run_monte_carlo(
                iterations=iterations,
                batch_size=batch_size,
                rho=rho,
                seed=int(seed["uint64_first_8_bytes_big_endian"]),
                shared_frailty=shared_frailty,
                oracle=oracle,
            )
        oracle_name = CompiledOracle.name
    else:
        raise ValueError("unknown smoke oracle engine")
    return {
        "protocol": PROTOCOL,
        "dataset_version": DATASET_VERSION,
        "status": "non_authorizing_smoke",
        "numpy_version": np.__version__,
        "bit_generator": BIT_GENERATOR,
        "seed": seed,
        "graph_sha256": graph_specification()["graph_sha256"],
        "oracle": {
            "engine": oracle_name,
            "embedded_c_source_sha256": (
                ORACLE_C_SOURCE_SHA256 if engine == "compiled" else None
            ),
            "self_test_against_python_reference": self_test,
        },
        "result": result,
    }


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("analytic", help="print M0 and Jeffreys analytic planning values")
    subparsers.add_parser("graph-audit", help="run exact degree and delete<=2 graph checks")

    smoke = subparsers.add_parser(
        "smoke", help="run a bounded, non-authorizing Monte Carlo smoke test"
    )
    smoke.add_argument("--iterations", type=int, default=DEFAULT_SMOKE_ITERATIONS)
    smoke.add_argument("--batch-size", type=int, default=250)
    smoke.add_argument("--rho", type=float, choices=(0.10, 0.20), default=0.10)
    smoke.add_argument("--shared-frailty", action="store_true")
    smoke.add_argument("--engine", choices=("python", "compiled"), default="python")

    search = subparsers.add_parser(
        "search", help="run the exact frozen 200k first-lattice-point search"
    )
    search.add_argument("--exact", action="store_true", required=True)
    search.add_argument("--project-root", type=Path, required=True)

    confirm = subparsers.add_parser(
        "confirm", help="run all three frozen 1M confirmation streams in fixed order"
    )
    confirm.add_argument("--exact", action="store_true", required=True)
    confirm.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["smoke"]
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command == "analytic":
        _print_json(
            {
                "protocol": PROTOCOL,
                "dataset_version": DATASET_VERSION,
                "analytic_models": analytic_capacity_report(),
            }
        )
        return 0
    if args.command == "graph-audit":
        _print_json(graph_robustness_report())
        return 0
    if args.command == "smoke":
        _print_json(
            smoke_report(
                iterations=args.iterations,
                batch_size=args.batch_size,
                rho=args.rho,
                shared_frailty=args.shared_frailty,
                engine=args.engine,
            )
        )
        return 0
    if args.command == "search":
        output = standard_formal_output(args.project_root, "search")
        payload = build_exact_search_artifact()
        write_json_exclusive_atomic(
            output,
            payload,
            post_link_action=lambda digest: _print_json(
                {
                    "output": str(output),
                    "sha256": digest,
                    "profile": "search",
                    "reference_match": True,
                }
            ),
        )
        return 0
    if args.command == "confirm":
        output = standard_formal_output(args.project_root, "confirm")
        payload = build_combined_confirmation_artifact()
        write_json_exclusive_atomic(
            output,
            payload,
            post_link_action=lambda digest: _print_json(
                {
                    "output": str(output),
                    "sha256": digest,
                    "profile": "combined_confirmation",
                    "reference_match": True,
                }
            ),
        )
        return 0
    parser.error("a capacity command is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
