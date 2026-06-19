# Experiment Log

This file is the chronological record of environment setup, reproduction attempts, failures, fixes, and conclusions. Every experiment entry should include date, command, environment, output location, and conclusion.

---

## 2026-06-16: Project Initialization

**Goal:** Create a clean working directory for video concept erasure causal-footprint experiments.

**Actions:**
- Created project structure under `/home/deepseek_VG/JUNCHI/video_concept_erasure_causal_footprint`.
- Added `README.md`, `environment.yml`, `docs/baseline_setup.md`, `prompts/causal_pilot.txt`, and `scripts/run_pilot.py`.
- Added tests for prompt parsing and dry-run manifest creation.

**Verification:**

```bash
python -m pytest tests/test_run_pilot.py -v
```

**Result:** 2 passed.

**Current limitation:** `scripts/run_pilot.py` only creates dry-run manifests. Heavy model inference still runs through baseline repos directly.

---

## 2026-06-16: Conda Environment Creation Attempt 1

**Command:**

```bash
/opt/miniconda3/bin/conda env create -f environment.yml
```

**Result:** Failed before solving because `defaults` channel requires Terms of Service acceptance:

```text
CondaToSNonInteractiveError: Terms of Service have not been accepted for https://repo.anaconda.com/pkgs/main and /pkgs/r
```

**Decision:** Do not accept ToS on the user's behalf. Remove `defaults` from `environment.yml` and add `nodefaults` so the environment uses only `pytorch`, `nvidia`, and `conda-forge` channels.

## 2026-06-16: Conda Environment Creation Attempt 2

**Command:**

```bash
/opt/miniconda3/bin/conda create -y -n vcecf --override-channels -c conda-forge python=3.10 pip
```

**Result:** Succeeded. Environment path: `/home/deepseek_VG/.conda/envs/vcecf`.

**Reason for command change:** `conda env create -f environment.yml` still injected global `defaults` channels and hit Anaconda ToS. The successful command explicitly overrides channels and uses only `conda-forge`.

## 2026-06-16: PyTorch Install Attempt 1

**Command:**

```bash
/home/deepseek_VG/.conda/envs/vcecf/bin/python -m pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

**Result:** Cancelled after ~11 minutes because the 2.2GB torch wheel stalled near 2.0GB at ~594 KB/s. No torch package was installed.

**Decision:** Keep the base `vcecf` environment. Install non-torch lightweight dependencies first, then retry torch as a separate resumable step or install the version required by the chosen baseline repo.

## 2026-06-16: Python User-Site Isolation Fix

**Issue discovered:**
During `pip install` inside `vcecf`, Python was still reading packages from `~/.local/lib/python3.10/site-packages`, which would pollute reproducibility.

**Fix applied:**
Added conda activation/deactivation hooks inside the environment:

- `PYTHONNOUSERSITE=1` on activate
- restored/unset on deactivate

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python - << 'PY'
import site
print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE)
PY
```

**Result:** `ENABLE_USER_SITE=False`

**Follow-up:** All package installs and experiment commands must be run with the isolated conda environment active.

## 2026-06-16: Conda Activation Hook Correction

**Issue discovered:** The first check output made the activate/deactivate hook contents look concatenated. To remove ambiguity, both hook files were rewritten explicitly.

**Fix applied:**
- `etc/conda/activate.d/project_isolation.sh` now only exports `PYTHONNOUSERSITE=1` and saves the previous value.
- `etc/conda/deactivate.d/project_isolation.sh` now only restores or unsets the previous value.

**Why it matters:** Environment hygiene must be reversible; activation hooks should not leave permanent shell state behind.

## 2026-06-16: Lightweight Package Install

**Command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pip install \
  pytest numpy pillow imageio tqdm pyyaml
```

**Result:** Succeeded.

**Installed tool-layer packages:** pytest, numpy, pillow, imageio, tqdm, pyyaml.

**Not installed yet:** torch, torchvision, torchaudio, diffusers, transformers, accelerate, xformers, imageio-ffmpeg, opencv-python. These are deferred until a baseline-specific setup because large downloads are slow/unstable on the current network.

## 2026-06-16: VideoEraser Clone

**Command:**

```bash
git clone --depth 1 https://github.com/bluedream02/VideoEraser.git baselines/external/VideoEraser
```

**Result:** Succeeded.

**Commit:** `ba19cceb561dda916614e609759eb5c5b54f1c83` (`Delete assets directory`).

**Initial read:** README supports AnimateDiff, ModelScope/ZeroScope, LaVie, and CogVideoX. For this project, ModelScope/ZeroScope is the likely first target because the repo has a direct `ModelScope/inference.py` entry.

## 2026-06-16: T2VUnlearning Clone Attempts

**Attempt 1:**

```bash
git clone --depth 1 https://github.com/VDIGPKU/T2VUnlearning.git baselines/external/T2VUnlearning
```

**Result:** Cancelled after ~4 minutes. Clone stalled during `index-pack` with only a partial `.git` directory.

**Attempt 2:**

```bash
curl -L --retry 2 --connect-timeout 20 --max-time 180 \
  -o /tmp/T2VUnlearning.zip \
  https://github.com/VDIGPKU/T2VUnlearning/archive/refs/heads/main.zip
```

**Result:** Cancelled. GitHub zip download timed out after ~3.5MB and retried from the beginning. Network too slow/unstable for this repo right now.

**Decision:** Cleaned partial directories. T2VUnlearning remains a P1 target, but it should not block P0 VideoEraser reproduction.

## 2026-06-16: T2VUnlearning Zip Import

**Source:** `/home/deepseek_VG/JUNCHI/video_concept_erasure_causal_footprint/T2VUnlearning-main.zip`

**Actions:**
- Inspected archive structure.
- Extracted to `baselines/external/T2VUnlearning/`.
- Confirmed official README and inference scripts are present.

**Repository state:**
- `README.md` marks model checkpoint and inference code as complete, training code as incomplete.
- Inference scripts exist for CogVideoX and HunyuanVideo.
- Negative prompting and SAFREE inference scripts are included.
- Package includes a bundled `diffusers/` source tree, so setup may need to follow the repo's exact install instructions rather than the project's shared environment.

**Practical note:** this baseline is now locally available, but reproduction should be driven by the repo's own README and may require a separate or isolated environment from `vcecf` if version conflicts appear.

## 2026-06-16: Baseline Readiness Checker

**Goal:** Add a lightweight check that verifies baseline source files and Python package availability without importing heavy ML packages or downloading model weights.

**Files added:**
- `scripts/check_baselines.py`
- `tests/test_check_baselines.py`

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests/test_check_baselines.py -v
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/check_baselines.py --output experiments/pilot_week1/baseline_readiness.json
```

**Result:** Tests passed. Readiness report shows both VideoEraser/ModelScope and T2VUnlearning source files are present, but runtime packages such as `torch`, `diffusers`, `transformers`, and `accelerate` are missing from the clean `vcecf` environment.

## 2026-06-19: Recovery Copy and GitHub Backup

Status: recovered from Codex conversation/log artifacts after the original /home/deepseek_VG/JUNCHI/video_concept_erasure_causal_footprint tree disappeared.

Kept for version control:
- project documentation and research notes;
- baseline orchestration scripts and lightweight tests;
- recovered CSV evidence for pilot rounds 1--3.

Not recovered and intentionally not tracked:
- generated videos, contact sheets, image review folders;
- external baseline repositories;
- model weights, adapters, checkpoints, and zip archives.

Repository hygiene decision: GitHub should store important code, docs, prompts, tests, and small CSV evidence only. Large media/model artifacts must remain outside git and be regenerated or downloaded when needed.

## 2026-06-19: Stable Git Working Copy and CogVideoX Clean Runner

**Goal:** Move active development from the volatile recovery copy into a stable Git-tracked path and rebuild the first runnable clean-source generation entry point.

**Stable path:**

```text
/home/deepseek_VG/JUNCHI/Video-causal
```

**GitHub remote:**

```text
https://github.com/Rosita777/Video-causal.git
```

**Network note:** Direct GitHub clone attempts failed twice with `GnuTLS recv error (-110)`. The stable copy was created from the already-synced local recovery repository while preserving `.git` and `origin`.

**Files added:**
- `scripts/generate_cogvideox_clean.py`
- `tests/test_generate_cogvideox_clean.py`

**Runner behavior:**
- `--dry-run` validates prompt parsing, planned video paths, seeds, generation parameters, and `generation_manifest.json` without importing heavy ML packages.
- Real generation lazily imports `torch`, `diffusers.CogVideoXPipeline`, and `diffusers.utils.export_to_video`.
- Default model ID is `zai-org/CogVideoX-2b`; local paths such as `models/CogVideoX-2b` can be passed with `--model`.
- Generated videos and manifests under `outputs/` remain outside git.

**Verification:**

```bash
python3 -m pytest tests/test_generate_cogvideox_clean.py -q
```

**Result:** `2 passed`.

## 2026-06-19: CogVideoX-2B Local Weights and Real Clean Smoke

**Goal:** Move from dry-run generation planning to real CogVideoX-2B clean-source video generation.

**Runtime fixes:**
- Reused `/home/deepseek_VG/.conda/envs/vcecf`.
- Fixed `transformers 4.51.3` import by downgrading `tokenizers` from `0.22.2` to `0.21.4`.
- Verified `CogVideoXPipeline` import with `diffusers 0.34.0`.
- PyTorch CUDA was only reliable when launching with `CUDA_VISIBLE_DEVICES=0`.

**Model download:**
- Direct `https://huggingface.co` access timed out.
- `HF_ENDPOINT=https://hf-mirror.com` worked.
- Downloaded `zai-org/CogVideoX-2b` to `models/CogVideoX-2b`.
- Model directory size after download: about 13G.
- `models/` remains ignored by git.

**Technical smoke:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_causal_screening.txt \
  --output-dir outputs/cogvideox_clean_tech_smoke \
  --model models/CogVideoX-2b \
  --limit 1 \
  --seed 42 \
  --steps 2 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Result:** succeeded and wrote one mp4 plus `generation_manifest.json`.

**Two-prompt clean smoke:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_clean_smoke.txt \
  --output-dir outputs/cogvideox_clean_v0_smoke \
  --model models/CogVideoX-2b \
  --limit 2 \
  --seed 100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Outputs:**
- `outputs/cogvideox_clean_v0_smoke/generation_manifest.json`
- `outputs/cogvideox_clean_v0_smoke/videos/000_a-realistic-video-of-a-red-ball-rolling-into-wooden-blocks-and-the-block_seed100.mp4`
- `outputs/cogvideox_clean_v0_smoke/videos/001_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed101.mp4`
- `outputs/cogvideox_clean_v0_smoke/review/contact_sheet.jpg`
- `outputs/cogvideox_clean_v0_smoke/review/annotation.csv`

**Initial contact-sheet screening:**
- `ice cube` / cola seed 101: usable clean source candidate; ice/liquid disturbance/bubbles are visible.
- `ball` / wooden blocks seed 100: not clean-valid; the red ball is visible but wooden blocks and the causal effect are absent.

**Decision:** Continue clean-source screening before applying erasure baselines. Invalid clean sources should be filtered out rather than interpreted as erasure failures.

## 2026-06-19: CogVideoX Clean Screening Round1 Seed200-205

**Goal:** Expand clean-source screening beyond the initial two-prompt smoke and prioritize templates that are likely to produce visible causal chains.

**Prompt file:**

```text
prompts/cogvideox_clean_screening_round1.txt
```

**Generation command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/cogvideox_clean_screening_round1_seed200 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Generated local artifacts:**
- `outputs/cogvideox_clean_screening_round1_seed200/generation_manifest.json`
- `outputs/cogvideox_clean_screening_round1_seed200/videos/`
- `outputs/cogvideox_clean_screening_round1_seed200/review/contact_sheet.jpg`
- `outputs/cogvideox_clean_screening_round1_seed200/review/annotation.csv`

These remain outside git.

**Tracked summary:**

```text
experiments/clean_screening/cogvideox_clean_screening_round1_seed200_summary.csv
```

**Initial contact-sheet screening:**

| Prompt ID | Clean-valid? | Notes |
| --- | --- | --- |
| `ice_cube_seed200` | yes | Ice cube and cola disturbance/bubbles are visible. |
| `bottle_seed201` | no | Bottle mouth and stream visible, but cup/filling effect is too weak or absent. |
| `pitcher_seed202` | no | Looks like a static glass/tube; pitcher and clear pouring event are absent. |
| `pipette_seed203` | no | Ink diffusion is strong, but pipette target source is not visible enough. |
| `stone_seed204` | yes | Stone/impact point and expanding ripples are visible. |
| `sugar_cube_seed205` | no | Sugar cube and swirl/dissolve effect are not visible. |

**Decision:** Use `ice_cube_seed200` and `stone_seed204` as immediate clean-valid candidates for first baseline runner tests. Continue generating more seeds for pitcher/bottle/pipette if those concepts are needed for broader coverage.
