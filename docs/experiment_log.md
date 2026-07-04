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

## 2026-06-19: Negative Prompt Round1 on CogVideoX Clean-Valid Sources

**Goal:** Run the first inference-time baseline on the current clean-valid CogVideoX-2B sources before implementing heavier baselines.

**Code change:** Extended `scripts/generate_cogvideox_clean.py` with:
- `--baseline clean` (default);
- `--baseline negative_prompt`, which passes each prompt's `target_concept` as `negative_prompt` to `CogVideoXPipeline`.

**Dry-run verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/generate_cogvideox_clean.py \
  --baseline negative_prompt \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/negative_prompt_round1_seed200_dryrun \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --dry-run
```

**Generation command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --baseline negative_prompt \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/negative_prompt_round1_seed200 \
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
- `outputs/negative_prompt_round1_seed200/generation_manifest.json`
- `outputs/negative_prompt_round1_seed200/videos/`
- `outputs/negative_prompt_round1_seed200/review/clean_valid_compare_contact_sheet.jpg`
- `outputs/negative_prompt_round1_seed200/review/clean_valid_compare_annotation.csv`

These remain outside git.

**Tracked summary:**

```text
experiments/baseline_runs/negative_prompt_round1_seed200_summary.csv
```

**Initial contact-sheet screening on clean-valid cases:**

| Prompt ID | Target visible? | Effect visible? | Outcome |
| --- | --- | --- | --- |
| `ice_cube_seed200` | no | yes | strict causal-footprint candidate |
| `stone_seed204` | no | yes | strict causal-footprint candidate |

**Decision:** Negative Prompt is now a reproduced baseline on CogVideoX-2B for the current clean-valid sources. Next baseline priority is SAFREE-CogVideoX on the same two cases, followed by VideoEraser/T2VUnlearning setup.

## 2026-06-20: Unified Baseline Suite Interface

**Motivation:** Future experiments should not reproduce baselines one at a time in an ad hoc way. Given a clean-valid prompt/seed set, the project should plan all required baselines together and make missing adapters explicit.

**File added:**
- `scripts/run_baseline_suite.py`
- `tests/test_run_baseline_suite.py`

**Suite command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_seed200 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --parallel \
  --dry-run
```

**Initial suite dry-run statuses before adapter restoration:**

| Baseline | Status |
| --- | --- |
| Negative Prompt | `ready` |
| SAFREE-CogVideoX | `blocked_missing_adapter` |
| VideoEraser | `blocked_missing_adapter` |
| T2VUnlearning | `blocked_missing_adapter` |

**Decision at this point:** The next engineering task was not another isolated run. It was to implement/restore adapters until SAFREE-CogVideoX, VideoEraser, and T2VUnlearning move from blocked to ready in the same suite interface. The SAFREE status is superseded by the later 2026-06-20 SAFREE adapter entry below.

`--parallel` is part of the suite contract. Once more adapters become ready, the same command can launch ready baselines together rather than forcing one-by-one reproduction.

## 2026-06-20: SAFREE-CogVideoX Adapter Restored

**Motivation:** SAFREE should be a first-class baseline in the unified suite, not a later one-off manual command.

**Files added/updated:**
- `scripts/adapters/run_safree_cogvideox.py`
- `scripts/run_baseline_suite.py`
- `scripts/check_baselines.py`
- `tests/test_run_safree_cogvideox.py`
- `tests/test_run_baseline_suite.py`
- `tests/test_check_baselines.py`

**External source state:** The official SAFREE CogVideoX pipeline was fetched locally into the ignored path:

```text
baselines/external/SAFREE/cogvideox/cogvideox_pipeline.py
```

The local wrapper injects each prompt row's `target_concept` into SAFREE's `CONCEPT_DICT` as `[target_concept]`, then passes the target string as the official pipeline's `concept` argument. This adapts SAFREE's safety-category interface to this project's arbitrary object/event concept-erasure prompts without treating it as Negative Prompt.

**Dry-run checks:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_seed200_safree_ready \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --parallel \
  --dry-run
```

Suite status from `outputs/baseline_suite_round1_seed200_safree_ready/suite_manifest.json`:

| Baseline | Status |
| --- | --- |
| Negative Prompt | `ready` |
| SAFREE-CogVideoX | `ready` |
| VideoEraser | `blocked_missing_adapter` |
| T2VUnlearning | `blocked_missing_adapter` |

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q
```

Result:

```text
17 passed in 0.34s
```

**Decision:** Future clean-valid video experiments should run the suite so Negative Prompt and SAFREE-CogVideoX launch together. The next adapter priorities are VideoEraser and then T2VUnlearning.


## 2026-06-20: Real Negative Prompt + SAFREE-CogVideoX Suite Run

**Goal:** Run the first real multi-baseline suite on the same CogVideoX-2B prompt/seed set instead of reproducing baselines one at a time.

**Initial failures and fixes:**
- `--enable-model-cpu-offload` failed in the sandbox with `RuntimeError: enable_model_cpu_offload requires accelerator, but not found`.
- Sandboxed PyTorch reported `cuda_available False` and `device_count 0`, even though `nvidia-smi` could see H800 GPUs.
- The same CUDA check outside the managed sandbox reported `cuda_available True`, `device_count 1`, `name NVIDIA H800`.
- SAFREE-CogVideoX failed as `fp16` with `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half` in the CogVideoX transformer time embedding path.
- A 1-step SAFREE GPU smoke with `--dtype fp32` succeeded, so the real suite used `fp32`.

**Successful command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --baseline negative_prompt \
  --baseline safree_cogvideox \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_seed200_real_gpu_fp32 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype fp32 \
  --enable-model-cpu-offload \
  --vae-tiling \
  --parallel
```

**Generated local artifacts:**
- `outputs/baseline_suite_round1_seed200_real_gpu_fp32/suite_manifest.json`
- `outputs/baseline_suite_round1_seed200_real_gpu_fp32/negative_prompt/generation_manifest.json`
- `outputs/baseline_suite_round1_seed200_real_gpu_fp32/safree_cogvideox/generation_manifest.json`
- 12 ignored `.mp4` files: 6 Negative Prompt and 6 SAFREE-CogVideoX videos.
- Review contact sheets for clean-valid `ice_cube_seed200` and `stone_seed204` under `outputs/baseline_suite_round1_seed200_real_gpu_fp32/review/`.

**Tracked summary:**

```text
experiments/baseline_runs/baseline_suite_round1_seed200_real_gpu_fp32_summary.csv
```

**Current status:** Generation succeeded. Manual visual review is pending; do not treat these rows as scientific outcomes until the contact sheets/videos are annotated.

## 2026-06-20: All Required Baselines Have Suite Interfaces

**Goal:** Make future experiments run from one baseline suite interface instead of one-off reproduction commands.

**Implemented interfaces:**
- `negative_prompt`: ready through `scripts/generate_cogvideox_clean.py --baseline negative_prompt`.
- `safree_cogvideox`: ready locally through `scripts/adapters/run_safree_cogvideox.py` when the ignored SAFREE pipeline is present.
- `videoeraser`: adapter added at `scripts/adapters/run_videoeraser_cogvideox.py`; current default status is `ready` through local `spea_arng_cogvideox_v0`, with optional `--mode external` for future official runners.
- `t2vunlearning`: adapter added at `scripts/adapters/run_t2vunlearning_cogvideox.py`; current default status is `ready` through local `receler_cogvideox_proxy_v0`, with optional `--mode external` for future official code/checkpoints.

**Important boundary:** These interfaces do not fake VideoEraser or T2VUnlearning outputs. They provide a stable dry-run manifest, external-file checks, and real-run delegation points. If a method generates weak, collapsed, or target-visible videos later, those are baseline outcomes to record, not reasons to remove the method.

**Path-handling fix:** The VideoEraser and T2VUnlearning adapters now resolve prompt, output, and local model paths before invoking external runners with the external repository as `cwd`. This prevents relative project paths from breaking after the subprocess changes working directory.

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests/test_run_baseline_suite.py tests/test_run_external_adapters.py -q
```

Result:

```text
9 passed
```

## 2026-06-20: VideoEraser Local Reimplementation v0

**Goal:** Stop blocking VideoEraser on unavailable or unstable external source code and provide a runnable CogVideoX baseline.

**Implementation:** `scripts/adapters/run_videoeraser_cogvideox.py` now defaults to `--mode local`, recorded as `spea_arng_cogvideox_v0`. The method is training-free: each prompt row gets an erased positive prompt where the `target_concept` is replaced by a neutral token, the original target concept is used as adversarial negative guidance, and prompt embeddings are displaced away from the original concept-bearing prompt using `--spea-strength`.

**Suite state after dry-run:**

```text
negative_prompt ready
safree_cogvideox ready
videoeraser ready local_reimplementation
t2vunlearning blocked_missing_external
```

**Successful smoke command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_videoeraser_cogvideox.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/videoeraser_local_gpu_smoke_fp32_limit1_step1_256x384 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 1 \
  --guidance-scale 6.0 \
  --num-frames 9 \
  --height 256 \
  --width 384 \
  --fps 8 \
  --dtype fp32 \
  --limit 1 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Artifacts:**
- `outputs/videoeraser_local_gpu_smoke_fp32_limit1_step1_256x384/generation_manifest.json`
- `outputs/videoeraser_local_gpu_smoke_fp32_limit1_step1_256x384/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`

**Resource note:** A full-size 480x720 / 49-frame / 1-step fp32 smoke failed with CUDA OOM on the crowded H800 node. Retrying full-size with `bf16`, model CPU offload, and VAE tiling succeeded later in the four-baseline suite smoke.

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q
```

Result:

```text
24 passed
```

## 2026-06-20: T2VUnlearning Local Reimplementation v0

**Goal:** Stop treating incomplete public training code/checkpoint availability as a blocker and provide a runnable CogVideoX T2VUnlearning baseline.

**Implementation:** `scripts/adapters/run_t2vunlearning_cogvideox.py` now defaults to `--mode local`, recorded as `receler_cogvideox_proxy_v0`. The local path mirrors the public inference contract: each prompt row records an unlearn concept and eraser rank; without a provided `--eraser-path`, generation uses a concept-suppressed prompt embedding plus target-concept negative guidance.

**Suite state after dry-run:**

```text
negative_prompt ready
safree_cogvideox ready
videoeraser ready local_reimplementation
t2vunlearning ready local_reimplementation
```

**Successful smoke command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_t2vunlearning_cogvideox.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-dir outputs/t2vunlearning_local_gpu_smoke_fp32_limit1_step1_256x384 \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 1 \
  --guidance-scale 6.0 \
  --num-frames 9 \
  --height 256 \
  --width 384 \
  --fps 8 \
  --dtype fp32 \
  --limit 1 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Artifacts:**
- `outputs/t2vunlearning_local_gpu_smoke_fp32_limit1_step1_256x384/generation_manifest.json`
- `outputs/t2vunlearning_local_gpu_smoke_fp32_limit1_step1_256x384/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q
```

Result:

```text
24 passed
```

## 2026-06-20: Full-Size Four-Baseline Suite Smoke

**Goal:** Test whether the current crowded H800 node can run all four baselines at full video shape, rather than stopping at 256x384 smoke tests.

**First attempt:** Full-size `bf16` suite failed on VideoEraser decode because `run_baseline_suite.py` did not pass `--enable-model-cpu-offload` and `--vae-tiling` through to the local VideoEraser/T2V adapter commands. A regression test now checks that local baseline commands inherit memory flags.

**Successful command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_all_local_bf16_limit1_step1_fullsize_seq_retry \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 1 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype bf16 \
  --limit 1 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Suite status:**

```text
negative_prompt ready
safree_cogvideox ready
videoeraser ready local_reimplementation
t2vunlearning ready local_reimplementation
```

**Generated ignored artifacts:**
- `outputs/baseline_suite_round1_all_local_bf16_limit1_step1_fullsize_seq_retry/negative_prompt/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`
- `outputs/baseline_suite_round1_all_local_bf16_limit1_step1_fullsize_seq_retry/safree_cogvideox/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`
- `outputs/baseline_suite_round1_all_local_bf16_limit1_step1_fullsize_seq_retry/videoeraser/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`
- `outputs/baseline_suite_round1_all_local_bf16_limit1_step1_fullsize_seq_retry/t2vunlearning/videos/000_a-realistic-close-up-video-of-a-clear-ice-cube-dropping-into-a-glass-of_seed200.mp4`

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q
```

Result:

```text
24 passed
```

## 2026-06-20: Full-Size Four-Baseline Suite, 10-Step Smoke

**Goal:** Move beyond 1-step smoke and test whether all four baselines can run at full video shape with a more meaningful denoising budget.

**Command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py   --prompts prompts/cogvideox_clean_screening_round1.txt   --output-root outputs/baseline_suite_round1_all_local_bf16_limit1_step10_fullsize_seq   --model models/CogVideoX-2b   --seed 200   --steps 10   --guidance-scale 6.0   --num-frames 49   --fps 8   --dtype bf16   --limit 1   --enable-model-cpu-offload   --vae-tiling
```

**Result:** Successful. Four full-size mp4 files and generation manifests were produced under:

```text
outputs/baseline_suite_round1_all_local_bf16_limit1_step10_fullsize_seq/
```

**Run shape:**

```text
negative_prompt: bf16, 10 steps, 49 frames, 1 prompt
safree_cogvideox: bf16, 10 steps, 49 frames, 1 prompt
videoeraser: bf16, 10 steps, 49 frames, 1 prompt
t2vunlearning: bf16, 10 steps, 49 frames, 1 prompt
```

**Interpretation:** The current node can run the complete four-baseline suite sequentially at full shape when using `bf16`, model CPU offload, and VAE tiling. Do not run these four baselines in parallel on the current crowded GPU allocation unless substantially more free memory is available.

## 2026-06-20: Full-Size Four-Baseline Suite, 20-Step Smoke

**Goal:** Confirm that the complete four-baseline CogVideoX-2B reproduction stack remains stable at a more useful denoising budget before expanding to more prompts.

**Command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_all_local_bf16_limit1_step20_fullsize_seq \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype bf16 \
  --limit 1 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Result:** Successful. All four baselines produced full-size mp4 files and manifests under:

```text
outputs/baseline_suite_round1_all_local_bf16_limit1_step20_fullsize_seq/
```

**Generated videos:**

```text
negative_prompt:   373735 bytes
safree_cogvideox:  628418 bytes
videoeraser:       249940 bytes
t2vunlearning:     277515 bytes
```

**Run shape:**

```text
Prompt: A realistic close-up video of a clear ice cube dropping into a glass of cola, and bubbles and splashes rise after the ice cube hits the drink.
Target concept: ice cube
Expected effect: bubbles and splashes rise

negative_prompt:   bf16, 20 steps, 480x720, 49 frames, 1 prompt
safree_cogvideox:  bf16, 20 steps, 480x720, 49 frames, 1 prompt
videoeraser:       bf16, 20 steps, 480x720, 49 frames, 1 prompt, local method spea_arng_cogvideox_v0
t2vunlearning:     bf16, 20 steps, 480x720, 49 frames, 1 prompt, local method receler_cogvideox_proxy_v0
```

**Interpretation:** The current reproduction interface is now runnable end-to-end for one causal prompt across negative prompt, SAFREE-CogVideoX, VideoEraser local reimplementation, and T2VUnlearning local proxy. The next scaling step should expand `--limit` across more clean causal templates, still sequentially, before attempting parallel execution on this crowded node.

## 2026-06-20: Full-Size Four-Baseline Suite, 20-Step, Six Clean Causal Prompts

**Goal:** Run the complete reproduction interface on every clean causal template currently in `prompts/cogvideox_clean_screening_round1.txt`, rather than validating only the first prompt.

**Command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_baseline_suite.py \
  --prompts prompts/cogvideox_clean_screening_round1.txt \
  --output-root outputs/baseline_suite_round1_all_local_bf16_limit6_step20_fullsize_seq \
  --model models/CogVideoX-2b \
  --seed 200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype bf16 \
  --limit 6 \
  --enable-model-cpu-offload \
  --vae-tiling
```

**Result:** Successful. The suite generated 24 full-size videos: 4 baselines x 6 causal prompts. All four generation manifests contain 6 items, and the suite manifest marks every job as `ready`.

**Targets covered:**

```text
ice cube, bottle, pitcher, pipette, stone, sugar cube
```

**Output root:**

```text
outputs/baseline_suite_round1_all_local_bf16_limit6_step20_fullsize_seq/
```

**Artifact counts and total mp4 sizes:**

```text
negative_prompt:   6 videos, 1068612 bytes total
safree_cogvideox:  6 videos, 1779554 bytes total
videoeraser:       6 videos,  967289 bytes total, local method spea_arng_cogvideox_v0
t2vunlearning:     6 videos, 1456692 bytes total, local method receler_cogvideox_proxy_v0
```

**Decode sanity check:** OpenCV successfully decoded every mp4 as 49 frames at 720x480 and 8 fps. The sugar-cube prompt produced near-black outputs across multiple baselines, with the VideoEraser sugar-cube output decoding as all-black frames (`mean=0.00`, `std=0.00`). Treat this as a generation-quality / prompt-robustness issue for the current baseline run, not an interface failure.

**Interpretation:** The current baseline suite now supports a real batched reproduction pass over the clean causal prompt set. The next step should be qualitative/automatic evaluation of these 24 outputs before increasing prompt count or inference steps; otherwise we risk spending GPU time on prompts such as sugar cube that already show poor base generation quality.

## 2026-06-20: Causal Footprint Mining Round 1, Prompt-Sharded Parallel T2V

**Goal:** Mine for stronger causal-footprint examples where the target source concept is visually absent or weak while the downstream effect remains visible. This specifically targets reviewer concerns that a causal failure might merely be ordinary incomplete erasure.

**Prompt set:** `prompts/causal_footprint_mining_round1.txt` contains 12 candidate causal templates. The first six were run in this pass:

```text
pebble -> circular ripples spread outward
raindrop -> circular ripple ring spreads outward
dye droplet -> red cloud blooms and spreads through water
match -> candle flame grows and keeps burning
hand -> desk lamp turns on and glows
finger -> dominoes topple one after another
```

**Run shape:** `CogVideoX-2B`, `bf16`, `480x720`, `49 frames`, `20 steps`, `guidance_scale=6.0`, `seed=300..305`, `limit=6`, model CPU offload and VAE tiling.

**Outputs:**

```text
outputs/causal_footprint_mining_round1_bf16_limit6_step20_fullsize_seq/
```

The completed output contains 30 videos: clean reference plus four erasure baselines across six prompts.

**Parallelization note:** The initial baseline suite ran sequentially for interface stability. For T2VUnlearning local proxy, the sequential job was stopped and replaced with six one-prompt shards on GPUs 0-5. Each shard used the same generation settings and seed `300 + prompt_index`, then the shard manifests were merged into the standard `t2vunlearning/generation_manifest.json`. This is the preferred pattern for future mining runs on this node: one CogVideoX process per GPU, because existing `dyme` resident processes already occupy roughly 40-46GB per H800 and two extra CogVideoX processes per card would risk OOM.

**Gallery and QC:**

```text
outputs/analysis_contact_sheets/causal_footprint_mining_round1_limit6_step20/video_gallery.html
outputs/analysis_contact_sheets/causal_footprint_mining_round1_limit6_step20/overview_middle_frames.png
outputs/analysis_contact_sheets/causal_footprint_mining_round1_limit6_step20/qc_metrics.tsv
```

**Initial QC interpretation:** `raindrop` and `dye droplet` were not flagged by the simple low-quality checks and should be inspected first for strong causal-footprint cases. `match`, `hand`, and `finger` have weak or nearly static clean-reference generations, so they should not be used as primary evidence unless visual inspection proves otherwise.

## 2026-06-20: Causal Footprint Mining Round 2 and Round 3 Expansion

**Goal:** Find more persuasive causal-footprint examples beyond the initial dye/water cases. The desired evidence is not simply "erasure failed"; it is a case where the source concept is absent or weak while a downstream causal footprint remains visible.

**Round 2 prompt set:**

```text
prompts/causal_footprint_mining_round2.txt
```

Round 2 contains 16 prompts. Clean-reference QC selected 8 prompts for full baseline reproduction:

```text
prompts/causal_footprint_mining_round2_cleanpass8.txt
source indices: 0, 1, 3, 10, 11, 12, 13, 15
targets: blue ink droplet, black ink droplet, oil droplet, pencil eraser, needle, magnet, fan, remote control
```

The full clean-pass run produced clean reference plus four erasure baselines for all 8 selected prompts:

```text
outputs/causal_footprint_mining_round2_cleanpass8_bf16_step20_fullsize_parallel/
outputs/analysis_contact_sheets/causal_footprint_mining_round2_cleanpass8_step20/video_gallery.html
outputs/analysis_contact_sheets/causal_footprint_mining_round2_cleanpass8_step20/qc_metrics.tsv
```

**Round 2 interpretation:** `blue/black ink`, `oil`, and `magnet` are worth inspection, but this round still leans heavily on diffusion-in-water effects. `fan` is mostly black/low quality and should not be used.

**Round 3 prompt set:**

```text
prompts/causal_footprint_mining_round3.txt
```

Round 3 contains 32 broader causal-footprint templates covering water disturbance, material traces, breakage, chain motion, light/electric state changes, magnetic/electrostatic effects, and deformation. Clean references were generated with 8 prompt shards on GPUs 0-7:

```text
outputs/causal_footprint_mining_round3_bf16_limit32_step20_fullsize_parallel/
outputs/analysis_contact_sheets/causal_footprint_mining_round3_clean_step20/clean_gallery.html
outputs/analysis_contact_sheets/causal_footprint_mining_round3_clean_step20/clean_ranked_shortlist.tsv
```

Clean QC selected 8 prompts for full baseline reproduction:

```text
prompts/causal_footprint_mining_round3_cleanpass8.txt
source indices: 0, 1, 2, 4, 13, 18, 30, 31
targets: pebble, raindrop, hailstone, shoe, baseball, soccer ball, magnet, comb
```

The full clean-pass run produced 40 videos: clean reference plus Negative Prompt, SAFREE-CogVideoX, VideoEraser local, and T2VUnlearning proxy for all 8 selected prompts.

```text
outputs/causal_footprint_mining_round3_cleanpass8_bf16_step20_fullsize_parallel/
outputs/analysis_contact_sheets/causal_footprint_mining_round3_cleanpass8_step20/video_gallery.html
outputs/analysis_contact_sheets/causal_footprint_mining_round3_cleanpass8_step20/overview_middle_frames.png
outputs/analysis_contact_sheets/causal_footprint_mining_round3_cleanpass8_step20/qc_metrics.tsv
```

**Initial Round 3 inspection priority:** Based on decode/QC only, inspect `raindrop`, `hailstone`, `baseball`, `soccer ball`, `magnet`, and `pebble` first. `shoe` has weaker temporal change; use it only if the visible footprint is clear. `SAFREE pebble` appears relatively static by QC and may be a weak row even if other baselines are useful.

**Parallelization correction:** The round2/round3 clean-pass reproductions were run as 8 prompt shards per baseline. That means each baseline used all 8 GPUs in parallel, but baselines themselves were still run one block at a time. This is stable but not the ideal utilization strategy requested for larger mining batches.

To fix this, added:

```text
scripts/run_parallel_baseline_jobs.py
tests/test_run_parallel_baseline_jobs.py
```

The new scheduler expands runs into `(prompt, baseline)` jobs and assigns them to GPU slots. Future mining should use this scheduler so all baselines are interleaved. Start with `--slots-per-gpu 1`; only test `--slots-per-gpu 2` on a small subset after checking GPU memory headroom, because one full-size CogVideoX-2B process can already consume a large fraction of an H800.

## 2026-06-20: Benchmark-First Research Direction

**Goal:** Convert the current qualitative causal-footprint evidence into a rigorous benchmark plan before designing a new erasure method.

**Motivation:** Round3 examples show the desired failure mode: the source concept can become weak or absent while downstream causal evidence remains, such as ripples, splash, cracks, net deformation, or footprints. A few examples are not enough for a paper-level claim; the next step is a benchmark that separates ordinary target-visible erasure failure from strict causal-footprint leakage.

**Design spec added:**

```text
docs/superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
```

**Core definitions:**

```text
C: source concept or event participant
E(C): direct visual evidence of C
F(C): causal footprint caused by C
```

**Key metric direction:**

```text
CFP@TPS<=1
```

This measures causal-footprint persistence only when target presence is already weak or absent, directly addressing the concern that examples might only be incomplete erasure.

**Documentation updates:**

- `README.md` now points to the benchmark design spec.
- `docs/research_notes.md` now records the benchmark-first framing and the `E(C)` vs `F(C)` distinction.
- `docs/current_open_questions.md` now prioritizes benchmark v0 prompt selection, strict-leakage thresholds, and canonical figure examples.

**Verification:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest tests -q
```

**Result:** `26 passed`.

## 2026-06-21: Evaluation Protocol v0 Research and Design

**Goal:** Decide how to evaluate causal-footprint leakage after the benchmark data construction strategy was defined.

**External evaluation lessons:** Recent video generation and video editing benchmarks argue against relying on a single coarse score such as FVD, IS, or global CLIPScore. The useful pattern is disentangled evaluation dimensions, fine-grained prompt categories, atomic checklist questions, MLLM-assisted scoring, and human calibration for temporal or physical judgments. Relevant sources reviewed include VBench, VBench++, VBench-2.0, EvalCrafter, FETV, T2V-CompBench, ETVA, VideoPhy, VideoPhy-2, PhyGenBench, CoVEBench, VEFX-Bench, and UVE.

**Claude/Opus discussion outcome:** Use MLLM scoring as a scalable first pass, but human-calibrate the protocol and human-adjudicate strict-leakage cases, unclear temporal/causal cases, and figure-selected examples. The key warning is that five-frame contact sheets are sufficient for browsing but not sufficient for final temporal/causal annotation.

**Spec update:**

```text
docs/superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
```

The spec now defines:

- annotation fields for target presence, footprint presence, quality, scene fidelity, timing, alternative visible causes, and causal incoherence;
- 0-3 scoring rubrics for TPS, FPS, QS, and SFS;
- a chronological MLLM/human chain-of-query prompt;
- strict leakage, target-visible failure, and quality-failure definitions;
- metric formulas including `CFP@TPS<=1`;
- cost-saving plan with MLLM first-pass scoring plus human calibration/adjudication.

## 2026-06-21: Data Construction Protocol v0

**Goal:** Address the concern that causal pairs such as `raindrop -> ripples` or `baseball -> cracks` could look hand-picked unless the benchmark explains where pairs come from and how they are filtered.

**Claude/Opus discussion outcome:** The benchmark should be framed as taxonomy-driven causal pair construction, not as a list of hand-written prompts. A valid pair must have an explicit causal mechanism, counterfactual dependence, temporal asymmetry, and visible footprint evidence. The data protocol should also include controls so the benchmark does not treat every natural ripple, crack, or deformation as causal leakage.

**Spec update:**

```text
docs/superpowers/specs/2026-06-20-causal-footprint-benchmark-v0-design.md
```

The spec now defines:

- a construction pipeline from mechanism taxonomy to candidate pairs to clean-source-gated benchmark rows;
- valid causal pair conditions and exclusion criteria;
- pair-level scores for exclusivity, counterfactual clarity, generatability, and erasure targetability;
- controlled source and counterfactual prompt templates;
- natural-footprint, no-footprint, and alternative-cause control prompts;
- v0 and paper-scale target sizes.

## 2026-06-21: Causal Footprint Candidate Pair Pool v0

**Goal:** Start implementing the benchmark data construction protocol with an auditable candidate pool rather than directly writing final benchmark items.

**Files added:**

```text
benchmarks/causal_footprint_v0/README.md
benchmarks/causal_footprint_v0/candidate_pairs.tsv
benchmarks/causal_footprint_v0/control_prompts.jsonl
```

**Candidate pool shape:**

```text
36 total candidate pairs
24 accepted_v0_slice
8 exploratory
4 rejected
```

Mechanism coverage is balanced:

```text
fluid_impact: 6
surface_trace: 6
fracture_damage: 6
elastic_deformation: 6
field_mediated: 6
agent_or_object_response: 6
```

Accepted v0 slice coverage:

```text
fluid_impact: 4
surface_trace: 5
fracture_damage: 4
elastic_deformation: 4
field_mediated: 3
agent_or_object_response: 4
```

**Controls:** `control_prompts.jsonl` currently contains 8 controls covering natural-footprint, no-footprint counterfactual, prior-footprint, and alternative-cause cases.

**Validation:** A local TSV/JSONL parse check found no duplicate `pair_id` values and no out-of-range pair-level scores.

**Interpretation:** This is still a candidate pool, not the final benchmark. The next step is to review the accepted slice, adjust scores/status if needed, then export accepted rows into the generation prompt format for clean-source screening.

## 2026-06-21: Causal Footprint v0 Accepted Slice Clean Generation

**Goal:** Export the accepted candidate pairs and run parallel clean-source generation before baseline erasure.

**Code added:**

```text
scripts/export_benchmark_prompts.py
tests/test_export_benchmark_prompts.py
```

**Scheduler update:** `scripts/run_parallel_baseline_jobs.py` now supports explicitly selected `--baseline clean` jobs while preserving the default four-erasure-baseline behavior.

**Review tool update:** `scripts/build_clean_source_review.py` now generates both `clean_source_screening.csv` and a readable `clean_gallery.html`. Each gallery row explicitly labels the baseline as `Clean reference`, shows the full source prompt, target concept, expected causal footprint, pair id, mechanism type, an mp4 link, and a five-frame preview strip.

**Exported prompts:**

```text
prompts/causal_footprint_v0_accepted24.txt
benchmarks/causal_footprint_v0/export_accepted24_manifest.json
```

**Clean run command shape:** 24 accepted candidates, CogVideoX-2B, `bf16`, `480x720`, `49 frames`, `20 steps`, `seed=1100..1123`, scheduled as one clean generation job per GPU across 8 H800 GPUs.

**Outputs:**

```text
outputs/causal_footprint_v0_clean_accepted24_bf16_step20_parallel/
outputs/causal_footprint_v0_clean_accepted24_bf16_step20_parallel/clean/generation_manifest.json
outputs/analysis_contact_sheets/causal_footprint_v0_clean_accepted24_step20/clean_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_clean_accepted24_step20/clean_overview_5frames.png
outputs/analysis_contact_sheets/causal_footprint_v0_clean_accepted24_step20/qc_metrics.tsv
```

**Clean review gallery command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_clean_source_review.py \
  --manifest outputs/causal_footprint_v0_clean_accepted24_bf16_step20_parallel/clean/generation_manifest.json \
  --metadata-manifest benchmarks/causal_footprint_v0/export_accepted24_manifest.json \
  --output-dir outputs/analysis_contact_sheets/causal_footprint_v0_clean_accepted24_step20 \
  --frames-per-video 5 \
  --thumb-width 220 \
  --thumb-height 124
```

**Result:** 24/24 clean generation jobs finished and produced mp4 files.

**Initial visual note:** Several prompts are clearly not clean-valid yet. Stronger-looking rows include water impact, some glass/crack cases, soccer-net deformation, tennis-racket deformation, magnet/filings, comb/paper, and possibly key/door. Weak rows include hand/clay, marker/whiteboard, hammer/tile, dropped cup, finger/rubber sheet, switch/lamp, and several rows where the target concept is absent even if the footprint appears. The next step is manual clean-source screening from the gallery before exporting final `items.jsonl`.

## 2026-06-21: Initial Clean-Source Gate Labels for Accepted24

**Annotation file:**

```text
experiments/clean_screening/causal_footprint_v0_clean_accepted24_initial_labels.csv
```

**Screening rule:** A source video is marked `valid` only when the target cause is visible, the expected causal footprint is visible, temporal order is reasonably clear, and the footprint plausibly depends on the target. Videos with visible footprint but missing target cause are marked `reject`, because they would confound later erasure analysis.

**Counts under strict initial screening:**

```text
valid: 5
weak: 5
reject: 14
```

**Valid rows for first baseline pass:**

```text
fluid_impact_pebble_pond_002
fracture_damage_rock_windshield_003
elastic_deformation_soccer_net_001
elastic_deformation_tennis_ball_racket_002
field_mediated_comb_paper_002
```

**Weak backup rows:** `fluid_impact_raindrop_puddle_001`, `fluid_impact_hailstone_water_003`, `fluid_impact_ink_droplet_glass_004`, `surface_trace_tire_mud_002`, and `field_mediated_magnet_filings_001`. These have usable target/footprint cues but unclear temporal ordering, cropped target visibility, or weak separation between cause and footprint.

**Interpretation:** The initial prompt pool is useful, but the clean-source gate is doing necessary filtering. The next practical step is to run more seeds or prompt variants for weak/rejected mechanisms before freezing v0; otherwise the final benchmark would overrepresent elastic deformation and underrepresent surface trace / agent-object response cases.

## 2026-06-21: Valid5 Four-Baseline Parallel Run

**Goal:** Run all required erasure baselines on the five strict clean-source-valid v0 rows before expanding the benchmark.

**Code update:** `scripts/export_benchmark_prompts.py` now accepts `--clean-labels` and `--clean-source-valid`, so clean-source gate labels can drive prompt export without manual copying.

**Exported valid5 prompts:**

```text
prompts/causal_footprint_v0_valid5.txt
benchmarks/causal_footprint_v0/export_valid5_manifest.json
```

**Valid5 pairs:**

```text
fluid_impact_pebble_pond_002
fracture_damage_rock_windshield_003
elastic_deformation_soccer_net_001
elastic_deformation_tennis_ball_racket_002
field_mediated_comb_paper_002
```

**Run command shape:** Four erasure baselines, 5 prompts each, CogVideoX-2B, `bf16`, `480x720`, `49 frames`, `20 steps`, `seed=2100..2104`, scheduled as one job per GPU across 8 H800 GPUs.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --prompts prompts/causal_footprint_v0_valid5.txt \
  --output-root outputs/baseline_suite_causal_footprint_v0_valid5_all_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 2100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --vae-slicing \
  --vae-tiling
```

**Outputs:**

```text
outputs/baseline_suite_causal_footprint_v0_valid5_all_step20_parallel/
outputs/baseline_suite_causal_footprint_v0_valid5_all_step20_parallel/parallel_job_manifest.json
outputs/analysis_contact_sheets/causal_footprint_v0_valid5_baseline_step20/baseline_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_valid5_baseline_step20/baseline_overview_midframes.png
experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv
```

**Result:** 20/20 jobs finished: 5 Negative Prompt, 5 SAFREE-CogVideoX, 5 VideoEraser local, and 5 T2V proxy videos.

**Initial visual note:** The gallery already shows strong candidate failure modes. In several rows, baseline outputs keep the causal footprint (water ripples, glass cracks, net/string deformation, or lifted paper scraps) even when the target cause is weakened, absent, or visually ambiguous. The next step is manual annotation from the baseline gallery, because some outputs are ordinary target leakage rather than clean target-erased causal-footprint leakage.

## 2026-06-21: Valid5 Baseline Manual Annotation

**Annotation file:** The valid5 baseline summary now includes manual labels for target visibility, causal-effect visibility, causeless-effect status, video quality, claim usability, and failure mode.

```text
experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv
```

**Local review page:**

```text
outputs/analysis_contact_sheets/causal_footprint_v0_valid5_baseline_step20/baseline_gallery_annotated.html
```

**Label policy:**

- `usable_for_claim=yes`: target cause is absent or effectively erased, while the causal footprint remains visible.
- `usable_for_claim=borderline`: footprint remains, but there is residual target/cause ambiguity or an alternative visible cause.
- `usable_for_claim=no`: ordinary target leakage, unclear footprint, or unusable output.

**Counts:**

```text
usable_for_claim=yes: 9
usable_for_claim=borderline: 3
usable_for_claim=no: 8
```

Strong `yes` cases by baseline:

```text
negative_prompt: 2
safree_cogvideox: 2
videoeraser: 3
t2vunlearning: 2
```

Strong examples:

```text
negative_prompt  + rock/windshield crack
negative_prompt  + tennis/racket deformation
safree_cogvideox + rock/windshield crack
safree_cogvideox + tennis/racket deformation
videoeraser      + soccer/net deformation
videoeraser      + tennis/racket deformation
videoeraser      + comb/paper scraps
t2vunlearning    + soccer/net deformation
t2vunlearning    + tennis/racket deformation
```

## 2026-06-21: Round4 Clean-Source Expansion48

**Motivation:** valid5 proves the problem but is too small for a benchmark claim. Round4 expands clean-source candidates using taxonomy-driven prompt variants before running more erasure baselines.

**Prompt sources:**

```text
benchmarks/causal_footprint_v0/round4_clean_expansion_prompts.tsv
prompts/causal_footprint_v0_round4_clean_expansion48.txt
```

**Design:** 48 clean-source prompts, 8 per mechanism type:

```text
fluid_impact: 8
surface_trace: 8
fracture_damage: 8
elastic_deformation: 8
field_mediated: 8
agent_or_object_response: 8
```

**Generation command shape:** CogVideoX-2B, `bf16`, `480x720`, `49 frames`, `20 steps`, `seed=3100..3147`, scheduled across 8 H800 GPUs with one clean generation job per GPU slot.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --baseline clean \
  --prompts prompts/causal_footprint_v0_round4_clean_expansion48.txt \
  --output-root outputs/causal_footprint_v0_round4_clean_expansion48_bf16_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 3100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --vae-slicing \
  --vae-tiling
```

**Result:** 48/48 clean jobs finished and produced mp4 files. Generated media remains under ignored `outputs/`.

**Review artifacts:**

```text
outputs/causal_footprint_v0_round4_clean_expansion48_bf16_step20_parallel/clean/generation_manifest.json
outputs/analysis_contact_sheets/causal_footprint_v0_round4_clean_expansion48_step20/clean_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round4_clean_expansion48_step20/clean_gallery_annotated.html
outputs/analysis_contact_sheets/causal_footprint_v0_round4_clean_expansion48_step20/clean_overview_5frames_annotated.png
```

**Review tool update:** `scripts/build_clean_source_review.py` now also accepts `--metadata-tsv`, so expansion TSV files can drive gallery labels without first creating a JSON manifest.

**Tracked initial labels:**

```text
experiments/clean_screening/causal_footprint_v0_round4_clean_expansion48_initial_labels.csv
```

**Initial clean-source counts:**

```text
yes: 9
borderline: 11
no: 28
```

**Clean-valid rows exported for next baseline run:**

```text
prompts/causal_footprint_v0_round4_clean_valid9.txt
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
```

The 9 current `yes` rows are:

```text
round4_fluid_impact_water_droplet_puddle_003
round4_fluid_impact_blue_ink_droplet_004
round4_surface_trace_bicycle_tire_mud_006
round4_fracture_rock_windshield_001
round4_elastic_soccer_net_variant_007
round4_elastic_tennis_racket_variant_008
round4_field_comb_paper_002
round4_field_fan_streamers_005
round4_field_hair_dryer_ribbons_006
```

**Interpretation:** Round4 confirms the benchmark needs an explicit clean-source gate. Strong categories remain water/ripple, windshield crack, soccer-net/racket deformation, and some field-mediated paper/ribbon motion. Surface trace and agent-object response still need prompt rewrites or additional seeds before they can support a balanced final v0 benchmark.

## 2026-06-21: Round4 Valid9 Four-Baseline Parallel Run

**Goal:** Run all required erasure baselines on the 9 clean-valid round4 sources so the benchmark claim is no longer supported by valid5 alone.

**Prompt file:**

```text
prompts/causal_footprint_v0_round4_clean_valid9.txt
```

**Run command shape:** Four erasure baselines, 9 prompts each, CogVideoX-2B, `bf16`, `480x720`, `49 frames`, `20 steps`, `seed=4100..4108`, scheduled across 8 H800 GPUs.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --prompts prompts/causal_footprint_v0_round4_clean_valid9.txt \
  --output-root outputs/baseline_suite_causal_footprint_v0_round4_valid9_all_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 4100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --vae-slicing \
  --vae-tiling
```

**Runtime note:** The first scheduler process was terminated after completing prompt indices 0-3, leaving 16/36 mp4 files. The interrupted manifest was preserved locally as:

```text
outputs/baseline_suite_causal_footprint_v0_round4_valid9_all_step20_parallel/parallel_job_manifest_interrupted_prompt0_3.json
```

The remaining prompt indices 4-8 were then resumed with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --prompts prompts/causal_footprint_v0_round4_clean_valid9.txt \
  --source-indices 4,5,6,7,8 \
  --output-root outputs/baseline_suite_causal_footprint_v0_round4_valid9_all_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 4100 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --vae-slicing \
  --vae-tiling
```

**Result:** 36/36 erasure videos finished: 9 Negative Prompt, 9 SAFREE-CogVideoX, 9 VideoEraser local, and 9 T2V proxy videos.

**Review artifacts:**

```text
outputs/baseline_suite_causal_footprint_v0_round4_valid9_all_step20_parallel/
outputs/analysis_contact_sheets/causal_footprint_v0_round4_valid9_baseline_step20/baseline_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round4_valid9_baseline_step20/baseline_gallery_annotated.html
outputs/analysis_contact_sheets/causal_footprint_v0_round4_valid9_baseline_step20/baseline_overview_5frames_annotated.png
```

**Tracked summary:**

```text
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

**Conservative erasure-output labels after re-review, excluding clean-reference rows:**

```text
usable_for_claim=yes: 15
usable_for_claim=borderline: 9
usable_for_claim=no: 12
```

Strong `yes` cases by baseline:

```text
negative_prompt: 3
safree_cogvideox: 3
videoeraser: 5
t2vunlearning: 4
```

**Interpretation:** Round4-valid9 strengthens the core observation. Stronger cases include ink plumes after droplet removal, tire tracks after tire removal, windshield cracks after rock removal, goal-net/racket deformation after ball removal, paper scraps after comb removal, and ribbons/streamers moving after fan or hair dryer removal. Some rows remain ordinary target leakage or residual-cause ambiguity; these are retained in the summary rather than silently filtered out.

**Re-review note:** Five labels were made more conservative after prompt-by-prompt inspection. Tennis-ball Negative Prompt, comb T2V proxy, comb SAFREE, and fan SAFREE were downgraded because target-like source cues remained visible. Tennis-ball SAFREE was moved from `no` to `borderline` because the footprint remains but a yellow residual-cause cue is ambiguous. This keeps the headline count focused on cleaner target-erased causal-footprint leakage rather than target leakage.

**Figure-candidate note:** After conservative re-review, the cleanest figure candidates are not rows where every baseline succeeds. Stronger candidates are `blue ink droplet -> blue plume`, `bicycle tire -> tire track`, `soccer ball -> net deformation`, and `fan/hair dryer -> streamer or ribbon motion`, because they show clear target-erased footprints in multiple baselines while keeping ordinary target leakage rows visible as negative cases.

## 2026-06-22: Formal Benchmark Items and First Metrics

**Goal:** Convert the current human-reviewed valid5 and round4-valid9 evidence into one benchmark source-of-truth file and reproducible metric tables.

**Input evidence:**

```text
benchmarks/causal_footprint_v0/export_valid5_manifest.json
experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv
benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json
experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv
```

**Commands:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_benchmark_items.py \
  --source valid5,benchmarks/causal_footprint_v0/export_valid5_manifest.json,experiments/baseline_runs/causal_footprint_v0_valid5_all_step20_parallel_summary.csv \
  --source round4_valid9,benchmarks/causal_footprint_v0/export_round4_clean_valid9_manifest.json,experiments/baseline_runs/causal_footprint_v0_round4_valid9_all_step20_parallel_summary.csv \
  --output benchmarks/causal_footprint_v0/items.jsonl

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_benchmark_metrics.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output-dir experiments/metrics
```

**Output artifacts:**

```text
benchmarks/causal_footprint_v0/items.jsonl
experiments/metrics/causal_footprint_v0_metrics_by_baseline.csv
experiments/metrics/causal_footprint_v0_metrics_by_mechanism.csv
experiments/metrics/causal_footprint_v0_metrics_summary.md
```

**Result:**

```text
benchmark items: 14
erasure outputs: 56
strict causal-footprint leakage: 24 / 56
borderline causal-footprint cases: 12 / 56
target-leakage failures: 14 / 56
```

Strict causal-footprint leakage by baseline:

```text
negative_prompt: 5 / 14
safree_cogvideox: 5 / 14
t2vunlearning: 6 / 14
videoeraser: 8 / 14
```

**Interpretation:** The current evidence now supports a benchmark-style problem statement rather than only selected examples. The headline count stays conservative by separating ordinary target leakage from target-erased causal-footprint leakage. The next research step is to expand clean-source coverage and calibrate automatic scoring against these human labels.

## 2026-06-22: Evaluator Calibration Harness

**Goal:** Standardize how future automatic video evaluators will be compared against the current 56-row human-labeled causal-footprint gold set.

**Design note:** This is not a VLM run. The purpose is to lock the gold schema, prediction schema, join key, and calibration metrics before plugging in any specific scorer.

**Gold export command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/export_calibration_gold.py \
  --items benchmarks/causal_footprint_v0/items.jsonl \
  --output experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv
```

**Calibration command using oracle-format smoke predictions:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/example_predictions.csv \
  --output-dir experiments/eval_calibration
```

**Tracked artifacts:**

```text
experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv
experiments/eval_calibration/example_predictions.csv
experiments/eval_calibration/calibration_metrics_by_label.csv
experiments/eval_calibration/calibration_confusion_matrix.csv
experiments/eval_calibration/calibration_metrics_summary.md
```

**Gold label support:**

```text
strict_leakage: 24
borderline: 12
target_leakage: 14
other_failure: 6
```

**Smoke result:** `example_predictions.csv` copies the human labels into the prediction schema, so strict leakage F1, relaxed leakage F1, and macro F1 are all 1.0000. This only verifies the calibration interface; it is not an automatic evaluator result.

**Next step:** plug in one real scorer that writes the required prediction schema:

```text
item_id,baseline,video_path,target_absent,effect_visible,quality_ok,pred_label,confidence,reason
```

## 2026-06-22: VLM Contact-Sheet Dry-Run Inputs

**Goal:** Prepare the first third-party VLM scorer input layer without making external API calls.

**Input gold file:**

```text
experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv
```

**Contact-sheet command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_vlm_eval_inputs.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --sheet-dir experiments/eval_calibration/frame_sheets \
  --output experiments/eval_calibration/vlm_inputs.csv \
  --frames-per-video 5 \
  --thumb-width 192 \
  --thumb-height 128
```

**Dry-run payload command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-jsonl experiments/eval_calibration/vlm_payloads_dryrun.jsonl \
  --dry-run
```

**Tracked artifacts:**

```text
experiments/eval_calibration/vlm_inputs.csv
experiments/eval_calibration/vlm_payloads_dryrun.jsonl
```

**Local generated media, ignored by git:**

```text
experiments/eval_calibration/frame_sheets/
```

**Result:**

```text
VLM input rows: 56
contact sheets generated: 56
reference sheets generated: 36
missing videos: 0
dry-run payloads: 56
```

**Interpretation:** The project now has a complete pre-API evaluator path: generated videos are represented as 5-frame contact sheets, model prompts are deterministic, and future third-party VLM responses can be converted into the existing prediction CSV schema for calibration. For `round4_valid9`, the VLM input also includes a clean-reference contact sheet; the older `valid5` rows do not have clean-reference videos.

## 2026-06-22: GPT-4o Scorer Attempt and GPT-4o-mini Fallback Smoke

**Goal:** Start real VLM judging with a mainstream OpenAI model.

**Preferred model:** `openai/gpt-4o`.

**Endpoint status:** The provided `https://api.360.cn/v1` endpoint lists `openai/gpt-4o`, but real image requests returned:

```text
当前分组 default 下对于模型 gpt-4o 无可用渠道
```

The same image request format worked with `openai/gpt-4o-mini`, so the blocker is model-channel availability rather than API key or image payload format.

**Fallback smoke command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/gpt4o_mini_sample8_raw.jsonl \
  --run-api \
  --model openai/gpt-4o-mini \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --limit 8 \
  --temperature 0 \
  --max-tokens 300 \
  --timeout 120

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv \
  --output-dir experiments/eval_calibration/gpt4o_mini_sample8 \
  --allow-partial
```

**Tracked artifacts:**

```text
experiments/eval_calibration/gpt4o_mini_sample8_predictions.csv
experiments/eval_calibration/gpt4o_mini_sample8_raw.jsonl
experiments/eval_calibration/gpt4o_mini_sample8/
```

**Calibration result:**

```text
matched predictions: 8
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7692
macro F1: 0.1000
```

**Interpretation:** `gpt-4o-mini` predicted `strict_leakage` for all 8 sample rows, including rows manually labeled as `target_leakage`, `borderline`, and `other_failure`. It is useful only as a pipeline smoke test and should not be used as the main judge. The next real scorer run should use full `openai/gpt-4o` once the endpoint has an available channel, or use another mainstream strong VLM as an explicitly documented fallback.

## 2026-06-22: Qwen-VL Fallback Scorer Trial

**Goal:** Replace the unavailable GPT-4o scorer with a usable mainstream VLM fallback, without using Doubao as the main judge.

**Endpoint candidates checked:**

```text
openai/gpt-4o: listed but no available channel for image requests
google/gemini-2.5-pro: returned truncated / non-JSON content through this OpenAI-compatible route
google/gemini-2.5-flash: returned truncated / non-JSON content through this OpenAI-compatible route
anthropic/claude-sonnet-4-6: could inspect the image, but did not reliably return the required JSON schema with the current prompt and token budget
alibaba/qwen-vl-max: stable, but over-predicted strict_leakage on the first 8 rows
qwen/qwen-vl-plus: stable and able to distinguish some target-leakage rows
```

**Qwen-VL-Max sample command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/qwen_vl_max_sample8_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/qwen_vl_max_sample8_raw.jsonl \
  --run-api \
  --model alibaba/qwen-vl-max \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --limit 8 \
  --temperature 0 \
  --max-tokens 300 \
  --timeout 120

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/qwen_vl_max_sample8_predictions.csv \
  --output-dir experiments/eval_calibration/qwen_vl_max_sample8 \
  --allow-partial
```

**Qwen-VL-Max sample result:**

```text
matched predictions: 8
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7692
macro F1: 0.1000
```

`alibaba/qwen-vl-max` predicted `strict_leakage` for all 8 sample rows, so it is not a good fallback under the current prompt.

**Qwen-VL-Plus full command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/qwen_vl_plus_full_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/qwen_vl_plus_full_raw.jsonl \
  --run-api \
  --model qwen/qwen-vl-plus \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --temperature 0 \
  --max-tokens 500 \
  --timeout 180

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/qwen_vl_plus_full_predictions.csv \
  --output-dir experiments/eval_calibration/qwen_vl_plus_full
```

**Artifact policy:**

```text
Qwen trial outputs were summarized here and removed from the tracked artifact set after the protocol moved to reference-aware Claude calibration.
```

**Qwen-VL-Plus full calibration result:**

```text
matched predictions: 56
strict leakage binary F1: 0.6761
relaxed leakage binary F1: 0.8675
macro F1: 0.3429
strict_leakage: precision 0.5106, recall 1.0000, F1 0.6761
target_leakage: precision 0.8889, recall 0.5714, F1 0.6957
```

**Predicted label distribution:**

```text
strict_leakage: 47
target_leakage: 9
borderline: 0
other_failure: 0
```

**Interpretation:** `qwen/qwen-vl-plus` is the best currently available fallback on this endpoint. It is useful as a high-recall leakage screener: it catches all human strict-leakage rows and most target-leakage rows. It is not yet a replacement for human labels because it collapses all `borderline` and `other_failure` cases into hard leakage decisions. The next evaluator step should recalibrate the prompt or split judging into staged questions so the model is allowed to say "ambiguous / not enough evidence" more often.

## 2026-06-22: Atomic VLM Protocol Trial

**Goal:** Reduce direct-label bias by asking the VLM for atomic visual facts instead of letting it choose the final benchmark label.

**Protocol change:** The current `scripts/evaluate_with_vlm.py` prompt asks the model to return:

```json
{
  "target_visible": "yes|no|partial",
  "effect_visible": "yes|no|partial",
  "separation_clear": "yes|no",
  "quality_ok": "yes|no",
  "confidence": 0.0,
  "reason": "short visual evidence"
}
```

The script derives the existing prediction CSV fields afterward:

- `target_visible = yes` -> `target_leakage`
- `target_visible = partial` -> `borderline`
- `effect_visible = partial` -> `borderline`
- `separation_clear = no` -> `borderline`
- `quality_ok = no` -> `other_failure`
- `target_visible = no` and `effect_visible = yes` -> `strict_leakage`

**Command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/qwen_vl_plus_atomic_sample8_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/qwen_vl_plus_atomic_sample8_raw.jsonl \
  --run-api \
  --model qwen/qwen-vl-plus \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --limit 8 \
  --temperature 0 \
  --max-tokens 500 \
  --timeout 180

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/qwen_vl_plus_atomic_sample8_predictions.csv \
  --output-dir experiments/eval_calibration/qwen_vl_plus_atomic_sample8 \
  --allow-partial
```

**Artifact policy:**

```text
This Qwen atomic sample was summarized here and not retained as a tracked artifact.
```

**Calibration result:**

```text
matched predictions: 8
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7692
macro F1: 0.1000
```

**Interpretation:** The atomic protocol did not fix the current `qwen/qwen-vl-plus` bias on the first 8 rows. It still marked every row as target absent, effect visible, and strict leakage. The direction remains conceptually cleaner than direct label prompting, but the prompt needs stronger ambiguity/negative-evidence calibration before running a full atomic evaluation.

## 2026-06-22: Reference-Aware Atomic VLM Trial

**Goal:** Test whether adding a clean-reference contact sheet helps `qwen/qwen-vl-plus` separate target visibility from downstream effects.

**Input coverage:**

```text
VLM rows: 56
output sheets: 56
reference sheets: 36
reference-backed subset: round4_valid9 only
```

**Full command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/qwen_vl_plus_reference_atomic_full_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/qwen_vl_plus_reference_atomic_full_raw.jsonl \
  --run-api \
  --model qwen/qwen-vl-plus \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --require-reference \
  --temperature 0 \
  --max-tokens 500 \
  --timeout 180

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/qwen_vl_plus_reference_atomic_full_predictions.csv \
  --output-dir experiments/eval_calibration/qwen_vl_plus_reference_atomic_full \
  --allow-partial
```

**Artifact policy:**

```text
Qwen reference-aware atomic outputs were summarized here and removed from the tracked artifact set after the Claude conservative cross-check was retained.
```

**Full calibration result:**

```text
matched predictions: 36
strict leakage binary F1: 0.6087
relaxed leakage binary F1: 0.8364
macro F1: 0.3060
strict_leakage: precision 0.4516, recall 0.9333, F1 0.6087
target_leakage: precision 1.0000, recall 0.4444, F1 0.6154
```

**Interpretation:** Clean-reference context helped define the target/effect visually, but did not solve the over-strict bias. Qwen catches nearly all human strict-leakage rows, but maps all borderline rows and all other-failure rows to leakage-like labels. It is useful as a high-recall screener, not as the final automatic judge.

## 2026-06-22: Claude Reference-Aware Atomic VLM Trial

**Goal:** Test a mainstream non-Qwen VLM on the same reference-aware atomic protocol.

**Command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/evaluate_with_vlm.py \
  --inputs experiments/eval_calibration/vlm_inputs.csv \
  --output-predictions experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --raw-output-jsonl experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl \
  --run-api \
  --model anthropic/claude-sonnet-4-6 \
  --api-config-file /home/deepseek_VG/JUNCHI/Diffusion-Personalization-Target-Alignment/token.txt \
  --require-reference \
  --temperature 0 \
  --max-tokens 1000 \
  --timeout 180

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/calibrate_evaluator.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --predictions experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --output-dir experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full \
  --allow-partial
```

**Tracked artifacts:**

```text
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_raw.jsonl
experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full/
```

**Calibration result:**

```text
matched predictions: 36
strict leakage binary F1: 0.4000
relaxed leakage binary F1: 0.7600
macro F1: 0.3438
strict_leakage: precision 0.8000, recall 0.2667, F1 0.4000
borderline: precision 0.1905, recall 0.4444, F1 0.2667
target_leakage: precision 0.4286, recall 0.3333, F1 0.3750
other_failure: precision 0.3333, recall 0.3333, F1 0.3333
```

**Interpretation:** Claude has the opposite failure mode from Qwen. It uses all four labels and gives useful visual reasons, but it is conservative: it often downgrades human strict-leakage rows to `borderline`, giving low strict-leakage recall. This makes it useful as a cross-check for ambiguity and target leakage, but not as the final automatic judge.

## 2026-06-23: Benchmark Evaluation V1 Manifest

**Goal:** Make the benchmark evaluation path explicit enough for paper tables and human review. The previous artifacts already had human gold rows, contact sheets, and VLM predictions, but they were spread across calibration folders. This step creates a single manifest and static review page.

**Commands:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_evaluation_manifest.py \
  --gold experiments/eval_calibration/causal_footprint_v0_gold_outputs.csv \
  --vlm-inputs experiments/eval_calibration/vlm_inputs.csv \
  --prediction claude=experiments/eval_calibration/claude_sonnet_4_6_reference_atomic_full_predictions.csv \
  --output experiments/evaluation/causal_footprint_v1_manifest.csv

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_annotation_review.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation \
  --project-root /home/deepseek_VG/JUNCHI/Video-causal

PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/compute_evaluation_metrics.py \
  --manifest experiments/evaluation/causal_footprint_v1_manifest.csv \
  --output-dir experiments/evaluation
```

**Artifacts:**

```text
experiments/evaluation/causal_footprint_v1_manifest.csv
experiments/evaluation/annotation_queue.csv
experiments/evaluation/review.html
experiments/evaluation/metrics_by_baseline.csv
experiments/evaluation/metrics_by_mechanism.csv
experiments/evaluation/model_agreement.csv
experiments/evaluation/metrics_summary.md
```

**Metric summary:**

```text
total outputs: 56
strict leakage: 24/56 (0.4286)
borderline: 12/56 (0.2143)
relaxed leakage: 36/56 (0.6429)
target leakage: 14/56 (0.2500)
other failure: 6/56 (0.1071)
Claude exact agreement: 12/36 (0.3333)
```

**By-baseline relaxed leakage:**

```text
negative_prompt: 7/14 (0.5000)
safree_cogvideox: 7/14 (0.5000)
t2vunlearning: 9/14 (0.6429)
videoeraser: 13/14 (0.9286)
```

**Interpretation:** The v1 evaluation layer supports the paper claim better than isolated contact-sheet examples: all four baselines have nonzero relaxed causal-footprint leakage, and VideoEraser has the strongest leakage rate in this current slice. Claude is retained as a disagreement-mining helper, not as ground truth.

## 2026-06-23: Round5 Taxonomy-Balanced Candidate Pool

**Goal:** Reduce benchmark over-dependence on the current water/ball-heavy slice by creating a larger physical causal-footprint prompt pool before running new clean-source generation.

**Artifacts:**

```text
benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv
prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt
```

**Composition:**

```text
fluid_impact: 10
surface_trace: 10
fracture_damage: 10
elastic_deformation: 10
field_mediated: 10
particle_dispersion: 10
```

**Design notes:**

- Round5 keeps only physical footprint mechanisms.
- It excludes the prior agent/semantic-response category, such as remote control, wall switch, and button-light prompts.
- It diversifies source concepts and media: pond, soup, tea, milk, snow, mud, fogged glass, phone screen, clay pot, spring, smoke, dust, glitter, cereal, and soil.
- The prompt file follows the existing `prompt | target | effect` format used by CogVideoX clean-source generation.

**Validation command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/generate_cogvideox_clean.py \
  --prompts prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt \
  --output-dir outputs/causal_footprint_v0_round5_taxonomy_expansion60_dryrun \
  --model zai-org/CogVideoX-2b \
  --seed 5200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --fps 8 \
  --dtype bf16 \
  --dry-run
```

**Validation result:**

```text
rows: 60
mechanisms: 6 x 10
unique ids: 60
prompt lines: 60
TSV/TXT consistency: pass
dry-run generation manifest: pass
```

**Next step:** completed by the following round5 clean-generation entry. Manual clean-source screening is now the active task.

## 2026-06-23: Round5 CogVideoX-2B Clean-Source Generation

**Goal:** Generate the full round5 clean-source candidate set so the next benchmark slice is not limited to water-drop and ball-net examples.

**Input:**

```text
benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv
prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt
```

**Primary command:**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 \
/home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --baseline clean \
  --prompts prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt \
  --output-root outputs/causal_footprint_v0_round5_taxonomy_expansion60_bf16_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 5200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,5,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --enable-sequential-cpu-offload \
  --vae-slicing \
  --vae-tiling
```

**Initial result:** 51 / 60 videos completed. Nine jobs assigned to GPU5 failed with CUDA OOM because unrelated `dyme` processes were already using most memory on all eight GPUs, with GPU5 additionally carrying another process.

**Retry command:** reran only failed prompt indices `5,8,16,24,31,39,43,48,56`, excluding GPU5.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 \
/home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/run_parallel_baseline_jobs.py \
  --baseline clean \
  --prompts prompts/causal_footprint_v0_round5_taxonomy_expansion60.txt \
  --source-indices 5,8,16,24,31,39,43,48,56 \
  --output-root outputs/causal_footprint_v0_round5_taxonomy_expansion60_bf16_step20_parallel \
  --model models/CogVideoX-2b \
  --seed 5200 \
  --steps 20 \
  --guidance-scale 6.0 \
  --num-frames 49 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --dtype bf16 \
  --gpus 0,1,2,3,4,6,7 \
  --slots-per-gpu 1 \
  --poll-interval 5 \
  --enable-sequential-cpu-offload \
  --vae-slicing \
  --vae-tiling
```

**Final result:**

```text
clean-source videos: 60 / 60
merged generation manifest:
outputs/causal_footprint_v0_round5_taxonomy_expansion60_bf16_step20_parallel/clean/generation_manifest.json
review gallery:
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_gallery.html
screening CSV:
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_source_screening.csv
frame strips: 60
```

**Review artifact command:**

```bash
PYTHONNOUSERSITE=1 /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/build_clean_source_review.py \
  --manifest outputs/causal_footprint_v0_round5_taxonomy_expansion60_bf16_step20_parallel/clean/generation_manifest.json \
  --metadata-tsv benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv \
  --output-dir outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20 \
  --frames-per-video 5 \
  --thumb-width 220 \
  --thumb-height 124
```

**Conclusion:** round5 clean-source generation is complete. The active next step is to annotate `clean_source_screening.csv` with valid, borderline, and failed clean-source labels, then export a clean-valid prompt slice for all four erasure baselines.

## 2026-06-23: Round5 Initial Clean-Source Labels

**Goal:** Pre-label the 60 generated round5 clean-source videos before deciding which prompts are safe enough for erasure-baseline runs.

**Input artifacts:**

```text
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/frame_strips/
outputs/analysis_contact_sheets/causal_footprint_v0_round5_taxonomy_expansion60_step20/clean_source_screening.csv
```

**Tracked label file:**

```text
experiments/clean_screening/causal_footprint_v0_round5_taxonomy_expansion60_initial_labels.csv
```

**Label policy:** `yes` is reserved for videos where the target, footprint, and temporal/dependence relation are all usable. `borderline` is used when the target and footprint are visible but the temporal chain is weak, cropped, static, or visually ambiguous. Blank/pure-color, target-missing, or footprint-missing generations are marked `no`.

**Summary:**

```text
total: 60
yes: 10
borderline: 11
no: 39
```

**By mechanism:**

```text
elastic_deformation: yes 0, borderline 1, no 9
field_mediated: yes 4, borderline 2, no 4
fluid_impact: yes 1, borderline 2, no 7
fracture_damage: yes 0, borderline 1, no 9
particle_dispersion: yes 4, borderline 3, no 3
surface_trace: yes 1, borderline 2, no 7
```

**Strict yes rows:**

```text
round5_fluid_coin_fountain_005
round5_surface_tire_puddle_010
round5_field_magnet_filings_001
round5_field_balloon_hair_003
round5_field_fan_streamers_005
round5_field_speaker_sand_008
round5_particle_chalk_eraser_003
round5_particle_snowball_wall_007
round5_particle_glitter_jar_008
round5_particle_soil_trowel_010
```

**Borderline rows:**

```text
round5_fluid_leaf_pond_001
round5_fluid_cherry_soda_006
round5_surface_sneaker_wet_sand_001
round5_surface_paw_mud_003
round5_fracture_ceramic_plate_floor_005
round5_elastic_boxing_glove_bag_006
round5_field_plastic_ruler_paper_004
round5_field_comb_confetti_010
round5_particle_flour_sifter_001
round5_particle_salt_shaker_002
round5_particle_seed_bag_006
```

**Interpretation:** Round5 confirms that broadening the taxonomy helps, but CogVideoX-2B still fails many physical prompts by producing blank/pure-color frames or static footprint-only scenes. For the next baseline run, the strict scientific slice is the 10 `yes` rows; the 11 `borderline` rows can be kept as an exploratory/backup slice but should not be mixed into the main claim without review.

## 2026-06-23: Round5 Clean-Valid Prompt Slice Export

**Goal:** Freeze prompt files for the next erasure-baseline run after clean-source screening.

**Inputs:**

```text
benchmarks/causal_footprint_v0/round5_taxonomy_expansion_prompts.tsv
experiments/clean_screening/causal_footprint_v0_round5_taxonomy_expansion60_initial_labels.csv
```

**Outputs:**

```text
prompts/causal_footprint_v0_round5_clean_yes10.txt
benchmarks/causal_footprint_v0/export_round5_clean_yes10_manifest.json
prompts/causal_footprint_v0_round5_clean_yes_borderline21.txt
benchmarks/causal_footprint_v0/export_round5_clean_yes_borderline21_manifest.json
```

**Counts:**

```text
yes10: 10 prompts
yes_borderline21: 21 prompts
```

**Decision:** use `prompts/causal_footprint_v0_round5_clean_yes10.txt` as the main scientific slice for the next four-baseline run. Keep the 21-row `yes + borderline` slice separate as backup/exploratory material.

## 2026-06-23: Round5 yes10 Four-Baseline Run

**Goal:** Run the strict clean-valid round5 slice on the four required erasure baselines and build a clean-reference-aligned review page.

**Prompt slice:**

```text
prompts/causal_footprint_v0_round5_clean_yes10.txt
benchmarks/causal_footprint_v0/export_round5_clean_yes10_manifest.json
```

**Generation settings:**

```text
model: models/CogVideoX-2b
seed: 6100 + prompt index
steps: 20
guidance_scale: 6.0
frames: 49
resolution: 720x480
dtype: bf16
baselines: negative_prompt, safree_cogvideox, videoeraser, t2vunlearning
```

**Runtime note:** The first mixed run exposed a memory issue in local VideoEraser/T2V proxy prompt encoding. The local adapters now encode prompts on CPU before enabling sequential/model CPU offload. This avoids T5 prompt-encoder OOM while keeping generation on the requested GPU/offload path.

**Result:**

```text
negative_prompt: 10 / 10
safree_cogvideox: 10 / 10
videoeraser: 10 / 10
t2vunlearning: 10 / 10
total erasure outputs: 40 / 40
```

**Review artifacts:**

```text
outputs/baseline_suite_causal_footprint_v0_round5_yes10_all_step20_parallel/
outputs/analysis_contact_sheets/causal_footprint_v0_round5_yes10_baseline_step20/baseline_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round5_yes10_baseline_step20/baseline_review.csv
experiments/baseline_runs/causal_footprint_v0_round5_yes10_all_step20_parallel_summary.csv
```

**Review tool update:** `scripts/build_baseline_review.py` is now the reusable review builder for prompt-slice baseline runs. It reads an exported clean-source manifest plus a baseline output root, keeps missing rows explicit, writes `baseline_review.csv`, and builds a clean-reference plus baseline HTML gallery.

**Interpretation:** This is the current main round5 evidence pool, but labels remain pending. The next scientific step is manual review of target visibility, footprint visibility, source/effect separation, and usable-for-claim status.

## 2026-06-23: Round5 borderline11 Exploratory Four-Baseline Run

**Goal:** Use idle GPUs to run the 11 round5 clean-source `borderline` rows separately, without mixing them into the strict main claim.

**Prompt slice:**

```text
prompts/causal_footprint_v0_round5_clean_borderline11.txt
benchmarks/causal_footprint_v0/export_round5_clean_borderline11_manifest.json
```

**Generation settings:**

```text
model: models/CogVideoX-2b
seed: 6300 + prompt index
steps: 20
guidance_scale: 6.0
frames: 49
resolution: 720x480
dtype: bf16
baselines: negative_prompt, safree_cogvideox, videoeraser, t2vunlearning
```

**Result:**

```text
negative_prompt: 11 / 11
safree_cogvideox: 11 / 11
videoeraser: 11 / 11
t2vunlearning: 11 / 11
total erasure outputs: 44 / 44
```

**Review artifacts:**

```text
outputs/baseline_suite_causal_footprint_v0_round5_borderline11_all_step20_parallel/
outputs/analysis_contact_sheets/causal_footprint_v0_round5_borderline11_baseline_step20/baseline_gallery.html
outputs/analysis_contact_sheets/causal_footprint_v0_round5_borderline11_baseline_step20/baseline_review.csv
experiments/baseline_runs/causal_footprint_v0_round5_borderline11_all_step20_parallel_summary.csv
```

**Interpretation:** These rows are exploratory because their clean references had weak temporal order, weak causal dependence, or partial target/effect visibility. They are useful for candidate mining but should not be merged into headline metrics unless later adjudicated.

## 2026-07-01: ZeroScope v2 Clean-Valid96 Four-Baseline Closure

**Goal:** Close the ZeroScope branch end-to-end on the v2 control-free benchmark protocol: clean-source gating, four erasure baselines, reference-aligned review sheets, VLM atomic labels, retry of API failures, and metric tables.

**Clean-source slice:**

```text
clean generation manifest:
outputs/zeroscope_v2_candidates304_clean_step20_f24_320x576_8gpu_s3/clean/generation_manifest.json

clean-gate VLM aggregate:
experiments/evaluation/zeroscope_v2_clean_gate_gpt54_6shards_retry_20260701_133602/aggregate_predictions_merged.csv

accepted prompts:
prompts/zeroscope_v2_clean_valid_gpt54_96.txt
benchmarks/causal_footprint_v2/zeroscope_clean_valid_gpt54_96_manifest.json
```

**Clean-gate result:** 96 / 304 candidate prompts passed the simplified v2 source criterion: target visible and causal footprint visible in the clean reference.

**Baseline generation settings:**

```text
base model: ZeroScope v2
resolution: 320x576
frames: 24
steps: 20
seed base: 3
baselines: negative_prompt, videoeraser, t2vunlearning, safree_zeroscope
```

**Baseline generation result:**

```text
output root:
outputs/zeroscope_v2_clean_valid96_baselines_step20_f24_320x576_8gpu_offload_s1_attached

negative_prompt: 96 / 96
videoeraser: 96 / 96
t2vunlearning: 96 / 96
safree_zeroscope: 96 / 96
total erasure outputs: 384 / 384

completion record:
experiments/baseline_runs/zeroscope_v2_clean_valid96_completion.json
```

**Runtime note:** The first high-concurrency run hit GPU OOM because the machine was already occupied by unrelated 8-GPU training. The final run used model CPU offload, then a sequential CPU-offload repair pass for missing rows.

**Review artifacts:**

```text
experiments/baseline_review/zeroscope_v2_clean_valid96_baselines_review/baseline_review.csv
experiments/baseline_review/zeroscope_v2_clean_valid96_baselines_review/baseline_gallery.html
experiments/baseline_review/zeroscope_v2_clean_valid96_baselines_review/frame_strips/
```

The review builder wrote 480 rows: 96 clean references plus 384 erasure outputs. Each video is represented by a 5-frame evenly sampled contact sheet.

**VLM evaluation protocol:**

```text
judge model: gpt-5.4
input per row: clean-reference contact sheet + erased-output contact sheet
atomic fields: target_visible, footprint_visible, footprint_match, separation_clear, video_quality, confidence, reason
derived final labels: strict_causal_footprint_leakage, erased_clean, target_leakage, borderline, other_failure
```

The first monolithic VLM run was stopped because it only wrote results after the full batch finished. The final run split the 384 rows into 32 shards, each with 12 rows. Thirteen API-level failures were retried separately and all were recovered.

**VLM artifacts:**

```text
sharded raw root:
experiments/evaluation/zeroscope_v2_clean_valid96_baselines_gpt54_sharded32_20260701

final merged labels:
experiments/evaluation/zeroscope_v2_clean_valid96_baselines_gpt54_sharded32_20260701/vlm_predictions_merged_retry1.csv

metrics:
experiments/metrics/zeroscope_v2_clean_valid96_baselines_gpt54_20260701/v2_metrics_summary.md
experiments/metrics/zeroscope_v2_clean_valid96_baselines_gpt54_20260701/v2_metrics_by_baseline.csv
experiments/metrics/zeroscope_v2_clean_valid96_baselines_gpt54_20260701/v2_metrics_by_mechanism.csv
```

**Final aggregate metrics:**

```text
total outputs: 384
target erased: 316 / 384 (0.8229)
footprint retained given target erased: 124 / 316 (0.3924)
strict causal-footprint leakage: 46 / 384 (0.1198)
strict leakage given target erased: 46 / 316 (0.1456)
erased clean: 149 / 384 (0.3880)
target leakage: 23 / 384 (0.0599)
borderline: 143 / 384 (0.3724)
other failure: 23 / 384 (0.0599)
```

**By baseline:**

```text
negative_prompt: strict 11 / 96, target erased 67 / 96
videoeraser: strict 21 / 96, target erased 87 / 96
t2vunlearning: strict 11 / 96, target erased 81 / 96
safree_zeroscope: strict 3 / 96, target erased 81 / 96
```

**By mechanism:** Strict leakage is most visible for `fluid_impact` (12 / 60) and `fracture_damage` (15 / 92). `surface_trace` has lower strict leakage (6 / 100) but many borderline rows, suggesting these prompts often produce ambiguous traces rather than clean source-target separation.

**Conclusion:** ZeroScope is now closed for the v2 protocol. It gives a second model family beyond CogVideoX-2B where current erasure baselines can remove the target in many cases while still retaining a downstream footprint. The strongest ZeroScope baseline for this failure mode is the local VideoEraser adapter; SAFREE-ZeroScope appears cleaner on strict leakage but has more erased-clean and quality-failure outcomes.

## 2026-07-02: ZeroScope MVP-0 Causal Chain Probe Dry-Run

**Goal:** Validate the causal-chain steering idea before committing to it as a
method. This step only builds the probe scaffold and dry-run generation matrix;
it does not claim real denoising steering success.

**Artifacts:**

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/prompts/source_prompts.txt
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/prompts/counterfactual_prompts.txt
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dry_run_generation_matrix/generation_manifest.json
```

**Probe construction:** 12 clean-valid ZeroScope v2 cases were selected from
the existing evaluated slice, prioritizing strict causal-footprint leakage while
round-robin balancing mechanism families. The selected slice covers
`fluid_impact`, `fracture_damage`, `elastic_deformation`,
`particle_dispersion`, `surface_trace`, and `field_mediated`.

**Dry-run matrix:** 96 planned rows were emitted: 12 cases times 8 conditions
(`target_negative`, `target_footprint_negative`,
`monolithic_counterfactual`, `cause_steering`, `mechanism_steering`,
`footprint_steering`, `full_chain_steering`, and `random_direction`).

**Noise control added during validation:** The builder now extracts a neutral
scene context from the source prompt when available and sanitizes
counterfactual/control contexts before creating minimal pairs. This prevents
obvious contradictions such as "no target is present, with target" or using a
footprint-only control as the background for a no-footprint pair.

**Gate:** Real ZeroScope steering is intentionally disabled in the runner until
the denoising-loop insertion path is inspected and written down. The next
scientific check is a one-prompt or few-prompt real steering smoke against the
prompt-only and random-direction controls.

## 2026-07-02: MVP-0 Real Runner Interface and Denoising-Loop Gate

**Decision:** The installed generation environment is `dyme` with
`diffusers==0.34.0`. Its `TextToVideoSDPipeline.__call__` exposes only the old
post-step callback API, so the MVP-0 steering implementation uses a focused
copied denoising loop. The steering insertion point is before
`scheduler.step(...)`, after the guided `noise_pred` is computed.

**Implemented support:** `scripts/adapters/run_mvp0_zeroscope_probe.py` now
supports non-dry-run rows through a copied ZeroScope loop. It encodes the main
prompt plus each selected cause/mechanism/footprint minimal pair, computes
positive-minus-negative residual directions, applies them inside a configured
timestep window, and records `alpha` plus `timestep_window` in the generation
manifest.

**Test coverage:**

```text
python -m pytest tests/test_run_mvp0_zeroscope_probe.py tests/test_build_mvp0_causal_chain_probe.py -q
11 passed
```

The tests cover dry-run contracts, strict-leakage balanced selection,
contradiction-free minimal-pair construction, pre-scheduler residual steering,
and CFG-consistent minimal-pair residuals.

**Ready smoke command once GPU memory is available:**

```bash
/home/deepseek_VG/.conda/envs/dyme/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_full_chain_item0 \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition full_chain_steering \
  --limit-items 1 \
  --steps 20 \
  --num-frames 24 \
  --height 320 \
  --width 576 \
  --alpha 0.5 \
  --timestep-window 4:14 \
  --enable-model-cpu-offload
```

**Current blocker:** The 8 H800 GPUs were already using roughly 72-75GB each
when this runner was prepared, so the real smoke was not launched in this
step. This is an engineering availability blocker, not a method result.

## 2026-07-02: Tiny Real Full-Chain Steering Smoke

**Goal:** Try a minimal real generation path despite heavy GPU occupancy, only
to check whether the copied ZeroScope steering loop can produce a readable
video artifact.

**Environment:** The first attempt in `dyme` failed before generation because
ZeroScope is stored as `.bin` weights and `transformers` now requires
`torch>=2.6` for this loading path. The successful attempt used `vcecf`
(`torch==2.6.0+cu124`, `diffusers==0.34.0`) with `PYTHONNOUSERSITE=1`.

**Command shape:** One item, one condition, reduced resolution and frame count:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_full_chain_item0_tiny \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition full_chain_steering \
  --limit-items 1 \
  --steps 4 \
  --num-frames 8 \
  --height 160 \
  --width 288 \
  --alpha 0.5 \
  --timestep-window 1:3 \
  --enable-model-cpu-offload \
  --vae-slicing
```

**Artifacts:**

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_full_chain_item0_tiny/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_full_chain_item0_tiny/videos/000_fluid-impact-pebble-pond-002_full_chain_steering_seed15000.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_full_chain_item0_tiny/first_frame.jpg
```

**Sanity check:** OpenCV read the MP4 successfully: 8 frames, 288x160, 8 fps.
The first frame was nonblank. This validates the runtime path only. The tiny
settings are too low-quality to evaluate causal-footprint suppression.

**Next gate:** Run a comparable four-condition smoke at standard ZeroScope
settings or near-standard settings once GPU memory is available:
`target_negative`, `target_footprint_negative`, `full_chain_steering`, and
`random_direction` for the same item and seed.

## 2026-07-02: Three-Condition Real Smoke on Least-Busy GPU

**Goal:** Produce a small comparable set for one probe item after selecting the
least memory-occupied GPU. This is still a runtime and visual sanity check, not
a method result.

**GPU selection:** `nvidia-smi` reported GPU 3 as the lowest-memory device at
query time, with 72533 / 81559 MiB used and 76% utilization. All GPUs were
heavily occupied, so this run used CPU offload and reduced video settings.

**Command:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320 \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition full_chain_steering \
  --limit-items 1 \
  --steps 8 \
  --num-frames 12 \
  --height 192 \
  --width 320 \
  --alpha 0.5 \
  --timestep-window 2:6 \
  --enable-model-cpu-offload \
  --vae-slicing
```

**Artifacts:**

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320/videos/000_fluid-impact-pebble-pond-002_target_negative_seed15000.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320/videos/000_fluid-impact-pebble-pond-002_target_footprint_negative_seed15000.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320/videos/000_fluid-impact-pebble-pond-002_full_chain_steering_seed15000.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_smoke_item0_3cond_s8_f12_192x320/contact_sheet.jpg
```

**Mechanical sanity check:** OpenCV loaded all three MP4 files. Each video has
12 frames at 320x192 and 8 fps. First-frame means were nonzero:
`target_negative=109.46`, `target_footprint_negative=122.25`,
`full_chain_steering=120.96`.

**Visual note:** At this low resolution the videos are semantically weak, but
the contact sheet is usable for eyeballing. The `target_negative` sample shows
stronger visible water/ripple texture, while `target_footprint_negative` and
`full_chain_steering` look closer to a calmer water/reflection scene. This is
only an informal observation and should not be treated as evidence of causal
footprint suppression.

**Next gate:** Run a stronger comparison at higher quality for 3-5 probe items,
then evaluate with the VLM leakage labels. The `random_direction` condition
should be implemented or audited before being used as a scientific control.

## 2026-07-02: Three-Item Batch Smoke and Prompt-Length Finding

**Goal:** Let the real runner "soak" on a slightly larger but still bounded
comparison: 3 probe items x 3 conditions. The purpose was to check whether
larger smoke runs are practical and whether the outputs are visually inspectable.

**GPU selection:** `nvidia-smi` showed all GPUs at 100% utilization. GPU 3 still
had the lowest memory usage at query time, so the run used GPU 3 with CPU
offload.

**Command shape:**

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python scripts/adapters/run_mvp0_zeroscope_probe.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_s10_f16_240x432 \
  --model models/zeroscope_v2_576w \
  --seed 15000 \
  --condition target_negative \
  --condition target_footprint_negative \
  --condition full_chain_steering \
  --limit-items 3 \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --alpha 0.5 \
  --timestep-window 2:8 \
  --enable-model-cpu-offload \
  --vae-slicing
```

**Artifacts:**

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_s10_f16_240x432/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_s10_f16_240x432/videos/*.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_s10_f16_240x432/contact_sheet.jpg
```

**Mechanical sanity check:** All 9 MP4 files loaded in OpenCV. Each has 16
frames at 432x240 and 8 fps. First frames were nonblank.

**Visual note:** The batch is useful for qualitative inspection. The first
fluid-impact item again shows a strong water-ripple pattern in `target_negative`
and a calmer water/reflection scene in the footprint-negative and steering
conditions. The fracture and elastic-deformation items are less clean but still
visually interpretable enough for manual triage. This remains a smoke result,
not evidence of method effectiveness.

**Important prompt-length finding:** During generation, CLIP warned that some
prompts exceeded the 77-token limit. A follow-up tokenizer audit showed that
only the long `source_prompt` fields exceed the limit: 10 of the 12 probe items
have source prompts between 82 and 94 CLIP tokens. The counterfactual, control,
and minimal-pair prompts are within the limit. This means the current
`target_negative` baseline can be silently truncated for most items, while the
steering residual prompts are not. Larger runs should wait until the source
prompt template is compressed or a no-truncation item subset is selected.

**Next gate:** Add a prompt-length audit to the probe-builder or runner, then
either compress source prompts to fit CLIP's 77-token limit or run the next
batch only on no-truncation items. After that, rerun the 3-item batch and send
the outputs through the VLM leakage evaluator.

## 2026-07-02: Compact Generation Prompts and Strict Audit

**Fix:** The probe builder now preserves the original `source_prompt` for
auditability and adds a compact `generation_prompt` for actual source-based
generation. The compact prompt keeps the neutral scene context, target, and
causal footprint, while removing long temporal scaffolding that was pushing
many prompts past CLIP's 77-token limit. The runner now uses
`generation_prompt` when present and keeps the original `source_prompt` in the
generation manifest for traceability.

**Guard:** The ZeroScope MVP-0 runner now supports strict prompt-length checks:

```bash
--strict-prompt-length --prompt-token-limit 77
```

The audit covers the main prompt, negative prompt, and active minimal-pair
prompts. In strict mode, any over-limit prompt fails before generation.

**Validation:**

```text
python -m pytest tests/test_build_mvp0_causal_chain_probe.py tests/test_run_mvp0_zeroscope_probe.py -q
15 passed
```

Rebuilding the 12-item probe and running the strict dry-run matrix succeeded:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dry_run_generation_matrix_strict_prompt_audit/generation_manifest.json
```

The original source prompts remain long for audit purposes, but all compact
generation prompts are below the CLIP limit:

```text
source range: 26-94 CLIP tokens
generation range: 26-54 CLIP tokens
```

**Rerun after fix:** The 3-item x 3-condition smoke was rerun with compact
prompts and `--strict-prompt-length`.

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_compact_s10_f16_240x432/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_compact_s10_f16_240x432/videos/*.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_batch_3items_3cond_compact_s10_f16_240x432/contact_sheet.jpg
```

No CLIP truncation warnings appeared during this rerun. OpenCV loaded all 9
videos successfully: each has 16 frames at 432x240 and 8 fps.

**Visual note:** The compact rerun looks cleaner than the earlier long-source
batch. On the fluid-impact item, `target_negative` still preserves a strong
ripple pattern, while `target_footprint_negative` and `full_chain_steering`
move toward pond vegetation/reflection scenes. The fracture and net examples
also become visually more stable. This is still a smoke observation; the next
real gate is VLM scoring.

## 2026-07-02: Random and Orthogonal Control Pilot

**Motivation:** Fable/reviewer feedback identified two non-negotiable controls:
a norm-matched random direction and an unrelated semantic direction. Without
these, a reviewer can argue that any residual perturbation of comparable norm
reduces motion/detail and therefore removes footprints without causal repair.

**Implementation:** The runner now includes:

```text
random_direction       -> gaussian_norm_matched control using the footprint residual norm
orthogonal_semantic    -> unrelated semantic direction: birds flying vs no birds
```

The random control encodes the footprint pair as a reference, computes its
residual norm per denoising step, samples a deterministic Gaussian direction
from the row seed and step index, and rescales it to the same norm before
applying the normal steering subtraction. The orthogonal control uses the same
steering machinery as real links but with an unrelated fixed minimal pair.

**Validation:**

```text
python -m pytest tests/test_run_mvp0_zeroscope_probe.py tests/test_build_mvp0_causal_chain_probe.py -q
19 passed
```

Strict dry-run with controls succeeded:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/dry_run_controls_strict_prompt_audit/generation_manifest.json
```

**Pilot run:** 3 items x 5 conditions, with strict prompt-length checking:

```text
target_negative
target_footprint_negative
full_chain_steering
random_direction
orthogonal_semantic
```

Artifacts:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_pilot_3items_5cond_controls_s10_f16_240x432/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_pilot_3items_5cond_controls_s10_f16_240x432/videos/*.mp4
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/real_pilot_3items_5cond_controls_s10_f16_240x432/contact_sheet.jpg
```

OpenCV loaded all 15 videos successfully: each has 16 frames at 432x240 and
8 fps.

**Visual note and risk:** The controls are informative but not yet favorable.
For the three pilot items, `random_direction` and `orthogonal_semantic` often
look close to `full_chain_steering` and `target_footprint_negative`. This may
mean the current steering strength/window is dominated by generic perturbation
or scene drift rather than causal-chain-specific repair. We should not scale or
claim success until VLM/low-level metrics show that full-chain steering reduces
footprint leakage more than both controls while preserving scene quality.

**Next gate:** Run VLM scoring and simple low-level proxies on this 15-video
pilot before launching a larger multi-seed batch. If controls match full-chain,
reduce alpha/window, average paraphrased causal directions, or reframe this
iteration as a leakage diagnostic rather than a repair result.

## 2026-07-02: Fable VLM and Low-Level Proxy Gate for Control Pilot

**Input:** The 3-item x 5-condition control pilot above was converted into
frame strips and paired with the corresponding clean-reference strips.

Artifacts:

```text
experiments/evaluation/mvp0_zeroscope_pilot_controls_fable_20260702/review.csv
experiments/evaluation/mvp0_zeroscope_pilot_controls_fable_20260702/frame_strips/*.jpg
experiments/evaluation/mvp0_zeroscope_pilot_controls_fable_20260702/fable_run/vlm_predictions.csv
experiments/evaluation/mvp0_zeroscope_pilot_controls_fable_20260702/low_level_proxy.csv
experiments/evaluation/mvp0_zeroscope_pilot_controls_fable_20260702/low_level_proxy_summary.csv
```

The dry-run produced 15 VLM payloads, and the fable run produced 15 prediction
rows.

**Fable summary:**

```text
target_negative:
  target_leakage=1, strict_causal_footprint_leakage=1, erased_clean=1

target_footprint_negative:
  erased_clean=2, target_leakage=1

full_chain_steering:
  erased_clean=2, strict_causal_footprint_leakage=1

random_direction:
  erased_clean=2, strict_causal_footprint_leakage=1

orthogonal_semantic:
  erased_clean=2, strict_causal_footprint_leakage=1
```

This is a useful negative/diagnostic result. `full_chain_steering` did not
separate from either norm-matched random steering or unrelated semantic
steering. On the fracture item, fable still reports footprint leakage for all
three steering/control conditions: no puck, but a crack remains. On the fluid
and elastic-net items, all three steering/control conditions are judged clean.
Thus the current pilot does not support a causal-specific repair claim.

**Low-level proxy summary:** Mean motion and frame-difference proxies do not
show a simple global-freezing explanation. Relative to `target_negative`,
`full_chain_steering` has mean frame-difference ratio 0.995 and mean-flow ratio
0.895, while `random_direction` has 1.098 and 0.986, and
`orthogonal_semantic` has 1.047 and 0.933. The problem is therefore not merely
that full-chain steering destroys all motion; the stronger issue is that its
semantic effect is not better than the controls in this pilot.

**Decision:** Do not scale this configuration as a positive method result.
The next experiment should make the method more discriminative before another
multi-seed run:

```text
1. Reduce alpha and/or narrow the denoising window.
2. Build paraphrase-averaged causal directions instead of one brittle pair.
3. Add a non-causal scene-preservation direction or regularizer.
4. Keep random and orthogonal controls in every batch.
5. Treat target-negative leakage as the main diagnostic, and require full-chain
   to beat both controls before claiming causal repair.
```

## 2026-07-02: Phase A Conservative Sweep Orchestrator and First Cell

**Implementation:** Added a Phase A sweep orchestrator for the MVP-0 ZeroScope
probe. It expands alpha/window grids into calls to the existing runner, writes
a `sweep_manifest.json`, and can execute all cells or a limited prefix. The
core diffusion runner was not changed.

Files:

```text
scripts/adapters/run_mvp0_zeroscope_sweep.py
tests/test_run_mvp0_zeroscope_sweep.py
docs/superpowers/specs/2026-07-02-causal-chain-steering-a-then-b-design.md
docs/superpowers/plans/2026-07-02-causal-chain-steering-phase-a-sweep.md
```

Validation:

```text
python -m pytest tests/test_run_mvp0_zeroscope_sweep.py tests/test_run_mvp0_zeroscope_probe.py tests/test_build_mvp0_causal_chain_probe.py -q
22 passed
```

The project directory is not a git repository, so there is no commit hash for
this implementation checkpoint.

**Dry-run sweep:** The 3 x 3 Phase A grid completed as dry-run:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_sweep_dry_run/sweep_manifest.json
```

The manifest reports 9 total cells and 9 completed cells. Spot checks confirmed
15 planned items per cell, default conditions correctly included in the runner
argv, and the expected alpha/window metadata in generated manifests.

**First real Phase A cell:** Ran the midpoint conservative setting:

```text
alpha = 0.25
timestep_window = 3:6
conditions = target_negative, target_footprint_negative, full_chain_steering,
             random_direction, orthogonal_semantic
```

Artifacts:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p25_window_3_6/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p25_window_3_6/videos/*.mp4
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/review.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/contact_sheet.jpg
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/frame_strips/*.jpg
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/fable_run/vlm_predictions.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/low_level_proxy.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p25_window_3_6_fable_20260702/low_level_proxy_summary.csv
```

OpenCV loaded all 15 videos successfully: each has 16 frames at 432x240 and
8 fps.

**Fable summary:**

```text
target_negative:
  borderline=3

target_footprint_negative:
  erased_clean=2, borderline=1

full_chain_steering:
  erased_clean=2, borderline=1

random_direction:
  erased_clean=2, borderline=1

orthogonal_semantic:
  erased_clean=2, borderline=1
```

The midpoint conservative cell improves the earlier fracture strict-leakage
failure, but the improvement is not causal-specific: `full_chain_steering`,
`random_direction`, and `orthogonal_semantic` still have the same label
distribution. Fable marks all full-chain outputs as `video_quality=yes`, but
the Phase A success gate is not met because full-chain does not beat either
control.

**Low-level proxy summary:** Full-chain does not look like global freezing in
this cell. Relative to `target_negative`, full-chain has mean frame-difference
ratio 1.192 and mean-flow ratio 1.034. Random has 1.096 and 0.983; orthogonal
semantic has 1.090 and 0.993. This suggests the failure is not just collapse or
motion suppression; the issue remains lack of semantic separation from
controls.

**Decision:** Do not scale this cell as a positive result. Continue Phase A
with at least one lower-strength cell, especially `alpha=0.15/window=2:5` or
`alpha=0.15/window=3:6`, before moving to Phase B paraphrase-averaged causal
directions.

## 2026-07-02: Phase A Low-Strength Cell and Stop Decision

**Second real Phase A cell:** Ran the lower-strength setting:

```text
alpha = 0.15
timestep_window = 2:5
conditions = target_negative, target_footprint_negative, full_chain_steering,
             random_direction, orthogonal_semantic
```

Artifacts:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p15_window_2_5/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_a_real_alpha_0p15_window_2_5/videos/*.mp4
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/review.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/contact_sheet.jpg
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/frame_strips/*.jpg
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/fable_run/vlm_predictions.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/low_level_proxy.csv
experiments/evaluation/mvp0_zeroscope_phase_a_alpha_0p15_window_2_5_fable_20260702/low_level_proxy_summary.csv
```

OpenCV loaded all 15 videos successfully: each has 16 frames at 432x240 and
8 fps.

**Fable summary:**

```text
target_negative:
  borderline=1, strict_causal_footprint_leakage=2

target_footprint_negative:
  erased_clean=1, strict_causal_footprint_leakage=2

full_chain_steering:
  erased_clean=1, target_leakage=2

random_direction:
  erased_clean=1, strict_causal_footprint_leakage=2

orthogonal_semantic:
  erased_clean=1, target_leakage=2
```

This lower-strength cell does not meet the Phase A gate. It removes strict
footprint leakage from full-chain only by reintroducing visible target leakage
on the fracture and elastic-net items. The unrelated semantic control shows
the same target-leakage pattern, while the random and target-footprint-negative
conditions keep strict footprint leakage. Therefore alpha/window tuning alone
does not produce a clean causal-specific separation.

**Low-level proxy summary:** There is still no obvious global collapse:
relative to `target_negative`, full-chain has mean frame-difference ratio
1.078 and mean-flow ratio 0.961; random has 1.096 and 0.988; orthogonal
semantic has 1.089 and 0.991. The failure is semantic, not a simple motion
suppression artifact.

**Decision:** Stop Phase A instead of sweeping all remaining cells. We now have
two informative settings:

```text
alpha=0.25/window=3:6: full-chain improves with controls, no separation.
alpha=0.15/window=2:5: full-chain avoids strict footprint leakage but leaks
                       the target like the unrelated semantic control.
```

The next step should be Phase B: paraphrase-averaged causal directions. The
implementation should preserve the same random and orthogonal controls, and
random should be norm-matched to the averaged footprint direction.

## 2026-07-02: Phase B Paraphrase-Averaged Causal Directions

**Implementation:** Added a Phase B manifest builder and runner support for
multi-pair link prompts. Each causal link now carries three minimal-pair
directions: the original pair plus two deterministic paraphrases. The runner
encodes every pair, predicts each residual direction, averages the
`positive - negative` directions, and applies the averaged link residual. The
old single-pair manifest shape remains supported.

Artifacts:

```text
docs/superpowers/plans/2026-07-02-causal-chain-steering-phase-b-paraphrase-averaging.md
scripts/build_mvp0_phase_b_paraphrase_probe.py
scripts/adapters/run_mvp0_zeroscope_probe.py
tests/test_build_mvp0_phase_b_paraphrase_probe.py
tests/test_run_mvp0_zeroscope_probe.py
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_paraphrase_probe_manifest.json
```

**Dry run:** The Phase B dry run passed on the first three items with strict
prompt-length checking:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_dry_run/generation_manifest.json
```

The dry-run manifest confirms that `full_chain_steering` uses three prompt
pairs for cause, mechanism, and footprint. The random control also uses the
three footprint paraphrases as its norm/source reference. The orthogonal
semantic control remains the original single unrelated semantic direction.

**Real cell:** Ran the same midpoint setting as the first Phase A cell:

```text
alpha = 0.25
timestep_window = 3:6
conditions = target_negative, target_footprint_negative, full_chain_steering,
             random_direction, orthogonal_semantic
limit_items = 3
```

Artifacts:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_real_alpha_0p25_window_3_6/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_real_alpha_0p25_window_3_6/videos/*.mp4
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/review.csv
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/contact_sheet.jpg
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/frame_strips/*.jpg
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/fable_run/vlm_predictions.csv
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/low_level_proxy.csv
experiments/evaluation/mvp0_zeroscope_phase_b_alpha_0p25_window_3_6_fable_20260702/low_level_proxy_summary.csv
```

OpenCV loaded all 15 generated videos successfully: each has 16 frames at
432x240 and 8 fps.

**Fable summary:**

```text
target_negative:
  borderline=2, strict_causal_footprint_leakage=1

target_footprint_negative:
  borderline=1, strict_causal_footprint_leakage=2

full_chain_steering:
  erased_clean=1, strict_causal_footprint_leakage=2

random_direction:
  erased_clean=2, target_leakage=1

orthogonal_semantic:
  erased_clean=2, strict_causal_footprint_leakage=1
```

This is the first partial positive signal. Compared with the Phase A midpoint
cell, paraphrase averaging separates full-chain from the random control:
`full_chain_steering` preserves the causal footprint in two of three items,
while `random_direction` preserves it in zero of three and leaks the target on
one item. However, the signal is not yet robust enough to claim a causal method:
the unrelated semantic control still produces one strict footprint-leakage
case, and the sample size is only three items. The current orthogonal control
also has a weaker construction than full-chain because it remains a single
semantic direction rather than a paraphrase-averaged, norm-matched control.

**Low-level proxy summary:** Phase B does not look like a simple global-motion
artifact. Relative to `target_negative`, full-chain has mean frame-difference
ratio 1.207 and mean-flow ratio 1.025. Random has 1.114 and 0.954; orthogonal
semantic has 1.090 and 0.938. The strongest full-chain semantic wins occur on
fluid impact and fracture damage; elastic deformation remains clean across
full-chain, random, and orthogonal controls.

**Decision:** Treat Phase B as promising but not paper-ready. The next
experiment should be B+ rather than immediate scaling: make the orthogonal
semantic control equally strong by paraphrase-averaging and norm-matching it,
then rerun the same `alpha=0.25/window=3:6` cell before expanding beyond three
items. A defensible success gate is: full-chain strict footprint leakage must
exceed both random and paraphrase-averaged orthogonal controls, with no
increase in target leakage and no large low-level proxy imbalance.

## 2026-07-02: Phase B+ Fair Orthogonal Control

**Implementation:** B+ keeps Phase B full-chain steering unchanged, but makes
the orthogonal semantic control fairer. The Phase B manifest builder now adds
three unrelated semantic minimal pairs under `orthogonal_semantic`. The runner
preserves manifest-provided orthogonal pairs, averages their residual
directions, encodes the averaged footprint direction as a reference, and scales
the orthogonal direction to that footprint-reference norm before applying
steering.

Artifacts:

```text
docs/superpowers/specs/2026-07-02-causal-chain-steering-b-plus-fair-controls-design.md
docs/superpowers/plans/2026-07-02-causal-chain-steering-b-plus-fair-controls.md
scripts/build_mvp0_phase_b_paraphrase_probe.py
scripts/adapters/run_mvp0_zeroscope_probe.py
tests/test_build_mvp0_phase_b_paraphrase_probe.py
tests/test_run_mvp0_zeroscope_probe.py
```

Focused tests pass:

```text
python -m pytest tests/test_run_mvp0_zeroscope_probe.py \
  tests/test_build_mvp0_causal_chain_probe.py \
  tests/test_run_mvp0_zeroscope_sweep.py \
  tests/test_build_mvp0_phase_b_paraphrase_probe.py -q

31 passed, 1 warning
```

**Dry run:** The B+ manifest and dry-run rows were generated successfully:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_fair_controls_probe_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_dry_run/generation_manifest.json
```

The dry-run manifest confirms that `orthogonal_semantic` rows now use three
orthogonal pairs and `control_reference=footprint`.

**Real cell:** Ran the same settings as Phase B:

```text
alpha = 0.25
timestep_window = 3:6
conditions = target_negative, target_footprint_negative, full_chain_steering,
             random_direction, orthogonal_semantic
limit_items = 3
```

Artifacts:

```text
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_real_alpha_0p25_window_3_6/generation_manifest.json
experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/phase_b_plus_real_alpha_0p25_window_3_6/videos/*.mp4
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/review.csv
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/contact_sheet.jpg
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/frame_strips/*.jpg
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/fable_run/vlm_predictions.csv
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/fable_run/vlm_raw_responses.jsonl
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/low_level_proxy.csv
experiments/evaluation/mvp0_zeroscope_phase_b_plus_alpha_0p25_window_3_6_fable_20260702/low_level_proxy_summary.csv
```

OpenCV loaded all 15 generated videos successfully: each has 16 frames at
432x240 and 8 fps. The first full fable batch stalled on two HTTPS responses;
those two rows were rerun individually and succeeded. The final merged
prediction file has 15 valid rows.

**Fable summary:**

```text
target_negative:
  erased_clean=1, strict_causal_footprint_leakage=2

target_footprint_negative:
  erased_clean=2, strict_causal_footprint_leakage=1

full_chain_steering:
  erased_clean=2, strict_causal_footprint_leakage=1

random_direction:
  erased_clean=2, strict_causal_footprint_leakage=1

orthogonal_semantic:
  erased_clean=2, strict_causal_footprint_leakage=1
```

This fails the B+ success gate. Once the orthogonal semantic control is
paraphrase-averaged and norm-matched, `full_chain_steering`,
`random_direction`, and `orthogonal_semantic` have the same label distribution.
The only strict footprint-leakage case for all three controls is the fracture
damage item; the fluid-impact and elastic-deformation items are erased cleanly
for all three. Therefore the Phase B signal was likely a control-strength
artifact rather than a causal-chain-specific effect.

**Low-level proxy summary:** Relative to `target_negative`, full-chain has mean
frame-difference ratio 1.207 and mean-flow ratio 1.025. Random has 1.097 and
0.941; fair orthogonal has 1.085 and 0.930. Full-chain is somewhat stronger in
low-level motion/appearance perturbation, but that extra perturbation does not
produce a semantic win over controls.

**Decision:** Do not scale Phase B/B+ as a positive method result. The fair
control check is useful: it prevents an overclaim. The next method iteration
should change the intervention mechanism, not just add more paraphrases or
more items. Plausible next directions are a localized/cross-attention-gated
intervention, a temporal-windowed footprint-only residual, or a verifier-guided
selection loop that accepts only edits preserving target absence while
separating full-chain from matched controls.

## 2026-07-02: Method B Attention Dependency Probe

**Implementation:** Built a white-box ZeroScope cross-attention dependency
probe as a diagnostic replacement for failed residual steering. The probe
resolves CLIP token spans for the target concept and causal footprint, wraps
UNet `attn2` cross-attention processors, records compact attention summaries
instead of full maps, and writes per-item JSONL/CSV traces.

Artifacts:

```text
docs/superpowers/specs/2026-07-02-causal-attention-dependency-probe-design.md
docs/superpowers/plans/2026-07-02-causal-attention-dependency-probe.md
scripts/adapters/run_zeroscope_attention_probe.py
tests/test_run_zeroscope_attention_probe.py
experiments/method_probe/zeroscope_attention_dependency_probe_20260702_dryrun/generation_manifest.json
experiments/method_probe/zeroscope_attention_dependency_probe_20260702_smoke_textcfg_v2/
experiments/method_probe/zeroscope_attention_dependency_probe_20260702_diag3_textcfg/
```

Focused tests pass in the `vcecf` environment:

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  -m pytest tests/test_run_zeroscope_attention_probe.py -q

10 passed
```

### 2026-07-03 C0.1 factorial gate dry run

**Implementation:** Added the C0.1 seed-matched factorial gate infrastructure.
The C0 runner now supports `--seeds-per-item`; the C0.1 review builder writes
a blinded human-review CSV plus a separate answer key; and the C0.1 scorer
joins human labels to the answer key and applies the 4/5 and 3/5 gate
thresholds.

The blind review artifact intentionally does not expose the variant label,
expected target/footprint states, prompt text, or raw video path. The answer key
keeps those fields for later scoring.

**Dry-run artifacts:**

```text
experiments/method_probe/c01_factorial_gate_20260703_dryrun/generation_manifest.json
experiments/evaluation/c01_factorial_gate_20260703_dryrun/blind_review.csv
experiments/evaluation/c01_factorial_gate_20260703_dryrun/answer_key.csv
experiments/evaluation/c01_factorial_gate_20260703_dryrun/synthetic_scores/
```

The dry run expanded 3 MVP-0 items into 60 planned rows:

```text
3 items x 5 seeds x 4 cells = 60 rows
```

Variant distribution is balanced: 15 rows each for `original`,
`remove_target`, `footprint_only`, and `target_only`. A leakage check over the
blind review CSV found zero exposed cell labels. A synthetic completed-review
smoke, with labels copied from the answer key, produced `3/3` gate passes as
expected.

**Tests:**

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c01_factorial_gate_review.py \
  tests/test_score_c01_factorial_gate.py \
  tests/test_build_c0_counterfactual_review.py \
  tests/test_run_zeroscope_attention_probe.py -q

26 passed
```

**Fable implementation review:** After implementation, `claude-fable-5`
reviewed C0.1 as a method and engineering advisor, not as a video judge. It
found no blocking issue before the real 60-video pilot, provided the result is
framed only as a generation-validity gate. The required pilot guardrails are:
manually spot-check 10 to 15 generated videos before bulk review, track
`uncertain` labels by reviewer and cell type, and include at least 10
overlapping rows for inter-rater agreement. Full notes are in
`docs/fable_c01_implementation_review_2026-07-03.md`.

**Decision:** The C0.1 gate infrastructure is ready for a real 60-video pilot,
but no GPU generation or C1 repair claim is made in this step. The real pilot
should run only after confirming the GPU and review plan.

**Important measurement fix:** The first smoke run mixed classifier-free
guidance unconditional and text-conditioned attention batches. The processor
now slices the text-conditioned half of the expanded video batch before
recording, so token masses correspond to the actual positive prompt branch.

**Dry run:** The first three MVP-0 items resolved target and footprint spans,
including BPE fragments such as `ripp` + `les</w>` and `outw` + `ard</w>`.
A full 12-item dry run also completed successfully:

```text
experiments/method_probe/zeroscope_attention_dependency_probe_20260702_dryrun_all/generation_manifest.json
```

**Smoke run:** Ran one item on GPU 3 with `steps=2`, `num_frames=4`,
`160x288`, `fp16`, and `--skip-video-export`. The run completed and produced
32 cross-attention records: 16 modules over 2 denoising steps.

**Three-item diagnostic:** Ran three MVP-0 items with `steps=4`, `num_frames=4`,
`160x288`, `fp16`, and `--skip-video-export`. Each item produced 64 records:
16 cross-attention modules over 4 denoising steps.

Mean text-conditioned attention mass:

```text
fluid_impact_pebble_pond_002:
  target=0.00368, footprint=0.00304, comparison=0.00451,
  chain/comparison=1.49

v2_fracture_damage_black_hockey_puck...:
  target=0.00493, footprint=0.00230, comparison=0.00417,
  chain/comparison=1.73

elastic_deformation_soccer_net_001:
  target=0.00415, footprint=0.00278, comparison=0.00475,
  chain/comparison=1.46
```

**Decision:** Method B passes the instrumentation gate and shows a weak but
consistent chain-token signal: target plus footprint mass exceeds matched
comparison-token mass in all three diagnostic items. This is not a repair or
editing result. It justifies one next intervention pass: attention masking or
reweighting on target+footprint tokens, with matched random-token and
matched layer/head controls. Do not claim causal repair until masking beats
those controls on generated-video evaluation.

## 2026-07-02: Method B2 Attention Mask Intervention

**Implementation:** Extended the ZeroScope attention probe into an
intervention runner. The processor can now multiplicatively suppress selected
cross-attention token columns, renormalize attention rows, and apply the
intervention only to the text-conditioned CFG half. Conditions currently
include `baseline`, `target_mask`, `footprint_mask`, `chain_mask`,
`comparison_token_mask`, and `random_token_mask`.

Artifacts:

```text
docs/superpowers/specs/2026-07-02-causal-attention-mask-b2-design.md
docs/superpowers/plans/2026-07-02-causal-attention-mask-b2.md
scripts/adapters/run_zeroscope_attention_probe.py
tests/test_run_zeroscope_attention_probe.py
experiments/method_probe/zeroscope_attention_mask_b2_20260702_dryrun/generation_manifest.json
experiments/method_probe/zeroscope_attention_mask_b2_20260702_smoke/
experiments/method_probe/zeroscope_attention_mask_b2_20260702_compact3/
experiments/evaluation/zeroscope_attention_mask_b2_compact3_fable_20260702/
```

Focused tests pass:

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  -m pytest tests/test_run_zeroscope_attention_probe.py -q

14 passed
```

**Dry run:** Three items over six conditions produced 18 manifest rows. Token
counts are matched for `chain_mask`, `comparison_token_mask`, and
`random_token_mask`; the fracture case masks 11 tokens in each matched-control
condition.

**Smoke:** One-item real smoke over `baseline`, `chain_mask`, and
`random_token_mask` completed with MP4s and attention traces. All videos are
decodable with 4 frames at 288x160. `chain_mask` drove both target and footprint
attention mass to zero while `random_token_mask` left chain-token mass nonzero.

**Compact 3-item matrix:** Ran three items over six B2 conditions with
`steps=4`, `num_frames=4`, `height=160`, `width=288`, and `mask_scale=0.0`.
All 18 videos are decodable and each row produced 64 cross-attention records.
Attention sanity checks passed:

```text
target_mask: target mass = 0 for all items
footprint_mask: footprint mass = 0 for all items
chain_mask: target mass = 0 and footprint mass = 0 for all items
random_token_mask: target/footprint masses remain nonzero
```

**Fable evaluation:** The first fable run used the proxy URL without `/v1` and
returned HTTP 403 fallbacks. Retried with the corrected `/v1` base URL and got
18 valid responses from `claude-fable-5`.

Label summary:

```text
baseline:              other_failure=2, strict_causal_footprint_leakage=1
target_mask:           other_failure=1, erased_clean=1, strict_causal_footprint_leakage=1
footprint_mask:        erased_clean=1, strict_causal_footprint_leakage=1, other_failure=1
chain_mask:            erased_clean=2, other_failure=1
comparison_token_mask: erased_clean=1, strict_causal_footprint_leakage=1, other_failure=1
random_token_mask:     other_failure=2, strict_causal_footprint_leakage=1
```

**Decision:** B2 hard attention masking works mechanically, but this compact
run is not a method result. The compact generation setting is too weak:
baseline is already judged `other_failure` in two of three items. The next B2
variant should keep the matched controls but use higher-quality generation and
a softer intervention, such as `mask_scale=0.25`, before making any claim about
chain-specific repair.

## 2026-07-02: Method B2 Soft Mask Quality Check

**Implementation note:** The attention recorder was optimized before this run.
The old summary path converted selected attention columns to Python lists,
which made quality-resolution runs extremely slow. The recorder now computes
the selected-column mean with tensor operations before calling `.item()`.

Focused tests pass:

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  -m pytest tests/test_run_zeroscope_attention_probe.py -q

15 passed
```

Artifacts:

```text
experiments/method_probe/zeroscope_attention_mask_b2_soft025_quality3_fast_20260702/
experiments/evaluation/zeroscope_attention_mask_b2_soft025_quality3_fable_20260702/
```

**Run:** Re-ran the first three MVP-0 items at higher quality than the compact
matrix: `steps=10`, `num_frames=16`, `height=240`, `width=432`, `fps=8`,
`guidance_scale=9.0`, `dtype=fp16`, and `mask_scale=0.25`. Conditions were
`baseline`, `chain_mask`, `comparison_token_mask`, and `random_token_mask`.
All 12 MP4s are decodable with 16 frames at 432x240. Each condition produced
160 attention-summary records.

**Attention sanity:** The intervention behaved mechanically as intended.
`chain_mask` reduced target and footprint attention mass to roughly one
quarter of baseline, while `comparison_token_mask` mainly reduced comparison
token mass and `random_token_mask` left target/footprint mass nonzero.

```text
fluid_impact_pebble_pond_002:
  baseline:              target=0.003476, footprint=0.002438, comparison=0.002959
  chain_mask:            target=0.000871, footprint=0.000609, comparison=0.002837
  comparison_token_mask: target=0.003498, footprint=0.002543, comparison=0.001720
  random_token_mask:     target=0.003781, footprint=0.002591, comparison=0.002122

v2_fracture_damage_black_hockey_puck...:
  baseline:              target=0.003771, footprint=0.001814, comparison=0.002554
  chain_mask:            target=0.001006, footprint=0.000480, comparison=0.002619
  comparison_token_mask: target=0.003896, footprint=0.001884, comparison=0.001183
  random_token_mask:     target=0.003879, footprint=0.001865, comparison=0.002148

elastic_deformation_soccer_net_001:
  baseline:              target=0.003098, footprint=0.002010, comparison=0.003321
  chain_mask:            target=0.000813, footprint=0.000525, comparison=0.003401
  comparison_token_mask: target=0.003215, footprint=0.002112, comparison=0.002218
  random_token_mask:     target=0.003278, footprint=0.002107, comparison=0.002269
```

**Fable evaluation:** The evaluator produced 12 valid predictions from
`claude-fable-5`.

Label summary:

```text
baseline:              target_leakage=1, strict_causal_footprint_leakage=1, erased_clean=1
chain_mask:            target_leakage=1, strict_causal_footprint_leakage=1, erased_clean=1
comparison_token_mask: target_leakage=1, strict_causal_footprint_leakage=1, erased_clean=1
random_token_mask:     strict_causal_footprint_leakage=3
```

Pair-level pattern:

```text
fluid_impact_pebble_pond_002:
  baseline / chain / comparison = target_leakage
  random = strict_causal_footprint_leakage

v2_fracture_damage_black_hockey_puck...:
  all four conditions = strict_causal_footprint_leakage

elastic_deformation_soccer_net_001:
  baseline / chain / comparison = erased_clean
  random = strict_causal_footprint_leakage
```

**Decision:** B2-soft is a useful negative result, not a repair result. The
white-box intervention is real, because chain-token attention is suppressed
while matched controls preserve it. However, the semantic outcome is not
chain-specific: `chain_mask` ties `baseline` and `comparison_token_mask`, and
`random_token_mask` even preserves the causal footprint in all three items.
Therefore simple cross-attention token-column reweighting is not sufficient as
the core method. The next method should move from "mask chain tokens" to a
stronger counterfactual or verifier-guided intervention: paired seed/noise
controls, explicit target-removal and footprint-removal objectives, and a
semantic verifier that rejects target leakage and footprint leakage separately.

## 2026-07-03: Method C0 Counterfactual Grid Pilot

**Fable pre-review:** Before implementing C0, `claude-fable-5` reviewed the
method shape. The critique is saved in
`docs/fable_c_method_review_2026-07-03.md`. The main concerns were VLM
circularity, same-seed comparisons not guaranteeing semantic control,
counterfactual prompt noise, weak negative controls, and verifier-guided search
turning into prompt hacking. The implemented C0 pilot therefore treats the
method as a counterfactual controllability audit, not as a repair claim.

**Implementation:** Added a four-cell counterfactual grid runner and review
builder:

```text
scripts/adapters/run_c0_counterfactual_grid.py
scripts/build_c0_counterfactual_review.py
```

For each item, the runner generates `original`, `remove_target`,
`footprint_only`, and `target_only` with the same seed. The expected target /
footprint states are:

```text
original:       yes / yes
remove_target:  no / no
footprint_only: no / yes
target_only:    yes / no
```

Focused and regression tests pass:

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c0_counterfactual_review.py \
  tests/test_run_zeroscope_attention_probe.py \
  tests/test_build_mvp0_causal_chain_probe.py \
  tests/test_build_mvp0_phase_b_paraphrase_probe.py \
  tests/test_run_mvp0_zeroscope_probe.py \
  tests/test_run_mvp0_zeroscope_sweep.py -q

51 passed
```

Artifacts:

```text
experiments/method_probe/c0_counterfactual_grid_20260703_dryrun/
experiments/method_probe/c0_counterfactual_grid_20260703_quality3/
experiments/evaluation/c0_counterfactual_grid_quality3_fable_20260703/
```

**Run:** Generated the first three MVP-0 items with `steps=10`,
`num_frames=16`, `height=240`, `width=432`, `fps=8`, `guidance_scale=9.0`,
`dtype=fp16`, and GPU 0. The dry run expanded 3 source items into 12 variant
items with a balanced four-cell grid. The real run produced 12 MP4s, and the
review builder produced 12 VLM input rows.

**Fable evaluation:** `claude-fable-5` returned 12 valid predictions. The old
erasure-oriented labels are not direct success/failure labels for C0, because
`target_only` intentionally keeps the target visible. Interpreted against the
C0 expected target/footprint states, the pass counts were:

```text
original:       1/3
remove_target:  1/3
footprint_only: 2/3
target_only:    2/3
```

Pair-level pattern:

```text
fluid_impact_pebble_pond_002:
  original misses the pebble but keeps ripples.
  remove_target is clean.
  footprint_only keeps ripples without pebble.
  target_only removes both, so target retention fails.

v2_fracture_damage_black_hockey_puck...:
  original succeeds.
  remove_target keeps the puck, so target removal fails.
  footprint_only gives cracks without puck.
  target_only gives puck without cracks.

elastic_deformation_soccer_net_001:
  original keeps the ball but does not generate net deformation.
  remove_target and footprint_only still show the ball.
  target_only gives ball without deformation.
```

**Decision:** C0 is a useful audit protocol and a better experimental direction
than B2 masking, but the first pilot is not a complete method. It demonstrates
that some chains can be separated by prompt-level counterfactuals, especially
the hockey-puck crack case, while other items expose base-model controllability
failures. The next C step should add a C0 scorer that selects items whose
`original` cell is valid before testing repair, then run verifier-guided prompt
search only within the four expected state constraints. This addresses fable's
critique by making prompt generation accountable to explicit target/footprint
state checks rather than relying on a single edited prompt.

### 2026-07-03 C0 scorer

**Implementation:** Added `scripts/score_c0_counterfactual_grid.py` with
variant-level and item-level scoring for the four C0 expected states. The scorer
writes:

```text
experiments/evaluation/c0_counterfactual_grid_quality3_fable_20260703/c0_scores/c0_variant_scores.csv
experiments/evaluation/c0_counterfactual_grid_quality3_fable_20260703/c0_scores/c0_item_scores.csv
experiments/evaluation/c0_counterfactual_grid_quality3_fable_20260703/c0_scores/c0_valid_originals.csv
experiments/evaluation/c0_counterfactual_grid_quality3_fable_20260703/c0_scores/c0_summary.json
```

**Pilot score:**

```text
total items:              3
original valid:           1/3
counterfactual pass:      0/3
full C0 grid pass:        0/3

variant pass counts:
  original:       1/3
  remove_target:  1/3
  footprint_only: 2/3
  target_only:    2/3

failure modes:
  invalid_original:      2
  failed:remove_target:  1
```

**Interpretation:** The scorer makes the next experiment clear. We should not
scale by running full four-cell grids on arbitrary items. First run a
base-validity screen to find items where `original` reliably contains both the
target and the footprint. Only those items should enter the full C0 grid and any
verifier-guided prompt search. In this pilot, the hockey-puck crack item is the
only valid starting point; it then fails specifically because `remove_target`
keeps the puck.

**Tests:**

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c0_counterfactual_grid.py \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c0_counterfactual_review.py -q

8 passed
```

### 2026-07-03 C0 base-validity screening setup

**Implementation:** Added original-only screening support to
`scripts/adapters/run_c0_counterfactual_grid.py` via `--variant-set original`.
This preserves the default four-cell grid, but lets us generate only the base
`original` cell for a larger candidate set. The scorer now also writes
`c0_valid_originals.csv`, which is the handoff list for full-grid follow-up.

**Dry-run screen:** Built an original-only screening manifest for all 12 current
MVP-0 candidates:

```text
experiments/method_probe/c0_base_validity_screen_20260703_dryrun/generation_manifest.json
experiments/evaluation/c0_base_validity_screen_20260703_dryrun/review.csv
```

The manifest contains 12 generation rows and `variant_grid=["original"]`.

**Pilot valid-original export:** Re-running the scorer on the existing 3-item
pilot exports one valid original item:

```text
v2_fracture_damage_black_hockey_puck_a_star_shaped_crack_spreads_across_t_side_033
```

**Next real screening command:**

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/c0_base_validity_screen_20260703_real_s10_f16_240x432 \
  --seed 34000 \
  --variant-set original \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --guidance-scale 9.0 \
  --dtype fp16 \
  --device cuda:0
```

Then build review rows, run fable, and score:

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/build_c0_counterfactual_review.py \
  --generation-manifest experiments/method_probe/c0_base_validity_screen_20260703_real_s10_f16_240x432/generation_manifest.json \
  --output-dir experiments/evaluation/c0_base_validity_screen_20260703_real_s10_f16_240x432_fable

PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/score_c0_counterfactual_grid.py \
  --predictions-csv experiments/evaluation/c0_base_validity_screen_20260703_real_s10_f16_240x432_fable/fable_run/vlm_predictions.csv \
  --output-dir experiments/evaluation/c0_base_validity_screen_20260703_real_s10_f16_240x432_fable/c0_scores
```

**Tests:**

```text
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python -m pytest \
  tests/test_score_c0_counterfactual_grid.py \
  tests/test_run_c0_counterfactual_grid.py \
  tests/test_build_c0_counterfactual_review.py -q

10 passed
```

### 2026-07-03 C0.1 real 60-video pilot generation

**Run:** Generated the real C0.1 seed-matched factorial gate pilot on physical
GPU 4 using `CUDA_VISIBLE_DEVICES=4`, ZeroScope, `steps=10`, `num_frames=16`,
`240x432`, `fp16`, model CPU offload, and VAE slicing.

```text
experiments/method_probe/c01_factorial_gate_20260703_real_s10_f16_240x432/
```

The run completed successfully and wrote:

```text
experiments/method_probe/c01_factorial_gate_20260703_real_s10_f16_240x432/generation_manifest.json
```

The manifest contains 60 planned/generated rows:

```text
3 items x 5 seeds x 4 cells = 60 videos
```

The four cells are balanced with 15 rows each for `original`,
`remove_target`, `footprint_only`, and `target_only`. All 60 referenced local
video files exist in the ignored `videos/` directory.

**Blinded review package:** Built the human review package with five-frame
strips:

```text
experiments/evaluation/c01_factorial_gate_20260703_real_s10_f16_240x432/blind_review.csv
experiments/evaluation/c01_factorial_gate_20260703_real_s10_f16_240x432/answer_key.csv
experiments/evaluation/c01_factorial_gate_20260703_real_s10_f16_240x432/review_manifest.json
experiments/evaluation/c01_factorial_gate_20260703_real_s10_f16_240x432/frame_strips/
```

Integrity checks:

```text
review_rows=60
answer_key_rows=60
frame_strip_count=60
variant_label_leaks_in_blind_review=0
nonempty_video_path_in_blind_review=0
review_ids_unique=true
review_and_key_ids_match=true
missing_frame_strips=0
```

**Next gate:** Do not score this as C0.1 yet. Per the fable method review, the
next step is a human spot-check of 10 to 15 generated rows before bulk review,
then reviewer labeling with `uncertain` tracking and at least 10 overlapping
rows for inter-rater agreement.

### 2026-07-04 C0.2 discrete factorial gate dry run

**Run:** Added a C0.2 prompt-template mode to the C0 runner and validated the
planned discrete-item pilot as a dry run.

```text
experiments/method_probe/c02_discrete_factorial_gate_20260704_dryrun/generation_manifest.json
```

Configuration:

```text
item_indices=3,4,8,10
prompt_template=c02_discrete
seed=52000
seeds_per_item=3
variant_grid=original,remove_target,footprint_only,target_only
```

Integrity checks:

```text
rows=48
probe_indices=3,4,8,10
variant_counts=12 each
seed_indices=0,1,2
dry_run=true
prompt_template_recorded=true
```

The C0.2 template uses item-specific surface and footprint phrase overrides so
each cell explicitly states target presence/absence and footprint
presence/absence. This is only a generation-validity pilot.

Real run command, completed on physical GPU 7:

```text
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/deepseek_VG/.conda/envs/vcecf/bin/python \
  scripts/adapters/run_c0_counterfactual_grid.py \
  --probe-manifest experiments/method_probe/zeroscope_mvp0_causal_chain_probe_20260702/probe_manifest.json \
  --output-dir experiments/method_probe/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432 \
  --item-indices 3,4,8,10 \
  --prompt-template c02_discrete \
  --seed 52000 \
  --seeds-per-item 3 \
  --steps 10 \
  --num-frames 16 \
  --height 240 \
  --width 432 \
  --guidance-scale 9.0 \
  --dtype fp16 \
  --device cuda:0 \
  --enable-model-cpu-offload \
  --vae-slicing
```

Generated manifest:

```text
experiments/method_probe/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/generation_manifest.json
```

Generation integrity checks:

```text
rows=48
videos=48
probe_indices=3,4,8,10
variant_counts=12 each
seed_indices=0,1,2
prompt_template=c02_discrete
dry_run=false
```

Blinded review and spot-check package:

```text
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/blind_review.csv
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/answer_key.csv
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/review_manifest.json
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/frame_strips/
experiments/evaluation/c02_discrete_factorial_gate_20260704_real_s10_f16_240x432/spotcheck_contact_sheets/
```

Review package integrity checks:

```text
blind_rows=48
answer_key_rows=48
ids_match=true
frame_strips=48
blind_variant_leaks=0
blind_video_path_empty=true
missing_frame_strips=0
spotcheck_item_count=4
spotcheck_missing_strip_count=0
spotcheck_sheets=4
```

Non-blind visual spot-check summary:

```text
item_3 makeup brush / pink powder cloud: weak; target and footprint collapse
  into general pink material/style.
item_4 garden rake / soil grooves: strongest; grooves are visible and discrete,
  though footprint-only sometimes grows tool-like structures.
item_8 hand / pillow dent: weak; hands and pillow texture drift, and the dent is
  not isolated cleanly.
item_10 marker / black whiteboard line: mixed; many cells contain general
  sketch/diagram lines, so target and footprint are not yet separated.
```

Interpretation: C0.2 succeeded mechanically and gave a useful validity gate,
but it does not yet justify a causal-method claim. The only promising cell
family is item 4, with item 10 as a possible narrower follow-up if the prompt is
made much simpler. Items 3 and 8 should be treated as negative evidence for
this generator/prompt regime. The next method step should either manually score
the blind rows for auditability or run a C0.3 narrowed gate over item 4, and
possibly a simplified item 10, before writing any stronger claim.

Fable method-advisor critique after the run: the conservative interpretation is
right, but the next step must avoid post-hoc cherry-picking. If we narrow only
to item 4 after seeing the sheets, the result is a demo unless the paper defines
an explicit target regime, such as low-entanglement rigid-object surface traces,
or pre-registers a larger item set and reports the full success rate. The main
attack points are selection bias, lack of ground-truth causal structure, lack of
quantitative counterfactual metrics, and the fact that entanglement failures in
items 3 and 8 are substantive negative evidence rather than small noise. Before
another generation pass, record success criteria for the gate and decide between
two honest paths: expand to a larger diverse item set, or make the method scope
narrow and test that scope directly. A stronger publishable version likely
needs a synthetic or controlled benchmark with known masks/causal structure plus
a re-insertion or consistency test, not only visual plausibility.
