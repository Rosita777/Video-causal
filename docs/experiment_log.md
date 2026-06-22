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
