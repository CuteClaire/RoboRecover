# Evaluating on LIBERO

[Back to README](../README.md)

This guide evaluates the fixed RoboRecover LIBERO starting points, not the upstream clean-start LIBERO episodes. Complete the README's data validation and overlay installation first. Commands assume absolute `ROBORECOVER_ROOT`, `ROBORECOVER_DATA`, `ROBORECOVER_RUNS`, and `PROJECTS` variables in every terminal.

## 1. Common simulator preparation

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in each process that runs simulation. Keep the model environment separate when using a server. Use MuJoCo **3.3.2**, robosuite **1.4.1**, and a compatible Python environment (3.10+). Pin these after installing the original model requirements and check for resolver conflicts:

```bash
python -m pip install 'mujoco==3.3.2' 'robosuite==1.4.1'
python -m pip check
export MUJOCO_GL=egl
python -c "import mujoco; from libero.libero import benchmark; print(mujoco.__version__); print(benchmark.get_benchmark_dict().keys()); assert mujoco.__version__ == '3.3.2'"
```

The model-specific installation references are in the [README table](../README.md#original-models-and-weights). MuJoCo's target version is a RoboRecover runtime requirement, not a statement that every upstream policy example uses the same version. In particular, do not copy OpenPI's Python 3.8 simulation lockfile unchanged.

Check LIBERO's configured asset, BDDL and init-state paths before starting a run. Switching between editable clones can leave stale paths in LIBERO's user configuration. Run configuration interactively once. No demonstration dataset download is required merely to replay starting points, but simulator assets and initial-state files are required.

### Input list and one-scenario smoke test

```bash
export LIBERO_LIST="$ROBORECOVER_RUNS/libero_test.txt"
head -n 1 "$LIBERO_LIST" > "$ROBORECOVER_RUNS/libero_smoke.txt"
export LIBERO_INPUT="$ROBORECOVER_RUNS/libero_smoke.txt"
export RUN_ID=smoke_r1
```

All commands below use `LIBERO_INPUT`. After a successful smoke test, switch to:

```bash
export LIBERO_INPUT="$LIBERO_LIST"
export RUN_ID=test_r1
```

Keep the `--filtered-root` and the list paths consistent. OpenPI, Being-H and UniFOLM reject samples outside the declared root. Do not add `--skip-state-mismatch-check` to make a failed smoke test proceed.

## 2. Pi0 and Pi0.5 (OpenPI)

**Original setup:** [OpenPI](https://github.com/Physical-Intelligence/openpi#installation) and its [LIBERO client/server guide](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero). Install the `openpi` overlay into the pinned checkout.

The policy server uses OpenPI's model environment. The evaluator uses a LIBERO environment with `openpi-client`, Pillow, ImageIO and the usual LIBERO dependencies. Install the client package from the same checkout:

```bash
export OPENPI_ROOT="$PROJECTS/openpi"
# In the pinned OpenPI checkout, set up the original model environment:
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
# In the LIBERO evaluation environment:
python -m pip install -e "$OPENPI_ROOT/packages/openpi-client"
```

### Terminal A: model server

From the OpenPI model environment, specify the model config and a compatible checkpoint. The official Pi0.5 LIBERO checkpoint is `gs://openpi-assets/checkpoints/pi05_libero/`; use a locally downloaded checkpoint path if needed.

```bash
cd "$OPENPI_ROOT"
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir gs://openpi-assets/checkpoints/pi05_libero/
```

For Pi0, use `--policy.config pi0_libero` with its matching checkpoint from the [OpenPI model table](https://github.com/Physical-Intelligence/openpi#pre-trained-models), for example `--policy.dir /absolute/path/to/pi0_libero`. Do not change only the model tag while continuing to serve Pi0.5.

### Terminal B: replay and evaluate

Activate the LIBERO environment, export the common variables and run:

```bash
cd "$OPENPI_ROOT"
export MUJOCO_GL=egl
python examples/libero/eval_filtered_repro.py \
  --filtered-root "$ROBORECOVER_DATA/libero/test" \
  --sample-file-list "$LIBERO_INPUT" \
  --host 127.0.0.1 --port 8000 --model-tag pi05 \
  --replan-steps 5 --num-steps-wait-after-replay 0 \
  --max-infer-steps -1 --no-save-video \
  --output-dir "$ROBORECOVER_RUNS/libero/pi05" --run-name "$RUN_ID"
```

For Pi0, connect to its server and change `--model-tag pi0` and the output directory. `--max-infer-steps -1` selects the suite-specific fresh inference budget. Optional RLDS recording is not required for evaluation and has additional dependencies; leave `--save-rlds` off.

## 3. Being-H0.5

Use the [pinned original environment instructions](https://github.com/BeingBeyond/Being-H/tree/594219eaa16097b86d460b3e1040cca0fc4e4b98) and [inference documentation](https://github.com/BeingBeyond/Being-H/blob/594219eaa16097b86d460b3e1040cca0fc4e4b98/docs/inference.md). The collected code uses Python 3.10 and the `BeingH` package. Install its requirements and FlashAttention as directed by the upstream project, then the `Being-H` overlay. Obtain [Being-H05-2B_libero](https://huggingface.co/BeingBeyond/Being-H05-2B_libero), not the generic pretraining checkpoint.

```bash
export BEINGH_ROOT="$PROJECTS/Being-H"
export BEINGH_CKPT=/absolute/path/to/Being-H05-2B_libero
```

For a new model environment, from the pinned checkout:

```bash
conda create -n beingh python=3.10 -y
conda activate beingh
cd "$BEINGH_ROOT"
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

If using this environment for simulation too, install LIBERO and apply the version checks in section 1 before proceeding. FlashAttention must match your CUDA/compiler/PyTorch combination; consult the original installation notes if compilation fails.

### Terminal A: Being-H server

```bash
cd "$BEINGH_ROOT"
python -m BeingH.inference.run_server_vla \
  --model-path "$BEINGH_CKPT" --port 18880 \
  --data-config-name libero_nonorm --dataset-name libero_posttrain \
  --embodiment-tag libero --seed 42 --prompt-template long \
  --max-view-num -1 --no-use-fixed-view --no-enable-rtc
```

### Terminal B: evaluator

Use an environment containing LIBERO and the same Being-H client package. These server-side normalization settings and client action interpretation must stay paired:

```bash
cd "$BEINGH_ROOT"
export PYTHONPATH="$BEINGH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
python -m BeingH.benchmark.libero.eval_filtered_repro_beingh \
  --filtered-root "$ROBORECOVER_DATA/libero/test" \
  --sample-file-list "$LIBERO_INPUT" \
  --host 127.0.0.1 --port 18880 \
  --action-type world_delta --data-config-name libero --exec-chunk-size 8 \
  --num-steps-wait-after-replay 0 --max-infer-steps -1 --no-save-video \
  --output-dir "$ROBORECOVER_RUNS/libero/beingh" --run-name "$RUN_ID"
```

Current upstream Being-H `main` has reorganized directories. Use the pinned revision instead of rewriting imports to match an unrelated version.

## 4. UniFOLM-VLA

Follow the [original installation and LIBERO instructions](https://github.com/unitreerobotics/unifolm-vla). The collected project's instructions use Python 3.10.18, CUDA 12.4, its specified LeRobot revision, and FlashAttention 2.5.6. Install its editable package and `experiments/LIBERO/libero_requirements.txt`, then recheck the common simulator versions. Do not substitute another model's LeRobot installation.

Install the `unifolm-vla` overlay and download both [LIBERO VLA weights](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero) and the [VLM backbone](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base). This evaluator loads the model directly; there is no separate server.

Original model-environment installation, from the pinned checkout:

```bash
conda create -n unifolm-vla python=3.10.18 -y
conda activate unifolm-vla
cd "$PROJECTS/unifolm-vla"
python -m pip install --no-deps 'lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68'
python -m pip install -e .
python -m pip install 'flash-attn==2.5.6' --no-build-isolation
python -m pip install -r experiments/LIBERO/libero_requirements.txt
# Install LIBERO editable from your chosen original checkout, then enforce section 1.
```

```bash
export UNIFOLM_ROOT="$PROJECTS/unifolm-vla"
export UNIFOLM_CKPT=/absolute/path/to/Unifolm-VLA-Libero
export UNIFOLM_VLM=/absolute/path/to/Unifolm-VLM-Base
cd "$UNIFOLM_ROOT"
export PYTHONPATH="$UNIFOLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
python -m experiments.LIBERO.eval_filtered_repro_unifolm \
  --pretrained-path "$UNIFOLM_CKPT" --vlm-pretrained-path "$UNIFOLM_VLM" \
  --filtered-root "$ROBORECOVER_DATA/libero/test" \
  --sample-file-list "$LIBERO_INPUT" \
  --resize-size 224 224 --window-size 2 \
  --num-steps-wait-after-replay 0 --max-infer-steps -1 --no-save-video \
  --output-dir "$ROBORECOVER_RUNS/libero/unifolm" --run-name "$RUN_ID"
```

Keep checkpoint configuration and action normalization files alongside the weights. A standalone parameter file is not a complete model package.

## 5. FastWAM

Follow [FastWAM environment setup and model preparation](https://github.com/yuantianyuan01/FastWAM). The collected upstream recipe uses Python 3.10 and PyTorch 2.7.1 with CUDA 12.8. Install the project editable, prepare its Wan/ActionDiT resources as described upstream, and install LIBERO with MuJoCo 3.3.2. Install the `FastWAM` overlay.

From [the official model release](https://huggingface.co/yuanty/fastwam), obtain **both** `libero_uncond_2cam224.pt` and `libero_uncond_2cam224_dataset_stats.json`. A weights-only download is insufficient.

The original model-environment commands are:

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
cd "$PROJECTS/FastWAM"
python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Also complete upstream **Model Preparation**, including the Wan resources/ActionDiT backbone required by the chosen config. Installing the package alone does not provide those resources. Install LIBERO and enforce section 1 in this same environment.

```bash
export FASTWAM_ROOT="$PROJECTS/FastWAM"
export FASTWAM_CKPT=/absolute/path/to/libero_uncond_2cam224.pt
export FASTWAM_STATS=/absolute/path/to/libero_uncond_2cam224_dataset_stats.json
cd "$FASTWAM_ROOT"
export PYTHONPATH="$FASTWAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
python experiments/libero/eval_filtered_repro_fastwam.py \
  --task-choice libero_uncond_2cam224_1e-4 --ckpt "$FASTWAM_CKPT" \
  --sample-file-list "$LIBERO_INPUT" --gpu-id 0 \
  --eval-mode seed_plus_replay --num-steps-wait-after-replay 0 \
  --max-infer-steps -1 --no-save-video \
  --output-dir "$ROBORECOVER_RUNS/libero/fastwam" --run-name "$RUN_ID" \
  EVALUATION.dataset_stats_path="$FASTWAM_STATS"
```

The last argument is a Hydra override, not an argparse flag: do not prepend `--`. `--dry-run` checks configuration/sample selection without inference; it is not a simulator/model smoke test. Use `--limit-samples 1` or the smoke input list for the latter.

## 6. Cosmos-Policy

Use [Cosmos SETUP.md](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md) and [LIBERO.md](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md). Install the `cosmos-policy` overlay into the pinned checkout. The original workflow uses Docker and the `cu128` extra / `libero` dependency group:

```bash
# Inside the prepared Cosmos container, from the Cosmos repository root:
uv sync --extra cu128 --group libero --python 3.10
.venv/bin/python -c "import mujoco; print(mujoco.__version__); assert mujoco.__version__ == '3.3.2'"
```

Mount the Cosmos checkout, RoboRecover, scenario tree and output directory into the container. Regenerate the input list inside the container so all paths are valid there. Export the common variables using container paths. If correcting a dependency after `uv sync`, run `.venv/bin/python` as below to avoid an automatic re-sync undoing the correction.

Download [Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B), including its checkpoint, `libero_dataset_statistics.json` and `libero_t5_embeddings.pkl`. The cached text embeddings must cover the instructions being evaluated. For explicit offline operation, use local paths:

```bash
export COSMOS_ROOT=/absolute/container/path/to/cosmos-policy
export COSMOS_CKPT=/absolute/container/path/to/checkpoint.pt
export COSMOS_STATS=/absolute/container/path/to/libero_dataset_statistics.json
export COSMOS_TEXT=/absolute/container/path/to/libero_t5_embeddings.pkl
cd "$COSMOS_ROOT"
export MUJOCO_GL=egl
.venv/bin/python -m cosmos_policy.experiments.robot.libero.run_libero_filtered_repro \
  --config cosmos_predict2_2b_480p_libero__inference_only \
  --ckpt_path "$COSMOS_CKPT" --config_file cosmos_policy/config/config.py \
  --dataset_stats_path "$COSMOS_STATS" --t5_text_embeddings_path "$COSMOS_TEXT" \
  --sample_file_list "$LIBERO_INPUT" --filtered_roots "$ROBORECOVER_DATA/libero/test" \
  --trials_per_sample 1 --num_workers 1 --worker_id 0 \
  --eval_mode seed_plus_replay_then_infer --num_steps_wait_after_replay 0 \
  --manifest_max_infer_steps -1 --manifest_save_video False \
  --use_wrist_image True --use_proprio True \
  --normalize_proprio True --unnormalize_actions True \
  --chunk_size 16 --num_open_loop_steps 16 \
  --use_jpeg_compression True --flip_images True \
  --num_denoising_steps_action 5 --ar_future_prediction False --ar_value_prediction False \
  --local_log_dir "$ROBORECOVER_RUNS/libero/cosmos" --manifest_run_name "$RUN_ID"
```

Cosmos uses **underscore-separated** CLI names and explicit boolean values. Do not replace these with the hyphenated options used by other adapters. Use the `.pt` file or a directory with exactly one `.pt`; a directory with several candidate weights is ambiguous.

## 7. Repeats, outputs and troubleshooting

Each command above schedules one trial per listed scenario, except when explicitly changing Cosmos's `--trials_per_sample`. For independent repeats of the other adapters, rerun with distinct `RUN_ID` values (`test_r1`, `test_r2`, ...), recording server/model random seeds. Do not duplicate list lines: some loaders deduplicate them. Do not reuse a completed output directory, because resume logic may skip those samples.

Retain all per-trial results; a one-scenario smoke run is not part of the full score. The standard fresh inference budgets used by these LIBERO adapters are 220/280/300/520 steps for Spatial/Object/Goal/10 respectively. Replaying the prefix is separate. Report deliberate changes to budgets or action-chunk size.

| Symptom | Check |
|---|---|
| `single_arm_env` import error | robosuite 1.5+ was installed; use the compatible 1.4.x environment. |
| `state_mismatch` | MuJoCo version, correct LIBERO clone/assets/init states, stored seed and episode. Keep the guard enabled. |
| EGL initialization error | Graphics driver/device visibility, headless EGL support and container GPU graphics configuration. |
| List paths missing | Regenerate with `prepare_data.py` after relocation or inside Docker. |
| Connection refused / protocol error | Correct model environment, server port, checkpoint loaded, matching client protocol. |
| Actions look wrong but requests succeed | Wrong embodiment checkpoint, statistics, action representation, or paired server/client normalization settings. |
| No new trials in output | Reused run directory/resume behavior; use a fresh run name. |

These commands have been checked against the collected evaluator interfaces. This does not replace a full clean-environment GPU inference validation for each model.
