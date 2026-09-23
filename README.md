# RoboRecover

**Evaluating robot policy recovery under execution deviations.**

RoboRecover provides recovery starting points and evaluation adapters for **LIBERO** and **RoboTwin 2.0**. Each scenario stores an initialization, an action prefix and an instruction. The evaluator reconstructs the deviation by replaying the prefix, hands control to your policy, and checks whether it completes the original task.

## What is included?

| Benchmark | Train / test starting points | Policy adapters |
|---|---|---|
| LIBERO | 800 / 200 | Pi0, Pi0.5, Being-H0.5, UniFOLM-VLA, FastWAM, Cosmos-Policy |
| RoboTwin 2.0 | 800 / 200 | Pi0.5, X-VLA, LingBot-VLA, SmolVLA; FastWAM and LingBot-VA configuration documented, recovery adapters pending |

- Fixed splits with per-file SHA256 checksums.
- Replay-then-inference evaluators using each policy's original implementation.
- Tools for installing evaluation extensions and validating data.
- One example JSON per benchmark.

**Dataset download:** the complete scenario archive is being prepared for Hugging Face. Until its link is published, full evaluation requires the exported archive from the authors. Model weights and simulator assets are obtained separately. RoboTwin FastWAM and LingBot-VA environment/configuration guidance is included, but their RoboRecover recovery entrypoints will be added later. The train split contains starting points, not complete policy-training demonstrations.

## Quick navigation

1. [Environment setup](#1-environment-setup)
2. [Install the evaluation extensions](#2-install-the-evaluation-extensions)
3. [Prepare the data](#3-prepare-the-data)
4. [LIBERO: all six policies](docs/LIBERO.md)
5. [RoboTwin: four adapters + two pending configurations](docs/ROBOTWIN.md)
6. [Results and protocol](#5-results-and-protocol)

## 1. Environment setup

Install the **original benchmark and model environments first**, then add RoboRecover. There is no single environment for every model: Python, PyTorch, CUDA and Transformers requirements differ. For client/server policies, run inference in the model environment and simulation in a separate benchmark environment. UniFOLM, FastWAM and Cosmos LIBERO adapters load the model within the evaluation process.

### Original benchmarks

| Benchmark | Source | Environment and assets |
|---|---|---|
| LIBERO | [Official repository](https://github.com/Lifelong-Robot-Learning/LIBERO) | [Installation](https://github.com/Lifelong-Robot-Learning/LIBERO#installation), [OpenPI LIBERO setup](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero) |
| RoboTwin 2.0 | [Official repository](https://github.com/RoboTwin-Platform/RoboTwin) | [Install and download](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), [usage guide](https://robotwin-platform.github.io/doc/usage/index.html) |

### Original models and weights

| Policy | Original environment / deployment | Weights |
|---|---|---|
| Pi0 / Pi0.5 | [OpenPI installation](https://github.com/Physical-Intelligence/openpi#installation), [LIBERO setup](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero) | [OpenPI model table](https://github.com/Physical-Intelligence/openpi#pre-trained-models); LIBERO uses `pi0_libero` / `pi05_libero`, not RoboTwin |
| Being-H0.5 | [Pinned README](https://github.com/BeingBeyond/Being-H/tree/594219eaa16097b86d460b3e1040cca0fc4e4b98), [inference guide](https://github.com/BeingBeyond/Being-H/blob/594219eaa16097b86d460b3e1040cca0fc4e4b98/docs/inference.md) | [Being-H05-2B_libero](https://huggingface.co/BeingBeyond/Being-H05-2B_libero) |
| UniFOLM-VLA | [Installation and LIBERO guide](https://github.com/unitreerobotics/unifolm-vla) | [Unifolm-VLA-Libero](https://huggingface.co/unitreerobotics/Unifolm-VLA-Libero), [VLM backbone](https://huggingface.co/unitreerobotics/Unifolm-VLM-Base) |
| FastWAM | [Environment and model preparation](https://github.com/yuantianyuan01/FastWAM#environment-setup) | [Weights and statistics](https://huggingface.co/yuanty/fastwam) |
| Cosmos-Policy | [Environment setup](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md), [LIBERO guide](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md) | [Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) |
| X-VLA | [Installation](https://github.com/2toinf/X-VLA), [RoboTwin deployment](https://github.com/2toinf/X-VLA/tree/main/evaluation/robotwin-2.0) | [X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) |
| LingBot-VLA | [Installation and RoboTwin guide](https://github.com/Robbyant/lingbot-vla) | [RoboTwin post-trained weights](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| SmolVLA | [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla) | [Base model](https://huggingface.co/lerobot/smolvla_base); provide your **RoboTwin-finetuned** checkpoint and processor/statistics files |
| LingBot-VA | [Official repository and RoboTwin setup](https://github.com/robbyant/lingbot-va) | [RoboTwin post-trained weights](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin); alternatively [LeRobot-format checkpoint/config guide](https://huggingface.co/docs/lerobot/main/lingbot_va) |

Follow the checked-out revision's instructions: upstream `main` may have changed its layout. In particular, the Being-H adapter targets the pinned `BeingH/` layout, not the current reorganized repository. Base revisions are in [dependencies.lock.json](dependencies.lock.json).

### Critical simulator settings

**LIBERO target runtime: MuJoCo `3.3.2`.** This matches the collected OpenPI runtime and FastWAM/Cosmos LIBERO configuration. OpenPI's upstream example lockfile pins an older MuJoCo and Python environment; do not use it unchanged for this runtime. After installing the original LIBERO dependencies in a compatible Python environment (3.10+), run:

```bash
python -m pip install 'mujoco==3.3.2' 'robosuite==1.4.1'
python -m pip check
python -c "import mujoco, robosuite; print(mujoco.__version__, robosuite.__version__); assert mujoco.__version__ == '3.3.2'"
python -c "import libero; print(libero.__file__)"
export MUJOCO_GL=egl
```

Keep robosuite on the LIBERO-compatible 1.4.x layout; 1.5 changes interfaces used here. Resolve dependency conflicts rather than ignoring `pip check`. Repeat the version check after installing another model package. For Cosmos use its `.venv/bin/python`; do not mix its RoboCasa and LIBERO dependency groups. Complete LIBERO's first-run asset/init-state path configuration interactively before background evaluation.

**RoboTwin uses SAPIEN, not MuJoCo.** Follow its original simulator and asset installation. The inspected replay environment uses Python 3.10, `sapien==3.0.0b1`, `mplib==0.2.1`. End-effector traces need cuRobo; do not disable it globally because a joint-action example succeeds. See [RoboTwin setup details](docs/ROBOTWIN.md#1-prepare-robotwin).

Use Linux with an NVIDIA GPU and working rendering. `nvidia-smi` does not test EGL/Vulkan. Containers need graphics-device access, not only CUDA compute. Platform-specific troubleshooting is included in both tutorials.

## 2. Install the evaluation extensions

```bash
git clone https://github.com/CuteClaire/RoboRecover.git
cd RoboRecover
export ROBORECOVER_ROOT="$PWD"
export PROJECTS=/absolute/path/to/projects
export ROBORECOVER_RUNS="$ROBORECOVER_ROOT/runs"
mkdir -p "$PROJECTS" "$ROBORECOVER_RUNS"
```

An **overlay** is a collection of evaluator/support files installed into the original project at their native paths. It does not install dependencies or weights. Use separate upstream checkouts instead of modifying a training workspace.

For example:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$PROJECTS/RoboTwin"
git -C "$PROJECTS/RoboTwin" checkout c3ddfa8b97d5519efa828b075999bd0006778e5e
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
python "$ROBORECOVER_ROOT/tools/install_overlay.py" \
  --component RoboTwin --checkout "$ROBOTWIN_ROOT"
# Review the dry run, then install:
python "$ROBORECOVER_ROOT/tools/install_overlay.py" \
  --component RoboTwin --checkout "$ROBOTWIN_ROOT" --apply
```

For a model, clone its URL, check out the revision in `dependencies.lock.json`, and repeat with `--component openpi`, `Being-H`, `unifolm-vla`, `FastWAM`, `cosmos-policy`, `X-VLA`, `lingbot-vla`, or `lerobot`. Names are case-sensitive. LIBERO itself has no overlay.

The installer checks revisions/hashes and backs up replaced files under `.roborecover-backup`. Do not apply these files to arbitrary upstream revisions. Export the absolute path variables above in **every terminal** used below.

## 3. Prepare the data

Extract the scenario archive as follows; filenames must match `splits/*.csv`:

```text
scenarios/
├── libero/{train,test}/      # 800 / 200 JSON files
└── robotwin/{train,test}/     # 800 / 200 JSON files
```

```bash
export ROBORECOVER_DATA=/absolute/path/to/scenarios
python "$ROBORECOVER_ROOT/tools/prepare_data.py" \
  --platform libero --split test --data-root "$ROBORECOVER_DATA" \
  --output "$ROBORECOVER_RUNS/libero_test.txt"
python "$ROBORECOVER_ROOT/tools/prepare_data.py" \
  --platform robotwin --split test --data-root "$ROBORECOVER_DATA" \
  --output "$ROBORECOVER_RUNS/robotwin_test.txt"
```

The tool validates membership, hashes and actions before writing input lists. Regenerate lists after moving data. For Docker, generate them **inside the container**, using mounted paths. `examples/` contains format examples, not a complete split.

## 4. Evaluate a policy

- **[LIBERO tutorial](docs/LIBERO.md):** original environments, checkpoints, server/client commands for Pi0/Pi0.5 and Being-H, in-process commands for UniFOLM/FastWAM/Cosmos, smoke tests and repeated trials.
- **[RoboTwin tutorial](docs/ROBOTWIN.md):** simulator setup, replay check, four runnable adapters, plus environment and checkpoint configuration for pending FastWAM/LingBot-VA adapters.

Start with one scenario. Use our recovery entrypoint, not an upstream clean-start benchmark script. Model ports are not interchangeable: HTTP, WebSocket and custom RPC clients need their matching server.

## 5. Results and protocol

Runs produce `results.jsonl`, `summary.json` and logs in the selected output directory. Cosmos uses `local_log_dir` / `manifest_run_name`; other adapters use `output-dir` / `run-name`. Keep each policy, repeat and protocol in a separate run directory.

Compute successes / scheduled trials within each scenario, then average scenarios equally. Report scenario count, repeat count, invalid trials, checkpoint/configuration and inference budget. A rendering/import/server failure must be investigated; if unresolved, retain it as a failed scheduled trial rather than silently dropping it.

- Preserve the original instruction, initialization and task-success criterion.
- Replay the prefix, then start policy inference with fresh scenario history. Replay does not consume the fresh inference budget.
- Use zero post-replay wait steps. Keep state-mismatch checks enabled.
- Preserve stored action arrays. Some LIBERO records have `len(actions) == annotated_step`; do not pad them.
- `seed_only` evaluates the original initialization and is not the recovery protocol.
- Do not pool all rollouts when scenarios have unequal repeat counts.

## Layout and release status

`docs/` contains the tutorials; `overlays/` the extensions; `splits/` the fixed membership; `tools/` data validation and installation; `code_manifest.json` code hashes; and `third_party_licenses/` upstream notices.

All 2000 local scenarios have passed data checks. The original RoboTwin environment passed 1000 schema/task checks and one real replay. Tutorial commands are checked against collected interfaces; full clean-install inference for every packaged model has not yet been rerun. Full dataset hosting is pending.

Repository-authored code uses [Apache-2.0](LICENSE). Third-party terms, scenario-data terms, simulator assets and model weights are separate; see [NOTICE.md](NOTICE.md). We thank the original benchmark and model authors linked above.
