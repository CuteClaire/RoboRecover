# Evaluating on RoboTwin 2.0

[Back to README](../README.md)

This guide covers the four included recovery adapters: **Pi0.5, X-VLA, LingBot-VLA and SmolVLA**. Complete data preparation and install the `RoboTwin` overlay plus the selected model overlay first. Export `PROJECTS`, `ROBORECOVER_ROOT`, `ROBORECOVER_DATA`, and `ROBORECOVER_RUNS` in each terminal.

## 1. Prepare RoboTwin

Follow the original [Install & Download guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), including the **assets and robot descriptions**, not just Python installation. Use the pinned checkout from `dependencies.lock.json` and apply the RoboTwin overlay afterward. Read the original project's CUDA/cuRobo notes before installing the planner.

The inspected replay environment is Python 3.10, SAPIEN 3.0.0b1 and MPLib 0.2.1. Check the actual interpreter after activating your RoboTwin environment:

```bash
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
cd "$ROBOTWIN_ROOT"
python -c "import sys; from importlib.metadata import version; print(sys.version); print('sapien', version('sapien')); print('mplib', version('mplib'))"
python -m pip check
```

RoboTwin uses SAPIEN rendering/physics, **not MuJoCo**. Its original documentation recommends CUDA 12.1; model environments may use other CUDA builds. Keep simulation and model servers separate rather than replacing the simulator's PyTorch stack to satisfy a policy.

End-effector (`action_type: ee`) traces need cuRobo. Joint-position replay and end-effector replay do not have identical dependencies. Set `ROBOTWIN_CUDA_HOME` to your installed toolkit when needed. Set `TORCH_CUDA_ARCH_LIST` for your GPU rather than relying on an adapter's historical `8.6` default. Compilation errors in the planner must be resolved before evaluation. Do not globally force `ROBOTWIN_SKIP_CUROBO_PLANNER=1` for the entire dataset.

Additional lightweight clients may be needed in the simulation environment: X-VLA uses Requests/SciPy, and LingBot uses its original WebSocket client. The collected LingBot client setup used `websockets==16.0` and `msgpack==1.2.1`; check compatibility with the pinned checkout rather than installing the policy's whole model environment into RoboTwin.

## 2. Check one starting point

Generate the list using the README's `prepare_data.py` command, then:

```bash
export ROBOTWIN_LIST="$ROBORECOVER_RUNS/robotwin_test.txt"
head -n 1 "$ROBOTWIN_LIST" > "$ROBORECOVER_RUNS/robotwin_smoke.txt"
export ROBOTWIN_INPUT="$ROBORECOVER_RUNS/robotwin_smoke.txt"
export RUN_ID=smoke_r1
cd "$ROBOTWIN_ROOT"
python script/replay_robotwin_trace.py \
  --sample-file-list "$ROBOTWIN_LIST" --verify-only
python script/replay_robotwin_trace.py \
  --sample-file-list "$ROBOTWIN_INPUT" \
  --output-dir "$ROBORECOVER_RUNS/robotwin/replay" --run-name "$RUN_ID"
```

`--verify-only` checks schema/task files, not rendering or physics. The real smoke replay should produce `replay_ok` and the expected replay length. That is not a policy-success result. Do not bypass an initial-state mismatch.

After a successful model smoke test using one of the sections below, switch to the full list:

```bash
export ROBOTWIN_INPUT="$ROBOTWIN_LIST"
export RUN_ID=test_r1
```

All evaluation clients below run **from the RoboTwin root in the RoboTwin environment**, even when the evaluator file lives in a model repository. This keeps task configs and asset paths resolvable. Servers run in their own original model environments. Use distinct ports if several policies run simultaneously.

## 3. X-VLA

### Original model environment and checkpoint

Use [X-VLA installation](https://github.com/2toinf/X-VLA) and [its RoboTwin deployment guide](https://github.com/2toinf/X-VLA/tree/main/evaluation/robotwin-2.0). The original recipe creates a Python 3.10 environment and installs `requirements.txt` (or uses `environment.yml`). Install the `X-VLA` overlay into the pinned checkout.

Use [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2), not X-VLA-Pt or the LIBERO checkpoint.

For a new model environment:

```bash
conda create -n xvla python=3.10 -y
conda activate xvla
cd "$PROJECTS/X-VLA"
python -m pip install -r requirements.txt
```

### Terminal A: HTTP model server (X-VLA environment)

```bash
export XVLA_ROOT="$PROJECTS/X-VLA"
cd "$XVLA_ROOT"
python -m deploy --model_path 2toINF/X-VLA-RoboTwin2 \
  --host 127.0.0.1 --port 8010
```

You can replace the model ID with a local downloaded checkpoint directory. Wait until loading finishes and the HTTP server is listening.

### Terminal B: evaluator (RoboTwin environment)

```bash
export XVLA_ROOT="$PROJECTS/X-VLA"
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
cd "$ROBOTWIN_ROOT"
python "$XVLA_ROOT/evaluation/robotwin-2.0/eval_ood_replay_xvla.py" \
  --sample-file-list "$ROBOTWIN_INPUT" \
  --xvla-host 127.0.0.1 --xvla-port 8010 \
  --eval-mode seed_plus_replay --repeat-idx 1 \
  --max-infer-steps -1 --num-steps-wait-after-replay 0 \
  --output-dir "$ROBORECOVER_RUNS/robotwin/xvla" --run-name "$RUN_ID"
```

The `--xvla-*` options are specific to the HTTP adapter. An OpenPI WebSocket server cannot be substituted at this port.

## 4. LingBot-VLA

### Original model environment and checkpoint

Follow [LingBot-VLA's installation and RoboTwin evaluation instructions](https://github.com/Robbyant/lingbot-vla). Install the `lingbot-vla` overlay. Download [the non-depth RoboTwin post-trained checkpoint](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin) and [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

Use the checkpoint/configuration corresponding to the pinned source version. Upstream LeRobot configuration formats have changed; changing packages or deleting config fields without understanding the migration can change inference behavior. A depth-model checkpoint requires its matching depth inputs and is not interchangeable with this example.

The original model setup uses Python 3.12, PyTorch 2.8 and CUDA 12.8:

```bash
conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla
cd "$PROJECTS/lingbot-vla"
bash install.sh
```

Run that project's installer only in the dedicated model environment, not in RoboTwin's Python 3.10 environment.

### Terminal A: WebSocket model server (LingBot environment)

```bash
export LINGBOT_ROOT="$PROJECTS/lingbot-vla"
export LINGBOT_CKPT=/absolute/path/to/lingbot-vla-4b-posttrain-robotwin
export QWEN25_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct
cd "$LINGBOT_ROOT"
python -m deploy.lingbot_vla_policy \
  --model_path "$LINGBOT_CKPT" --port 8020 \
  --use_length 25 --num_denoising_step 10
```

If the checkpoint configuration's normalization file is not available at its recorded location, add `--norm_path /absolute/path/to/matching_robotwin_norm.json`. Do not substitute statistics from a different embodiment.

### Terminal B: evaluator (RoboTwin environment)

```bash
export LINGBOT_ROOT="$PROJECTS/lingbot-vla"
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
cd "$ROBOTWIN_ROOT"
python "$LINGBOT_ROOT/evaluation/robotwin-2.0/eval_ood_replay_lingbot.py" \
  --sample-file-list "$ROBOTWIN_INPUT" \
  --model-server-host 127.0.0.1 --model-server-port 8020 \
  --eval-mode seed_plus_replay --repeat-idx 1 \
  --max-infer-steps -1 --num-steps-wait-after-replay 0 \
  --output-dir "$ROBORECOVER_RUNS/robotwin/lingbot" --run-name "$RUN_ID"
```

## 5. SmolVLA

### Original model environment and checkpoint

Set up the pinned LeRobot checkout using the [SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla), including its model dependencies. Install the `lerobot` overlay and retain the original policy/processor code. Do not install a second, incompatible LeRobot revision into this environment.

The published [SmolVLA base model](https://huggingface.co/lerobot/smolvla_base) is not a RoboTwin specialist. Supply a RoboTwin-finetuned checkpoint directory with its configuration, processor files and normalization statistics. Author-trained weights are not part of this release.

The pinned LeRobot package requires Python 3.12 or later. For its model environment:

```bash
cd "$PROJECTS/lerobot"
uv sync --python 3.12 --extra smolvla
source .venv/bin/activate
```

Use that activated environment for Terminal A only; use RoboTwin's environment for Terminal B.

### Terminal A: custom RPC server (SmolVLA environment)

```bash
export LEROBOT_ROOT="$PROJECTS/lerobot"
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
export SMOLVLA_CKPT=/absolute/path/to/robotwin_smolvla/pretrained_model
cd "$LEROBOT_ROOT"
python robotwin_ood_eval/smolvla_model_server.py \
  --model-path "$SMOLVLA_CKPT" --host 127.0.0.1 --port 8030 \
  --device cuda --action-chunk-size 50 --inference-mode chunk
```

### Terminal B: evaluator (RoboTwin environment)

```bash
export LEROBOT_ROOT="$PROJECTS/lerobot"
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
cd "$ROBOTWIN_ROOT"
python "$LEROBOT_ROOT/robotwin_ood_eval/eval_ood_replay_smolvla.py" \
  --sample-file-list "$ROBOTWIN_INPUT" \
  --model-server-host 127.0.0.1 --model-server-port 8030 \
  --eval-mode seed_plus_replay --repeat-idx 1 \
  --max-infer-steps -1 --num-steps-wait-after-replay 0 \
  --output-dir "$ROBORECOVER_RUNS/robotwin/smolvla" --run-name "$RUN_ID"
```

Keep action-chunk size and inference mode consistent across reported runs. This custom RPC endpoint is not the LingBot WebSocket service.

## 6. Pi0.5

### Original model environment and checkpoint layout

This adapter uses RoboTwin's bundled `policy/pi05` OpenPI integration, **not** the standalone LIBERO WebSocket server. Follow [RoboTwin's policy instructions](https://github.com/RoboTwin-Platform/RoboTwin/tree/main/policy) and [OpenPI installation](https://github.com/Physical-Intelligence/openpi#installation). From the pinned RoboTwin checkout, install the bundled model package in its own environment:

```bash
cd "$ROBOTWIN_ROOT/policy/pi05"
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

The client remains in the RoboTwin simulator environment. The server uses the bundled `.venv/bin/python`; its OpenPI modules include the integration extensions installed by the RoboTwin overlay. A generic OpenPI install may lack the `robotwin_repo_id` argument used here.

Provide a RoboTwin-finetuned Pi0.5 checkpoint. A LIBERO checkpoint has different observation/action conventions and cannot be substituted. This loader expects the following layout **relative to the RoboTwin root**:

```text
policy/pi05/checkpoints/<train-config>/<model-name>/<checkpoint-id>/
├── params/                    # Trained OpenPI parameters
└── assets/
    └── <robotwin-repo-id>/     # Matching normalization/config assets
```

Preserve the complete checkpoint export, including assets. The loader selects an asset entry; avoid multiple unrelated entries in `assets/`. The train-config must exist in the bundled OpenPI configuration registry. The collected config name below is included by the overlay; use another only when it matches your checkpoint.

### Terminal A: Pi0.5 RPC server (bundled model environment)

```bash
export ROBOTWIN_ROOT="$PROJECTS/RoboTwin"
export PI05_CONFIG=pi05_base_finetune_on_robotwin_clean_randomized_joint_training
export PI05_MODEL=pi05_robotwin2
export PI05_CHECKPOINT=final
cd "$ROBOTWIN_ROOT"
policy/pi05/.venv/bin/python script/pi05_model_server_v3.py \
  --host 127.0.0.1 --port 8040 \
  --train-config-name "$PI05_CONFIG" --model-name "$PI05_MODEL" \
  --checkpoint-id "$PI05_CHECKPOINT" --pi0-step 32
```

### Terminal B: evaluator (RoboTwin environment)

Export the same three checkpoint/config variables and run:

```bash
cd "$ROBOTWIN_ROOT"
python script/eval_ood_replay_then_infer.py \
  --sample-file-list "$ROBOTWIN_INPUT" \
  --model-server-host 127.0.0.1 --model-server-port 8040 \
  --train-config-name "$PI05_CONFIG" --model-name "$PI05_MODEL" \
  --checkpoint-id "$PI05_CHECKPOINT" --pi0-step 32 \
  --eval-mode seed_plus_replay --repeat-idx 1 \
  --max-infer-steps -1 --num-steps-wait-after-replay 0 \
  --output-dir "$ROBORECOVER_RUNS/robotwin/pi05" --run-name "$RUN_ID"
```

These commands require compatible weights supplied by the user; they do not download or create a RoboTwin checkpoint.

## 7. Repeated trials and output inspection

Run one policy on one GPU/server first. For repetitions, change **both** `--repeat-idx` and `--run-name` for each complete pass, for example `1/test_r1`, `2/test_r2`, `3/test_r3`. Keep simulation initialization fixed to the scenario and record any model sampling-seed changes separately. Do not run different clients against one stateful server concurrently unless its isolation behavior has been validated.

The output directory contains `results.jsonl`, `summary.json` and logs. Check `eval_success_at_end`, `status`, `reason`, replay/inference step counts and exceptions. `replay_ok` alone is not policy success, and a one-step inference smoke test reaching its budget is not a full benchmark failure rate.

`--max-infer-steps -1` leaves the normal task horizon in control; `--max-infer-steps 1` can be used solely to test one model action. Do not leave the smoke-test budget enabled for a reported full evaluation. Keep post-replay waits at zero and initial-state checks enabled.

## Troubleshooting

| Symptom | Action |
|---|---|
| `failed to find a rendering device` | Verify SAPIEN/Vulkan graphics access, driver setup and container graphics capabilities. A working CUDA model server does not prove simulator rendering works. |
| Missing asset / embodiment / YAML | Finish original asset download; run the evaluator from RoboTwin root; check `ROBOTWIN_ROOT`. |
| cuRobo build error | Check toolkit/compiler/PyTorch CUDA compatibility and GPU architecture; do not disable planning for `ee` traces. |
| `state_mismatch` | Check task configs, asset versions, simulator revisions and stored initialization. Do not bypass the guard. |
| Model request hangs | Check the server terminal, protocol and port; large first-load/JIT delays differ from a dead server. |
| Pi0.5 missing assets / unexpected `robotwin_repo_id` | Use bundled `policy/pi05` environment and the complete checkpoint layout, not a standalone OpenPI environment. |
| SmolVLA processor mismatch | Supply matching finetuned policy, processor and statistics; do not combine a base model with another checkpoint's statistics. |
| LingBot configuration fields rejected | Pinned code/checkpoint compatibility, especially upstream LeRobot format migration. |

The commands are matched to the collected source interfaces; full clean-environment execution of every packaged policy is still pending. Model servers should remain on trusted local networks, not exposed as public services.
