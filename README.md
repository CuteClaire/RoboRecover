# RoboRecover

Evaluation toolkit for robot policy recovery under execution deviations in LIBERO and RoboTwin.

> **Development snapshot.** Evaluation extensions, fixed splits and format examples are available. Full dataset hosting and end-to-end environment/model-server instructions are being prepared. This is not yet a one-command, fully validated release. No model weights are included.

## Evaluation scope

| Platform | Train / test starting points | Collected policy adapters |
|---|---|---|
| LIBERO | 800 / 200 | Pi0, Pi0.5, Being-H, UniFOLM, Cosmos-Policy, FastWAM |
| RoboTwin | 800 / 200 | Pi0.5, X-VLA, LingBot-VLA, SmolVLA |

RoboTwin FastWAM and LingBot-VA adapters are not included. The training split contains evaluation starting points, not complete policy-training demonstrations.

## Repository layout

- `overlays/`: evaluation extensions and support files, organized by upstream repository and native file location.
- `splits/`: fixed membership, relative scenario paths, checksums and replay lengths.
- `examples/`: one JSON format example per platform; not the full test set.
- `tools/prepare_data.py`: validate scenario checksums and generate evaluator input lists.
- `tools/install_overlay.py`: inspect/install an overlay into a pinned upstream checkout, with backups.
- `dependencies.lock.json`: collected upstream revisions; unresolved entries are explicitly marked.
- `third_party_licenses/`, `NOTICE.md`: original notices and redistribution caveats.

## 1. Prepare the environments

Clone this repository and set an absolute location:

```bash
git clone https://github.com/CuteClaire/RoboRecover.git
cd RoboRecover
export ROBORECOVER_ROOT="$PWD"
```

Each model uses its own upstream environment. Do not combine all model dependencies into a single environment. Use the repository URLs and revisions in `dependencies.lock.json`, install the upstream simulator/model dependencies and acquire their separately distributed assets and weights. Dedicated, clean upstream checkouts are recommended.

For example, prepare the pinned RoboTwin checkout:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /absolute/path/to/RoboTwin
git -C /absolute/path/to/RoboTwin checkout c3ddfa8b97d5519efa828b075999bd0006778e5e
export ROBOTWIN_ROOT=/absolute/path/to/RoboTwin
python "$ROBORECOVER_ROOT/tools/install_overlay.py" \
  --component RoboTwin --checkout "$ROBOTWIN_ROOT"
# Review the dry-run output, then install:
python "$ROBORECOVER_ROOT/tools/install_overlay.py" \
  --component RoboTwin --checkout "$ROBOTWIN_ROOT" --apply
```

Repeat the overlay installation for the selected model checkout. Existing files are backed up under `.roborecover-backup` inside that checkout. Installation refuses a mismatched or unresolved base revision. UniFOLM's base revision is currently unresolved, so its installation remains pending.

Installing overlays does **not** install Python packages, simulator assets or weights. Model-server setup and clean-install environment validation are still being completed.

## 2. Prepare evaluation data

**Full dataset download: pending Hugging Face publication.** The complete scenarios are intentionally not stored in Git. Until publication, these steps require an author-provided exported scenario tree. No private workspace paths are required.

The scenario root should contain `libero/train`, `libero/test`, `robotwin/train`, and `robotwin/test`, with the filenames listed in `splits/*.csv`.

```bash
export ROBORECOVER_DATA=/absolute/path/to/scenarios
python "$ROBORECOVER_ROOT/tools/prepare_data.py" \
  --platform robotwin --split test --data-root "$ROBORECOVER_DATA" \
  --output "$ROBORECOVER_ROOT/runs/robotwin_test.txt"
python "$ROBORECOVER_ROOT/tools/prepare_data.py" \
  --platform libero --split test --data-root "$ROBORECOVER_DATA" \
  --output "$ROBORECOVER_ROOT/runs/libero_test.txt"
```

The validator checks the full split, SHA256 hashes, action counts, dimensions and finite numeric values. It stops on missing or modified data.

## 3. Evaluate RoboTwin

Activate the configured RoboTwin environment and start with a schema check, then a single simulator replay:

```bash
cd "$ROBOTWIN_ROOT"
python script/replay_robotwin_trace.py \
  --sample-file-list "$ROBORECOVER_ROOT/runs/robotwin_test.txt" --verify-only
python script/replay_robotwin_trace.py \
  --sample-file-list "$ROBORECOVER_ROOT/runs/robotwin_test.txt" --limit 1 \
  --output-dir "$ROBORECOVER_ROOT/runs" --run-name replay_smoke
```

Replay requires a functioning GPU rendering device and simulator assets. `--verify-only` does not test rendering or physics. Do not disable state-mismatch checks to hide a setup failure.

After starting a compatible Pi0.5 model RPC server, evaluate using its port (8000 is an example):

```bash
python script/eval_ood_replay_then_infer.py \
  --sample-file-list "$ROBORECOVER_ROOT/runs/robotwin_test.txt" \
  --model-server-host 127.0.0.1 --model-server-port 8000 \
  --eval-mode seed_plus_replay --repeat-idx 1 \
  --output-dir "$ROBORECOVER_ROOT/runs" --run-name pi05_r1
```

Other native entrypoints, run from their respective model checkout:

| Policy | Entrypoint | Server arguments |
|---|---|---|
| X-VLA | `evaluation/robotwin-2.0/eval_ood_replay_xvla.py` | `--xvla-host`, `--xvla-port` |
| LingBot-VLA | `evaluation/robotwin-2.0/eval_ood_replay_lingbot.py` | `--model-server-host`, `--model-server-port` |
| SmolVLA | `robotwin_ood_eval/eval_ood_replay_smolvla.py` | `--model-server-host`, `--model-server-port` |

All accept `--sample-file-list`; inspect each entrypoint's `--help` for model-specific configuration. Ports do not imply interchangeable protocols: X-VLA, LingBot-VLA and RPC-based servers have different interfaces. Weights and normalization configuration must match the chosen adapter.

## 4. Evaluate LIBERO

Install the OpenPI overlay, activate its LIBERO evaluation environment, and start a compatible OpenPI WebSocket policy server using the chosen Pi0/Pi0.5 weights. From the OpenPI checkout:

```bash
python examples/libero/eval_filtered_repro.py \
  --filtered-root "$ROBORECOVER_DATA/libero/test" \
  --sample-file-list "$ROBORECOVER_ROOT/runs/libero_test.txt" \
  --host 127.0.0.1 --port 8000 --model-tag pi05 \
  --num-steps-wait-after-replay 0 --no-save-video \
  --output-dir "$ROBORECOVER_ROOT/runs" --run-name libero_pi05_r1
```

Collected alternative entrypoints:

| Policy | Entrypoint in its upstream checkout |
|---|---|
| Being-H | `BeingH/benchmark/libero/eval_filtered_repro_beingh.py` |
| UniFOLM | `experiments/LIBERO/eval_filtered_repro_unifolm.py` |
| FastWAM | `experiments/libero/eval_filtered_repro_fastwam.py` |
| Cosmos-Policy | `cosmos_policy/experiments/robot/libero/run_libero_filtered_repro.py` |

These models have different checkpoint/configuration arguments; complete tested launch recipes are pending. Their source availability must not be interpreted as verified clean-install execution. Optional OpenPI RLDS export additionally requires the provided writer and its dependencies; ordinary evaluation does not import that writer.

## Results and protocol

Evaluators write per-scenario results and summary files under the selected output/run directory. Preserve status, success, replay steps, inference steps and repeat identifier. For multiple repeats, use a distinct run name for each run.

Replay the stored prefix, then hand control to the policy using the original instruction and success criterion. Replay does not consume the fresh inference budget. Do not add post-replay wait steps unless defining a separate evaluation setting. Use `seed_only` only when intentionally evaluating from the original initialization instead of the recovery starting point.

Compute each scenario's success rate over its scheduled trials, then average scenarios equally. Invalid scheduled trials count as failures; report them separately as well. Never silently drop failed runs or pool unequal repeat counts across scenarios.

LIBERO actions are 7D. RoboTwin stored actions may be 14D or 16D and must be interpreted by the native replay implementation. Some LIBERO records use `len(actions) == annotated_step`; retain the stored prefix unchanged rather than padding it. The fixed test sets each cover nine stage/deviation groups.

## Validation status

The author workspace checks found all 2000 scenarios, validated RoboTwin's 1000 scenarios through its schema/task-file checker, and successfully replayed one 24-step RoboTwin example. These checks predate this portable overlay packaging. Clean-environment all-model inference and full simulator replay of this release have **not** yet been completed.

See `NOTICE.md` for licensing scope. Third-party licenses are preserved; model weights, simulator assets and full scenario-data licensing are separate from this repository's Apache-2.0 license.
