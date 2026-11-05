# World Model (H-JEPA) for PI-Desktop

A learned world model for the agent and its sub-agents. This plugin follows the Joint-Embedding
Predictive Architecture (JEPA) and its hierarchical form (H-JEPA): the model predicts
**representations, not text**.

As the agent works, it records each step. From those steps the model learns:

- what each action, sub-agent delegation and intent does to the state;
- how often each one ends in failure.

Then it changes what the agent does:

- **Before a fork:** `wm_plan` imagines each candidate (an approach, a plan, or a hand-off to
  `explorer`, `fixer`, …). It ranks them by predicted failure, distance to the goal and
  uncertainty.
- **After a step:** `wm_observe` measures how **surprising** the real result was and tells the
  agent to `continue` or `replan`.
- **When unsure:** if the predictors disagree, the agent is told to `probe` first. One way is the
  bundled cheap, read-only `world-model-scout` sub-agent.

## Install

1. Download `eskwisit.world-model-hjepa-<version>.piplug` from the
   [releases](https://github.com/eskwisit/pi-world-model-hjepa/releases), then install it from the
   PI-Desktop plugins page. To develop, load this folder as a development plugin instead.
2. Grant the permissions it asks for (see [Data and permissions](#data-and-permissions)).
3. Start a new session, so that the `world-model-scout` sub-agent and the skills are picked up.
4. Optional: pick the H-JEPA model and where it runs in the plugin's settings
   ([Settings](#settings)).

## The model

Everything is pure JavaScript. There are no dependencies, no network access and no processes.
The model is in [lib/hjepa.js](lib/hjepa.js).

| Part | Level 1: one step | Level 2: one intent (a chunk of steps) |
|---|---|---|
| Input | state text → signed feature hashing (256-d) | level-1 target latent of the chunk's first state |
| Online encoder | `enc1`: 256→64→32 | `enc2`: 32→32→16 |
| Target encoder | `tgt1`: EMA(0.99) of `enc1`, stop-grad | `tgt2`: EMA of `enc2` |
| Action / intent embedding | `act1`: 256→16 | `act2`: 256→16 |
| Predictor | 3× residual MLP `z + f(z, a)`: an ensemble whose spread is the uncertainty | 3× residual MLP `h + f(h, intent)` |
| Loss | latent MSE against the target encoder, plus a VICReg variance term against collapse | same |
| Cost head | `cost1(z′)` → P(failure), trained on outcome labels | `cost2(h′)` |

- **Surprise** is the latent prediction error divided by its running average, so 1 is a typical
  step.
- **Uncertainty** is the ensemble spread divided by its running average.
- **Planning** encodes the state, rolls each option's steps through the predictors and its intent
  through level 2. It then scores
  `failure + goalWeight·goalDistance + riskAversion·max(0, uncertainty − 1)`.
- **Cold start:** while the model has fewer than ~50 labelled steps, the predicted failure is
  blended with a per-agent Beta(1, 1) success rate.

## Tools

| Tool | What it does |
|---|---|
| `wm_observe` | Records one step (`episode, state, action, result, agent, intent, ok`) or ends an episode (`outcome`). It trains a few online minibatches and returns `surprise`, `uncertainty`, `predictedFailure` and `advice`. |
| `wm_plan` | Ranks 1–12 `options` (`{label, agent, intent, steps}`) from `state` toward `goal`. It returns `recommended`, `advice` (`go` / `probe` / `close call` / `cold start`), `trust` and per-level details. |
| `wm_train` | Trains on all recorded experience for 1–50 epochs, bounded to ~80 s. |
| `wm_status` | Shows counts, episode outcomes, per-agent reliability and settings. `reset: "model" \| "all"`. |
| `wm_hjepa` | Official H-JEPA release: stages the installer (`plan`), returns an evaluation command for the configured compute target (`run`), and reports the installation and last run (`status`). |

The **World Model** skill tells the agent when to call these tools and how to record sub-agent
reports. Sub-agents never call the tools themselves: the orchestrator does.

## Official H-JEPA release

The plugin also ships a way to install the official H-JEPA release:

- **Checkpoints:** [huggingface.co/jepa-world-models/h-jepa](https://huggingface.co/jepa-world-models/h-jepa), with the
  [code](https://github.com/kevinghst/H-JEPA) and the
  [evaluation data](https://huggingface.co/datasets/jepa-world-models/h-jepa). These come from
  Zhang, Terver, Rabbat, LeCun and Balestriero, arXiv 2610.06805.
- **What the models are:** visual robot-planning world models (AntMaze, FourRoom, Cube, Push-T,
  DROID). They run in the release's own Python/PyTorch environment, not inside the plugin, and they
  don't replace the text model above.

Asking the agent to "run H-JEPA" installs whatever the chosen model needs by itself (see
[Picking the world model](#picking-the-world-model)). To install ahead of time, ask it to "install
H-JEPA" and the **H-JEPA install** skill takes over:

1. `wm_hjepa {action:"plan"}` (the model chosen in the settings), or
   `{action:"plan", env, models, seeds}` for other checkpoints, stages
   [scripts/install-hjepa.sh](scripts/install-hjepa.sh) as `.world-model/hjepa/install-hjepa.sh`.
   It returns the download size and the dry-run, install and check commands.
2. The agent runs them with Bash. The script does the following, following the release README:
   - clones the code at a pinned commit (`97054ec`);
   - creates a Python 3.11 environment (conda, uv or `python3.11 -m venv`);
   - runs `pip install -e ".[train,env]"`;
   - downloads the selected checkpoints at a pinned revision (`08fc8a2`) into
     `$HJEPA_HOME/ckpts`;
   - downloads the evaluation tasks (`cc05cb8`) and verifies their SHA-256 checksums;
   - loads one checkpoint as a smoke test.
3. `wm_hjepa {action:"status"}` reports the result of the check, the model chosen in the
   settings, the compute target and the last run.

You can also run it by hand:

```bash
bash scripts/install-hjepa.sh --help
bash scripts/install-hjepa.sh --dry-run                      # default: ant / hjepa_l3 / seed 42, ~295 MB
bash scripts/install-hjepa.sh --env sim --models all --seeds 42
bash scripts/install-hjepa.sh --check
```

| Selection | Checkpoints | Download |
|---|---|---|
| default (`ant`, `hjepa_l3`, `42`) | 1 | ~295 MB with eval tasks |
| one env, all models, all seeds | 21 | 1.7–2.7 GB |
| `--env all --models all --seeds all` | 93 | ~12.4 GB, plus ~1.7 GB of eval tasks |

Notes:

- **What is not installed:** the training datasets (tens of GB). See the release README, sections
  3.1 and 4.1.
- **Platforms:** the release targets NVIDIA GPUs (CUDA 12.8 wheels on Linux). On macOS arm64,
  `decord` has no wheel, so the script installs the rest of `[train]` plus `decord2` (or
  `eva-decord`) in its place.
- **Security:** checkpoints are pickled modules. They are loaded with
  `torch.load(weights_only=False)`, exactly as the release does, which executes code. Load only
  the pinned files.
- **Licences:** the code and checkpoints are MIT, the data is CC BY 4.0. Cite DROID for `droid/*`.

### Picking the world model

Four settings pick which official checkpoint is installed and evaluated:

| Setting | Choices |
|---|---|
| **H-JEPA environment** | Visual AntMaze, FourRoom Distractors, OGBench Cube, Push-T (simulated, 50 tasks each, scored by **success rate**), or DROID (a real robot arm, offline on 16 recorded clips, scored by **Fréchet fidelity**: how closely the planned path follows the recorded one, 0–100) |
| **H-JEPA model** | Recommended (H-JEPA 3 levels; DROID 2 levels), H-JEPA 2/3/4 levels, HWM 2/3/4 levels (the paper's baseline), LeWM (flat). DROID has LeWM, H-JEPA 2 and HWM 2 only. |
| **H-JEPA training seed** | 1st, 2nd or 3rd training run: 42/43/44, DROID 1/1000/10000 |
| **Evaluation size** | tasks or clips per run, 0 = all |

Then ask the agent to "run H-JEPA". `wm_hjepa {action:"run"}` builds one command that does three
things on the target set by **H-JEPA compute** (below):

1. installs whatever that checkpoint needs and is missing: code, Python environment, checkpoint,
   evaluation tasks or DROID clips;
2. evaluates it;
3. records the score.

### Running evaluations: this computer's GPU or a remote NVIDIA GPU

As released, `eval.py` needs CUDA: it calls `model.to("cuda")`, its configs set `device: cuda`, and
it forces Linux-only headless rendering. The **H-JEPA compute** setting picks where
`wm_hjepa {action:"run"}` runs it. [scripts/hjepa-run.sh](scripts/hjepa-run.sh) does the work for
both targets.

- **This computer's GPU (`local`, default).** [scripts/hjepa_local.py](scripts/hjepa_local.py)
  launches `eval.py` and adapts it in memory; the clone is not modified:
  - `cuda` goes to the Apple GPU (`mps`) or the CPU (**Local device** setting);
  - every `device: cuda` in the config is overridden;
  - float64 becomes float32 on `mps`;
  - MuJoCo renders with `glfw` on macOS.

  When the machine does have an NVIDIA GPU, `eval.py` runs unchanged.
  - **Measured on an Apple M2:** one `ant_l3` task (AntMaze, 3-level H-JEPA, seed 42) took
    628–684 s over two runs and succeeded both times. All 50 tasks would take ~9 h, so run it
    in the background.
  - Numbers are close to, not identical with, the paper's CUDA runs.
  - **Measured on an Apple M2:** DROID H-JEPA 2 levels (seed 1, planner seed 1) planned
    clip 0 in 343 s, with Fréchet fidelity 9.75% on that one clip. All 16 clips would take
    ~1.5 h. One clip is not comparable with the paper's mean over 16 clips and 3 seeds.
  - **DROID on a Mac:** `decord`, the video reader DROID needs, has no Apple Silicon wheel. The
    installer puts `decord2` (or `eva-decord`) in its place; both import as `decord`.
- **Remote NVIDIA GPU (`remote`).** Over key-authenticated SSH to **Remote GPU host**:
  - checks Linux and `nvidia-smi`;
  - with `setup` (the default from `wm_hjepa`), copies the pinned installer and installs what the
    run needs and is missing under **Remote HJEPA_HOME**;
  - runs `eval.py` unchanged with `MUJOCO_GL=egl`;
  - copies the results back with `scp`.

  This is the paper's setup. Any machine you can SSH into works: a rented cloud GPU, a lab server.

Results land in `.world-model/hjepa/results/<run-id>/` (`metrics.yaml`, `planning_compute.json`,
`eval_config.yaml`, or `eval.csv` and `plan_config.yaml` for DROID, plus `run.json`).
`wm_hjepa {action:"status"}` reports the last run: its metric, score and count. Runs resume:
rerunning the same command skips the finished tasks.

```bash
bash scripts/hjepa-run.sh --tasks 1                                   # this Mac, Apple GPU
bash scripts/hjepa-run.sh --target remote --host me@gpu-box --setup  # remote NVIDIA, 50 tasks
bash scripts/hjepa-run.sh --help
```

## Settings

| Key | Default | Meaning |
|---|---|---|
| `onlineSteps` | 2 | Minibatches trained after each `wm_observe` (0 turns online learning off). |
| `riskAversion` | 0.3 | Weight on uncertainty in `wm_plan`. |
| `goalWeight` | 0.5 | Weight on the latent distance to the goal. |
| `surpriseThreshold` | 2.5 | Relative error above which `wm_observe` advises `replan`. |
| `maxTransitions` | 4000 | The most recent steps that are kept. |
| `hjepaEnv` | `ant` | Which H-JEPA world model: `ant`, `fourroom`, `cube`, `pusht` or `droid`. |
| `hjepaModel` | `auto` | `auto` (recommended), `hjepa_l2`–`l4`, `hwm_l2`–`l4`, `lewm`. |
| `hjepaSeed` | `first` | Training seed slot: `first`, `second`, `third`. |
| `hjepaTasks` | 0 | Tasks (≤ 50) or DROID clips (≤ 16) per run; 0 = all. |
| `hjepaCompute` | `local` | Where H-JEPA evaluations run: `local` (this computer's GPU) or `remote` (NVIDIA over SSH). |
| `hjepaLocalDevice` | `auto` | `auto`, `mps` or `cpu`, for local runs without an NVIDIA GPU. |
| `hjepaRemoteHost` | (empty) | SSH destination of the NVIDIA machine (key authentication only). |
| `hjepaRemoteHome` | `hjepa` | Folder on the remote host, relative to its home directory. |

## Data and permissions

- `agent.prompt.inject` indexes the two skills. `agent.tool.register` adds the five tools.
- `.world-model/hjepa/` holds the staged installer, its log and `state.json`. The installer runs
  only when the agent runs it with Bash (or when you do). It writes outside the workspace, to
  `$HJEPA_HOME` (default `~/hjepa`).
- `fs.read` and `fs.write` are scoped to `.world-model/**` in the workspace:
  - `experience.json` holds the compact texts the agent chose to log;
  - `model.json` holds the weights (~650 KB).

  Add `.world-model/` to your `.gitignore` if you don't want them committed.
- Node `fs` is used only to copy `world-model-scout.md` to `~/.agents/subagents` and to remove it
  on unload, unless you edited it. After the first install, start a new session so the scout
  shows up in the agent list.

## Limits

This is a small model that learns from your own sessions. It is not a pretrained one.

- It starts knowing nothing. Its signals become useful after a few dozen labelled steps, and only
  if the agent writes states in a consistent format.
- Feature hashing sees words and word pairs, not meaning. Paraphrases look different to it.
- Treat its output as a prior. Instructions, code and test results win.

## Tests

```bash
node --test tests/
```

The suite covers the following, all without a host:

- gradient checks against finite differences;
- collapse prevention;
- loss reduction on a toy environment;
- surprise on unexpected results;
- weight serialisation;
- every tool;
- install and removal of the sub-agent;
- an end-to-end case where the model learns to route a bug fix to `fixer` rather than `explorer`.
- the H-JEPA installer:
  - its size catalog against the tool's;
  - dry runs of the staged commands;
  - `--check` and status, including paths with spaces and quotes.
- the evaluation runner, end to end:
  - locally, with a fake release;
  - over SSH, with fake `ssh`/`scp` and a fake GPU host;
  - its refusals: a host without NVIDIA, a missing release, bad input;
- the CUDA→MPS/CPU adapter, against the installed PyTorch (skipped when `~/hjepa` is absent).

## License

Apache-2.0
