---
name: H-JEPA install
description: the user asks to install, download, set up, check or run the official H-JEPA world models (huggingface.co/jepa-world-models/h-jepa, github.com/kevinghst/H-JEPA checkpoints)
---

# Install the official H-JEPA release

The release has three parts:

- **Code:** <https://github.com/kevinghst/H-JEPA> (MIT).
- **Checkpoints:** <https://huggingface.co/jepa-world-models/h-jepa> (MIT): 93 PyTorch models,
  12.4 GB in total.
- **Evaluation data:** <https://huggingface.co/datasets/jepa-world-models/h-jepa> (CC BY 4.0).

They are **visual robot-planning world models**: pixels and actions in Visual AntMaze, FourRoom
Distractors, OGBench Cube, Push-T and DROID. They do not read agent trajectories. The plugin's
`wm_observe` / `wm_plan` keep using its own small text H-JEPA. Say so if the user seems to expect
otherwise.

Tool: `wm_hjepa`. It may be on-demand under a plugin-prefixed name; if it is inactive, call
`ToolSearch` with `wm_hjepa`. The tool never runs processes. You run its commands with Bash, from the
workspace root.

## 1. The selection comes from the plugin settings

The user picks the world model in the plugin settings. `wm_hjepa {action:"status"}` shows the
choice as `model` (or a `model.error` when two settings don't fit together: relay it).

| Setting | Choices |
|---|---|
| **H-JEPA environment** | `ant` Visual AntMaze, `fourroom` FourRoom Distractors, `cube` OGBench Cube, `pusht` Push-T, `droid` DROID (real robot arm, recorded videos) |
| **H-JEPA model** | Recommended (H-JEPA 3 levels; DROID: 2 levels), H-JEPA 2/3/4 levels, HWM 2/3/4 levels, LeWM flat. DROID has only LeWM, H-JEPA 2, HWM 2. |
| **H-JEPA training seed** | 1st / 2nd / 3rd: 42/43/44, or DROID 1/1000/10000 |
| **Evaluation size** | tasks (of 50) or DROID clips (of 16) per run; 0 = all |

- **Metrics:** the simulated environments report the **success rate**. DROID reports **Fréchet
  fidelity**: how closely the planned robot path follows the recorded one, from 0 (no motion) to
  100 (exact). DROID runs offline on 16 recorded clips.
- **When the user names something in chat** ("run DROID", "the 4-level model"), pass it as
  arguments (`env`, `model`, `seed`, `tasks`) for that run. Suggest they save it in the settings
  if they want it as the default.
- **Large installs:** if the user asks for "all" or a large set (several GB), confirm once, with the
  size `plan` reports. A single checkpoint is 45–350 MB.

## 2. Install only (optional)

`run` (section 3) installs what it needs by itself. Use `plan` to install ahead of time, or several
checkpoints at once.

1. Call `wm_hjepa {action:"plan"}` (the settings' model) or `{action:"plan", env, models, seeds,
   home?, codeDir?, python?}`. It writes the
   pinned installer to `.world-model/hjepa/install-hjepa.sh` and returns `commands`, `downloadMB`,
   `requirements` and `notes`.
2. Run `commands.dryRun` and check the printed steps.
   - If it says no conda/uv/python3.11 was found, ask the user to install one (e.g. `brew install
     uv`), or to pick `python`. **Never install a toolchain without asking.**
3. Run `commands.install` with Bash and a long timeout (3600 s). The output goes to
   `.world-model/hjepa/install.log`.
   - Delegating the run to the `test-runner` sub-agent keeps pip and download logs out of your
     context. Tell it to run the exact command and report only the last error.
   - Never run two installs into the same home at once.
4. On failure, read the end of `.world-model/hjepa/install.log`. Fix the cause (network, disk,
   missing toolchain) and run the install command again: it resumes.
   - Clones, environments and downloads that already exist are reused.
   - `hf` skips files it already has.

   Don't edit the pinned revisions.
5. Call `wm_hjepa {action:"status"}`. `status: "installed"` means:
   - the code is at the pinned commit;
   - `torch` and `hydra` import;
   - every selected checkpoint and its `normalizer.pt` exist;
   - the evaluation tasks passed their SHA-256 check.

## 3. Run an evaluation (installs what is missing first)

The plugin setting **H-JEPA compute** picks where evaluations run. `wm_hjepa {action:"status"}`
shows it as `compute`.

| Setting | Where | How |
|---|---|---|
| `local` (default): this computer's GPU | Apple GPU (`mps`) or CPU, chosen by **Local device**; an NVIDIA GPU is used directly when present | `hjepa_local.py` redirects the release's CUDA calls in memory. The clone is not modified. |
| `remote`: remote NVIDIA GPU | the SSH host in **Remote GPU host**, under **Remote HJEPA_HOME** | installs what is missing there, runs `eval.py` unchanged, copies results back |

1. Call `wm_hjepa {action:"run"}`.
   - With no arguments it evaluates the checkpoint chosen in the settings. `env`, `model`,
     `seed`, `tasks`, `planSeed` (DROID), `compute`, `device` and `host` override them for one
     run.
   - The result has `selection`, `config`, `metric`, `count`, `unit`, `estimate`, `commands`.
   - Suggest `tasks: 1` for a first check when the model hasn't run on this target before.
2. Run `commands.dryRun`, then `commands.run` as a **background Bash task**, and tell the user the
   `estimate`.
   - The run first installs whatever is missing for that model: code, Python environment,
     checkpoint, evaluation tasks or DROID clips, and the DROID video reader on a Mac. This
     happens locally or on the remote host. Nothing is reinstalled when it's all there.
   - The output goes to `log`. A run is resumable: rerunning the same command skips the tasks or
     clips already done.
3. When it ends, call `wm_hjepa {action:"status"}`. `lastRun` has `status`, `metric`, `score` (%),
   `count`, `unit` and `device`. The full results are in `results`.

Local runs on `mps`/`cpu` are close to the paper's CUDA setup but not identical (float32 on the
Apple GPU, other kernels). For paper-faithful numbers, use `remote`.

**Remote setup, before the first run:**
- The user sets **Remote GPU host** to an SSH destination that works without a password prompt
  (`ssh <host> true` must succeed).
- The host runs Linux with a working `nvidia-smi`, git, and conda, uv or python3.11.
- If any of this is missing, the script stops with a clear message: relay it. Never handle
  passwords or private keys yourself, and never install anything on a remote host the user didn't
  configure.

For training, the full paper reproductions and DROID, point the user to the release README. The
training datasets are tens of GB and are not part of this install.

## Safety and limits

- **The checkpoints are pickled modules.** The installer loads them with
  `torch.load(weights_only=False)`, exactly as the release does, and that executes code. Load only
  the pinned files. Never load a checkpoint from another source this way.
- **The release targets NVIDIA GPUs** (CUDA 12.8 wheels on Linux).
- **macOS arm64:** `decord`, the video reader DROID needs, has no Apple Silicon wheel. The
  installer puts `decord2` (macOS 14+) or `eva-decord` (macOS 11+) in its place; both import as
  `decord`.
- **`eval.py` needs CUDA as released.** The pinned code calls `model.to("cuda")`, its planner
  configs set `device: cuda`, and MPS has no float64. Always run evaluations through
  `wm_hjepa {action:"run"}`, never `eval.py` directly on a Mac. Don't edit the pinned clone.
- **Licences and citation:** the code and checkpoints are MIT, the data is CC BY 4.0. Remind the
  user to cite the paper (arXiv 2610.06805), and DROID when they use `droid/*`.
