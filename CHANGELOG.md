# Changelog

## 0.1.0

- First release:
  - A hierarchical JEPA world model (step level and intent level) that learns from agent and
    sub-agent steps.
  - Four tools: `wm_observe`, `wm_plan`, `wm_train` and `wm_status`.
  - The World Model skill.
  - The read-only `world-model-scout` sub-agent.
  - The installer for the official H-JEPA release (`wm_hjepa` tool, the H-JEPA install skill and
    `scripts/install-hjepa.sh`): pinned code, checkpoints and evaluation tasks from Hugging Face,
    checksums, and a smoke test.
  - H-JEPA evaluations on this computer's GPU (Apple GPU / CPU through an in-memory CUDA adapter) or
    on a remote NVIDIA GPU over SSH, chosen by the new "H-JEPA compute" setting
    (`wm_hjepa {action:"run"}`, `scripts/hjepa-run.sh`, `scripts/hjepa_local.py`).
  - Settings to pick the H-JEPA world model (environment including DROID, model, training seed,
    evaluation size). `wm_hjepa {action:"run"}` installs whatever that model needs and is missing
    (locally or on the remote host), then evaluates it.
  - DROID evaluation with Fréchet fidelity. On Apple Silicon it uses `decord2`/`eva-decord` in
    place of `decord`, and the CUDA adapter now also covers tensors created with `device="cuda"`.
