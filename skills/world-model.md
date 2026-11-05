---
name: World Model
description: a multi-step task where you choose between approaches or sub-agents (explorer, fixer, test-runner…), or the user asks to plan with, train or check the world model / H-JEPA
---

# World Model (H-JEPA)

A small hierarchical JEPA learns, from what you record, what actions and sub-agent delegations do
to the state and how often they end in failure. It predicts **representations, not text**:

- **Step level**: one action → next state.
- **Intent level**: one subgoal (a chunk of steps) → its end state.

An ensemble of predictors gives the **uncertainty**. The prediction error on what really happened
gives the **surprise**.

Tools: `wm_observe`, `wm_plan`, `wm_train`, `wm_status`. They may be on-demand under a
plugin-prefixed name; if they are inactive, call `ToolSearch` with `wm_`.

It is a learned prior, not an oracle:

- The user's instructions, the code and test results always win over its ranking.
- Never pick a destructive or out-of-scope option because it scored best.

Keep the overhead small:

- Use it for tasks with ≥3 steps or a real choice. Skip it for one-shot answers.
- State, action and result texts are compact (≤600 chars, facts not prose, no secrets, no file
  dumps). Write states the same way every time, e.g. `tests: 3 failing (auth); files: login.ts
  edited; blocker: none`. A consistent format is what makes the latents comparable.

## Loop

1. **Start.** Pick an episode id (`<task-slug>-<n>`). On the first call of the session, call
   `wm_status`. If `coldStart` is true, the model is still learning: follow its advice loosely,
   but record everything anyway.
2. **Plan before a fork.** Before you choose between approaches or decide whom to delegate to, call
   `wm_plan` with:
   - `state` and `goal`, written like states;
   - 2–5 `options` of the form `{label, agent, intent, steps}`. `agent` is `main` or the sub-agent
     you would hand it to; `steps` are the 1–6 concrete moves.

   Read `advice`:
   - `go: …`: take the recommended option unless you have evidence against it.
   - `probe`: the model is unsure. Reduce the uncertainty cheaply first: one Read/Grep, or the
     **world-model-scout** sub-agent for a wider look. Observe the result, then plan again.
   - `close call`: decide by cost, or ask the user if it matters.
   - `cold start`: prefer the cheapest informative option.
3. **Observe after every meaningful step.** That is a tool batch, a test run, or a sub-agent
   report. Call `wm_observe {episode, state, action, result, agent, intent, ok}`:
   - `state` is before the step and `result` is after it.
   - For a sub-agent, `agent` is its name, `action` is the brief in one line and `result` is its
     report in one line.
   - `ok` says whether the step achieved its purpose.
   - Keep `intent` the same for all steps that serve one subgoal.

   Read `advice`:
   - `continue`.
   - `replan`: the step failed, or the result surprised the model (`surprise` above the
     threshold). Re-check your assumptions and call `wm_plan` again before the next step.
4. **End.** Call `wm_observe {episode, outcome: "success" | "failure"}`. This labels the episode and
   trains the cost heads. Record failures honestly: they are the most useful data.
5. **Train.** Every ~10 episodes, or when the user asks, call `wm_train {epochs: 5}` and report
   the first/last losses in one line.

## Sub-agents

The world model sits with the orchestrator (you). Sub-agents never call the `wm_` tools.

- Use `wm_plan` to choose which sub-agent gets a task, and on which plan.
- Record each sub-agent report with `wm_observe`. This builds the per-agent reliability
  (`agentPrior`), which `wm_plan` blends in while the model is still cold.
- When you run several sub-agents in parallel, record each report as its own step, in the order
  the reports arrived.

## Report

In your final message, add at most one line about the world model. Mention it only if it changed a
decision (e.g. "World model flagged high uncertainty → scouted first"). Never paste its raw
JSON.

## Status and reset

- `wm_status` shows counts, episode outcomes, per-agent stats and settings.
- `wm_status {reset: "model"}` keeps the experience and retrains from scratch on the next
  `wm_train`.
- `{reset: "all"}` also forgets the experience. Do it only if the user asks.
- The data lives in `.world-model/` in the workspace. Tell the user to add it to `.gitignore` if
  they don't want it committed.
