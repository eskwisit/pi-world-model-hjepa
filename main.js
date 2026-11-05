"use strict";

/**
 * World Model (H-JEPA) for PI-Desktop agents and sub-agents.
 *
 * The agent records what it sees and does (wm_observe); a small hierarchical JEPA learns, in latent
 * space, what each action and each intent does to the state and how likely it is to end in failure.
 * Before committing to an approach or a sub-agent the agent asks it to imagine the options
 * (wm_plan); after each step the model reports how surprising the result was, which is the signal
 * to keep going or to replan.
 *
 * Files: the experience and the weights live in the workspace under .world-model/ through the
 * host's scoped pi.fs API. node:fs is used only to copy the world-model-scout sub-agent to
 * ~/.agents/subagents (where PI-Desktop loads sub-agents from) and remove it on unload.
 */

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { HJEPA, round } = require(path.join(__dirname, "lib", "hjepa.js"));

const DIR = ".world-model";
const MODEL_FILE = `${DIR}/model.json`;
const EXP_FILE = `${DIR}/experience.json`;
const MAX_TEXT = 600;
const COLD_START = 20; // labeled transitions before the cost heads are trusted
const CHUNK = 4; // steps per level-2 chunk when no intent is given
const BATCH = 16;
const TRAIN_BUDGET_MS = 80_000;
const MARKER = "Use only from the world-model skill.";
const SUBAGENTS = ["world-model-scout.md"];
const HJEPA_DIR = `${DIR}/hjepa`;
const HJEPA_SCRIPT = `${HJEPA_DIR}/install-hjepa.sh`;
const HJEPA_STATE = `${HJEPA_DIR}/state.json`;
const HJEPA_LOG = `${HJEPA_DIR}/install.log`;
const HJEPA_LAST_RUN = `${HJEPA_DIR}/last-run.json`;

const DEFAULTS = {
  onlineSteps: 2,
  riskAversion: 0.3,
  goalWeight: 0.5,
  surpriseThreshold: 2.5,
  maxTransitions: 4000,
};

let subagentStatus = "not installed";
const registered = [];

// ---------- serialisation of tool calls (files are read-modify-write) ----------
let queue = Promise.resolve();
function serial(fn) {
  const run = queue.then(fn, fn);
  queue = run.catch(() => {});
  return run;
}

// ---------- settings ----------
async function settings() {
  let s = {};
  try {
    s = (await pi.plugin.getSettings()) || {};
  } catch {
    // defaults
  }
  const out = { ...DEFAULTS };
  for (const k of Object.keys(DEFAULTS)) if (typeof s[k] === "number" && Number.isFinite(s[k])) out[k] = s[k];
  out.onlineSteps = clampInt(out.onlineSteps, 0, 20);
  out.maxTransitions = clampInt(out.maxTransitions, 100, 20000);
  return out;
}

function clampInt(v, lo, hi) {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}

// ---------- storage ----------
async function readJSON(p) {
  try {
    return JSON.parse(await pi.fs.readText(p));
  } catch {
    return null;
  }
}

async function loadExperience() {
  const e = await readJSON(EXP_FILE);
  if (!e || !Array.isArray(e.transitions)) return { version: 1, transitions: [], episodes: {} };
  if (!e.episodes || typeof e.episodes !== "object") e.episodes = {};
  return e;
}

async function saveExperience(e) {
  await pi.fs.writeText(EXP_FILE, JSON.stringify(e));
}

async function loadModel() {
  const j = await readJSON(MODEL_FILE);
  return (j && HJEPA.fromJSON(j)) || new HJEPA(1);
}

async function saveModel(m) {
  await pi.fs.writeText(MODEL_FILE, JSON.stringify(m.toJSON()));
}

// ---------- text ----------
function text(v) {
  if (v == null) return "";
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s.length > MAX_TEXT ? s.slice(0, MAX_TEXT) : s;
}

/** The agent is part of the action: the same action by another sub-agent may end differently. */
function actionText(t) {
  return t.agent ? `agent ${t.agent} ${t.action}` : t.action;
}

// ---------- datasets ----------
function episodeMap(exp) {
  const by = new Map();
  for (const t of exp.transitions) {
    if (!by.has(t.episode)) by.set(t.episode, []);
    by.get(t.episode).push(t);
  }
  return by;
}

/** y for a step: explicit ok wins; else success episodes label every step 0, failed ones their last 3 steps 1. */
function stepLabel(t, idx, len, outcome) {
  if (t.ok === true) return 0;
  if (t.ok === false) return 1;
  if (outcome === "success") return 0;
  if (outcome === "failure" && idx >= len - 3) return 1;
  return undefined;
}

function lowSamples(exp) {
  const out = [];
  for (const [id, steps] of episodeMap(exp)) {
    const outcome = exp.episodes[id] && exp.episodes[id].outcome;
    steps.forEach((t, i) => out.push({ s: t.state, a: actionText(t), s2: t.result, y: stepLabel(t, i, steps.length, outcome) }));
  }
  return out;
}

/** Level-2 chunks: consecutive steps with the same intent, or CHUNK steps when there is none. */
function chunks(steps) {
  const out = [];
  let cur = null;
  for (const t of steps) {
    const key = t.intent || null;
    if (!cur || cur.key !== key || (key === null && cur.steps.length >= CHUNK)) {
      cur = { key, steps: [] };
      out.push(cur);
    }
    cur.steps.push(t);
  }
  return out;
}

function highSamples(exp) {
  const out = [];
  for (const [id, steps] of episodeMap(exp)) {
    const outcome = exp.episodes[id] && exp.episodes[id].outcome;
    const cs = chunks(steps);
    cs.forEach((c, i) => {
      const first = c.steps[0];
      const last = c.steps[c.steps.length - 1];
      let y;
      if (c.steps.some((t) => t.ok === false)) y = 1;
      else if (outcome === "success") y = 0;
      else if (outcome === "failure" && i === cs.length - 1) y = 1;
      else if (c.steps.every((t) => t.ok === true)) y = 0;
      const intent = c.key || c.steps.map(actionText).join(" then ");
      out.push({ s: first.state, intent: first.agent && c.key ? `agent ${first.agent} ${intent}` : intent, s2: last.result, y });
    });
  }
  return out;
}

function labeledCount(exp) {
  return lowSamples(exp).filter((x) => x.y !== undefined).length;
}

/** Beta(1,1) prior per agent over step outcomes. */
function agentStats(exp) {
  const stats = {};
  for (const t of exp.transitions) {
    if (!t.agent) continue;
    const a = (stats[t.agent] = stats[t.agent] || { steps: 0, ok: 0, failed: 0 });
    a.steps += 1;
    if (t.ok === true) a.ok += 1;
    if (t.ok === false) a.failed += 1;
  }
  for (const a of Object.values(stats)) a.failureRate = round((a.failed + 1) / (a.ok + a.failed + 2));
  return stats;
}

// ---------- training ----------
function shuffled(arr, rand) {
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function pick(arr, n, rand, must) {
  const out = must ? [must] : [];
  for (let i = out.length; i < n && arr.length; i++) out.push(arr[Math.floor(rand() * arr.length)]);
  return out;
}

function avgStats(list) {
  const out = {};
  if (!list.length) return out;
  for (const k of Object.keys(list[0])) out[k] = round(list.reduce((s, x) => s + x[k], 0) / list.length, 5);
  return out;
}

// ---------- tools ----------
async function wmObserve(args = {}) {
  return serial(async () => {
    const episode = text(args.episode).trim();
    if (!episode) return { error: "episode is required" };
    const cfg = await settings();
    const exp = await loadExperience();
    const ep = (exp.episodes[episode] = exp.episodes[episode] || { started: new Date().toISOString() });
    if (args.goal) ep.goal = text(args.goal);
    const hasStep = args.state != null && args.action != null && args.result != null;
    if (!hasStep && !args.outcome) return { error: "give state, action and result for a step, or outcome to end the episode" };

    const model = await loadModel();
    const out = { episode };
    let newSample = null;
    if (hasStep) {
      const t = {
        episode,
        t: exp.transitions.filter((x) => x.episode === episode).length,
        state: text(args.state),
        action: text(args.action),
        result: text(args.result),
      };
      if (args.agent) t.agent = text(args.agent).slice(0, 80);
      if (args.intent) t.intent = text(args.intent).slice(0, 200);
      if (typeof args.ok === "boolean") t.ok = args.ok;
      const cold = labeledCount(exp) < COLD_START;
      const sp = model.surprise(t.state, actionText(t), t.result);
      exp.transitions.push(t);
      if (exp.transitions.length > cfg.maxTransitions) exp.transitions.splice(0, exp.transitions.length - cfg.maxTransitions);
      newSample = { s: t.state, a: actionText(t), s2: t.result, y: t.ok === undefined ? undefined : t.ok ? 0 : 1 };
      out.surprise = round(sp.surprise, 2);
      out.uncertainty = round(sp.uncertainty, 2);
      out.predictedFailure = round(sp.failure);
      out.coldStart = cold || model.steps === 0;
      if (t.ok === false) out.advice = "replan: the step failed; record why in the next state and call wm_plan before retrying";
      else if (!out.coldStart && sp.surprise > cfg.surpriseThreshold)
        out.advice = "replan: the result diverged from what the world model expected; re-check assumptions and call wm_plan again";
      else out.advice = "continue";
    }
    if (args.outcome === "success" || args.outcome === "failure") {
      ep.outcome = args.outcome;
      ep.ended = new Date().toISOString();
      out.ended = args.outcome;
    }

    // online learning: a few small batches that always include the newest step
    const steps = args.outcome ? Math.max(cfg.onlineSteps, 4) : cfg.onlineSteps;
    if (steps > 0 && exp.transitions.length) {
      const low = lowSamples(exp);
      const high = highSamples(exp);
      const rand = Math.random;
      const st = [];
      for (let i = 0; i < steps; i++) st.push(model.trainBatch(pick(low, BATCH, rand, newSample), pick(high, Math.min(BATCH, high.length), rand)));
      out.trained = { batches: steps, loss: avgStats(st) };
      await saveModel(model);
    }
    await saveExperience(exp);
    out.transitions = exp.transitions.length;
    return out;
  });
}

function normOption(o, i) {
  if (typeof o === "string") return { label: o };
  if (!o || typeof o !== "object") return null;
  const label = text(o.label || o.intent || (Array.isArray(o.steps) && o.steps.join(" then ")) || `option ${i + 1}`);
  return {
    label,
    agent: o.agent ? text(o.agent).slice(0, 80) : undefined,
    intent: o.intent ? text(o.intent).slice(0, 200) : undefined,
    steps: Array.isArray(o.steps) ? o.steps.slice(0, 12).map(text) : undefined,
  };
}

async function wmPlan(args = {}) {
  return serial(async () => {
    if (args.state == null) return { error: "state is required" };
    const options = (Array.isArray(args.options) ? args.options : []).map(normOption).filter(Boolean).slice(0, 12);
    if (!options.length) return { error: "give 1-12 options" };
    const cfg = await settings();
    const risk = typeof args.riskAversion === "number" ? args.riskAversion : cfg.riskAversion;
    const exp = await loadExperience();
    const model = await loadModel();
    const labeled = labeledCount(exp);
    const trust = model.steps === 0 ? 0 : Math.min(1, labeled / (COLD_START * 2.5));
    const agents = agentStats(exp);
    const state = text(args.state);
    const goal = args.goal ? text(args.goal) : undefined;

    const ranked = options.map((o) => {
      const prefix = o.agent ? `agent ${o.agent} ` : "";
      const im = model.imagine(
        state,
        { label: prefix + o.label, intent: o.intent ? prefix + o.intent : undefined, steps: o.steps && o.steps.map((s) => prefix + s) },
        goal,
      );
      const modelFailure = (im.failure1 + im.failure2) / 2;
      const prior = o.agent && agents[o.agent] ? agents[o.agent].failureRate : 0.5;
      const failure = trust * modelFailure + (1 - trust) * prior;
      const uncertainty = (im.uncertainty1 + im.uncertainty2) / 2;
      const goalDistance = goal ? (im.goal1 + im.goal2) / 2 : 0;
      const score = failure + cfg.goalWeight * goalDistance + risk * Math.max(0, uncertainty - 1);
      return {
        label: o.label,
        agent: o.agent,
        score: round(score),
        failure: round(failure),
        goalDistance: goal ? round(goalDistance) : undefined,
        uncertainty: round(uncertainty, 2),
        levels: {
          step: { failure: round(im.failure1), uncertainty: round(im.uncertainty1, 2), goal: goal ? round(im.goal1) : undefined },
          intent: { failure: round(im.failure2), uncertainty: round(im.uncertainty2, 2), goal: goal ? round(im.goal2) : undefined },
        },
        agentPrior: o.agent && agents[o.agent] ? agents[o.agent] : undefined,
        trace: o.steps ? im.trace : undefined,
      };
    });
    ranked.sort((a, b) => a.score - b.score);
    const best = ranked[0];
    let advice;
    if (trust < 0.4) advice = "cold start: the ranking is mostly priors; prefer the cheapest informative option and record the outcome";
    else if (best.uncertainty > 1.5) advice = "probe: the world model is unsure here; gather information first (e.g. the world-model-scout sub-agent), observe it, then plan again";
    else if (ranked.length > 1 && ranked[1].score - best.score < 0.05) advice = "close call: pick by cost or ask the user";
    else advice = `go: ${best.label}`;
    return { recommended: best.label, advice, trust: round(trust, 2), ranked };
  });
}

async function wmTrain(args = {}) {
  return serial(async () => {
    const epochs = clampInt(typeof args.epochs === "number" ? args.epochs : 5, 1, 50);
    const exp = await loadExperience();
    const low = lowSamples(exp);
    const high = highSamples(exp);
    if (!low.length) return { error: "no experience yet: record steps with wm_observe first" };
    const model = await loadModel();
    const started = Date.now();
    const perEpoch = [];
    let stopped = false;
    for (let e = 0; e < epochs && !stopped; e++) {
      const L = shuffled(low, Math.random);
      const Hs = shuffled(high, Math.random);
      const nb = Math.ceil(L.length / BATCH);
      const st = [];
      for (let b = 0; b < nb; b++) {
        const hb = Hs.length ? Hs.slice(((b * BATCH) % Hs.length), ((b * BATCH) % Hs.length) + BATCH) : [];
        st.push(model.trainBatch(L.slice(b * BATCH, (b + 1) * BATCH), hb));
        if (Date.now() - started > TRAIN_BUDGET_MS) {
          stopped = true;
          break;
        }
      }
      perEpoch.push(avgStats(st));
    }
    await saveModel(model);
    return {
      epochs: perEpoch.length,
      stoppedEarly: stopped,
      samples: { step: low.length, intent: high.length, labeled: low.filter((x) => x.y !== undefined).length },
      first: perEpoch[0],
      last: perEpoch[perEpoch.length - 1],
      ms: Date.now() - started,
    };
  });
}

async function wmStatus(args = {}) {
  return serial(async () => {
    if (args.reset === "model" || args.reset === "all") {
      await saveModel(new HJEPA(1));
      if (args.reset === "all") await saveExperience({ version: 1, transitions: [], episodes: {} });
    }
    const exp = await loadExperience();
    const model = await loadModel();
    const eps = Object.values(exp.episodes);
    const labeled = labeledCount(exp);
    return {
      reset: args.reset || undefined,
      transitions: exp.transitions.length,
      labeled,
      coldStart: labeled < COLD_START || model.steps === 0,
      episodes: {
        total: eps.length,
        success: eps.filter((e) => e.outcome === "success").length,
        failure: eps.filter((e) => e.outcome === "failure").length,
        open: eps.filter((e) => !e.outcome).length,
      },
      agents: agentStats(exp),
      model: { steps: model.steps, dims: model.dims, errorScale: { step: round(model.err1, 5), intent: round(model.err2, 5) } },
      files: [EXP_FILE, MODEL_FILE],
      subagents: subagentStatus,
      officialHjepa: await hjepaSummary(),
      settings: await settings(),
    };
  });
}

// ---------- official H-JEPA release (huggingface.co/jepa-world-models/h-jepa) ----------
/** Catalog of the pinned release; sizes in MB per checkpoint run. Kept equal to scripts/install-hjepa.sh (tested). */
const RELEASE = {
  model: "https://huggingface.co/jepa-world-models/h-jepa",
  code: "https://github.com/kevinghst/H-JEPA",
  data: "https://huggingface.co/datasets/jepa-world-models/h-jepa",
  sim: { models: ["lewm", "hjepa_l2", "hjepa_l3", "hjepa_l4", "hwm_l2", "hwm_l3", "hwm_l4"], seeds: ["42", "43", "44"], model: "hjepa_l3", seed: "42" },
  droid: { models: ["lewm", "hjepa_l2", "hwm_l2"], seeds: ["1", "1000", "10000"], model: "hjepa_l2", seed: "1" },
  envs: ["ant", "fourroom", "cube", "pusht", "droid"],
  mb: {
    ant: { lewm: 50, hjepa_l2: 95, hjepa_l3: 140, hjepa_l4: 185, hwm_l2: 93, hwm_l3: 137, hwm_l4: 181 },
    fourroom: { lewm: 59, hjepa_l2: 66, hjepa_l3: 87, hjepa_l4: 108, hwm_l2: 66, hwm_l3: 86, hwm_l4: 106 },
    cube: { lewm: 45, hjepa_l2: 93, hjepa_l3: 141, hjepa_l4: 189, hwm_l2: 93, hwm_l3: 140, hwm_l4: 187 },
    pusht: { lewm: 45, hjepa_l2: 93, hjepa_l3: 141, hjepa_l4: 189, hwm_l2: 93, hwm_l3: 140, hwm_l4: 187 },
    droid: { lewm: 216, hjepa_l2: 348, hwm_l2: 345 },
  },
  tasksMb: { ant: 155, fourroom: 574, cube: 152, pusht: 565, droid: 252 },
};

const list = (v) => (Array.isArray(v) ? v : String(v || "").split(",")).map((x) => String(x).trim()).filter(Boolean);
const shq = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;

/** Expand and validate a selection the same way the script does. Returns { runs, mb } or { error }. */
function hjepaSelection(args) {
  const envArg = list(args.env || "ant");
  const envs = envArg.length === 1 && envArg[0] === "all" ? RELEASE.envs : envArg.length === 1 && envArg[0] === "sim" ? RELEASE.envs.slice(0, 4) : envArg;
  const runs = [];
  let mb = 0;
  for (const e of envs) {
    if (!RELEASE.envs.includes(e)) return { error: `unknown environment '${e}' (ant, fourroom, cube, pusht, droid, sim, all)` };
    const cat = e === "droid" ? RELEASE.droid : RELEASE.sim;
    const ms = args.models == null || args.models === "default" ? [cat.model] : list(args.models)[0] === "all" ? cat.models : list(args.models);
    const ss = args.seeds == null || args.seeds === "default" ? [cat.seed] : list(args.seeds)[0] === "all" ? cat.seeds : list(args.seeds).map(String);
    for (const m of ms) {
      if (!cat.models.includes(m)) return { error: `model '${m}' does not exist for ${e} (have: ${cat.models.join(", ")})` };
      for (const sd of ss) {
        if (!cat.seeds.includes(sd)) return { error: `seed '${sd}' does not exist for ${e} (have: ${cat.seeds.join(", ")})` };
        runs.push(`${e}/${m}/${sd}`);
        mb += RELEASE.mb[e][m];
      }
    }
    if (!args.skipTasks) mb += RELEASE.tasksMb[e];
  }
  return { envs, runs, mb };
}

/** Planner config for a checkpoint, following the release README (section 5). */
function hjepaPlanner(run) {
  const [env, model] = run.split("/");
  const m = /^(?:hjepa|hwm)_(l\d)$/.exec(model);
  return `${env}_${m ? m[1] : "flat"}`;
}

async function readHjepaState() {
  return readJSON(HJEPA_STATE);
}

async function hjepaSummary() {
  const st = await readHjepaState();
  const run = await readJSON(HJEPA_LAST_RUN);
  const lastRun = run ? { runId: run.runId, status: run.status, metric: run.metric, score: run.score, count: run.count, device: run.device } : undefined;
  const compute = (await computeSettings()).target;
  const picked = await modelSettings();
  const model = picked.error ? { error: picked.error } : { env: picked.env, model: picked.model, seed: picked.seed, tasks: picked.tasks || "all" };
  return st ? { status: st.status, checked: st.checked, runs: st.runs, model, compute, lastRun } : { status: "not installed", model, compute, lastRun };
}

async function wmHjepa(args = {}) {
  return serial(async () => {
    const action = args.action || "plan";
    if (action === "status") {
      const st = await readHjepaState();
      const lastRun = await readJSON(HJEPA_LAST_RUN);
      const compute = await computeSettings();
      const model = await modelSettings();
      if (!st) return { status: "not installed", model, compute, lastRun: lastRun || undefined, next: "call wm_hjepa {action:'run'}: it installs what the selected model needs, then evaluates it" };
      return {
        ...st,
        model,
        compute,
        lastRun: lastRun || undefined,
        next: "call wm_hjepa {action:'run'} for the evaluation command on the configured compute target",
        recheck: `bash ${HJEPA_SCRIPT} --check --state ${HJEPA_STATE}${st.home ? ` --home ${shq(st.home)}` : ""}${st.codeDir ? ` --code-dir ${shq(st.codeDir)}` : ""}${selectionFlags(st.runs)}`,
      };
    }
    if (action === "run") return hjepaRun(args);
    if (action !== "plan") return { error: "action must be plan, run or status" };
    if (args.env == null && args.models == null && args.seeds == null) {
      // No selection given: install the world model picked in the settings.
      const picked = await modelSettings();
      if (picked.error) return { error: picked.error };
      args = { ...args, env: picked.env, models: picked.model, seeds: picked.seed };
    }
    const sel = hjepaSelection(args);
    if (sel.error) return { error: sel.error };
    if (args.python && !["auto", "conda", "uv", "venv"].includes(args.python)) return { error: "python must be auto, conda, uv or venv" };
    const script = fs.readFileSync(path.join(__dirname, "scripts", "install-hjepa.sh"), "utf8");
    await pi.fs.writeText(HJEPA_SCRIPT, script);

    const flags = [];
    flags.push(`--env ${shq(sel.envs.join(","))}`);
    if (args.models != null && args.models !== "default") flags.push(`--models ${shq(list(args.models).join(","))}`);
    if (args.seeds != null && args.seeds !== "default") flags.push(`--seeds ${shq(list(args.seeds).join(","))}`);
    if (args.home) flags.push(`--home ${shq(args.home)}`);
    if (args.codeDir) flags.push(`--code-dir ${shq(args.codeDir)}`);
    if (args.python) flags.push(`--python ${args.python}`);
    if (args.skipCode) flags.push("--skip-code");
    if (args.skipTasks) flags.push("--skip-tasks");
    flags.push(`--state ${HJEPA_STATE}`);
    const base = `bash ${HJEPA_SCRIPT} ${flags.join(" ")}`;
    return {
      staged: HJEPA_SCRIPT,
      selection: sel.runs,
      downloadMB: sel.mb,
      commands: {
        dryRun: `${base} --dry-run`,
        install: `${base} > ${HJEPA_LOG} 2>&1`,
        check: `${base} --check`,
      },
      evalPlanner: Object.fromEntries(sel.runs.map((r) => [r, hjepaPlanner(r)])),
      requirements: [
        "git, and one of conda / uv / python3.11 (the release requires Python 3.11)",
        "network access to github.com, huggingface.co and PyPI",
        `disk: ~${sel.mb} MB of downloads plus the Python environment (torch, MuJoCo, JAX…)`,
      ],
      notes: [
        "Pinned: code 97054ec, checkpoints 08fc8a2, eval data cc05cb8. Code and checkpoints MIT, data CC BY 4.0 (cite DROID for droid/*).",
        "The checkpoints are pickled modules loaded with torch.load(weights_only=False), as in the release README: they execute code, so only load these pinned files.",
        "The release targets NVIDIA GPUs (CUDA 12.8 wheels on Linux). On macOS arm64, decord (a video reader in [train]) has no wheel and is skipped; video code such as DROID may need it.",
        "eval.py needs CUDA as released. Run evaluations with wm_hjepa {action:'run'}: it uses the 'H-JEPA compute' setting (this computer's GPU through hjepa_local.py, or a remote NVIDIA GPU over SSH).",
        "These are visual robot-planning world models (pixels + actions in AntMaze, FourRoom, Cube, Push-T, DROID). They do not read agent trajectories: wm_observe / wm_plan keep using the plugin's own text H-JEPA.",
        "Training datasets (tens of GB) are not downloaded; see the release README sections 3.1 and 4.1.",
      ],
      links: { model: RELEASE.model, code: RELEASE.code, data: RELEASE.data },
    };
  });
}

// ---------- running evaluations: this computer's GPU or a remote NVIDIA GPU ----------
const COMPUTE_TARGETS = ["local", "remote"];
const LOCAL_DEVICES = ["auto", "mps", "cpu"];
const HOST_RE = /^[A-Za-z0-9_.@:][A-Za-z0-9_.@:-]*$/;
const REMOTE_HOME_RE = /^(~\/)?[A-Za-z0-9_./][A-Za-z0-9_./-]*$/;
const OUT_RE = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;
const OVERRIDE_RE = /^[A-Za-z0-9_.=+/:,-]+$/;
const RUN_FILES = ["hjepa-run.sh", "hjepa_local.py", "install-hjepa.sh"];

/** The compute settings: target local|remote, local device, remote host and home. */
async function computeSettings() {
  const s = await rawSettings();
  return {
    target: COMPUTE_TARGETS.includes(s.hjepaCompute) ? s.hjepaCompute : "local",
    device: LOCAL_DEVICES.includes(s.hjepaLocalDevice) ? s.hjepaLocalDevice : "auto",
    host: typeof s.hjepaRemoteHost === "string" ? s.hjepaRemoteHost.trim() : "",
    remoteHome: typeof s.hjepaRemoteHome === "string" && s.hjepaRemoteHome.trim() ? s.hjepaRemoteHome.trim() : "hjepa",
  };
}

async function rawSettings() {
  try {
    return (await pi.plugin.getSettings()) || {};
  } catch {
    return {};
  }
}

const SEED_SLOTS = { first: 0, second: 1, third: 2 };

/**
 * The world model the user picked in the settings: environment, model ("auto" = the release's
 * recommended one for that environment), training seed slot and evaluation size (0 = all).
 */
async function modelSettings() {
  const s = await rawSettings();
  const env = RELEASE.envs.includes(s.hjepaEnv) ? s.hjepaEnv : "ant";
  const cat = env === "droid" ? RELEASE.droid : RELEASE.sim;
  const model = typeof s.hjepaModel === "string" && s.hjepaModel !== "auto" ? s.hjepaModel : cat.model;
  const seed = cat.seeds[SEED_SLOTS[s.hjepaSeed] ?? 0];
  const tasks = typeof s.hjepaTasks === "number" && Number.isInteger(s.hjepaTasks) && s.hjepaTasks > 0 ? s.hjepaTasks : null;
  const out = { env, model, seed, tasks };
  if (!cat.models.includes(model)) {
    out.error = `the 'H-JEPA model' setting (${model}) is not trained for ${env}: ${env} has ${cat.models.join(", ")}. Pick one of those, or 'Recommended'.`;
  }
  return out;
}

/** Measured seconds per task/clip on an Apple M2 GPU (mps), by eval config. */
const MEASURED_M2 = { ant_l3: 656, droid_l2: 350 };

async function hjepaRun(args) {
  const cfg = await computeSettings();
  const target = args.compute || cfg.target;
  if (!COMPUTE_TARGETS.includes(target)) return { error: "compute must be local or remote" };

  // What to evaluate: the arguments if given, else the 'H-JEPA environment / model / seed' settings.
  const picked = await modelSettings();
  const fromSettings = args.env == null;
  if (fromSettings && picked.error && args.model == null) return { error: picked.error };
  const envs = list(args.env || picked.env);
  if (envs.length !== 1 || ["all", "sim"].includes(envs[0])) return { error: "run evaluates one checkpoint: give one env" };
  const models = args.model != null ? list(args.model) : args.models != null ? list(args.models) : fromSettings ? [picked.model] : [];
  const seeds = args.seed != null ? list(String(args.seed)) : args.seeds != null ? list(String(args.seeds)) : fromSettings ? [picked.seed] : [];
  if (models.length > 1 || seeds.length > 1 || models[0] === "all" || seeds[0] === "all") return { error: "run evaluates one checkpoint: give one model and one seed" };
  const sel = hjepaSelection({ env: envs[0], models: models[0] || "default", seeds: seeds[0] || "default" });
  if (sel.error) return { error: sel.error };
  const [env, m, sd] = sel.runs[0].split("/");
  const droid = env === "droid";
  const max = droid ? 16 : 50;
  const unit = droid ? "clip" : "task";
  const tasks = args.tasks != null ? Number(args.tasks) : fromSettings ? picked.tasks : null;
  if (tasks != null && !(Number.isInteger(tasks) && tasks >= 1 && tasks <= max)) {
    return { error: `tasks must be an integer from 1 to ${max} for ${env} (${droid ? "16 DROID clips" : "50 tasks"}); 0 in the 'Evaluation size' setting means all` };
  }
  const planSeed = args.planSeed != null ? Number(args.planSeed) : 1;
  if (droid && !(Number.isInteger(planSeed) && planSeed >= 0)) return { error: "planSeed must be a non-negative integer" };
  if (args.out != null && !OUT_RE.test(args.out)) return { error: "out may only contain letters, digits, '_', '-', '.'" };
  const overrides = Array.isArray(args.overrides) ? args.overrides.map(String) : [];
  const bad = overrides.find((o) => !OVERRIDE_RE.test(o));
  if (bad) return { error: `override '${bad}' contains unsupported characters` };
  const device = args.device || cfg.device;
  if (!LOCAL_DEVICES.includes(device)) return { error: "device must be auto, mps or cpu" };
  const host = args.host || cfg.host;
  const remoteHome = args.remoteHome || cfg.remoteHome;
  if (target === "remote") {
    if (!host) return { error: "no remote host: set the plugin setting 'Remote GPU host' (an SSH destination reachable with a key, e.g. user@gpu-box or an ~/.ssh/config alias), or pass host" };
    if (!HOST_RE.test(host)) return { error: "host may only contain letters, digits and _ . @ : -" };
    if (!REMOTE_HOME_RE.test(remoteHome) || remoteHome.includes("..")) return { error: "remote home must be a plain path (letters, digits, _ . / -)" };
  }
  if (args.python && !["auto", "conda", "uv", "venv"].includes(args.python)) return { error: "python must be auto, conda, uv or venv" };

  for (const f of RUN_FILES) await pi.fs.writeText(`${HJEPA_DIR}/${f}`, fs.readFileSync(path.join(__dirname, "scripts", f), "utf8"));

  const out = args.out || `eval_pi_${target}${tasks ? `_n${tasks}` : ""}`;
  const runId = `${target}-${env}_${m}-s${sd}-${droid ? `p${planSeed}-` : ""}${out}`;
  const setup = args.setup !== false;
  const flags = [`--target ${target}`, `--env ${env}`, `--model ${m}`, `--seed ${sd}`];
  if (droid) flags.push(`--plan-seed ${planSeed}`);
  if (tasks) flags.push(`--tasks ${tasks}`);
  if (args.out) flags.push(`--out ${out}`);
  if (target === "local") {
    flags.push(`--device ${device}`);
    if (args.home) flags.push(`--home ${shq(args.home)}`);
  } else {
    flags.push(`--host ${host}`, `--remote-home ${shq(remoteHome.replace(/^~\//, ""))}`);
  }
  if (setup) flags.push("--setup");
  if (args.python) flags.push(`--python ${args.python}`);
  const base = `bash ${HJEPA_DIR}/hjepa-run.sh ${flags.join(" ")}${overrides.length ? ` -- ${overrides.join(" ")}` : ""}`;
  const log = `${HJEPA_DIR}/run-${runId}.log`;
  const n = tasks || max;
  const config = `${env}_${hjepaPlanner(sel.runs[0]).split("_")[1]}`;
  const perUnit = MEASURED_M2[config];
  const estimate =
    target === "remote"
      ? "Speed depends on the remote GPU. The first run also installs what is missing there."
      : perUnit
        ? `Measured on an Apple M2 GPU (mps): ~${Math.round(perUnit / 60)} min per ${unit} of ${config}, so ~${Math.round((n * perUnit) / 360) / 10} h for ${n} ${unit}(s). cpu is slower; other machines differ.`
        : `Not measured for ${config} yet: run 1 ${unit} first (tasks: 1) to time it. For scale, ant_l3 took ~11 min per task on an Apple M2 GPU.`;
  return {
    target,
    where: target === "local" ? `this computer (device ${device}; an NVIDIA GPU is used directly when present)` : `${host} over SSH (remote home ${remoteHome})`,
    selection: { source: fromSettings && args.model == null && args.seed == null ? "settings" : "arguments", env, model: m, seed: sd, planSeed: droid ? planSeed : undefined },
    config,
    checkpoint: sel.runs[0],
    metric: droid ? "Frechet fidelity (% match of the planned to the recorded robot path; 0 = no motion, 100 = exact)" : "success rate (% of tasks where the planner reached the goal)",
    count: n,
    unit,
    runId,
    staged: RUN_FILES.map((f) => `${HJEPA_DIR}/${f}`),
    setup: setup ? "installs whatever this run needs and is missing first (code, Python environment, checkpoint, eval tasks or DROID clips)" : "off: fails if something is missing",
    commands: { dryRun: `${base} --dry-run`, run: `${base} > ${log} 2>&1` },
    log,
    results: `${HJEPA_DIR}/results/${runId}/`,
    record: HJEPA_LAST_RUN,
    background: true,
    estimate,
    notes:
      target === "local"
        ? [
            "No NVIDIA GPU: eval.py runs through hjepa_local.py, which sends 'cuda' to mps/cpu, casts float64 to float32 on mps and keeps MUJOCO_GL=glfw on macOS. The clone is not modified.",
            "Results are close to, not identical with, the paper's CUDA runs.",
            droid ? "DROID on macOS arm64 reads its videos with decord2 (or eva-decord), installed in place of decord, which has no Apple Silicon wheel." : undefined,
          ].filter(Boolean)
        : [
            "Needs key-based SSH (BatchMode: no password prompt), Linux with a working nvidia-smi, git, and conda / uv / python3.11 on the host.",
            "eval.py runs unchanged there (the paper's setup). Results are copied back over scp.",
            "If the SSH connection drops, rerun the same command: completed tasks are skipped.",
          ],
  };
}

function selectionFlags(runs) {
  if (!Array.isArray(runs) || !runs.length) return "";
  const envs = [...new Set(runs.map((r) => r.split("/")[0]))];
  const models = [...new Set(runs.map((r) => r.split("/")[1]))];
  const seeds = [...new Set(runs.map((r) => r.split("/")[2]))];
  return ` --env ${shq(envs.join(","))} --models ${shq(models.join(","))} --seeds ${shq(seeds.join(","))}`;
}

// ---------- sub-agent install ----------
const subagentSource = () => path.join(__dirname, "subagents");
const subagentTarget = () => path.join(os.homedir(), ".agents", "subagents");

function installSubagents() {
  const dst = subagentTarget();
  fs.mkdirSync(dst, { recursive: true });
  const changed = [];
  for (const name of SUBAGENTS) {
    const body = fs.readFileSync(path.join(subagentSource(), name));
    const target = path.join(dst, name);
    let current = null;
    try {
      current = fs.readFileSync(target);
    } catch {
      // missing
    }
    if (!current || (!current.equals(body) && current.toString("utf8").includes(MARKER))) {
      fs.writeFileSync(target, body);
      changed.push(name);
    }
  }
  return changed;
}

/** Remove the sub-agents this plugin installed, unless the user edited them. */
function removeSubagents() {
  for (const name of SUBAGENTS) {
    try {
      const target = path.join(subagentTarget(), name);
      if (fs.readFileSync(target).equals(fs.readFileSync(path.join(subagentSource(), name)))) fs.rmSync(target, { force: true });
    } catch {
      // already gone
    }
  }
}

// ---------- registration ----------
const EPISODE = { type: "string", description: "Episode id: one per task, e.g. fix-login-bug-1" };

const TOOLS = [
  {
    name: "wm_observe",
    description:
      "World model: record one step of an agent or sub-agent (state before, action, result after, ok) or end an episode (outcome). Learns online and returns surprise and advice (continue / replan). Use from the world-model skill.",
    risk: "low",
    schema: {
      type: "object",
      properties: {
        episode: EPISODE,
        goal: { type: "string", description: "The episode goal, once, short" },
        state: { type: "string", description: "Compact state before the step (≤600 chars): what is known, what is failing" },
        action: { type: "string", description: "What was done, e.g. 'run tests', 'delegate fix to fixer'" },
        result: { type: "string", description: "Compact state after the step (≤600 chars)" },
        agent: { type: "string", description: "Who acted: 'main' or the sub-agent name, e.g. explorer" },
        intent: { type: "string", description: "The subgoal this step serves; consecutive steps with one intent form a level-2 chunk" },
        ok: { type: "boolean", description: "Did the step achieve what it was for" },
        outcome: { type: "string", enum: ["success", "failure"], description: "End the episode with this outcome" },
      },
      required: ["episode"],
    },
    execute: wmObserve,
  },
  {
    name: "wm_plan",
    description:
      "World model: imagine 1-12 candidate actions, plans or sub-agent delegations from the current state and rank them by predicted failure, distance to the goal and uncertainty (H-JEPA, step and intent level). Use from the world-model skill.",
    risk: "low",
    schema: {
      type: "object",
      properties: {
        state: { type: "string", description: "Compact current state (≤600 chars)" },
        goal: { type: "string", description: "Desired end state, phrased like a state" },
        options: {
          type: "array",
          description: "1-12 options: { label, agent?, intent?, steps? }",
          items: {
            type: "object",
            properties: {
              label: { type: "string", description: "Short name of the option, e.g. 'delegate the fix to fixer'" },
              agent: { type: "string", description: "Who would act: 'main' or a sub-agent name" },
              intent: { type: "string", description: "The subgoal it serves (level 2)" },
              steps: { type: "array", items: { type: "string" }, description: "Concrete steps to roll out (level 1), ≤12" },
            },
            required: ["label"],
          },
        },
        riskAversion: { type: "number", description: "Weight on uncertainty; default from settings" },
      },
      required: ["state", "options"],
    },
    execute: wmPlan,
  },
  {
    name: "wm_train",
    description: "World model: train the H-JEPA on all recorded experience for a few epochs (bounded to ~80 s). Use from the world-model skill.",
    risk: "low",
    schema: { type: "object", properties: { epochs: { type: "integer", description: "1-50, default 5" } } },
    execute: wmTrain,
  },
  {
    name: "wm_status",
    description: "World model: experience and episode counts, per-agent reliability, model state; reset 'model' or 'all'. Use from the world-model skill.",
    risk: "low",
    schema: { type: "object", properties: { reset: { type: "string", enum: ["model", "all"], description: "Reset the weights, or the weights and the experience" } } },
    execute: wmStatus,
  },
  {
    name: "wm_hjepa",
    description:
      "World model: install and run the official H-JEPA release (huggingface.co/jepa-world-models/h-jepa). plan stages the pinned installer; run returns an evaluation command for this computer's GPU or a remote NVIDIA GPU (setting); status. Use from the hjepa-install skill.",
    risk: "low",
    schema: {
      type: "object",
      properties: {
        action: { type: "string", enum: ["plan", "run", "status"], description: "plan (default): stage the installer; run: stage and return an evaluation command for the configured compute target; status" },
        compute: { type: "string", enum: ["local", "remote"], description: "run: override the 'H-JEPA compute' setting" },
        model: { type: "string", description: "run: one model, e.g. hjepa_l3 (default: the 'H-JEPA model' setting)" },
        seed: { type: "string", description: "run: one training seed, e.g. 42 or 1000 (default: the 'H-JEPA training seed' setting)" },
        planSeed: { type: "integer", description: "run, droid: planner seed (the paper uses 1, 2, 3); default 1" },
        tasks: { type: "integer", description: "run: the first N tasks (of 50) or DROID clips (of 16); default: the 'Evaluation size' setting (0 = all)" },
        out: { type: "string", description: "run: output.dir name; rerunning the same one resumes" },
        device: { type: "string", enum: ["auto", "mps", "cpu"], description: "run, local: override the 'Local device' setting" },
        host: { type: "string", description: "run, remote: override the 'Remote GPU host' setting" },
        remoteHome: { type: "string", description: "run, remote: override the 'Remote HJEPA_HOME' setting" },
        setup: { type: "boolean", description: "run: install whatever the run needs and is missing first, locally or on the host (default true)" },
        overrides: { type: "array", items: { type: "string" }, description: "run: extra hydra overrides, e.g. eval.chunk_size=5" },
        env: { type: "string", description: "ant, fourroom, cube, pusht, droid (plan: comma-separated, sim or all; run: one). Default: the 'H-JEPA environment' setting" },
        models: { type: "string", description: "lewm, hjepa_l2..l4, hwm_l2..l4 (droid: lewm, hjepa_l2, hwm_l2), comma-separated, or all; default hjepa_l3 (droid hjepa_l2)" },
        seeds: { type: "string", description: "42, 43, 44 (droid: 1, 1000, 10000), comma-separated, or all; default 42 (droid 1)" },
        home: { type: "string", description: "HJEPA_HOME data root; default $HJEPA_HOME or ~/hjepa" },
        codeDir: { type: "string", description: "Where to clone the code; default <home>/H-JEPA" },
        python: { type: "string", enum: ["auto", "conda", "uv", "venv"], description: "Environment manager; default auto" },
        skipCode: { type: "boolean", description: "Only download checkpoints (code and environment already set up)" },
        skipTasks: { type: "boolean", description: "Do not download the evaluation tasks" },
      },
    },
    execute: wmHjepa,
  },
];

async function onLoad() {
  try {
    const changed = installSubagents();
    subagentStatus = changed.length ? `installed or updated: ${changed.join(", ")}` : "installed and up to date";
  } catch (e) {
    subagentStatus = `could not install to ${subagentTarget()}: ${e && e.message}`;
  }
  for (const tool of TOOLS) {
    await pi.agent.registerTool({
      name: tool.name,
      description: tool.description,
      risk: tool.risk,
      schema: tool.schema,
      execute: async (args, ctx) => {
        if (ctx && typeof ctx.log === "function") ctx.log(`${tool.name} called`);
        try {
          return await tool.execute(args || {}, ctx);
        } catch (e) {
          return { error: String((e && e.message) || e) };
        }
      },
    });
    registered.push(tool.name);
  }
}

async function onUnload() {
  for (const name of registered.splice(0)) {
    try {
      await pi.agent.unregisterTool(name);
    } catch {
      // the host drops plugin tools on unload anyway
    }
  }
  try {
    removeSubagents();
  } catch {
    // best effort
  }
}

module.exports = { onLoad, onUnload };
