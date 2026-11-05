"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { ROOT, setup } = require("./helpers.js");

const SCRIPTS = path.join(ROOT, "scripts");
const REAL_PY = path.join(os.homedir(), "hjepa", "env", "bin", "python");

function writeExec(file, body) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, body, { mode: 0o755 });
}

/** A fake installed release in `home`: a python that behaves like eval.py (writes metrics). */
function fakeRelease(home, { env = "ant", model = "hjepa_l3", seed = "42" } = {}) {
  writeExec(
    path.join(home, "env", "bin", "python"),
    `#!/bin/bash
[ "$1" != -c ] || exit 0   # import probes (decord)
printf '%s\\n' "$@" > "$HJEPA_HOME/argv.txt"
env | grep -E '^(HJEPA_|MUJOCO_GL)' | sort > "$HJEPA_HOME/env.txt"
pwd > "$HJEPA_HOME/cwd.txt"
out=""; pol=""
for a in "$@"; do case "$a" in output.dir=*) out="\${a#output.dir=}";; policy=*) pol="\${a#policy=}";; esac; done
d="$(dirname "$pol")/$out"; mkdir -p "$d"
printf 'success_rate: 50.0\\nevaluation_time: 12.5\\n' > "$d/metrics.yaml"
echo '{}' > "$d/planning_compute.json"; echo 'seed: 1' > "$d/eval_config.yaml"
printf 'ate/end_distance_xyz,frechet/skill_mean,n_episodes\\r\\n0.05,0.4321,1\\r\\n' > "$d/eval.csv"; echo 'seed: 1' > "$d/plan_config.yaml"   # eval.csv is CRLF, like the csv module writes
[ -z "\${FAKE_FAIL:-}" ] || exit 7
`,
  );
  const tasksDir = { cube: "ogbench" }[env] || env;
  fs.mkdirSync(path.join(home, "H-JEPA", "h_jepa", "assets", "eval_trajs", tasksDir), { recursive: true });
  fs.writeFileSync(path.join(home, "H-JEPA", "h_jepa", "assets", "eval_trajs", tasksDir, "tasks.pt"), "x");
  fs.mkdirSync(path.join(home, "droid", "droid_raw"), { recursive: true });
  const ck = path.join(home, "ckpts", env, `${env}_${model}`, `seed${seed}`);
  fs.mkdirSync(ck, { recursive: true });
  fs.writeFileSync(path.join(ck, `${env}_${model}_object.ckpt`), "x");
  fs.writeFileSync(path.join(ck, "normalizer.pt"), "x");
}

/** Fake ssh / scp: the "remote host" is the folder $FAKE_REMOTE. nvidia-smi is optional. */
function fakeRemote(base, { gpu = true } = {}) {
  const bin = path.join(base, "fakebin");
  const remote = path.join(base, "remote");
  fs.mkdirSync(remote, { recursive: true });
  writeExec(
    path.join(bin, "ssh"),
    `#!/bin/bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done
echo "$1" >> "$FAKE_REMOTE/hosts.txt"; shift
cd "$FAKE_REMOTE" && exec /bin/bash -c "$*"
`,
  );
  writeExec(
    path.join(bin, "scp"),
    `#!/bin/bash
ops=()
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) ops+=("$1"); shift;; esac; done
map() { case "$1" in *:*) printf '%s/%s' "$FAKE_REMOTE" "\${1#*:}";; *) printf '%s' "$1";; esac; }
cp -R "$(map "\${ops[0]}")" "$(map "\${ops[1]}")"
`,
  );
  if (gpu) writeExec(path.join(bin, "nvidia-smi"), "#!/bin/bash\necho 'GPU 0: NVIDIA Fake A100 (UUID: GPU-0)'\n");
  return { bin, remote };
}

function sh(dirs, line, env = {}) {
  const r = spawnSync("/bin/bash", ["-c", line], {
    cwd: dirs.workspace,
    env: { HOME: dirs.home, PATH: "/usr/bin:/bin", ...env },
    encoding: "utf8",
  });
  return { code: r.status, out: r.stdout + r.stderr };
}

const readJSON = (f) => JSON.parse(fs.readFileSync(f, "utf8"));

test("run defaults to this computer and stages the three scripts", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const r = await pi.call("wm_hjepa", { action: "run", tasks: 2 });
  assert.equal(r.target, "local");
  assert.equal(r.checkpoint, "ant/hjepa_l3/42");
  assert.equal(r.config, "ant_l3");
  assert.equal(r.runId, "local-ant_hjepa_l3-s42-eval_pi_local_n2");
  assert.equal(r.background, true);
  assert.match(r.estimate, /~11 min per task of ant_l3, so ~0\.4 h for 2 task/);
  assert.equal(r.selection.source, "settings");
  assert.equal(r.unit, "task");
  for (const f of ["hjepa-run.sh", "hjepa_local.py", "install-hjepa.sh"]) {
    assert.equal(fs.readFileSync(path.join(dirs.workspace, ".world-model/hjepa", f), "utf8"), fs.readFileSync(path.join(SCRIPTS, f), "utf8"));
  }
  assert.match(r.commands.run, /^bash \.world-model\/hjepa\/hjepa-run\.sh --target local --env ant --model hjepa_l3 --seed 42 --tasks 2 --device auto --setup > \.world-model\/hjepa\/run-local-ant_hjepa_l3-s42-eval_pi_local_n2\.log 2>&1$/);
});

test("run validates its input", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  const err = async (a) => (await pi.call("wm_hjepa", { action: "run", ...a })).error;
  assert.match(await err({ env: "ant,cube" }), /one env/);
  assert.match(await err({ env: "all" }), /one env/);
  assert.match(await err({ models: "lewm,hjepa_l2" }), /one model/);
  assert.match(await err({ env: "droid", model: "hjepa_l3" }), /does not exist for droid/);
  assert.match(await err({ tasks: 0 }), /1 to 50/);
  assert.match(await err({ tasks: 2.5 }), /1 to 50/);
  assert.match(await err({ out: "../x" }), /out may only/);
  assert.match(await err({ overrides: ["a;b"] }), /unsupported characters/);
  assert.match(await err({ device: "cuda" }), /auto, mps or cpu/);
  assert.match(await err({ compute: "cloud" }), /local or remote/);
  assert.match(await err({ compute: "remote" }), /Remote GPU host/);
  assert.match(await err({ compute: "remote", host: "-oProxyCommand=x" }), /host may only/);
  assert.match(await err({ compute: "remote", host: "a b" }), /host may only/);
  assert.match(await err({ compute: "remote", host: "gpu", remoteHome: "../etc" }), /plain path/);
});

test("the settings choose the target: remote host, remote home and local device", async (t) => {
  const { pi, restore } = await setup({ settings: { hjepaCompute: "remote", hjepaRemoteHost: "me@gpu-box", hjepaRemoteHome: "~/models/hj" } });
  t.after(restore);
  const r = await pi.call("wm_hjepa", { action: "run", env: "cube", model: "lewm", seed: "43", overrides: ["eval.chunk_size=5"] });
  assert.equal(r.target, "remote");
  assert.match(r.where, /me@gpu-box/);
  assert.equal(r.config, "cube_flat");
  assert.match(r.commands.run, /--target remote --env cube --model lewm --seed 43 --host me@gpu-box --remote-home 'models\/hj' --setup -- eval\.chunk_size=5 > /);
  assert.equal(r.background, true);
  const noSetup = await pi.call("wm_hjepa", { action: "run", setup: false });
  assert.doesNotMatch(noSetup.commands.run, /--setup/);
  const local = await pi.call("wm_hjepa", { action: "run", compute: "local", device: "cpu", tasks: 1 });
  assert.match(local.commands.run, /--target local .*--device cpu --setup/);
  const st = await pi.call("wm_hjepa", { action: "status" });
  assert.deepEqual(st.compute, { target: "remote", device: "auto", host: "me@gpu-box", remoteHome: "~/models/hj" });
});

test("local run end to end: launcher, environment, results and last-run record", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const home = path.join(dirs.base, "hj home");
  fakeRelease(home);
  const r = await pi.call("wm_hjepa", { action: "run", tasks: 2, home });
  const run = sh(dirs, r.commands.run);
  assert.equal(run.code, 0, fs.readFileSync(path.join(dirs.workspace, r.log), "utf8"));
  const argv = fs.readFileSync(path.join(home, "argv.txt"), "utf8").trim().split("\n");
  const linuxGpu = false; // the sandbox PATH has no nvidia-smi
  assert.equal(argv[0], linuxGpu ? "eval.py" : path.join(fs.realpathSync(dirs.workspace), ".world-model/hjepa/hjepa_local.py"));
  assert.deepEqual(argv.slice(1, 7), ["eval.py", "--config-name", "ant_l3", "seed=42", "output.dir=eval_pi_local_n2", "eval.num_eval=2"]);
  assert.equal(argv[7], `policy=${path.join(home, "ckpts/ant/ant_hjepa_l3/seed42/ant_hjepa_l3_object.ckpt")}`);
  const env = fs.readFileSync(path.join(home, "env.txt"), "utf8");
  assert.match(env, /HJEPA_DEVICE=auto/);
  assert.match(env, new RegExp(`HJEPA_HOME=${home.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
  if (process.platform === "darwin") assert.match(env, /MUJOCO_GL=glfw/);
  assert.equal(fs.realpathSync(fs.readFileSync(path.join(home, "cwd.txt"), "utf8").trim()), fs.realpathSync(path.join(home, "H-JEPA", "h_jepa")));
  const res = path.join(dirs.workspace, r.results);
  for (const f of ["metrics.yaml", "planning_compute.json", "eval_config.yaml", "run.json"]) assert.ok(fs.existsSync(path.join(res, f)), f);
  const last = readJSON(path.join(dirs.workspace, ".world-model/hjepa/last-run.json"));
  assert.equal(last.status, "done");
  assert.equal(last.successRate, 50);
  assert.equal(last.evaluationTime, 12.5);
  assert.equal(last.count, 2);
  assert.equal(last.metric, "success_rate");
  assert.equal(last.runId, r.runId);
  const st = await pi.call("wm_hjepa", { action: "status" });
  assert.equal(st.lastRun.successRate, 50);
  assert.equal((await pi.call("wm_status", {})).officialHjepa.lastRun.status, "done");
});

test("local run: a failing eval is recorded as failed; a missing install is reported", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const missing = await pi.call("wm_hjepa", { action: "run", tasks: 1, setup: false, home: path.join(dirs.base, "nothing") });
  const m = sh(dirs, missing.commands.run);
  assert.equal(m.code, 2);
  assert.match(fs.readFileSync(path.join(dirs.workspace, missing.log), "utf8"), /missing in .*: code python checkpoint\. Rerun with --setup/);
  const home = path.join(dirs.base, "hj");
  fakeRelease(home);
  const r = await pi.call("wm_hjepa", { action: "run", tasks: 1, home });
  const run = sh(dirs, r.commands.run, { FAKE_FAIL: "1" });
  assert.equal(run.code, 7);
  assert.equal(readJSON(path.join(dirs.workspace, ".world-model/hjepa/last-run.json")).status, "failed");
});

test("remote run end to end over (fake) SSH: preflight, unchanged eval.py, results copied back", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { hjepaCompute: "remote", hjepaRemoteHost: "me@gpu-box" } });
  t.after(restore);
  const { bin, remote } = fakeRemote(dirs.base);
  fakeRelease(path.join(remote, "hjepa"), { env: "pusht", model: "hwm_l2", seed: "44" });
  const r = await pi.call("wm_hjepa", { action: "run", env: "pusht", model: "hwm_l2", seed: "44", tasks: 3, setup: false });
  const run = sh(dirs, r.commands.run, { PATH: `${bin}:/usr/bin:/bin`, FAKE_REMOTE: remote });
  const log = fs.readFileSync(path.join(dirs.workspace, r.log), "utf8");
  assert.equal(run.code, 0, log);
  assert.match(log, /remote: GPU 0: NVIDIA Fake A100/);
  const rh = path.join(remote, "hjepa");
  const argv = fs.readFileSync(path.join(rh, "argv.txt"), "utf8").trim().split("\n");
  assert.deepEqual(argv.slice(0, 6), ["eval.py", "--config-name", "pusht_l2", "seed=44", "output.dir=eval_pi_remote_n3", "eval.num_eval=3"]);
  assert.match(fs.readFileSync(path.join(rh, "env.txt"), "utf8"), /MUJOCO_GL=egl/);
  assert.doesNotMatch(fs.readFileSync(path.join(rh, "env.txt"), "utf8"), /HJEPA_DEVICE/);
  assert.match(fs.readFileSync(path.join(remote, "hosts.txt"), "utf8"), /^me@gpu-box$/m);
  const res = path.join(dirs.workspace, r.results);
  assert.equal(fs.readFileSync(path.join(res, "metrics.yaml"), "utf8"), "success_rate: 50.0\nevaluation_time: 12.5\n");
  const last = readJSON(path.join(dirs.workspace, ".world-model/hjepa/last-run.json"));
  assert.equal(last.status, "done");
  assert.equal(last.host, "me@gpu-box");
  assert.equal(last.device, "cuda (NVIDIA Fake A100)");
  assert.equal(last.successRate, 50);
});

test("remote run refuses a host without an NVIDIA GPU, and one without the release", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { hjepaCompute: "remote", hjepaRemoteHost: "cpu-box" } });
  t.after(restore);
  const noGpu = fakeRemote(path.join(dirs.base, "a"), { gpu: false });
  const r = await pi.call("wm_hjepa", { action: "run", tasks: 1, setup: false });
  let run = sh(dirs, r.commands.run, { PATH: `${noGpu.bin}:/usr/bin:/bin`, FAKE_REMOTE: noGpu.remote });
  assert.equal(run.code, 2);
  assert.match(fs.readFileSync(path.join(dirs.workspace, r.log), "utf8"), /not an NVIDIA GPU machine/);
  assert.equal(readJSON(path.join(dirs.workspace, ".world-model/hjepa/last-run.json")).status, "failed");
  const empty = fakeRemote(path.join(dirs.base, "b"));
  run = sh(dirs, r.commands.run, { PATH: `${empty.bin}:/usr/bin:/bin`, FAKE_REMOTE: empty.remote });
  assert.equal(run.code, 2);
  assert.match(fs.readFileSync(path.join(dirs.workspace, r.log), "utf8"), /missing on cpu-box: code python checkpoint\. Rerun with --setup/);
});

test("remote --setup copies the pinned installer and runs it on the host first (dry run)", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { hjepaCompute: "remote", hjepaRemoteHost: "gpu", hjepaRemoteHome: "/data/hj" } });
  t.after(restore);
  const r = await pi.call("wm_hjepa", { action: "run", env: "droid", python: "uv" });
  const run = sh(dirs, r.commands.dryRun);
  assert.equal(run.code, 0, run.out);
  assert.match(run.out, /scp -q .* \S*\.world-model\/hjepa\/install-hjepa\.sh gpu:\/data\/hj\/\.pi\/install-hjepa\.sh/);
  assert.match(run.out, /bash '\/data\/hj'\/\.pi\/install-hjepa\.sh --home '\/data\/hj' '--env' 'droid' '--models' 'hjepa_l2' '--seeds' '1' '--python' 'uv'/);
  assert.match(run.out, /eval\.py '--config-name' 'droid_l2' 'seed=1' 'output\.dir=eval_pi_remote'/);
  assert.equal(r.runId, "remote-droid_hjepa_l2-s1-p1-eval_pi_remote");
});

test("hjepa_local.py redirects cuda and float64 and overrides every device: cuda (needs the installed release)", { skip: !fs.existsSync(REAL_PY) && "release not installed in ~/hjepa" }, () => {
  const code = `
import sys, torch
sys.argv = ["x"]
sys.path.insert(0, ${JSON.stringify(SCRIPTS)})
import hjepa_local as L
dev = "mps" if torch.backends.mps.is_available() else "cpu"
L.patch_torch(dev)
m = torch.nn.Linear(2, 2).to("cuda")
assert next(m.parameters()).device.type == dev, next(m.parameters()).device
m2 = torch.nn.Linear(2, 2).cuda()
assert next(m2.parameters()).device.type == dev
t = torch.zeros(3, dtype=torch.float64).to(torch.device("cuda"))
assert t.device.type == dev
assert t.dtype == (torch.float32 if dev == "mps" else torch.float64), t.dtype
z = torch.zeros(1, 1, 3, device="cuda")
assert z.device.type == dev, z.device
f = torch.ones(2, device="cuda", dtype=torch.float64)
assert f.device.type == dev and f.dtype == (torch.float32 if dev == "mps" else torch.float64)
assert torch.zeros(2, device=None).device.type == "cpu"
c = torch.zeros(2).to("cpu", torch.float16)
assert c.dtype == torch.float16 and c.device.type == "cpu"
import os
os.chdir(os.path.expanduser("~/hjepa/H-JEPA/h_jepa"))
o = L.device_overrides(["--config-name", "ant_l3", "solver.solvers.level2.device=cpu"], "mps")
assert o == ["solver.solvers.level1.device=mps", "solver.solvers.level3.device=mps"], o
assert L.device_overrides(["--config-name=cube_flat"], "cpu") == ["solver.device=cpu"], L.device_overrides(["--config-name=cube_flat"], "cpu")
print("ok", dev)
`;
  const r = spawnSync(REAL_PY, ["-c", code], { encoding: "utf8", env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  assert.match(r.stdout, /^ok /);
});

test("settings pick the world model: environment, recommended model, seed slot, evaluation size", async (t) => {
  const { pi, restore } = await setup({ settings: { hjepaEnv: "droid", hjepaSeed: "second", hjepaTasks: 3 } });
  t.after(restore);
  const r = await pi.call("wm_hjepa", { action: "run" });
  assert.deepEqual(r.selection, { source: "settings", env: "droid", model: "hjepa_l2", seed: "1000", planSeed: 1 });
  assert.equal(r.config, "droid_l2");
  assert.equal(r.count, 3);
  assert.equal(r.unit, "clip");
  assert.match(r.metric, /Frechet fidelity/);
  assert.match(r.commands.run, /--env droid --model hjepa_l2 --seed 1000 --plan-seed 1 --tasks 3 --device auto --setup/);
  assert.match(r.estimate, /~6 min per clip of droid_l2, so ~0\.3 h for 3 clip/);
  const plan = await pi.call("wm_hjepa", { action: "plan" });
  assert.deepEqual(plan.selection, ["droid/hjepa_l2/1000"]);
  const st = await pi.call("wm_hjepa", { action: "status" });
  assert.equal(st.model.env, "droid");
  assert.equal((await pi.call("wm_status", {})).officialHjepa.model.seed, "1000");
  // arguments override the settings; an env argument drops the settings' model and seed
  const ant = await pi.call("wm_hjepa", { action: "run", env: "ant", tasks: 1 });
  assert.deepEqual(ant.selection, { source: "arguments", env: "ant", model: "hjepa_l3", seed: "42", planSeed: undefined });
});

test("settings that do not fit together are refused with a clear message", async (t) => {
  const bad = await setup({ settings: { hjepaEnv: "droid", hjepaModel: "hjepa_l3" } });
  t.after(bad.restore);
  assert.match((await bad.pi.call("wm_hjepa", { action: "run" })).error, /'H-JEPA model' setting \(hjepa_l3\) is not trained for droid: droid has lewm, hjepa_l2, hwm_l2/);
  assert.match((await bad.pi.call("wm_hjepa", { action: "plan" })).error, /not trained for droid/);
  const big = await setup({ settings: { hjepaEnv: "droid", hjepaTasks: 20 } });
  t.after(big.restore);
  assert.match((await big.pi.call("wm_hjepa", { action: "run" })).error, /1 to 16 for droid/);
  const sim = await setup({ settings: { hjepaEnv: "cube", hjepaModel: "hwm_l4", hjepaSeed: "third", hjepaTasks: 0 } });
  t.after(sim.restore);
  const r = await sim.pi.call("wm_hjepa", { action: "run" });
  assert.deepEqual([r.config, r.checkpoint, r.count], ["cube_l4", "cube/hwm_l4/44", 50]);
});

test("DROID local run end to end: planner seed, clips, Frechet fidelity recorded", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { hjepaEnv: "droid", hjepaTasks: 2 } });
  t.after(restore);
  const home = path.join(dirs.base, "hj");
  fakeRelease(home, { env: "droid", model: "hjepa_l2", seed: "1" });
  const r = await pi.call("wm_hjepa", { action: "run", planSeed: 3, home });
  const run = sh(dirs, r.commands.run);
  assert.equal(run.code, 0, fs.readFileSync(path.join(dirs.workspace, r.log), "utf8"));
  const argv = fs.readFileSync(path.join(home, "argv.txt"), "utf8").trim().split("\n");
  assert.deepEqual(argv.slice(1, 6), ["eval.py", "--config-name", "droid_l2", "seed=3", "output.dir=eval_pi_local_n2"]);
  assert.equal(argv[6], "num_eval=2");
  const res = path.join(dirs.workspace, r.results);
  assert.ok(fs.existsSync(path.join(res, "eval.csv")) && fs.existsSync(path.join(res, "plan_config.yaml")));
  const last = readJSON(path.join(dirs.workspace, ".world-model/hjepa/last-run.json"));
  assert.equal(last.metric, "frechet_fidelity");
  assert.equal(last.score, 43.21);
  assert.equal(last.frechetFidelity, 43.21);
  assert.equal(last.planSeed, 3);
  assert.equal(last.unit, "clip");
  assert.equal(last.count, 1); // n_episodes from eval.csv
  assert.equal(last.installed, "");
});

test("with setup, a missing checkpoint or clip set is installed first (dry run)", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { hjepaEnv: "droid" } });
  t.after(restore);
  const home = path.join(dirs.base, "hj");
  fakeRelease(home, { env: "ant" }); // code and Python, but no DROID checkpoint
  fs.rmSync(path.join(home, "droid"), { recursive: true });
  const r = await pi.call("wm_hjepa", { action: "run", tasks: 1, home });
  const run = sh(dirs, r.commands.dryRun);
  assert.equal(run.code, 0, run.out);
  assert.match(run.out, /missing locally: checkpoint droid-clips; installing with install-hjepa\.sh/);
  assert.match(run.out, /install-hjepa\.sh --home \S+ --env droid --models hjepa_l2 --seeds 1 --python auto --state /);
  const ok = await pi.call("wm_hjepa", { action: "run", env: "ant", tasks: 1, home });
  assert.doesNotMatch(sh(dirs, ok.commands.dryRun).out, /install-hjepa/);
});
