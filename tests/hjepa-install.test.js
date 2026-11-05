"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync, spawnSync } = require("node:child_process");
const { ROOT, setup } = require("./helpers.js");

const SCRIPT = path.join(ROOT, "scripts", "install-hjepa.sh");

/** Run a command line the way the agent does: bash, from the workspace root. */
function sh(dirs, line) {
  const r = spawnSync("/bin/bash", ["-c", line], { cwd: dirs.workspace, env: { ...process.env, HOME: dirs.home, PATH: "/usr/bin:/bin" }, encoding: "utf8" });
  return { code: r.status, out: r.stdout + r.stderr };
}

/** The script's own size estimate for a selection (dry run, no network). */
function scriptMBStderr(args) {
  const r = spawnSync("/bin/bash", [SCRIPT, "--dry-run", "--skip-code", "--skip-smoke", "--home", "/tmp/hjepa-test", ...args], { encoding: "utf8", env: { ...process.env, PATH: "/usr/bin:/bin" } });
  const m = /download: ~(\d+) MB/.exec(r.stderr);
  return { code: r.status, mb: m ? Number(m[1]) : null, err: r.stderr };
}

test("the installer is valid bash", () => {
  execFileSync("/bin/bash", ["-n", SCRIPT]);
  assert.match(execFileSync("/bin/bash", [SCRIPT, "--help"], { encoding: "utf8" }), /--env ENVS/);
});

test("plan stages the shipped installer and returns the default selection", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const r = await pi.call("wm_hjepa", {});
  assert.equal(r.staged, ".world-model/hjepa/install-hjepa.sh");
  assert.equal(fs.readFileSync(path.join(dirs.workspace, r.staged), "utf8"), fs.readFileSync(SCRIPT, "utf8"));
  assert.deepEqual(r.selection, ["ant/hjepa_l3/42"]);
  assert.equal(r.downloadMB, 295);
  assert.deepEqual(r.evalPlanner, { "ant/hjepa_l3/42": "ant_l3" });
  assert.match(r.commands.install, /--state \.world-model\/hjepa\/state\.json > \.world-model\/hjepa\/install\.log 2>&1$/);
  assert.match(r.links.model, /huggingface\.co\/jepa-world-models\/h-jepa$/);
  assert.ok(r.notes.some((n) => /weights_only=False/.test(n)));
});

test("plan validates the selection like the script does", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  assert.match((await pi.call("wm_hjepa", { env: "mars" })).error, /unknown environment 'mars'/);
  assert.match((await pi.call("wm_hjepa", { env: "droid", models: "hjepa_l3" })).error, /does not exist for droid/);
  assert.match((await pi.call("wm_hjepa", { seeds: "7" })).error, /seed '7'/);
  assert.match((await pi.call("wm_hjepa", { python: "pipx" })).error, /python must be/);
  assert.match((await pi.call("wm_hjepa", { action: "nope" })).error, /plan, run or status/);
  const lewm = await pi.call("wm_hjepa", { env: "fourroom,droid", models: "lewm", seeds: "all" });
  assert.equal(lewm.selection.length, 6);
  assert.equal(lewm.evalPlanner["droid/lewm/1000"], "droid_flat");
});

test("the tool's size catalog matches the script's for every selection shape", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  const cases = [
    [{}, []],
    [{ env: "all", models: "all", seeds: "all" }, ["--env", "all", "--models", "all", "--seeds", "all"]],
    [{ env: "sim", models: "all" }, ["--env", "sim", "--models", "all"]],
    [{ env: "droid" }, ["--env", "droid"]],
    [{ env: "cube,pusht", models: "lewm,hwm_l4", seeds: "43,44", skipTasks: true }, ["--env", "cube,pusht", "--models", "lewm,hwm_l4", "--seeds", "43,44", "--skip-tasks"]],
  ];
  for (const [args, flags] of cases) {
    const tool = await pi.call("wm_hjepa", args);
    const script = scriptMBStderr(flags);
    assert.equal(script.code, 0, script.err);
    assert.equal(tool.downloadMB, script.mb, JSON.stringify(args));
  }
  // the whole release: 84 simulated + 9 DROID checkpoints, ~12.4 GB plus the evaluation tasks
  const all = await pi.call("wm_hjepa", { env: "all", models: "all", seeds: "all", skipTasks: true });
  assert.equal(all.selection.length, 93);
  assert.ok(all.downloadMB > 12000 && all.downloadMB < 12800, String(all.downloadMB));
});

test("the staged dry-run command runs from the workspace root without changing anything", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const home = path.join(dirs.base, "hj home");
  const r = await pi.call("wm_hjepa", { home, env: "pusht", models: "lewm" });
  const run = sh(dirs, r.commands.dryRun);
  assert.equal(run.code, 0, run.out);
  assert.match(run.out, /--include pusht\/pusht_lewm\/seed42\/\*/);
  assert.match(run.out, /--revision 08fc8a21208671a87741f2960aeba9b5db4420e0/);
  assert.match(run.out, /checkout --quiet 97054ecaff83e13839195ec4b92d74c08a0127c5/);
  assert.equal(fs.existsSync(home), false);
});

test("status: not installed, then the --check result and a quoted recheck", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  assert.equal((await pi.call("wm_hjepa", { action: "status" })).status, "not installed");
  const home = path.join(dirs.base, "it's here");
  // fake a downloaded checkpoint so the check finds it
  const ck = path.join(home, "ckpts", "ant", "ant_hjepa_l3", "seed42");
  fs.mkdirSync(ck, { recursive: true });
  fs.writeFileSync(path.join(ck, "ant_hjepa_l3_object.ckpt"), "x");
  fs.writeFileSync(path.join(ck, "normalizer.pt"), "x");
  const plan = await pi.call("wm_hjepa", { home });
  const check = sh(dirs, plan.commands.check);
  assert.equal(check.code, 1, check.out); // code and Python env are missing
  assert.match(check.out, /checkpoints {2}1\/1 present/);
  const st = await pi.call("wm_hjepa", { action: "status" });
  assert.equal(st.status, "incomplete");
  assert.equal(st.home, home);
  assert.deepEqual(st.runs, ["ant/hjepa_l3/42"]);
  assert.match(st.next, /wm_hjepa \{action:'run'\}/);
  assert.equal(st.compute.target, "local");
  const re = sh(dirs, st.recheck);
  assert.equal(re.code, 1, re.out);
  assert.equal((await pi.call("wm_status", {})).officialHjepa.status, "incomplete");
});
