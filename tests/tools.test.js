"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { setup } = require("./helpers.js");

const read = (dirs, rel) => JSON.parse(fs.readFileSync(path.join(dirs.workspace, rel), "utf8"));

test("wm_observe validates its input", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  assert.match((await pi.call("wm_observe", {})).error, /episode/);
  assert.match((await pi.call("wm_observe", { episode: "e1", state: "s" })).error, /state, action and result/);
});

test("wm_observe records a step, learns online and persists both files", async (t) => {
  const { dirs, pi, restore } = await setup();
  t.after(restore);
  const r = await pi.call("wm_observe", {
    episode: "e1", goal: "tests pass", state: "tests: 2 failing", action: "edit login.ts", result: "tests: 0 failing",
    agent: "fixer", intent: "fix auth", ok: true,
  });
  assert.equal(r.advice, "continue");
  assert.equal(r.coldStart, true);
  assert.equal(r.transitions, 1);
  assert.equal(r.trained.batches, 2);
  const exp = read(dirs, ".world-model/experience.json");
  assert.equal(exp.transitions[0].agent, "fixer");
  assert.equal(exp.episodes.e1.goal, "tests pass");
  assert.ok(read(dirs, ".world-model/model.json").steps >= 2);
  const end = await pi.call("wm_observe", { episode: "e1", outcome: "success" });
  assert.equal(end.ended, "success");
  assert.equal(read(dirs, ".world-model/experience.json").episodes.e1.outcome, "success");
});

test("a failed step always advises replanning", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  const r = await pi.call("wm_observe", { episode: "e", state: "a", action: "b", result: "c", ok: false });
  assert.match(r.advice, /^replan/);
});

test("long text is truncated and the experience is capped", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { maxTransitions: 100, onlineSteps: 0 } });
  t.after(restore);
  for (let i = 0; i < 105; i++) await pi.call("wm_observe", { episode: `e${i % 3}`, state: "x".repeat(2000), action: `a${i}`, result: "r" });
  const exp = read(dirs, ".world-model/experience.json");
  assert.equal(exp.transitions.length, 100);
  assert.equal(exp.transitions[0].action, "a5");
  assert.equal(exp.transitions[0].state.length, 600);
});

test("wm_plan validates input and is cold before any training", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  assert.match((await pi.call("wm_plan", { options: ["a"] })).error, /state/);
  assert.match((await pi.call("wm_plan", { state: "s", options: [] })).error, /options/);
  const r = await pi.call("wm_plan", { state: "s", options: ["a", { label: "b", agent: "fixer", steps: ["x", "y"] }] });
  assert.equal(r.trust, 0);
  assert.match(r.advice, /^cold start/);
  assert.equal(r.ranked.length, 2);
  assert.equal(r.ranked.find((o) => o.label === "b").trace.length, 2);
});

test("concurrent calls do not lose steps", async (t) => {
  const { dirs, pi, restore } = await setup({ settings: { onlineSteps: 1 } });
  t.after(restore);
  await Promise.all(Array.from({ length: 8 }, (_, i) => pi.call("wm_observe", { episode: "p", state: "s", action: `a${i}`, result: "r" })));
  assert.equal(read(dirs, ".world-model/experience.json").transitions.length, 8);
});

test("wm_train needs experience, then reports losses", async (t) => {
  const { pi, restore } = await setup({ settings: { onlineSteps: 0 } });
  t.after(restore);
  assert.match((await pi.call("wm_train", {})).error, /no experience/);
  for (let i = 0; i < 20; i++) await pi.call("wm_observe", { episode: `e${i}`, state: "s", action: "a", result: `r${i % 2}`, ok: i % 2 === 0 });
  const r = await pi.call("wm_train", { epochs: 3 });
  assert.equal(r.epochs, 3);
  assert.equal(r.samples.step, 20);
  assert.equal(r.samples.labeled, 20);
  assert.ok(typeof r.last.pred1 === "number");
});

test("wm_status counts episodes and agents, and reset clears", async (t) => {
  const { pi, restore } = await setup({ settings: { onlineSteps: 0 } });
  t.after(restore);
  await pi.call("wm_observe", { episode: "a", state: "s", action: "x", result: "r", agent: "fixer", ok: true });
  await pi.call("wm_observe", { episode: "a", outcome: "success" });
  await pi.call("wm_observe", { episode: "b", state: "s", action: "x", result: "r", agent: "fixer", ok: false });
  let s = await pi.call("wm_status", {});
  assert.deepEqual(s.episodes, { total: 2, success: 1, failure: 0, open: 1 });
  assert.equal(s.agents.fixer.steps, 2);
  assert.equal(s.agents.fixer.failureRate, 0.5);
  assert.equal(s.settings.onlineSteps, 0);
  s = await pi.call("wm_status", { reset: "all" });
  assert.equal(s.transitions, 0);
  assert.equal(s.model.steps, 0);
});

test("end to end: after experience, wm_plan routes a fix to the sub-agent that fixes things", async (t) => {
  const { pi, restore } = await setup({ settings: { onlineSteps: 0 } });
  t.after(restore);
  const bugs = ["login", "upload", "search", "billing", "export", "parser"];
  for (let i = 0; i < 36; i++) {
    const bug = bugs[i % bugs.length];
    const good = i % 2 === 0;
    const agent = good ? "fixer" : "explorer";
    await pi.call("wm_observe", {
      episode: `fix-${i}`,
      state: `tests: 1 failing (${bug}); files: unchanged`,
      action: `fix the ${bug} bug`,
      agent,
      intent: "fix the failing test",
      result: good ? `tests: 0 failing; files: ${bug}.ts edited` : `tests: 1 failing (${bug}); files: unchanged; report: read only, no edit`,
      ok: good,
    });
    await pi.call("wm_observe", { episode: `fix-${i}`, outcome: good ? "success" : "failure" });
  }
  await pi.call("wm_train", { epochs: 30 });
  const r = await pi.call("wm_plan", {
    state: "tests: 1 failing (search); files: unchanged",
    goal: "tests: 0 failing; files: search.ts edited",
    options: [
      { label: "fix the search bug", agent: "explorer", intent: "fix the failing test" },
      { label: "fix the search bug", agent: "fixer", intent: "fix the failing test" },
    ],
  });
  assert.equal(r.ranked[0].agent, "fixer", JSON.stringify(r.ranked));
  assert.ok(r.trust >= 0.4);
  assert.ok(r.ranked[0].failure < r.ranked[1].failure);
});
