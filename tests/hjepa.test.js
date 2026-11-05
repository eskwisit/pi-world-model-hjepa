"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { HJEPA, MLP, featurize, rng, cosDist, varianceReg, dims } = require("../lib/hjepa.js");

test("featurize is deterministic, L2-normalised and order-sensitive through bigrams", () => {
  const a = featurize("run the tests");
  assert.equal(a.length, dims.D);
  assert.deepEqual(Array.from(a), Array.from(featurize("Run the TESTS!")));
  const n = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  assert.ok(Math.abs(n - 1) < 1e-9);
  assert.notDeepEqual(Array.from(a), Array.from(featurize("tests the run")));
  assert.ok(featurize("").every((v) => v === 0));
  assert.ok(featurize({ tests: "failing", files: ["a.ts"] }).some((v) => v !== 0));
});

test("MLP backward matches finite differences", () => {
  const m = new MLP([5, 4, 3], ["tanh", "linear"], rng(3));
  const x = Float64Array.from([0.3, -0.2, 0.5, 0, 1]);
  const w = [1, -2, 0.5];
  const loss = () => m.forward(x).y.reduce((s, v, i) => s + v * w[i], 0);
  const f = m.forward(x);
  const g = m.zeroGrads();
  const gx = m.backward(f.cache, w, g);
  for (let l = 0; l < m.layers.length; l++) {
    for (let i = 0; i < m.layers[l].W.length; i++) {
      const o = m.layers[l].W[i];
      m.layers[l].W[i] = o + 1e-6;
      const up = loss();
      m.layers[l].W[i] = o - 1e-6;
      const down = loss();
      m.layers[l].W[i] = o;
      assert.ok(Math.abs((up - down) / 2e-6 - g[l].W[i]) < 1e-6);
    }
  }
  for (let i = 0; i < x.length; i++) {
    const o = x[i];
    x[i] = o + 1e-6;
    const up = loss();
    x[i] = o - 1e-6;
    const down = loss();
    x[i] = o;
    assert.ok(Math.abs((up - down) / 2e-6 - gx[i]) < 1e-6);
  }
});

test("variance regulariser pushes collapsed latents apart and ignores spread ones", () => {
  const collapsed = [Float64Array.from([0, 0]), Float64Array.from([0.01, 0])];
  const r = varianceReg(collapsed);
  assert.ok(r.loss > 0);
  assert.ok(r.grads[0][0] > 0 && r.grads[1][0] < 0); // gradient descent moves them apart
  const spread = varianceReg([Float64Array.from([-2, 2]), Float64Array.from([2, -2])]);
  assert.equal(spread.loss, 0);
});

test("cosDist is 0 for the same direction, 1 for opposite and for a zero vector", () => {
  assert.ok(cosDist([1, 2], [2, 4]) < 1e-12);
  assert.ok(Math.abs(cosDist([1, 0], [-1, 0]) - 1) < 1e-12);
  assert.equal(cosDist([0, 0], [1, 0]), 1);
});

/** A toy environment: "edit X" works, "delete X" breaks the tests. */
function toyWorld(seed) {
  const r = rng(seed);
  const files = ["main", "utils", "config", "readme", "tests", "server"];
  return () => {
    const f = files[Math.floor(r() * files.length)];
    const bad = r() < 0.3;
    return {
      s: `workspace clean ${f} present`,
      a: bad ? `delete ${f}` : `edit ${f}`,
      s2: bad ? `error ${f} missing tests fail` : `${f} modified tests pass`,
      y: bad ? 1 : 0,
    };
  };
}

function trained(seed = 1, batches = 250) {
  const m = new HJEPA(seed);
  const sample = toyWorld(seed + 4);
  let first;
  let last;
  for (let e = 0; e < batches; e++) {
    const b = Array.from({ length: 16 }, sample);
    const st = m.trainBatch(b, b.map((x) => ({ s: x.s, intent: x.a, s2: x.s2, y: x.y })));
    if (e === 0) first = st;
    last = st;
  }
  return { m, first, last };
}

test("training lowers the latent prediction loss and learns which actions fail", () => {
  const { m, first, last } = trained();
  assert.ok(last.pred1 < first.pred1 / 5, `pred1 ${first.pred1} -> ${last.pred1}`);
  assert.ok(last.cost1 < first.cost1, `cost1 ${first.cost1} -> ${last.cost1}`);
  const s = "workspace clean main present";
  const good = m.imagine(s, { label: "edit main" }, "main modified tests pass");
  const bad = m.imagine(s, { label: "delete main" }, "main modified tests pass");
  assert.ok(good.failure1 < 0.2 && bad.failure1 > 0.8, `${good.failure1} vs ${bad.failure1}`);
  assert.ok(good.failure2 < bad.failure2, "the intent level ranks them the same way");
  assert.ok(good.goal1 < bad.goal1, "the good action lands closer to the goal");
});

test("an off-distribution result is far more surprising than an expected one", () => {
  const { m } = trained();
  const s = "workspace clean main present";
  const expected = m.surprise(s, "edit main", "main modified tests pass");
  const odd = m.surprise(s, "edit main", "kernel panic network outage database corrupted");
  assert.ok(odd.surprise > 10 * expected.surprise, `${expected.surprise} vs ${odd.surprise}`);
});

test("multi-step imagination returns a trace per step", () => {
  const { m } = trained(2, 60);
  const im = m.imagine("workspace clean main present", { label: "x", steps: ["edit main", "edit utils", "delete config"] });
  assert.equal(im.trace.length, 3);
  assert.equal(im.trace[2].step, "delete config");
});

test("toJSON/fromJSON round-trips predictions; other dimensions are refused", () => {
  const { m } = trained(3, 40);
  const j = JSON.parse(JSON.stringify(m.toJSON()));
  const back = HJEPA.fromJSON(j);
  const s = "workspace clean main present";
  const a = m.imagine(s, { label: "delete main" }, "goal");
  const b = back.imagine(s, { label: "delete main" }, "goal");
  assert.ok(Math.abs(a.failure1 - b.failure1) < 1e-4);
  assert.ok(Math.abs(a.goal2 - b.goal2) < 1e-4);
  assert.equal(back.steps, m.steps);
  assert.equal(HJEPA.fromJSON({ ...j, dims: { ...j.dims, Z: 8 } }), null);
  assert.equal(HJEPA.fromJSON(null), null);
});

test("the same seed gives the same model", () => {
  const a = trained(5, 10).m.imagine("s", { label: "edit main" });
  const b = trained(5, 10).m.imagine("s", { label: "edit main" });
  assert.equal(a.failure1, b.failure1);
});
