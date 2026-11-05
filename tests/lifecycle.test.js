"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { ROOT, setup } = require("./helpers.js");

const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, "manifest.json"), "utf8"));
const scout = (home) => path.join(home, ".agents", "subagents", "world-model-scout.md");

test("onLoad registers exactly the tools the manifest declares, with the same schemas", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  const declared = manifest.contributes.agentTools;
  assert.deepEqual([...pi.tools.keys()].sort(), declared.map((d) => d.name).sort());
  for (const d of declared) {
    assert.deepEqual(pi.tools.get(d.name).schema, d.schema, d.name);
    assert.equal(pi.tools.get(d.name).description, d.description);
  }
});

test("the manifest scopes file access to .world-model/** and ships the skill", () => {
  assert.deepEqual(manifest.fs.read.scope, [".world-model/**"]);
  assert.deepEqual(manifest.fs.write.scope, [".world-model/**"]);
  for (const s of manifest.contributes.skills) {
    const body = fs.readFileSync(path.join(ROOT, s), "utf8");
    const desc = /^description: (.*)$/m.exec(body)[1];
    assert.ok(desc.length <= 240);
  }
});

test("the scout sub-agent is installed on load and removed on unload", async (t) => {
  const { dirs, pi, main, restore } = await setup();
  t.after(restore);
  assert.ok(fs.existsSync(scout(dirs.home)));
  assert.match((await pi.call("wm_status", {})).subagents, /installed/);
  await main.onUnload();
  assert.equal(fs.existsSync(scout(dirs.home)), false);
  assert.equal(pi.tools.size, 0);
});

test("a scout the user edited is neither overwritten nor removed", async (t) => {
  const { dirs, main, restore, pi } = await setup();
  t.after(restore);
  fs.writeFileSync(scout(dirs.home), "---\nname: world-model-scout\n---\nmine\n");
  await main.onUnload();
  const again = require("./helpers.js").loadMain(pi);
  await again.onLoad();
  assert.match(fs.readFileSync(scout(dirs.home), "utf8"), /mine/);
  await again.onUnload();
  assert.ok(fs.existsSync(scout(dirs.home)));
});

test("a tool error comes back as { error } instead of throwing", async (t) => {
  const { pi, restore } = await setup();
  t.after(restore);
  pi.fs.writeText = async () => {
    throw new Error("disk full");
  };
  const r = await pi.call("wm_observe", { episode: "e", state: "s", action: "a", result: "r" });
  assert.match(r.error, /disk full/);
});
