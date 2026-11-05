"use strict";

/**
 * Test helpers for World Model. No dependencies: main.js is compiled as a fresh CommonJS module per
 * test with its own fake `pi` global; the workspace and HOME live in a temporary folder.
 */

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const MAIN = path.join(ROOT, "main.js");

/** Compile a fresh copy of main.js bound to `pi`. */
function loadMain(pi) {
  const src = fs.readFileSync(MAIN, "utf8");
  const m = new module.constructor(MAIN, module);
  m.filename = MAIN;
  m.paths = module.paths;
  m.__pi = pi;
  const bind = "const pi = module.__pi;";
  const directive = /^\s*(["'])use strict\1;?/;
  m._compile(directive.test(src) ? src.replace(directive, (d) => `${d} ${bind}`) : `${bind}\n${src}`, MAIN);
  return m.exports;
}

function sandbox() {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "world-model-hjepa-test-"));
  const dirs = { base, workspace: path.join(base, "workspace"), home: path.join(base, "home") };
  for (const d of [dirs.workspace, dirs.home]) fs.mkdirSync(d, { recursive: true });
  return dirs;
}

/** A fake `pi`: pi.fs over the workspace restricted to .world-model/**, settings in memory. */
function fakePi(dirs, { settings = {} } = {}) {
  const abs = (p) => {
    if (!/^\.world-model\//.test(p)) throw new Error(`outside the fs scope: ${p}`);
    const full = path.resolve(dirs.workspace, p);
    if (path.relative(dirs.workspace, full).startsWith("..")) throw new Error(`outside the workspace: ${p}`);
    return full;
  };
  const tools = new Map();
  const unregistered = [];
  const store = { ...settings };
  return {
    tools,
    unregistered,
    store,
    fs: {
      readText: async (p) => fs.readFileSync(abs(p), "utf8"),
      writeText: async (p, content) => {
        fs.mkdirSync(path.dirname(abs(p)), { recursive: true });
        fs.writeFileSync(abs(p), content);
      },
    },
    plugin: { getSettings: async () => JSON.parse(JSON.stringify(store)) },
    agent: {
      registerTool: async (tool) => tools.set(tool.name, tool),
      unregisterTool: async (name) => {
        unregistered.push(name);
        tools.delete(name);
      },
    },
    call(name, args) {
      return tools.get(name).execute(args, { sessionId: "s1", log: () => {} });
    },
  };
}

function withHome(home) {
  const prev = process.env.HOME;
  process.env.HOME = home;
  return () => {
    if (prev === undefined) delete process.env.HOME;
    else process.env.HOME = prev;
  };
}

/** Load the plugin in a fresh sandbox; returns { dirs, pi, main, restore }. */
async function setup(opts) {
  const dirs = sandbox();
  const restore = withHome(dirs.home);
  const pi = fakePi(dirs, opts);
  const main = loadMain(pi);
  await main.onLoad();
  return { dirs, pi, main, restore };
}

module.exports = { ROOT, MAIN, loadMain, sandbox, fakePi, withHome, setup };
