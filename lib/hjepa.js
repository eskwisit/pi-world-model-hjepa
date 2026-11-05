"use strict";

/**
 * A small, dependency-free Hierarchical JEPA (H-JEPA) world model for agent trajectories.
 *
 * Observations and actions are text. They are turned into fixed-size vectors by signed feature
 * hashing (unigrams + bigrams), then:
 *
 *   Level 1 (fine, one step):
 *     z  = enc1(s)                 online encoder
 *     z' = tgt1(s')                target encoder = EMA of enc1, no gradient (stop-grad)
 *     ẑ' = z + pred1_k(z, act1(a))  K residual predictors (ensemble -> epistemic uncertainty)
 *     loss = |ẑ' − z'|² + variance regulariser on z (VICReg-style, against collapse)
 *     cost1(z') -> P(failure)      scored on the outcome latent, trained on outcome labels
 *
 *   Level 2 (coarse, one intent / subgoal = a chunk of steps):
 *     h  = enc2(tgt1(first state of chunk))
 *     h' = tgt2(tgt1(last result of chunk))
 *     ĥ' = h + pred2_k(h, act2(intent))
 *     cost2(h') -> P(failure)
 *
 * The model never predicts text: it predicts representations, which is the point of JEPA.
 * Planning rolls the predictors forward in latent space and scores options by predicted failure,
 * latent distance to a goal and ensemble disagreement.
 */

const D = 256; // hashed input features
const Z = 32; // level-1 latent
const A = 16; // action / intent embedding
const H = 16; // level-2 latent
const K = 3; // ensemble size
const EMA = 0.99;
const VAR_GAMMA = 0.5;
const VAR_WEIGHT = 0.5;
const VERSION = 1;

// ---------- random ----------
function rng(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- text -> features ----------
function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function flatten(value, prefix, out) {
  if (value == null) return out;
  if (typeof value === "object") {
    if (Array.isArray(value)) for (const v of value) flatten(v, prefix, out);
    else for (const [k, v] of Object.entries(value)) flatten(v, k, out);
    return out;
  }
  const text = String(value);
  if (prefix) out.push(`${prefix} ${text}`);
  else out.push(text);
  return out;
}

function tokens(value) {
  const text = typeof value === "string" ? value : flatten(value, "", []).join(" \n ");
  return text.toLowerCase().match(/[a-z0-9_]+/g) || [];
}

/** Signed feature hashing of unigrams and bigrams, L2-normalised. */
function featurize(value) {
  const x = new Float64Array(D);
  const toks = tokens(value);
  const add = (s, w) => {
    const h = fnv1a(s);
    x[h % D] += (h & 0x80000000 ? -1 : 1) * w;
  };
  for (let i = 0; i < toks.length; i++) {
    add(toks[i], 1);
    if (i + 1 < toks.length) add(`${toks[i]}\u0001${toks[i + 1]}`, 0.5);
  }
  let n = 0;
  for (let i = 0; i < D; i++) n += x[i] * x[i];
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < D; i++) x[i] /= n;
  return x;
}

// ---------- MLP with manual backprop and Adam ----------
class MLP {
  /** sizes: [in, h1, ..., out]; acts: activation per layer ("tanh" | "linear"). */
  constructor(sizes, acts, rand, scale = 1) {
    this.sizes = sizes;
    this.acts = acts;
    this.layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l];
      const nout = sizes[l + 1];
      const W = new Float64Array(nin * nout);
      const lim = (Math.sqrt(6 / (nin + nout))) * scale;
      if (rand) for (let i = 0; i < W.length; i++) W[i] = (rand() * 2 - 1) * lim;
      this.layers.push({ W, b: new Float64Array(nout), nin, nout });
    }
  }

  forward(x) {
    const inputs = [];
    const outs = [];
    let h = x;
    for (let l = 0; l < this.layers.length; l++) {
      const { W, b, nin, nout } = this.layers[l];
      inputs.push(h);
      const y = new Float64Array(nout);
      for (let o = 0; o < nout; o++) {
        let s = b[o];
        const row = o * nin;
        for (let i = 0; i < nin; i++) {
          const v = h[i];
          if (v !== 0) s += W[row + i] * v;
        }
        y[o] = this.acts[l] === "tanh" ? Math.tanh(s) : s;
      }
      outs.push(y);
      h = y;
    }
    return { y: h, cache: { inputs, outs } };
  }

  zeroGrads() {
    return this.layers.map((L) => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));
  }

  /** Accumulate gradients for dL/dy into `grads`; returns dL/dx. */
  backward(cache, gy, grads) {
    let g = gy;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const { W, nin, nout } = this.layers[l];
      const x = cache.inputs[l];
      const y = cache.outs[l];
      const gpre = new Float64Array(nout);
      for (let o = 0; o < nout; o++) gpre[o] = this.acts[l] === "tanh" ? g[o] * (1 - y[o] * y[o]) : g[o];
      const gx = new Float64Array(nin);
      const GW = grads[l].W;
      const Gb = grads[l].b;
      for (let o = 0; o < nout; o++) {
        const go = gpre[o];
        if (go === 0) continue;
        Gb[o] += go;
        const row = o * nin;
        for (let i = 0; i < nin; i++) {
          GW[row + i] += go * x[i];
          gx[i] += go * W[row + i];
        }
      }
      g = gx;
    }
    return g;
  }

  /** Adam step with gradient averaging over `n` samples and global-norm clipping. */
  step(grads, n, lr, t, clip = 5) {
    if (!this.m) {
      this.m = this.zeroGrads();
      this.v = this.zeroGrads();
    }
    let norm = 0;
    for (const g of grads) {
      for (const v of g.W) norm += (v / n) ** 2;
      for (const v of g.b) norm += (v / n) ** 2;
    }
    norm = Math.sqrt(norm);
    const k = norm > clip ? clip / norm : 1;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** t;
    const c2 = 1 - b2 ** t;
    for (let l = 0; l < this.layers.length; l++) {
      for (const key of ["W", "b"]) {
        const P = this.layers[l][key];
        const G = grads[l][key];
        const M = this.m[l][key];
        const V = this.v[l][key];
        for (let i = 0; i < P.length; i++) {
          const g = (G[i] / n) * k;
          M[i] = b1 * M[i] + (1 - b1) * g;
          V[i] = b2 * V[i] + (1 - b2) * g * g;
          P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + 1e-8);
        }
      }
    }
  }

  copyFrom(other) {
    for (let l = 0; l < this.layers.length; l++) {
      this.layers[l].W.set(other.layers[l].W);
      this.layers[l].b.set(other.layers[l].b);
    }
  }

  /** this = tau * this + (1 - tau) * other */
  emaFrom(other, tau) {
    for (let l = 0; l < this.layers.length; l++) {
      for (const key of ["W", "b"]) {
        const P = this.layers[l][key];
        const O = other.layers[l][key];
        for (let i = 0; i < P.length; i++) P[i] = tau * P[i] + (1 - tau) * O[i];
      }
    }
  }

  toJSON() {
    const r = (a) => Array.from(a, (v) => Math.round(v * 1e6) / 1e6);
    return { sizes: this.sizes, acts: this.acts, layers: this.layers.map((L) => ({ W: r(L.W), b: r(L.b) })) };
  }

  static fromJSON(j) {
    const m = new MLP(j.sizes, j.acts, null);
    j.layers.forEach((L, l) => {
      m.layers[l].W.set(L.W);
      m.layers[l].b.set(L.b);
    });
    return m;
  }
}

// ---------- vector helpers ----------
const concat = (a, b) => {
  const out = new Float64Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
};
const add = (a, b) => a.map((v, i) => v + b[i]);
const mse = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
};
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
function cosDist(a, b) {
  let ab = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  if (aa === 0 || bb === 0) return 1;
  return (1 - ab / Math.sqrt(aa * bb)) / 2; // [0, 1]
}
function meanStd(vectors) {
  const n = vectors.length;
  const dim = vectors[0].length;
  const mean = new Float64Array(dim);
  for (const v of vectors) for (let i = 0; i < dim; i++) mean[i] += v[i] / n;
  let spread = 0;
  for (let i = 0; i < dim; i++) {
    let s = 0;
    for (const v of vectors) s += (v[i] - mean[i]) ** 2;
    spread += Math.sqrt(s / n);
  }
  return { mean, std: spread / dim };
}

/** VICReg variance term: mean_j relu(gamma − std_j). Returns loss and per-sample gradients. */
function varianceReg(zs) {
  const n = zs.length;
  const dim = zs[0].length;
  const grads = zs.map(() => new Float64Array(dim));
  if (n < 2) return { loss: 0, grads };
  let loss = 0;
  for (let j = 0; j < dim; j++) {
    let m = 0;
    for (const z of zs) m += z[j] / n;
    let v = 0;
    for (const z of zs) v += (z[j] - m) ** 2 / n;
    const sd = Math.sqrt(v + 1e-4);
    if (sd < VAR_GAMMA) {
      loss += (VAR_GAMMA - sd) / dim;
      for (let i = 0; i < n; i++) grads[i][j] = -((zs[i][j] - m) / (n * sd)) / dim;
    }
  }
  return { loss, grads };
}

// ---------- the model ----------
class HJEPA {
  constructor(seed = 1) {
    const r = rng(seed);
    this.version = VERSION;
    this.dims = { D, Z, A, H, K };
    this.enc1 = new MLP([D, 64, Z], ["tanh", "linear"], r);
    this.tgt1 = new MLP([D, 64, Z], ["tanh", "linear"], null);
    this.tgt1.copyFrom(this.enc1);
    this.act1 = new MLP([D, A], ["tanh"], r);
    this.pred1 = Array.from({ length: K }, () => new MLP([Z + A, 64, Z], ["tanh", "linear"], r, 0.5));
    this.cost1 = new MLP([Z, 16, 1], ["tanh", "linear"], r);
    this.enc2 = new MLP([Z, 32, H], ["tanh", "linear"], r);
    this.tgt2 = new MLP([Z, 32, H], ["tanh", "linear"], null);
    this.tgt2.copyFrom(this.enc2);
    this.act2 = new MLP([D, A], ["tanh"], r);
    this.pred2 = Array.from({ length: K }, () => new MLP([H + A, 32, H], ["tanh", "linear"], r, 0.5));
    this.cost2 = new MLP([H, 16, 1], ["tanh", "linear"], r);
    this.steps = 0;
    this.rand = rng(seed * 7919 + 13);
    // running scale of one-step prediction error, used to normalise surprise and uncertainty
    this.err1 = null;
    this.err2 = null;
    this.unc1 = null;
    this.unc2 = null;
  }

  modules() {
    return ["enc1", "tgt1", "act1", "cost1", "enc2", "tgt2", "act2", "cost2"];
  }

  // ----- inference -----
  encode(s) {
    return this.enc1.forward(featurize(s)).y;
  }
  target(s) {
    return this.tgt1.forward(featurize(s)).y;
  }
  /** One latent step at level 1: ensemble mean, spread. */
  step1(z, a) {
    const ea = this.act1.forward(featurize(a)).y;
    const preds = this.pred1.map((p) => add(z, p.forward(concat(z, ea)).y));
    const { mean, std } = meanStd(preds);
    return { z: mean, unc: std };
  }
  step2(h, intent) {
    const e = this.act2.forward(featurize(intent)).y;
    const preds = this.pred2.map((p) => add(h, p.forward(concat(h, e)).y));
    const { mean, std } = meanStd(preds);
    return { h: mean, unc: std };
  }
  fail1(z) {
    return sigmoid(this.cost1.forward(z).y[0]);
  }
  fail2(h) {
    return sigmoid(this.cost2.forward(h).y[0]);
  }
  high(z) {
    return this.enc2.forward(z).y;
  }

  /** Surprise of an observed transition: error vs the running error scale (1 ≈ typical). */
  surprise(s, a, s2) {
    const z = this.encode(s);
    const { z: zhat, unc } = this.step1(z, a);
    const err = mse(zhat, this.target(s2));
    const scale = this.err1 || err || 1;
    return {
      error: err,
      surprise: scale > 0 ? err / scale : 0,
      uncertainty: this.unc1 ? unc / this.unc1 : 1,
      failure: this.fail1(zhat),
    };
  }

  /**
   * Score one option. option = { label, intent?, steps?: string[] }.
   * Returns predicted failure, goal distance and relative uncertainty (1 ≈ typical).
   */
  imagine(state, option, goal) {
    const z0 = this.encode(state);
    const steps = option.steps && option.steps.length ? option.steps : [option.label];
    let z = z0;
    let unc = 0;
    const trace = [];
    for (const a of steps) {
      const r = this.step1(z, a);
      z = r.z;
      unc += r.unc;
      trace.push({ step: a, failure: round(this.fail1(z)), uncertainty: round(this.unc1 ? r.unc / this.unc1 : 1) });
    }
    unc /= steps.length;
    const out = {
      failure1: this.fail1(z),
      uncertainty1: this.unc1 ? unc / this.unc1 : 1,
      trace,
    };
    const intent = option.intent || option.label;
    const h0 = this.high(z0);
    const r2 = this.step2(h0, intent);
    out.failure2 = this.fail2(r2.h);
    out.uncertainty2 = this.unc2 ? r2.unc / this.unc2 : 1;
    if (goal) {
      const zg = this.target(goal);
      out.goal1 = cosDist(z, zg);
      out.goal2 = cosDist(r2.h, this.tgt2.forward(zg).y);
    }
    return out;
  }

  // ----- training -----
  /**
   * One minibatch update.
   * low:  [{ s, a, s2, y? }]       y = 1 failure, 0 success, undefined unlabeled
   * high: [{ s, intent, s2, y? }]  s / s2 = first state / last result of a chunk
   */
  trainBatch(low, high, lr = 3e-3) {
    this.steps += 1;
    const t = this.steps;
    const stats = { pred1: 0, var1: 0, cost1: 0, pred2: 0, var2: 0, cost2: 0 };

    if (low.length) {
      const gEnc = this.enc1.zeroGrads();
      const gAct = this.act1.zeroGrads();
      const gPred = this.pred1.map((p) => p.zeroGrads());
      const gCost = this.cost1.zeroGrads();
      const encs = low.map((x) => this.enc1.forward(featurize(x.s)));
      const zs = encs.map((e) => e.y);
      const vr = varianceReg(zs);
      stats.var1 = vr.loss;
      let nCost = 0;
      let errSum = 0;
      let uncSum = 0;
      low.forEach((x, i) => {
        const z = zs[i];
        const fa = this.act1.forward(featurize(x.a));
        const target = this.tgt1.forward(featurize(x.s2)).y;
        const gz = new Float64Array(Z);
        for (let j = 0; j < Z; j++) gz[j] = VAR_WEIGHT * vr.grads[i][j];
        const ga = new Float64Array(A);
        const inp = concat(z, fa.y);
        const preds = [];
        this.pred1.forEach((p, k) => {
          const f = p.forward(inp);
          const zhat = add(z, f.y);
          preds.push(zhat);
          if (this.rand() > 0.8) return; // bootstrap: keeps ensemble members distinct
          const l = mse(zhat, target);
          stats.pred1 += l / (low.length * K);
          const gy = zhat.map((v, j) => (2 * (v - target[j])) / Z / K);
          const gin = p.backward(f.cache, gy, gPred[k]);
          for (let j = 0; j < Z; j++) gz[j] += gy[j] + gin[j];
          for (let j = 0; j < A; j++) ga[j] += gin[Z + j];
        });
        const ms = meanStd(preds);
        errSum += mse(ms.mean, target);
        uncSum += ms.std;
        this.enc1.backward(encs[i].cache, gz, gEnc);
        this.act1.backward(fa.cache, ga, gAct);
        if (x.y === 0 || x.y === 1) {
          const c = this.cost1.forward(target); // cost of the outcome latent; no gradient into encoders
          const p = sigmoid(c.y[0]);
          stats.cost1 += -(x.y * Math.log(p + 1e-9) + (1 - x.y) * Math.log(1 - p + 1e-9));
          this.cost1.backward(c.cache, [p - x.y], gCost);
          nCost += 1;
        }
      });
      const n = low.length;
      this.enc1.step(gEnc, n, lr, t);
      this.act1.step(gAct, n, lr, t);
      this.pred1.forEach((p, k) => p.step(gPred[k], n, lr, t));
      if (nCost) {
        this.cost1.step(gCost, nCost, lr, t);
        stats.cost1 /= nCost;
      }
      this.tgt1.emaFrom(this.enc1, EMA);
      this.err1 = runAvg(this.err1, errSum / n);
      this.unc1 = runAvg(this.unc1, uncSum / n);
    }

    if (high.length) {
      const gEnc = this.enc2.zeroGrads();
      const gAct = this.act2.zeroGrads();
      const gPred = this.pred2.map((p) => p.zeroGrads());
      const gCost = this.cost2.zeroGrads();
      const encs = high.map((x) => this.enc2.forward(this.tgt1.forward(featurize(x.s)).y));
      const hs = encs.map((e) => e.y);
      const vr = varianceReg(hs);
      stats.var2 = vr.loss;
      let nCost = 0;
      let errSum = 0;
      let uncSum = 0;
      high.forEach((x, i) => {
        const h = hs[i];
        const fa = this.act2.forward(featurize(x.intent));
        const target = this.tgt2.forward(this.tgt1.forward(featurize(x.s2)).y).y;
        const gh = new Float64Array(H);
        for (let j = 0; j < H; j++) gh[j] = VAR_WEIGHT * vr.grads[i][j];
        const ga = new Float64Array(A);
        const inp = concat(h, fa.y);
        const preds = [];
        this.pred2.forEach((p, k) => {
          const f = p.forward(inp);
          const hhat = add(h, f.y);
          preds.push(hhat);
          if (this.rand() > 0.8) return;
          stats.pred2 += mse(hhat, target) / (high.length * K);
          const gy = hhat.map((v, j) => (2 * (v - target[j])) / H / K);
          const gin = p.backward(f.cache, gy, gPred[k]);
          for (let j = 0; j < H; j++) gh[j] += gy[j] + gin[j];
          for (let j = 0; j < A; j++) ga[j] += gin[H + j];
        });
        const ms = meanStd(preds);
        errSum += mse(ms.mean, target);
        uncSum += ms.std;
        this.enc2.backward(encs[i].cache, gh, gEnc);
        this.act2.backward(fa.cache, ga, gAct);
        if (x.y === 0 || x.y === 1) {
          const c = this.cost2.forward(target);
          const p = sigmoid(c.y[0]);
          stats.cost2 += -(x.y * Math.log(p + 1e-9) + (1 - x.y) * Math.log(1 - p + 1e-9));
          this.cost2.backward(c.cache, [p - x.y], gCost);
          nCost += 1;
        }
      });
      const n = high.length;
      this.enc2.step(gEnc, n, lr, t);
      this.act2.step(gAct, n, lr, t);
      this.pred2.forEach((p, k) => p.step(gPred[k], n, lr, t));
      if (nCost) {
        this.cost2.step(gCost, nCost, lr, t);
        stats.cost2 /= nCost;
      }
      this.tgt2.emaFrom(this.enc2, EMA);
      this.err2 = runAvg(this.err2, errSum / n);
      this.unc2 = runAvg(this.unc2, uncSum / n);
    }
    return stats;
  }

  toJSON() {
    const out = { version: this.version, dims: this.dims, steps: this.steps, err1: this.err1, err2: this.err2, unc1: this.unc1, unc2: this.unc2 };
    for (const k of this.modules()) out[k] = this[k].toJSON();
    out.pred1 = this.pred1.map((p) => p.toJSON());
    out.pred2 = this.pred2.map((p) => p.toJSON());
    return out;
  }

  /** Restore a saved model; returns null when it was saved with other dimensions or version. */
  static fromJSON(j, seed = 1) {
    if (!j || j.version !== VERSION || !j.dims || j.dims.D !== D || j.dims.Z !== Z || j.dims.A !== A || j.dims.H !== H || j.dims.K !== K) return null;
    const m = new HJEPA(seed + (j.steps || 0));
    for (const k of m.modules()) m[k] = MLP.fromJSON(j[k]);
    m.pred1 = j.pred1.map(MLP.fromJSON);
    m.pred2 = j.pred2.map(MLP.fromJSON);
    m.steps = j.steps || 0;
    for (const k of ["err1", "err2", "unc1", "unc2"]) m[k] = typeof j[k] === "number" ? j[k] : null;
    return m;
  }
}

function runAvg(prev, x, beta = 0.95) {
  if (!Number.isFinite(x)) return prev;
  return prev == null ? x : beta * prev + (1 - beta) * x;
}

function round(x, d = 3) {
  return Number.isFinite(x) ? Math.round(x * 10 ** d) / 10 ** d : x;
}

module.exports = { HJEPA, MLP, featurize, tokens, fnv1a, rng, cosDist, varianceReg, meanStd, mse, round, dims: { D, Z, A, H, K } };
