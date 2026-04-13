// inference.js — Pure-JS port of MAIN v20 forward pass.

export function basis(x, nKnots, xMin, xMax) {
  const t = Math.max(0, Math.min(1, (x - xMin) / Math.max(xMax - xMin, 1e-8)));
  const k = nKnots + 1;
  const B = new Array(k).fill(0);
  if (t === 1) { B[k - 1] = 1; return B; }
  const h = 1 / nKnots;
  const seg = Math.min(nKnots - 1, Math.floor(t / h));
  const lo = seg * h, hi = (seg + 1) * h;
  B[seg]     = (hi - t) / h;
  B[seg + 1] = (t - lo) / h;
  return B;
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

function firing(xByName, mf) {
  const [c0, c1, w0r, c2, c3, w1r, c4, c5, w2r] = mf;
  const w0 = Math.max(w0r, 0.03), w1 = Math.max(w1r, 0.03), w2 = Math.max(w2r, 0.03);
  const g = (x, c, w) => Math.exp(-0.5 * Math.pow((x - c) / w, 2));
  const rf = xByName['ρf'], rsv = xByName['ρsv'], rp = xByName['ρp'];
  const fw = [
    g(rf, c0, w0) * g(rsv, 0, w0) * g(rp, 0, w0),
    g(rf, c2, w1) * g(rsv, c3, w1),
    g(rf, c4, w2) * g(rp,  c5, w2),
    g(rf, 0,  w0),
  ].map(v => v + 1e-8);
  const s = fw[0] + fw[1] + fw[2] + fw[3];
  return fw.map(v => v / s);
}

function ruleLog(M, r, xByName) {
  const { rule_feats, rule_pairs, n_knots, n_int_knots, k_u, k_i, phi, psi, xmn, xmx } = M;
  const feats = rule_feats[r];

  // unary: per-feature basis chunks + trailing bias
  let sum = 0;
  const phiR = phi[r];
  for (let fi = 0; fi < feats.length; fi++) {
    const f = feats[fi];
    const B = basis(xByName[f], n_knots, xmn[f], xmx[f]);
    const s = fi * k_u;
    for (let j = 0; j < k_u; j++) sum += B[j] * phiR[s + j];
  }
  sum += phiR[phiR.length - 1];  // bias

  // interaction: tensor-product grids (row-major, k_i × k_i)
  const pairs = rule_pairs[r] || [];
  const psiR = psi[r] || [];
  for (let pi = 0; pi < pairs.length; pi++) {
    const [fi, fj] = pairs[pi];
    const Bi = basis(xByName[fi], n_int_knots, xmn[fi], xmx[fi]);
    const Bj = basis(xByName[fj], n_int_knots, xmn[fj], xmx[fj]);
    const s = pi * k_i * k_i;
    for (let a = 0; a < k_i; a++) {
      for (let b = 0; b < k_i; b++) {
        sum += Bi[a] * Bj[b] * psiR[s + a * k_i + b];
      }
    }
  }
  return sum;
}

export function predict(M, xByName) {
  const w = firing(xByName, M.mf);
  let logVu = 0;
  for (let r = 0; r < 4; r++) logVu += w[r] * ruleLog(M, r, xByName);
  return Math.exp(Math.max(-10, Math.min(12, logVu)));
}

export function classifyConfig(xByName) {
  const rf = xByName['ρf'], rsv = xByName['ρsv'], rp = xByName['ρp'];
  if (rf > 0 && rsv === 0 && rp === 0)                return 0;  // fiber-only
  if (rf > 0 && rsv > 0)                              return 1;  // fiber+stirrup (triple → 1)
  if (rf > 0 && rsv === 0 && rp > 0)                  return 2;  // fiber+prestress
  return 3;                                                        // no-fiber
}

export function xArrayToDict(xArr, allFeats) {
  const out = {};
  for (let i = 0; i < allFeats.length; i++) out[allFeats[i]] = xArr[i];
  return out;
}
