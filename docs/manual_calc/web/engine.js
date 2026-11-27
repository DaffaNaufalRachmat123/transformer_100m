/* engine.js - port JavaScript dari sim.py (model mini + forward/backward/AdamW/inferensi).
 * Dipakai oleh animasi di manual_calc.html supaya bobot bisa di-generate ulang di browser.
 * Aritmatika float64 (Number JS) sama seperti numpy. Tidak ada dependensi.
 */
(function (G) {
  'use strict';
  const E = {};
  const D = 4, HQ = 2, HKV = 1, DH = 2, FFN = 8, THETA = 500000.0;
  const KV = HKV * DH;
  E.dims = { D, HQ, HKV, DH, FFN, KV };

  // ------------------------------------------------------------------ util matriks
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const mm = (A, B) => {                       // A[n][k] @ B[k][m]
    const n = A.length, k = B.length, m = B[0].length, out = zeros(n, m);
    for (let i = 0; i < n; i++) for (let t = 0; t < k; t++) { const a = A[i][t]; if (a === 0) continue; const Bt = B[t], o = out[i]; for (let j = 0; j < m; j++) o[j] += a * Bt[j]; }
    return out;
  };
  const mmT = (A, B) => {                      // A[n][k] @ B[m][k]^T
    const n = A.length, m = B.length, k = A[0].length, out = zeros(n, m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { let s = 0; for (let t = 0; t < k; t++) s += A[i][t] * B[j][t]; out[i][j] = s; }
    return out;
  };
  const tmm = (A, B) => {                      // A[k][n]^T @ B[k][m]
    const k = A.length, n = A[0].length, m = B[0].length, out = zeros(n, m);
    for (let t = 0; t < k; t++) for (let i = 0; i < n; i++) { const a = A[t][i]; if (a === 0) continue; for (let j = 0; j < m; j++) out[i][j] += a * B[t][j]; }
    return out;
  };
  const vm = (x, W) => mm([x], W)[0];          // vektor @ matriks
  const vmT = (x, W) => mmT([x], W)[0];        // vektor @ matriks^T
  const outer = (a, b) => a.map(x => b.map(y => x * y));
  const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  const clone = A => A.map(r => Array.isArray(r) ? r.slice() : r);
  const cloneP = P => { const o = {}; for (const k in P) o[k] = Array.isArray(P[k][0]) ? P[k].map(r => r.slice()) : P[k].slice(); return o; };
  E.util = { zeros, mm, mmT, tmm, vm, vmT, outer, addM, cloneP };

  // ------------------------------------------------------------------ PRNG + rumus pembangkit bobot
  // mulberry32 -> U(0,1);  Box-Muller: z = sqrt(-2 ln u1) * cos(2 pi u2);  bobot = sigma * z
  E.mulberry32 = function (a) {
    return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  E.boxMuller = (u1, u2) => Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  E.sigmas = () => ({ E: 0.02, Wq: 1 / Math.sqrt(D), Wk: 1 / Math.sqrt(D), Wv: 1 / Math.sqrt(D), Wo: 1 / Math.sqrt(D),
                      W1: 1 / Math.sqrt(D), W2: 1 / Math.sqrt(FFN), W3: 1 / Math.sqrt(D) });
  E.shapes = V => ({ E: [V, D], Wq: [D, D], Wk: [D, KV], Wv: [D, KV], Wo: [D, D], W1: [D, FFN], W2: [FFN, D], W3: [D, FFN] });
  E.ORDER = ['E', 'Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'W3'];

  /** Bangkitkan bobot baru. Mengembalikan {P, Z, U}: Z = z mentah, U = pasangan (u1,u2) tiap angka. */
  E.generateParams = function (V, seed) {
    const rnd = E.mulberry32(seed >>> 0), sg = E.sigmas(), sh = E.shapes(V);
    const P = {}, Z = {}, U = {};
    for (const k of E.ORDER) {
      const [r, c] = sh[k]; P[k] = zeros(r, c); Z[k] = zeros(r, c); U[k] = zeros(r, c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        let u1 = rnd(); if (u1 < 1e-12) u1 = 1e-12; const u2 = rnd();
        const z = E.boxMuller(u1, u2);
        Z[k][i][j] = z; U[k][i][j] = [u1, u2]; P[k][i][j] = sg[k] * z;
      }
    }
    P.ga = new Array(D).fill(1); P.gf = new Array(D).fill(1); P.gF = new Array(D).fill(1);
    return { P, Z, U };
  };

  E.makeModel = function (ids, P) {
    const idx = {}; ids.forEach((t, i) => { idx[t] = i; });
    return { ids: ids.slice(), idx, V: ids.length, P };
  };

  // ------------------------------------------------------------------ rms / rope
  function rms(x, g, eps = 1e-6) {
    const y = [], n = [], rs = [];
    for (const row of x) {
      let ms = 0; for (const v of row) ms += v * v; ms /= row.length;
      const r = 1 / Math.sqrt(ms + eps), nn = row.map(v => v * r);
      n.push(nn); rs.push(r); y.push(nn.map((v, i) => v * g[i]));
    }
    return { y, n, rs };
  }
  function rmsBwd(dy, g, n, rs) {
    const dx = [], dg = new Array(dy[0].length).fill(0);
    for (let i = 0; i < dy.length; i++) {
      let mean = 0; for (let j = 0; j < dy[i].length; j++) mean += dy[i][j] * g[j] * n[i][j];
      mean /= dy[i].length;
      dx.push(dy[i].map((v, j) => rs[i] * (v * g[j] - n[i][j] * mean)));
      for (let j = 0; j < dy[i].length; j++) dg[j] += dy[i][j] * n[i][j];
    }
    return { dx, dg };
  }
  function rope(x, heads, rev = false, off = 0) {
    const S = x.length, out = x.map(r => r.slice());
    for (let p = 0; p < S; p++) for (let j = 0; j < DH / 2; j++) {
      const a = (p + off) / Math.pow(THETA, (2 * j) / DH), c = Math.cos(a); let s = Math.sin(a); if (rev) s = -s;
      for (let h = 0; h < heads; h++) {
        const i0 = h * DH + 2 * j, x0 = x[p][i0], x1 = x[p][i0 + 1];
        out[p][i0] = x0 * c - x1 * s; out[p][i0 + 1] = x0 * s + x1 * c;
      }
    }
    return out;
  }
  E.rope = rope; E.rms = rms;

  const softmaxRow = (arr, n) => { let mx = -Infinity; for (let j = 0; j < n; j++) if (arr[j] > mx) mx = arr[j]; const e = new Array(arr.length).fill(0); let z = 0; for (let j = 0; j < n; j++) { e[j] = Math.exp(arr[j] - mx); z += e[j]; } for (let j = 0; j < n; j++) e[j] /= z; return e; };

  // ------------------------------------------------------------------ forward
  function forward(m, toks) {
    const P = m.P, S = toks.length, idx = toks.map(t => m.idx[t]);
    const x0 = idx.map(i => P.E[i].slice());
    const r1 = rms(x0, P.ga), n1 = r1.y;
    const qpre = mm(n1, P.Wq), kpre = mm(n1, P.Wk), v = mm(n1, P.Wv);
    const Q = rope(qpre, HQ), K = rope(kpre, HKV);
    const sc = 1 / Math.sqrt(DH), att = zeros(S, D), PM = {}, SC = {};
    for (let h = 0; h < HQ; h++) {
      const kv = Math.floor(h / (HQ / HKV)), s = zeros(S, S), p = zeros(S, S);
      for (let i = 0; i < S; i++) {
        for (let j = 0; j < S; j++) { let d = 0; for (let t = 0; t < DH; t++) d += Q[i][h * DH + t] * K[j][kv * DH + t]; s[i][j] = d * sc; }
        const w = softmaxRow(s[i], i + 1); p[i] = w;
        for (let j = 0; j <= i; j++) for (let t = 0; t < DH; t++) att[i][h * DH + t] += w[j] * v[j][kv * DH + t];
      }
      PM[h] = p; SC[h] = s;
    }
    const o = mm(att, P.Wo), xm = addM(x0, o);
    const r2 = rms(xm, P.gf), n2 = r2.y;
    const gate = mm(n2, P.W1), up = mm(n2, P.W3);
    const sg = up.map(r => r.map(u => 1 / (1 + Math.exp(-u))));
    const su = up.map((r, i) => r.map((u, j) => u * sg[i][j]));
    const ff = gate.map((r, i) => r.map((g, j) => g * su[i][j]));
    const down = mm(ff, P.W2), x1 = addM(xm, down);
    const rf = rms(x1, P.gF), xf = rf.y;
    const logits = mmT(xf, P.E);
    return { idx, x0, n1, nn1: r1.n, rs1: r1.rs, qpre, kpre, Q, K, V: v, P: PM, S: SC, att, o, xm, n2, nn2: r2.n, rs2: r2.rs,
             gate, up, su, ff, down, x1, xf, nnf: rf.n, rsf: rf.rs, logits };
  }
  E.forward = forward;

  function softmaxFull(lg) { return lg.map(r => softmaxRow(r, r.length)); }

  // ------------------------------------------------------------------ loss + gradien seluruh batch
  function lossGrad(m, seqs, masks, END) {
    const P = m.P, cs = seqs.map(t => forward(m, t));
    let tot = 0; for (const mk of masks) for (const v of mk) tot += v;
    const G = {}; for (const k in P) G[k] = Array.isArray(P[k][0]) ? zeros(P[k].length, P[k][0].length) : new Array(P[k].length).fill(0);
    let L = 0; const infos = [];
    for (let q = 0; q < seqs.length; q++) {
      const t = seqs[q], mk = masks[q], c = cs[q], S = t.length;
      const p = softmaxFull(c.logits);
      const tg = t.map((_, i) => (i + 1 < S ? m.idx[t[i + 1]] : m.idx[END]));
      const li = new Array(S).fill(0);
      const dl = p.map((r, i) => { const o = r.slice(); o[tg[i]] -= 1; const f = mk[i] / tot; return o.map(v => v * f); });
      for (let i = 0; i < S; i++) { li[i] = -Math.log(p[i][tg[i]]) * mk[i]; L += li[i]; }
      infos.push({ loss: li, p, tg, dl, c });

      G.E = addM(G.E, tmm(dl, c.xf));
      const dxf = mm(dl, P.E);
      let r = rmsBwd(dxf, P.gF, c.nnf, c.rsf); const dx1 = r.dx; G.gF = G.gF.map((v, i) => v + r.dg[i]);
      const dxm = clone(dx1);
      const dff = mmT(dx1, P.W2); G.W2 = addM(G.W2, tmm(c.ff, dx1));
      const sg = c.up.map(rr => rr.map(u => 1 / (1 + Math.exp(-u))));
      const dsilu = c.su.map((rr, i) => rr.map((s, j) => s + sg[i][j] * (1 - s)));
      const dgate = dff.map((rr, i) => rr.map((d, j) => d * c.su[i][j]));
      const dup = dff.map((rr, i) => rr.map((d, j) => d * c.gate[i][j] * dsilu[i][j]));
      G.W1 = addM(G.W1, tmm(c.n2, dgate)); G.W3 = addM(G.W3, tmm(c.n2, dup));
      const dn2 = addM(mmT(dgate, P.W1), mmT(dup, P.W3));
      r = rmsBwd(dn2, P.gf, c.nn2, c.rs2); G.gf = G.gf.map((v, i) => v + r.dg[i]);
      for (let i = 0; i < S; i++) for (let j = 0; j < D; j++) dxm[i][j] += r.dx[i][j];
      const dx0 = clone(dxm);
      G.Wo = addM(G.Wo, tmm(c.att, dxm));
      const datt = mmT(dxm, P.Wo);
      const dQ = zeros(S, D), dK = zeros(S, KV), dV = zeros(S, KV), sc = 1 / Math.sqrt(DH);
      for (let h = 0; h < HQ; h++) {
        const kv = Math.floor(h / (HQ / HKV)), pp = c.P[h];
        for (let i = 0; i < S; i++) {
          const dp = new Array(i + 1).fill(0); let Dv = 0;
          for (let j = 0; j <= i; j++) { let d = 0; for (let tt = 0; tt < DH; tt++) d += datt[i][h * DH + tt] * c.V[j][kv * DH + tt]; dp[j] = d; Dv += pp[i][j] * d; }
          for (let j = 0; j <= i; j++) {
            const ds = pp[i][j] * (dp[j] - Dv) * sc;
            for (let tt = 0; tt < DH; tt++) {
              dV[j][kv * DH + tt] += pp[i][j] * datt[i][h * DH + tt];
              dQ[i][h * DH + tt] += ds * c.K[j][kv * DH + tt];
              dK[j][kv * DH + tt] += ds * c.Q[i][h * DH + tt];
            }
          }
        }
      }
      const dQr = rope(dQ, HQ, true), dKr = rope(dK, HKV, true);
      G.Wq = addM(G.Wq, tmm(c.n1, dQr)); G.Wk = addM(G.Wk, tmm(c.n1, dKr)); G.Wv = addM(G.Wv, tmm(c.n1, dV));
      const dn1 = addM(addM(mmT(dQr, P.Wq), mmT(dKr, P.Wk)), mmT(dV, P.Wv));
      r = rmsBwd(dn1, P.ga, c.nn1, c.rs1); G.ga = G.ga.map((v, i) => v + r.dg[i]);
      for (let i = 0; i < S; i++) for (let j = 0; j < D; j++) dx0[i][j] += r.dx[i][j];
      for (let i = 0; i < S; i++) { const row = G.E[c.idx[i]]; for (let j = 0; j < D; j++) row[j] += dx0[i][j]; }
    }
    return { L: L / tot, G, infos, tot };
  }
  E.lossGrad = lossGrad;

  // ------------------------------------------------------------------ jadwal LR + AdamW
  const H = { LR0: 3e-2, WARMUP: 20, STEPS: 600, CLIP: 1.0, WD: 0.01, B1: 0.9, B2: 0.95, EPS: 1e-8 };
  E.hyper = H;
  E.DECAY = ['E', 'Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'W3'];
  E.sched = function (step, total = H.STEPS, lr0 = H.LR0, warm = H.WARMUP) {
    let lr = lr0 * Math.min(1, (step + 1) / Math.max(1, warm));
    if (step >= warm && total > warm) {
      const p = Math.min(1, Math.max(0, (step - warm) / (total - warm)));
      lr = Math.max(lr0 * 0.5 * (1 + Math.cos(Math.PI * p)), lr0 * 0.1);
    }
    return lr;
  };
  const sumsq = G => { let s = 0; for (const k in G) { const g = G[k]; if (Array.isArray(g[0])) for (const r of g) for (const v of r) s += v * v; else for (const v of g) s += v * v; } return s; };
  const mapT = (a, f) => Array.isArray(a[0]) ? a.map((r, i) => r.map((v, j) => f(v, i, j))) : a.map((v, i) => f(v, i));

  function adamState(P) { const st = {}; for (const k in P) st[k] = { m: mapT(P[k], () => 0), v: mapT(P[k], () => 0) }; return st; }
  /** satu langkah training; mengembalikan info + mengubah m.P. rec = true -> catat detail Wk[0][0] dan gamma_attn. */
  function stepOnce(m, st, seqs, masks, step, total, END, rec) {
    const { L, G } = lossGrad(m, seqs, masks, END);
    const gn = Math.sqrt(sumsq(G)), coef = gn > H.CLIP ? H.CLIP / (gn + 1e-6) : 1;
    const lr = E.sched(step, total), t = step + 1, out = { step, L, gn, coef, lr, t };
    if (rec) out.norms = {}, out.G = G;
    for (const k in m.P) {
      const w = E.DECAY.includes(k) ? H.WD : 0, s = st[k];
      if (rec) { const g = G[k]; out.norms[k] = Math.sqrt(sumsq({ x: g })); }
      const isMat = Array.isArray(m.P[k][0]);
      const upd = (g, mi, vi, p) => {
        const gc = g * coef, mn = H.B1 * mi + (1 - H.B1) * gc, vn = H.B2 * vi + (1 - H.B2) * gc * gc;
        const mh = mn / (1 - Math.pow(H.B1, t)), vh = vn / (1 - Math.pow(H.B2, t));
        const u = lr * mh / (Math.sqrt(vh) + H.EPS), dec = lr * w * p;
        return { gc, mn, vn, mh, vh, u, dec, pn: p - u - dec };
      };
      if (isMat) {
        for (let i = 0; i < m.P[k].length; i++) for (let j = 0; j < m.P[k][0].length; j++) {
          const r = upd(G[k][i][j], s.m[i][j], s.v[i][j], m.P[k][i][j]);
          if (rec && k === 'Wk' && i === 0 && j === 0) out.Wk = { w: m.P[k][i][j], g: G[k][i][j], gc: r.gc, m: r.mn, v: r.vn, mh: r.mh, vh: r.vh, upd: r.u, dec: r.dec, wn: r.pn };
          s.m[i][j] = r.mn; s.v[i][j] = r.vn; m.P[k][i][j] = r.pn;
        }
      } else {
        if (rec && k === 'ga') out.ga = [];
        for (let i = 0; i < m.P[k].length; i++) {
          const r = upd(G[k][i], s.m[i], s.v[i], m.P[k][i]);
          if (rec && k === 'ga') out.ga.push({ w: m.P[k][i], gc: r.gc, upd: r.u });
          s.m[i] = r.mn; s.v[i] = r.vn; m.P[k][i] = r.pn;
        }
      }
    }
    return out;
  }
  E.train = function (m, seqs, masks, END, steps = H.STEPS) {
    const st = adamState(m.P), hist = [];
    for (let s = 0; s < steps; s++) { const r = stepOnce(m, st, seqs, masks, s, steps, END, false); hist.push({ step: s, L: r.L, gn: r.gn, lr: r.lr, coef: r.coef }); }
    return hist;
  };
  E.adamTrace = function (m0, seqs, masks, END, n = 3) {
    const m = { ids: m0.ids, idx: m0.idx, V: m0.V, P: cloneP(m0.P) }, st = adamState(m.P), rows = [];
    for (let s = 0; s < n; s++) rows.push(stepOnce(m, st, seqs, masks, s, H.STEPS, END, true));
    return rows;
  };

  // ------------------------------------------------------------------ backward manual satu posisi
  E.backwardOne = function (m, seq, pos) {
    const P = m.P, c = forward(m, seq), S = seq.length, R = {};
    const lg = c.logits[pos], p = softmaxRow(lg, lg.length), tg = m.idx[seq[pos + 1]];
    const dl = p.slice(); dl[tg] -= 1; Object.assign(R, { p, tg, dl });
    const xf = c.xf[pos];
    R.dE_out = outer(dl, xf);
    const dxf = vm(dl, P.E); R.dxf = dxf;
    const nn = c.nnf[pos], rs = c.rsf[pos];
    let mean = 0; for (let i = 0; i < D; i++) mean += dxf[i] * P.gF[i] * nn[i]; mean /= D;
    const dx1 = dxf.map((v, i) => rs * (v * P.gF[i] - nn[i] * mean));
    Object.assign(R, { nnf: nn, rsf: rs, meanf: mean, dx1, dgF: dxf.map((v, i) => v * nn[i]) });
    const ff = c.ff[pos]; R.dW2 = outer(ff, dx1);
    const dff = vmT(dx1, P.W2); R.dff = dff;
    const up = c.up[pos], gate = c.gate[pos], su = c.su[pos];
    const sg = up.map(u => 1 / (1 + Math.exp(-u))), dsilu = su.map((s, k) => s + sg[k] * (1 - s));
    const dgate = dff.map((d, k) => d * su[k]), dup = dff.map((d, k) => d * gate[k] * dsilu[k]);
    Object.assign(R, { sg, dsilu, dgate, dup });
    const n2 = c.n2[pos]; R.dW1 = outer(n2, dgate); R.dW3 = outer(n2, dup);
    const a = vmT(dgate, P.W1), b = vmT(dup, P.W3), dn2 = a.map((v, i) => v + b[i]); R.dn2 = dn2;
    const nn2 = c.nn2[pos], rs2 = c.rs2[pos];
    let mean2 = 0; for (let i = 0; i < D; i++) mean2 += dn2[i] * P.gf[i] * nn2[i]; mean2 /= D;
    const dxm_n = dn2.map((v, i) => rs2 * (v * P.gf[i] - nn2[i] * mean2)), dxm = dx1.map((v, i) => v + dxm_n[i]);
    Object.assign(R, { rs2, mean2, dxm_n, dxm, nn2 });
    const att = c.att[pos]; R.dWo = outer(att, dxm);
    const datt = vmT(dxm, P.Wo); R.datt = datt;
    const sc = 1 / Math.sqrt(DH), dQ = new Array(D).fill(0), dK = zeros(pos + 1, DH), dV = zeros(pos + 1, DH), heads = [];
    for (let h = 0; h < HQ; h++) {
      const kvh = Math.floor(h / (HQ / HKV)), pp = c.P[h][pos].slice(0, pos + 1), q = c.Q[pos].slice(h * DH, (h + 1) * DH), dO = datt.slice(h * DH, (h + 1) * DH);
      const dp = [], ds = []; let Dv = 0;
      for (let j = 0; j <= pos; j++) { let d = 0; for (let t = 0; t < DH; t++) d += c.V[j][kvh * DH + t] * dO[t]; dp.push(d); Dv += pp[j] * d; }
      const dq = new Array(DH).fill(0);
      for (let j = 0; j <= pos; j++) {
        const dsj = pp[j] * (dp[j] - Dv) * sc; ds.push(dsj);
        for (let t = 0; t < DH; t++) { dq[t] += dsj * c.K[j][kvh * DH + t]; dK[j][t] += dsj * q[t]; dV[j][t] += pp[j] * dO[t]; dQ[h * DH + t] += dsj * c.K[j][kvh * DH + t]; }
      }
      heads.push({ dO, p: pp, dp, D: Dv, ds, dq });
    }
    R.heads = heads; R.dQ = dQ;
    const dQp = rope([dQ], HQ, true, pos)[0], dKp = rope(dK, HKV, true);
    Object.assign(R, { dK, dKp, dV, dQp });
    const n1 = c.n1.slice(0, pos + 1);
    R.dWq = outer(c.n1[pos], dQp); R.dWk = tmm(n1, dKp); R.dWv = tmm(n1, dV);
    const dn1 = zeros(pos + 1, D);
    const tq = vmT(dQp, P.Wq); for (let i = 0; i < D; i++) dn1[pos][i] += tq[i];
    for (let j = 0; j <= pos; j++) { const a1 = vmT(dKp[j], P.Wk), b1 = vmT(dV[j], P.Wv); for (let i = 0; i < D; i++) dn1[j][i] += a1[i] + b1[i]; }
    const dx0 = zeros(pos + 1, D), dga = new Array(D).fill(0);
    for (let j = 0; j <= pos; j++) {
      const nn1 = c.nn1[j], rs1 = c.rs1[j]; let mean1 = 0;
      for (let i = 0; i < D; i++) mean1 += dn1[j][i] * P.ga[i] * nn1[i]; mean1 /= D;
      for (let i = 0; i < D; i++) { dx0[j][i] = rs1 * (dn1[j][i] * P.ga[i] - nn1[i] * mean1); dga[i] += dn1[j][i] * nn1[i]; }
    }
    for (let i = 0; i < D; i++) dx0[pos][i] += dxm[i];
    Object.assign(R, { dx0, dga, dgf: dn2.map((v, i) => v * nn2[i]), dn1 });
    const dE = R.dE_out.map(r => r.slice());
    for (let j = 0; j <= pos; j++) for (let i = 0; i < D; i++) dE[c.idx[j]][i] += dx0[j][i];
    R.dE = dE;
    return { R, c };
  };

  // ------------------------------------------------------------------ inferensi greedy + KV cache
  E.generate = function (m, prompt, END, maxn = 24) {
    const P = m.P, sc = 1 / Math.sqrt(DH), kc = [], vc = [], out = [], trace = [];
    const step = (tok, pos) => {
      const x = P.E[m.idx[tok]], n1 = rms([x], P.ga).y[0];
      const q = rope([vm(n1, P.Wq)], HQ, false, pos)[0], k = rope([vm(n1, P.Wk)], HKV, false, pos)[0], v = vm(n1, P.Wv);
      kc.push(k); vc.push(v);
      const att = new Array(D).fill(0);
      for (let h = 0; h < HQ; h++) {
        const kvh = Math.floor(h / (HQ / HKV)), s = kc.map(kk => { let d = 0; for (let t = 0; t < DH; t++) d += kk[kvh * DH + t] * q[h * DH + t]; return d * sc; });
        const w = softmaxRow(s, s.length);
        for (let j = 0; j < w.length; j++) for (let t = 0; t < DH; t++) att[h * DH + t] += w[j] * vc[j][kvh * DH + t];
      }
      const o = vm(att, P.Wo), xm = x.map((v2, i) => v2 + o[i]);
      const n2 = rms([xm], P.gf).y[0], gt = vm(n2, P.W1), up = vm(n2, P.W3);
      const ff = gt.map((g, i) => g * (up[i] / (1 + Math.exp(-up[i]))));
      const dn = vm(ff, P.W2), x1 = xm.map((v2, i) => v2 + dn[i]);
      return mmT([rms([x1], P.gF).y[0]], P.E)[0];
    };
    let lg = null;
    prompt.forEach((t, i) => { lg = step(t, i); });
    let pos = prompt.length;
    for (let n = 0; n < maxn; n++) {
      const pr = softmaxRow(lg, lg.length); let nx = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[nx]) nx = i;
      const order = lg.map((v, i) => i).sort((a, b) => lg[b] - lg[a]).slice(0, 3);
      const tok = m.ids[nx];
      trace.push({ pos, tok, p: pr[nx], logit: lg[nx], top: order.map(i => [m.ids[i], lg[i], pr[i]]), cache: kc.length });
      if (tok === END) break;
      out.push(tok); lg = step(tok, pos); pos++;
    }
    return { out, trace };
  };

  E.softmax = a => softmaxRow(a, a.length);
  G.Engine = E;
  if (typeof module !== 'undefined') module.exports = E;
})(typeof window !== 'undefined' ? window : globalThis);
