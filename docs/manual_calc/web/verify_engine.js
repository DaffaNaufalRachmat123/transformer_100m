// verify_engine.js - dipanggil oleh verify_engine.py
const fs = require('fs');
const E = require('./engine.js');
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const m = E.makeModel(d.ids, d.P);
let worst = 0, fail = false;
const flat = a => (Array.isArray(a) ? a.flat(Infinity) : [a]);
function cmp(name, a, b, tol = 1e-9) {
  const x = flat(a), y = flat(b); let w = 0;
  if (x.length !== y.length) { console.log('PANJANG BEDA', name, x.length, y.length); fail = true; return; }
  for (let i = 0; i < x.length; i++) w = Math.max(w, Math.abs(x[i] - y[i]));
  worst = Math.max(worst, w);
  if (w > tol) { fail = true; console.log('GAGAL', name, w); } else console.log('ok   ', name.padEnd(14), w.toExponential(2));
}
const lg = E.lossGrad(m, [d.A, d.B], [d.mA, d.mB], d.END);
cmp('loss', lg.L, d.expect.L);
const c = E.forward(m, d.A);
cmp('logits', c.logits, d.expect.logits); cmp('xf', c.xf, d.expect.xf);
for (const k in d.expect.G) cmp('grad ' + k, lg.G[k], d.expect.G[k]);

// backward manual satu posisi vs loss_grad dengan mask satu posisi
const { R } = E.backwardOne(m, d.A, d.POS);
const G1 = d.expect.G1;
cmp('bwd dE', R.dE, G1.E); cmp('bwd dWq', R.dWq, G1.Wq); cmp('bwd dWk', R.dWk, G1.Wk); cmp('bwd dWv', R.dWv, G1.Wv);
cmp('bwd dWo', R.dWo, G1.Wo); cmp('bwd dW1', R.dW1, G1.W1); cmp('bwd dW2', R.dW2, G1.W2); cmp('bwd dW3', R.dW3, G1.W3);
cmp('bwd dgF', R.dgF, G1.gF); cmp('bwd dga', R.dga, G1.ga); cmp('bwd dgf', R.dgf, G1.gf);

// 3 langkah AdamW
const rows = E.adamTrace(m, [d.A, d.B], [d.mA, d.mB], d.END, 3);
rows.forEach((r, i) => { const e = d.expect.adam[i]; cmp('adam L' + i, r.L, e.L); cmp('adam gn' + i, r.gn, e.gn); cmp('adam Wk' + i, r.Wk.wn, e.wk); });

// training penuh + inferensi
const t0 = Date.now();
const mt = E.makeModel(d.ids, JSON.parse(JSON.stringify(d.P)));
const h = E.train(mt, [d.A, d.B], [d.mA, d.mB], d.END);
console.log('train 600 langkah:', Date.now() - t0, 'ms; loss akhir', h[h.length - 1].L.toExponential(3));
const ia = d.A.indexOf(3), ib = d.B.indexOf(3);
const oa = E.generate(mt, d.A.slice(0, ia + 1), d.END).out, ob = E.generate(mt, d.B.slice(0, ib + 1), d.END).out;
const same = (x, y) => JSON.stringify(x) === JSON.stringify(y);
console.log('inferensi A sama:', same(oa, d.expect.outA), ' B sama:', same(ob, d.expect.outB));
if (!same(oa, d.expect.outA) || !same(ob, d.expect.outB)) fail = true;
console.log(fail ? 'GAGAL' : 'SEMUA COCOK', 'selisih maks', worst.toExponential(2));
process.exit(fail ? 1 : 0);
