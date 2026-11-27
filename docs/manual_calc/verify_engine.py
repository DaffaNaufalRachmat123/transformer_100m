"""verify_engine.py - mencocokkan web/engine.js (JavaScript) dengan sim.py (numpy).

    python3 docs/manual_calc/verify_engine.py

Bobot numpy seed 42 dikirim ke node, lalu loss, gradien, backward manual, langkah AdamW,
dan hasil inferensi dibandingkan. Selisih harus ~1e-12.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import sim  # noqa: E402

sc = sim.build_scenario()
A, B, UNIQ = sc["a"], sc["b"], sc["uniq"]
mA, mB = sim.mask_repo_new(A), sim.mask_repo_new(B)
m = sim.Mini(UNIQ)
POS = A.index(sim.ASSIST)

L, G, _, tot = sim.loss_grad(m, [A, B], [mA, mB])
c = sim.forward(m, A)

# backward satu posisi lewat loss_grad dengan mask satu posisi (rujukan)
mk1 = [0] * len(A); mk1[POS] = 1
L1, G1, _, _ = sim.loss_grad(m, [A], [mk1])

# 3 langkah AdamW
m2 = sim.Mini(UNIQ)
st = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in m2.P.items()}
adam = []
for step in range(3):
    Ls, Gs, _, _ = sim.loss_grad(m2, [A, B], [mA, mB])
    gn = float(np.sqrt(sum((g * g).sum() for g in Gs.values())))
    coef = sim.CLIP / (gn + 1e-6) if gn > sim.CLIP else 1.0
    lr = sim.sched(step); t = step + 1
    for k in m2.P:
        g = Gs[k] * coef
        mm_, vv = st[k]
        mm_[:] = sim.B1 * mm_ + (1 - sim.B1) * g
        vv[:] = sim.B2 * vv + (1 - sim.B2) * g * g
        mh, vh = mm_ / (1 - sim.B1 ** t), vv / (1 - sim.B2 ** t)
        w = sim.WD if k in sim.DECAY else 0.0
        m2.P[k] = m2.P[k] - lr * mh / (np.sqrt(vh) + sim.EPS) - lr * w * m2.P[k]
    adam.append(dict(L=Ls, gn=gn, coef=coef, lr=lr, wk=float(m2.P["Wk"][0, 0])))

# model terlatih penuh (600 langkah) -> inferensi
mt = sim.Mini(UNIQ)
sim.train(mt, [A, B], [mA, mB])
pa, pb = A[:POS + 1], B[:B.index(sim.ASSIST) + 1]
out_a, _ = sim.generate(mt, pa)
out_b, _ = sim.generate(mt, pb)

payload = dict(
    ids=UNIQ, A=A, B=B, mA=mA, mB=mB, POS=POS, END=sim.END,
    P={k: v.tolist() for k, v in m.P.items()},
    expect=dict(L=L, tot=tot, logits=c["logits"].tolist(), xf=c["xf"].tolist(),
                G={k: v.tolist() for k, v in G.items()}, G1={k: v.tolist() for k, v in G1.items()},
                adam=adam, outA=out_a, outB=out_b),
)
tmp = HERE / "web" / "_verify.json"
tmp.write_text(json.dumps(payload))
try:
    r = subprocess.run(["node", str(HERE / "web" / "verify_engine.js"), str(tmp)], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)
finally:
    tmp.unlink(missing_ok=True)
