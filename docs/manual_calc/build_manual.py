"""
build_manual.py - membuat docs/manual_calc/manual_calc.html

    python3 docs/manual_calc/build_manual.py

Semua angka di halaman dihitung ulang oleh sim.py saat skrip ini dijalankan.
Font di-embed (base64) supaya halaman terbaca tanpa internet.
"""
import base64
import html
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import sim  # noqa: E402

ROOT = sim.ROOT
esc = html.escape

# ----------------------------------------------------------------------------
# helper HTML
# ----------------------------------------------------------------------------
def f5(v, nd=5):
    return f"{float(v):.{nd}f}"


def num(v, nd=5):
    return f'<span class="n">{float(v):.{nd}f}</span>'


def tint(v, scale=1.0):
    """warna sel diverging halus: biru = positif, merah = negatif."""
    a = min(abs(v) / scale, 1.0) * 0.55
    if a < 0.02:
        return ""
    rgb = "56,189,248" if v >= 0 else "251,113,133"
    return f' style="background:rgba({rgb},{a:.2f})"'


def table(headers, rows, cls="", numeric_from=1, heat=None, caption=None):
    """rows: list of list of str/number. heat: skala heatmap untuk kolom numerik (None=off)."""
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = []
    numre = re.compile(r"^(<b>)?-?\d[\d.]*(e[+-]?\d+)?(</b>)?$")
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            if isinstance(c, (int, float, np.floating, np.integer)) and not isinstance(c, bool):
                st = tint(float(c), heat) if heat else ""
                s = f"{c:.5f}" if isinstance(c, (float, np.floating)) else str(c)
                tds.append(f'<td class="r"{st}>{s}</td>')
            elif isinstance(c, str) and i >= numeric_from and numre.match(c):
                st = tint(float(re.sub(r"<.*?>", "", c)), heat) if heat else ""
                tds.append(f'<td class="r"{st}>{c}</td>')
            else:
                tds.append(f"<td>{c}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tw"><table class="{cls}">{cap}<thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def mat(name, M, rowlab=None, collab=None, heat=None, nd=5):
    M = np.atleast_2d(M)
    heat = heat or max(float(np.abs(M).max()), 1e-9)
    ch = "".join(f"<th>{c}</th>" for c in (collab or [f"k{j}" for j in range(M.shape[1])]))
    rows = []
    for i, r in enumerate(M):
        lab = rowlab[i] if rowlab else f"{i}"
        tds = "".join(f'<td class="r"{tint(v, heat)}>{v:.{nd}f}</td>' for v in r)
        rows.append(f'<tr><th class="rl">{lab}</th>{tds}</tr>')
    return (f'<div class="tw"><table class="mat"><caption>{name} &nbsp;<span class="dim">'
            f'({M.shape[0]}&times;{M.shape[1]})</span></caption><thead><tr><th></th>{ch}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def vec(name, v, nd=5):
    v = np.ravel(v)
    return f'<div class="vec"><span class="vn">{name}</span> = (' + ", ".join(f"{x:.{nd}f}" for x in v) + ")</div>"


def box(kind, title, body):
    return f'<div class="box {kind}"><div class="bt">{title}</div><div class="bb">{body}</div></div>'


def human(body):
    return box("human", "ARTI SEDERHANA", body)


def dev(body):
    return box("dev", "UNTUK PROGRAMMER", body)


def formula(body):
    return box("formula", "RUMUS", body)


def warn(title, body):
    return box("warn", title, body)


def code(s):
    return f"<code>{esc(s)}</code>"


def fold(title, inner, open_=True):
    return f'<details{" open" if open_ else ""}><summary>{title}</summary>{inner}</details>'


def sec(num_, title, inner, lead=""):
    return (f'<section id="s{num_}"><h2><span class="sn">{num_}</span>{title}</h2>'
            f'{"<p class=lead>" + lead + "</p>" if lead else ""}{inner}</section>')


def sub(title, inner):
    return f"<h3>{title}</h3>{inner}"


def tokname(i):
    return esc(sim.show_tok(i))


def svg_chart(series, w=720, h=260, logy=False, ylabel="", xlabel="langkah", marks=None, color="#38bdf8"):
    xs = [p[0] for p in series]
    ys = [p[1] for p in series]
    if logy:
        ys = [math.log10(max(y, 1e-9)) for y in ys]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 46
    sx = lambda x: pad + (x - x0) / max(x1 - x0, 1e-9) * (w - pad - 14)
    sy = lambda y: h - 34 - (y - y0) / max(y1 - y0, 1e-9) * (h - 34 - 16)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    grid = ""
    nt = 5
    for i in range(nt + 1):
        yv = y0 + (y1 - y0) * i / nt
        lab = f"1e{yv:.1f}" if logy else f"{yv:.2f}"
        grid += (f'<line x1="{pad}" y1="{sy(yv):.1f}" x2="{w-14}" y2="{sy(yv):.1f}" class="gl"/>'
                 f'<text x="{pad-6}" y="{sy(yv)+4:.1f}" class="ax" text-anchor="end">{lab}</text>')
    for i in range(6):
        xv = x0 + (x1 - x0) * i / 5
        grid += f'<text x="{sx(xv):.1f}" y="{h-16}" class="ax" text-anchor="middle">{int(xv)}</text>'
    mk = ""
    for (mx, label) in (marks or []):
        mk += (f'<line x1="{sx(mx):.1f}" y1="14" x2="{sx(mx):.1f}" y2="{h-34}" class="mk"/>'
               f'<text x="{sx(mx)+4:.1f}" y="26" class="ax">{label}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{grid}{mk}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round"/>'
            f'<text x="{pad}" y="12" class="ax">{ylabel}</text>'
            f'<text x="{w-14}" y="{h-2}" class="ax" text-anchor="end">{xlabel}</text></svg>')


# ----------------------------------------------------------------------------
# skenario
# ----------------------------------------------------------------------------
sc = sim.build_scenario()
A, B = sc["a"], sc["b"]
UNIQ = sc["uniq"]
mA_old, mB_old = sim.mask_repo_old(A), sim.mask_repo_old(B)
mA, mB = sim.mask_repo_new(A), sim.mask_repo_new(B)
SEQS, MASKS = [A, B], [mA, mB]
m0 = sim.Mini(UNIQ)
P0 = m0.copy_params()
nparam = sum(v.size for v in m0.P.values())
idxA_assist = A.index(sim.ASSIST)   # 15
idxB_assist = B.index(sim.ASSIST)
POS = idxA_assist                   # posisi yang dibedah: <|assistant|> -> kata pertama jawaban
TGT_TOK = A[POS + 1]
L0, G0, INF0, TOT0 = sim.loss_grad(m0, SEQS, MASKS)
LOLD, _, _, TOTOLD = sim.loss_grad(sim.Mini(UNIQ), SEQS, [mA_old, mB_old])
c0 = sim.forward(m0, A)

# ----------------------------------------------------------------------------
# training utama (mask baru)
# ----------------------------------------------------------------------------
mT = sim.Mini(UNIQ)
HIST = sim.train(mT, SEQS, MASKS)
pa = A[:idxA_assist + 1]
pb = B[:idxB_assist + 1]
ans_a = A[idxA_assist + 1:-1]
ans_b = B[idxB_assist + 1:-1]
out_a, tr_a = sim.generate(mT, pa)
out_b, tr_b = sim.generate(mT, pb)
LFINAL, _, INFF, _ = sim.loss_grad(mT, SEQS, MASKS)

# gradient check
rng = np.random.default_rng(0)
worst = 0.0
mchk = sim.Mini(UNIQ)
_, Gc, _, _ = sim.loss_grad(mchk, SEQS, MASKS)
for k in mchk.P:
    for _ in range(3):
        ix = tuple(int(rng.integers(0, s)) for s in mchk.P[k].shape)
        e = 1e-6
        o = mchk.P[k][ix]
        mchk.P[k][ix] = o + e
        lp = sim.loss_grad(mchk, SEQS, MASKS)[0]
        mchk.P[k][ix] = o - e
        lm = sim.loss_grad(mchk, SEQS, MASKS)[0]
        mchk.P[k][ix] = o
        worst = max(worst, abs(Gc[k][ix] - (lp - lm) / (2 * e)))

# ----------------------------------------------------------------------------
# eksperimen kontrol
# ----------------------------------------------------------------------------
def run_exp(seed, ffn_override=None, mask_fn=sim.mask_repo_new):
    old_ffn = sim.FFN
    if ffn_override:
        sim.FFN = ffn_override
    try:
        mm = sim.Mini(UNIQ, seed=seed)
        sim.train(mm, SEQS, [mask_fn(A), mask_fn(B)])
        r = {}
        for nm, pr, tg in (("a", pa, ans_a), ("b", pb, ans_b)):
            o, _ = sim.generate(mm, pr)
            o2, _ = sim.generate(mm, pr[1:])           # tanpa BOS
            r[nm] = (o == tg, o2 == tg, o)
        return mm, r
    finally:
        sim.FFN = old_ffn

EXP = {}
for tag, fn in (("old", sim.mask_repo_old), ("new", sim.mask_repo_new)):
    EXP[tag] = []
    for ffn in (4, 8):
        for seed in range(1, 9):
            _, r = run_exp(seed, ffn, fn)
            EXP[tag].append((ffn, seed, r))

# eksperimen hard-block: tambah kalimat C dengan token berulang
TEXT_C = "<|user|>Say it: ha ha ha<|assistant|>ha ha ha<|end|>"
C = [sim.BOS] + sim.enc.encode(TEXT_C, add_bos=False, add_eos=False)
uniq3 = sorted(set(A) | set(B) | set(C))
mc = sim.Mini(uniq3)
sim.train(mc, [A, B, C], [mA, mB, sim.mask_repo_new(C)])
ic = C.index(sim.ASSIST)
pc, ans_c = C[:ic + 1], C[ic + 1:-1]
out_c0, _ = sim.generate(mc, pc, hard_block=0)
out_c3, _ = sim.generate(mc, pc, hard_block=3)
out_c3r, _ = sim.generate(mc, pc, hard_block=3, rep=1.3)
out_c13, _ = sim.generate(mc, pc, hard_block=0, rep=1.3)

# perturbasi
PERT = []
pa_h = [t if t != 4766 else 11099 for t in pa]
PERT.append(("EN&rarr;JA, kata <b>cat</b> diganti <b>house</b>", sim.generate(mT, pa_h)[0]))
pb_x = [t for t in pb if t != 21873]
PERT.append(("JA&rarr;EN, kata <b>猫</b> dihapus dari sumber", sim.generate(mT, pb_x)[0]))
pb_swap = pb[:]
PERT.append(("JA&rarr;EN, instruksi ditukar jadi &lsquo;日本語に翻訳&rsquo; (sumber tetap Jepang)",
             sim.generate(mT, pa[:pa.index(7611 if False else 74)] + [74] + pb[pb.index(14132):])[0]
             if False else sim.generate(mT, A[:A.index(74) + 1] + pb[pb.index(14132):])[0]))


def toks_to_text(ts):
    b = b"".join(sim.glyph_bytes(sim.TOKS[t]) for t in ts)
    return b.decode("utf-8", "replace")


# ----------------------------------------------------------------------------
# backward tangan untuk satu posisi (diverifikasi terhadap loss_grad penuh)
# ----------------------------------------------------------------------------
def backward_one(m, seq, pos):
    P = m.P
    c = sim.forward(m, seq)
    S = len(seq)
    lg = c["logits"][pos]
    ex = np.exp(lg - lg.max())
    p = ex / ex.sum()
    tg = m.idx[seq[pos + 1]]
    dl = p.copy(); dl[tg] -= 1
    R = {"p": p, "tg": tg, "dl": dl}
    xf = c["xf"][pos]
    R["dE_out"] = np.outer(dl, xf)
    dxf = dl @ P["E"]; R["dxf"] = dxf
    nn, rs = c["nnf"][pos], c["rsf"][pos]
    mean = (dxf * P["gF"] * nn).mean()
    dx1 = rs * (dxf * P["gF"] - nn * mean)
    R.update(nnf=nn, rsf=float(rs[0]) if np.ndim(rs) else float(rs), meanf=mean, dx1=dx1, dgF=dxf * nn)
    ff = c["ff"][pos]
    R["dW2"] = np.outer(ff, dx1)
    dff = dx1 @ P["W2"].T; R["dff"] = dff
    up, gate, su = c["up"][pos], c["gate"][pos], c["su"][pos]
    sg = 1 / (1 + np.exp(-up)); dsilu = su + sg * (1 - su)
    dgate, dup = dff * su, dff * gate * dsilu
    R.update(sg=sg, dsilu=dsilu, dgate=dgate, dup=dup)
    n2 = c["n2"][pos]
    R["dW1"] = np.outer(n2, dgate); R["dW3"] = np.outer(n2, dup)
    dn2 = dgate @ P["W1"].T + dup @ P["W3"].T; R["dn2"] = dn2
    nn2, rs2 = c["nn2"][pos], float(np.ravel(c["rs2"][pos])[0])
    mean2 = (dn2 * P["gf"] * nn2).mean()
    dxm_n = rs2 * (dn2 * P["gf"] - nn2 * mean2)
    dxm = dx1 + dxm_n
    R.update(rs2=rs2, mean2=mean2, dxm_n=dxm_n, dxm=dxm)
    att = c["att"][pos]
    R["dWo"] = np.outer(att, dxm)
    datt = dxm @ P["Wo"].T; R["datt"] = datt
    sc_ = 1 / math.sqrt(sim.DH)
    dQ = np.zeros(sim.D); dK = np.zeros((S, sim.DH)); dV = np.zeros((S, sim.DH))
    heads = []
    for h in range(sim.HQ):
        sl = slice(h * sim.DH, (h + 1) * sim.DH)
        pp = c["P"][h][pos][:pos + 1]
        V_, K_, q = c["V"][:pos + 1], c["K"][:pos + 1], c["Q"][pos][sl]
        dO = datt[sl]
        dp = V_ @ dO
        Dv = float((pp * dp).sum())
        ds = pp * (dp - Dv) * sc_
        dQ[sl] += ds @ K_
        dK[:pos + 1] += np.outer(ds, q)
        dV[:pos + 1] += np.outer(pp, dO)
        heads.append(dict(dO=dO, p=pp, dp=dp, D=Dv, ds=ds, dq=ds @ K_))
    R["heads"] = heads; R["dQ"] = dQ
    dQp = sim.rope(dQ[None], sim.HQ, True, off=pos)[0]
    dKp = sim.rope(dK[:pos + 1], sim.HKV, True)
    R.update(dK=dK[:pos + 1], dKp=dKp, dV=dV[:pos + 1], dQp=dQp)
    n1 = c["n1"]
    R["dWq"] = np.outer(n1[pos], dQp)
    R["dWk"] = n1[:pos + 1].T @ dKp
    R["dWv"] = n1[:pos + 1].T @ dV[:pos + 1]
    dn1 = np.zeros((pos + 1, sim.D))
    dn1[pos] += dQp @ P["Wq"].T
    for j in range(pos + 1):
        dn1[j] += dKp[j] @ P["Wk"].T + dV[j] @ P["Wv"].T
    dx0 = np.zeros((pos + 1, sim.D))
    for j in range(pos + 1):
        nn1 = c["nn1"][j]; rs1 = float(np.ravel(c["rs1"][j])[0])
        mean1 = (dn1[j] * P["ga"] * nn1).mean()
        dx0[j] = rs1 * (dn1[j] * P["ga"] - nn1 * mean1)
    dx0[pos] += dxm
    R["dx0"] = dx0
    R["dga"] = sum((dn1[j] * c["nn1"][j]) for j in range(pos + 1))
    R["dgf"] = dn2 * nn2
    dE = R["dE_out"].copy()
    np.add.at(dE, c["idx"][:pos + 1], dx0)
    R["dE"] = dE
    return R, c

# verifikasi terhadap implementasi penuh
mk_one = [0] * len(A); mk_one[POS] = 1
Lone, Gone, _, _ = sim.loss_grad(m0, [A], [mk_one])
RB, cB = backward_one(m0, A, POS)
chk = {
    "dE": np.abs(RB["dE"] - Gone["E"]).max(), "dWq": np.abs(RB["dWq"] - Gone["Wq"]).max(),
    "dWk": np.abs(RB["dWk"] - Gone["Wk"]).max(), "dWv": np.abs(RB["dWv"] - Gone["Wv"]).max(),
    "dWo": np.abs(RB["dWo"] - Gone["Wo"]).max(), "dW1": np.abs(RB["dW1"] - Gone["W1"]).max(),
    "dW2": np.abs(RB["dW2"] - Gone["W2"]).max(), "dW3": np.abs(RB["dW3"] - Gone["W3"]).max(),
    "dgF": np.abs(RB["dgF"] - Gone["gF"]).max(), "dga": np.abs(RB["dga"] - Gone["ga"]).max(),
    "dgf": np.abs(RB["dgf"] - Gone["gf"]).max(),
}
assert max(chk.values()) < 1e-12, chk

# ----------------------------------------------------------------------------
# adam trace (Wk[0,0], gamma_attn[0])
# ----------------------------------------------------------------------------
def adam_trace(steps=3):
    m = sim.Mini(UNIQ)
    st = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in m.P.items()}
    rows = []
    for step in range(steps):
        L, G, _, _ = sim.loss_grad(m, SEQS, MASKS)
        gn = math.sqrt(sum((g * g).sum() for g in G.values()))
        coef = sim.CLIP / (gn + 1e-6) if gn > sim.CLIP else 1.0
        lr = sim.sched(step); t = step + 1
        rec = {"step": step, "L": L, "gn": gn, "coef": coef, "lr": lr, "t": t}
        for k in m.P:
            g = G[k] * coef
            mm, vv = st[k]
            mm[:] = sim.B1 * mm + (1 - sim.B1) * g
            vv[:] = sim.B2 * vv + (1 - sim.B2) * g * g
            mh, vh = mm / (1 - sim.B1 ** t), vv / (1 - sim.B2 ** t)
            w = sim.WD if k in sim.DECAY else 0.0
            if k == "Wk":
                ix = (0, 0)
                rec["Wk"] = dict(w=m.P[k][ix], g=G[k][ix], gc=g[ix], m=mm[ix], v=vv[ix], mh=mh[ix], vh=vh[ix],
                                 upd=lr * mh[ix] / (math.sqrt(vh[ix]) + sim.EPS), dec=lr * w * m.P[k][ix])
            if k == "ga":
                rec["ga"] = [(m.P[k][i], g[i], lr * mh[i] / (math.sqrt(vh[i]) + sim.EPS)) for i in range(sim.D)]
            m.P[k] = m.P[k] - lr * mh / (np.sqrt(vh) + sim.EPS) - lr * w * m.P[k]
        rec["Wk_new"] = m.P["Wk"][0, 0]
        rec["norms"] = {k: float(np.sqrt((g * g).sum())) for k, g in G.items()}
        rows.append(rec)
    return rows

ADAM = adam_trace(3)

# ----------------------------------------------------------------------------
# font embed
# ----------------------------------------------------------------------------
def font_css():
    d = HERE / "fonts"
    def b64(n):
        return base64.b64encode((d / n).read_bytes()).decode()
    return f"""
@font-face{{font-family:'Space Grotesk';font-weight:500 700;font-style:normal;font-display:swap;
src:url(data:font/woff2;base64,{b64('SpaceGrotesk-latin.woff2')}) format('woff2');}}
@font-face{{font-family:'Inter';font-weight:400 700;font-style:normal;font-display:swap;
src:url(data:font/woff2;base64,{b64('Inter-latin.woff2')}) format('woff2');}}
@font-face{{font-family:'JetBrains Mono';font-weight:400 700;font-style:normal;font-display:swap;
src:url(data:font/woff2;base64,{b64('JetBrainsMono-latin.woff2')}) format('woff2');
unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;}}
@font-face{{font-family:'JetBrains Mono';font-weight:400 700;font-style:normal;font-display:swap;
src:url(data:font/woff2;base64,{b64('JetBrainsMono-latin-ext.woff2')}) format('woff2');
unicode-range:U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,U+2113,U+2C60-2C7F,U+A720-A7FF;}}
"""


from sections import build_all  # noqa: E402

if __name__ == "__main__":
    ctx = {k: v for k, v in globals().items() if not k.startswith("__")}
    html_out = build_all(ctx)
    out = HERE / "manual_calc.html"
    out.write_text(html_out, encoding="utf-8")
    print("tertulis:", out, f"({out.stat().st_size/1024:.0f} KB)")
