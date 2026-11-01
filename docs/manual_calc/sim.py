"""
sim.py - perhitungan manual EN<->JA: model mini + mesin hitung.

Dipakai oleh build_manual.py untuk menghasilkan manual_calc.html. Semua angka di
halaman berasal dari file ini, jadi bisa direproduksi:

    python3 docs/manual_calc/build_manual.py

Yang disimulasikan mengikuti kode repo (bukan kernel CUDA-nya):
  - tokenizer      : scripts/bpe_reference.py (identik dengan bpe/bpe_tokenizer.cpp)
  - mask SFT       : transformer/train_cuda_unified.cu  build_assistant_mask()
  - forward        : transformer/train_step.cuh, cuda/*.cu(h)
  - backward/AdamW : cuda_kernels.cu, attn_kernels.cuh, train_cuda_unified.cu
Aritmatika float64 (numpy), bukan FP16 seperti GPU.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from bpe_reference import BPEReference, BYTE_ENCODER  # noqa: E402

# ----------------------------------------------------------------------------
# Data: dua kalimat, dua arah
# ----------------------------------------------------------------------------
EN = "I have a cat in my house"
JA = "私の家には猫がいます"
SYS = "You are a multilingual translation assistant."
TEXT_EN2JA = f"<|system|>{SYS}<|user|>日本語に翻訳: {EN}<|assistant|>{JA}<|end|>"
TEXT_JA2EN = f"<|system|>{SYS}<|user|>Translate to English: {JA}<|assistant|>{EN}<|end|>"

# Ukuran model mini (rumus identik dengan model nyata)
D, HQ, HKV, DH, FFN = 4, 2, 1, 2, 8
THETA = 500000.0
SEED = 42

# Hyperparameter SFT (mengikuti train_cuda_unified.cu; LR/langkah di-skala)
LR0, WARMUP, STEPS = 3e-2, 20, 600
CLIP, WD = 1.0, 0.01
B1, B2, EPS = 0.9, 0.95, 1e-8

enc = BPEReference(ROOT / "vocab_9")
INV_BYTE = {v: k for k, v in BYTE_ENCODER.items()}
TOKS = enc.tokens
BOS, END, ASSIST = 0, 4, 3


def glyph_bytes(tok: str) -> bytes:
    return bytes(INV_BYTE[c] for c in tok if c in INV_BYTE)


def show_tok(tid: int) -> str:
    """Teks yang bisa dibaca manusia untuk satu ID token."""
    t = TOKS[tid]
    if t.startswith("<|"):
        return t
    b = glyph_bytes(t)
    try:
        return b.decode("utf-8").replace(" ", "\u2423")
    except UnicodeDecodeError:
        return "[" + " ".join(f"{x:02X}" for x in b) + "]"


def encode_full(text, bos=True):
    ids = enc.encode(text, add_bos=bos, add_eos=False)
    return ids


# ----------------------------------------------------------------------------
# Mask SFT (salinan logika build_assistant_mask di train_cuda_unified.cu)
# ----------------------------------------------------------------------------
def mask_repo_old(tokens):
    m = [0] * len(tokens)
    ina = False
    for i, t in enumerate(tokens):
        if t == ASSIST:
            ina = True
        elif t == END:
            if ina:
                m[i] = 1
            ina = False
        if ina and t != ASSIST:
            m[i] = 1
    return m


def mask_repo_new(tokens):
    m = [0] * len(tokens)
    ina = False
    for i, t in enumerate(tokens):
        if t == ASSIST:
            ina = True
            m[i] = 1
        elif t == END:
            ina = False
        elif ina:
            m[i] = 1
    return m


# ----------------------------------------------------------------------------
# Model mini
# ----------------------------------------------------------------------------
class Mini:
    def __init__(self, vocab_ids, seed=SEED):
        self.ids = list(vocab_ids)
        self.idx = {t: i for i, t in enumerate(self.ids)}
        self.V = len(self.ids)
        r = np.random.default_rng(seed)
        kv = HKV * DH
        self.P = {
            "E": r.normal(0, 1, (self.V, D)) * 0.02,
            "Wq": r.normal(0, 1, (D, D)) / math.sqrt(D),
            "Wk": r.normal(0, 1, (D, kv)) / math.sqrt(D),
            "Wv": r.normal(0, 1, (D, kv)) / math.sqrt(D),
            "Wo": r.normal(0, 1, (D, D)) / math.sqrt(D),
            "W1": r.normal(0, 1, (D, FFN)) / math.sqrt(D),
            "W2": r.normal(0, 1, (FFN, D)) / math.sqrt(FFN),
            "W3": r.normal(0, 1, (D, FFN)) / math.sqrt(D),
            "ga": np.ones(D), "gf": np.ones(D), "gF": np.ones(D),
        }

    def copy_params(self):
        return {k: v.copy() for k, v in self.P.items()}


def rms(x, g, eps=1e-6):
    ms = (x * x).mean(-1, keepdims=True)
    rs = 1.0 / np.sqrt(ms + eps)
    n = x * rs
    return n * g, n, rs


def rms_bwd(dy, g, n, rs):
    mean = (dy * g * n).mean(-1, keepdims=True)
    return rs * (dy * g - n * mean), (dy * n).sum(0)


def rope(x, heads, rev=False, off=0):
    S = x.shape[0]
    x = x.reshape(S, heads, DH)
    out = x.copy()
    for p in range(S):
        for j in range(DH // 2):
            a = (p + off) / THETA ** (2 * j / DH)
            c, s = math.cos(a), math.sin(a)
            if rev:
                s = -s
            x0, x1 = x[p, :, 2 * j], x[p, :, 2 * j + 1]
            out[p, :, 2 * j] = x0 * c - x1 * s
            out[p, :, 2 * j + 1] = x0 * s + x1 * c
    return out.reshape(S, heads * DH)


def forward(m, toks):
    P = m.P
    idx = np.array([m.idx[t] for t in toks])
    c = {"idx": idx}
    x0 = P["E"][idx]
    n1, nn1, rs1 = rms(x0, P["ga"])
    qpre, kpre, v = n1 @ P["Wq"], n1 @ P["Wk"], n1 @ P["Wv"]
    Q, K = rope(qpre, HQ), rope(kpre, HKV)
    S = len(toks)
    sc = 1 / math.sqrt(DH)
    att = np.zeros((S, D))
    PM, SC = {}, {}
    for h in range(HQ):
        kv = h // (HQ // HKV)
        q = Q[:, h * DH:(h + 1) * DH]
        k = K[:, kv * DH:(kv + 1) * DH]
        vv = v[:, kv * DH:(kv + 1) * DH]
        s = q @ k.T * sc
        sm = np.where(np.tril(np.ones((S, S))) > 0, s, -1e30)
        e = np.exp(sm - sm.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        PM[h], SC[h] = p, s
        att[:, h * DH:(h + 1) * DH] = p @ vv
    o = att @ P["Wo"]
    xm = x0 + o
    n2, nn2, rs2 = rms(xm, P["gf"])
    gate, up = n2 @ P["W1"], n2 @ P["W3"]
    sg = 1 / (1 + np.exp(-up))
    su = up * sg
    ff = gate * su
    down = ff @ P["W2"]
    x1 = xm + down
    xf, nnf, rsf = rms(x1, P["gF"])
    logits = xf @ P["E"].T
    c.update(x0=x0, n1=n1, nn1=nn1, rs1=rs1, qpre=qpre, kpre=kpre, Q=Q, K=K, V=v,
             P=PM, S=SC, att=att, o=o, xm=xm, n2=n2, nn2=nn2, rs2=rs2, gate=gate,
             up=up, su=su, ff=ff, down=down, x1=x1, xf=xf, nnf=nnf, rsf=rsf,
             logits=logits)
    return c


def loss_grad(m, seqs, masks):
    P = m.P
    cs = [forward(m, t) for t in seqs]
    tot = sum(sum(mk) for mk in masks)
    G = {k: np.zeros_like(v) for k, v in P.items()}
    L = 0.0
    infos = []
    for t, mk, c in zip(seqs, masks, cs):
        S = len(t)
        lg = c["logits"]
        ex = np.exp(lg - lg.max(1, keepdims=True))
        p = ex / ex.sum(1, keepdims=True)
        tg = np.array([m.idx[t[i + 1]] if i + 1 < S else m.idx[END] for i in range(S)])
        dl = p.copy()
        dl[np.arange(S), tg] -= 1
        mkv = np.array(mk, float)
        dl *= mkv[:, None] / tot
        li = -np.log(p[np.arange(S), tg]) * mkv
        L += li.sum()
        infos.append(dict(loss=li, p=p, tg=tg, dl=dl, c=c))

        G["E"] += dl.T @ c["xf"]
        dxf = dl @ P["E"]
        dx1, dg = rms_bwd(dxf, P["gF"], c["nnf"], c["rsf"])
        G["gF"] += dg
        dxm = dx1.copy()
        dff = dx1 @ P["W2"].T
        G["W2"] += c["ff"].T @ dx1
        sg = 1 / (1 + np.exp(-c["up"]))
        su = c["su"]
        dsilu = su + sg * (1 - su)
        dgate = dff * su
        dup = dff * c["gate"] * dsilu
        G["W1"] += c["n2"].T @ dgate
        G["W3"] += c["n2"].T @ dup
        dn2 = dgate @ P["W1"].T + dup @ P["W3"].T
        dx, dg = rms_bwd(dn2, P["gf"], c["nn2"], c["rs2"])
        G["gf"] += dg
        dxm += dx
        dx0 = dxm.copy()
        G["Wo"] += c["att"].T @ dxm
        datt = dxm @ P["Wo"].T
        dQ = np.zeros((S, D))
        dK = np.zeros((S, HKV * DH))
        dV = np.zeros_like(dK)
        sc = 1 / math.sqrt(DH)
        for h in range(HQ):
            kv = h // (HQ // HKV)
            sl = slice(h * DH, (h + 1) * DH)
            sk = slice(kv * DH, (kv + 1) * DH)
            q, k, v = c["Q"][:, sl], c["K"][:, sk], c["V"][:, sk]
            pp, dO = c["P"][h], datt[:, sl]
            dV[:, sk] += pp.T @ dO
            dp = dO @ v.T
            Dv = (pp * dp).sum(1, keepdims=True)
            ds = pp * (dp - Dv) * sc
            dQ[:, sl] += ds @ k
            dK[:, sk] += ds.T @ q
        dQ = rope(dQ, HQ, True)
        dK = rope(dK, HKV, True)
        G["Wq"] += c["n1"].T @ dQ
        G["Wk"] += c["n1"].T @ dK
        G["Wv"] += c["n1"].T @ dV
        dn1 = dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
        dx, dg = rms_bwd(dn1, P["ga"], c["nn1"], c["rs1"])
        G["ga"] += dg
        dx0 += dx
        np.add.at(G["E"], c["idx"], dx0)
    return L / tot, G, infos, tot


def sched(step, total=STEPS, lr0=LR0, warm=WARMUP):
    lr = lr0 * min(1.0, (step + 1) / max(1, warm))
    if step >= warm and total > warm:
        p = min(1.0, max(0.0, (step - warm) / (total - warm)))
        lr = max(lr0 * 0.5 * (1 + math.cos(math.pi * p)), lr0 * 0.1)
    return lr


DECAY = ("E", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "W3")


def train(m, seqs, masks, steps=STEPS, record_every=None):
    st = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in m.P.items()}
    hist = []
    for step in range(steps):
        L, G, _, _ = loss_grad(m, seqs, masks)
        gn = math.sqrt(sum((g * g).sum() for g in G.values()))
        coef = CLIP / (gn + 1e-6) if gn > CLIP else 1.0
        lr = sched(step, steps)
        t = step + 1
        for k in m.P:
            g = G[k] * coef
            mm, vv = st[k]
            mm[:] = B1 * mm + (1 - B1) * g
            vv[:] = B2 * vv + (1 - B2) * g * g
            mh, vh = mm / (1 - B1 ** t), vv / (1 - B2 ** t)
            w = WD if k in DECAY else 0.0
            m.P[k] = m.P[k] - lr * mh / (np.sqrt(vh) + EPS) - lr * w * m.P[k]
        hist.append((step, L, gn, lr, coef))
    return hist


def generate(m, prompt, maxn=24, hard_block=0, rep=1.0):
    """Greedy + KV-cache, meniru Transformer::forward_one_token."""
    P = m.P
    sc = 1 / math.sqrt(DH)
    kc, vc, out, trace, recent = [], [], [], [], []

    def step(tok, pos):
        x = P["E"][m.idx[tok]]
        n1, _, _ = rms(x[None], P["ga"])
        q = rope(n1 @ P["Wq"], HQ, off=pos)
        k = rope(n1 @ P["Wk"], HKV, off=pos)
        v = n1 @ P["Wv"]
        kc.append(k[0]); vc.append(v[0])
        K, V = np.array(kc), np.array(vc)
        att = np.zeros(D)
        for h in range(HQ):
            kv = h // (HQ // HKV)
            s = K[:, kv * DH:(kv + 1) * DH] @ q[0, h * DH:(h + 1) * DH] * sc
            e = np.exp(s - s.max()); w = e / e.sum()
            att[h * DH:(h + 1) * DH] = w @ V[:, kv * DH:(kv + 1) * DH]
        xm = x + att @ P["Wo"]
        n2, _, _ = rms(xm[None], P["gf"])
        ff = (n2 @ P["W1"]) * ((n2 @ P["W3"]) / (1 + np.exp(-(n2 @ P["W3"]))))
        x1 = xm + (ff @ P["W2"])[0]
        xf, _, _ = rms(x1[None], P["gF"])
        return (xf @ P["E"].T)[0]

    lg = None
    for i, t in enumerate(prompt):
        lg = step(t, i)
    pos = len(prompt)
    for n in range(maxn):
        lg = lg.copy()
        cnt = {}
        for t in recent:
            cnt[t] = cnt.get(t, 0) + 1
        if rep > 1:
            for t, cc in cnt.items():
                pen = rep + (cc - 1) * 0.2
                i = m.idx[t]
                lg[i] = lg[i] / pen if lg[i] > 0 else lg[i] * pen
        if hard_block > 0:
            for t, cc in cnt.items():
                if cc >= hard_block and t >= 16:
                    lg[m.idx[t]] = -1e30
            if recent and recent[-1] >= 16:
                lg[m.idx[recent[-1]]] = -1e30
        ex = np.exp(lg - lg.max()); p = ex / ex.sum()
        nx = int(np.argmax(lg)); tok = m.ids[nx]
        order = np.argsort(-lg)[:3]
        trace.append(dict(pos=pos, tok=tok, p=float(p[nx]), logit=float(lg[nx]),
                          top=[(m.ids[i], float(lg[i]), float(p[i])) for i in order]))
        if tok == END:
            break
        out.append(tok); recent.append(tok)
        lg = step(tok, pos); pos += 1
    return out, trace


# ----------------------------------------------------------------------------
# Persiapan data standar (dipakai build_manual.py)
# ----------------------------------------------------------------------------
def build_scenario():
    full_a = encode_full(TEXT_EN2JA)            # dengan system prompt + BOS (36 token, tanpa EOS ganda)
    full_b = encode_full(TEXT_JA2EN)
    # versi simulasi: tanpa system prompt supaya muat dicetak
    def short(text):
        i = text.index("<|user|>")
        return [BOS] + enc.encode(text[i:], add_bos=False, add_eos=False)
    a, b = short(TEXT_EN2JA), short(TEXT_JA2EN)
    uniq = sorted(set(a) | set(b))
    return dict(full_a=full_a, full_b=full_b, a=a, b=b, uniq=uniq)
