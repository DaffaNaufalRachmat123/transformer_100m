"""sec_b.py - bagian 7..13 (backward, optimizer, kurva, inferensi, eksperimen, model nyata)."""
import math

import numpy as np

import sec_a
from sec_a import cells, chip, chips, setup as _setup_a


def setup(ctx):
    globals().update(ctx)
    _setup_a(ctx)


def v5(v):
    return ", ".join(f"{x:.5f}" for x in np.ravel(v))


# ---------------------------------------------------------------------------
# 7. Backward
# ---------------------------------------------------------------------------
def s7():
    R, c = RB, cB
    P = m0.P
    pos = POS
    tg = R["tg"]
    # --- 7.1
    dl = R["dl"]
    rows = [[i, tokname(uid) + (" &larr; target" if i == tg else ""), f"{R['p'][i]:.5f}", f"{dl[i]:.5f}"]
            for i, uid in enumerate(UNIQ)]
    s71 = (
        human(f"Backward menjawab: &lsquo;kalau bobot digeser sedikit, ke arah mana loss turun?&rsquo; Dimulai dari ujung. Untuk logit token target, "
              f"gradiennya <b>p &minus; 1</b> (negatif: naikkan skornya). Untuk token lain gradiennya <b>p</b> (positif: turunkan).")
        + formula("&part;loss/&part;logit<sub>v</sub> = p<sub>v</sub> &minus; 1[v = target]")
        + fold("gradien semua logit", table(["baris", "token", "p", "dlogit"], rows, cls="cmp", numeric_from=2))
        + dev("Dalam batch nyata gradien ini dikali 1/jumlah posisi bermask (<code>launch_scale_fp16(d_logits, 1/tok_count)</code> di <code>train_step_ce</code>). "
              "Di sini hanya satu posisi diisolasi, jadi pembaginya 1. Jalur SFT tidak memakai loss-scale 4096 seperti pretraining; itu hanya aman di float64, "
              "di FP16 GPU gradien kecil bisa lenyap.")
    )
    # --- 7.2
    s72 = (
        human("Karena tabel embedding dipakai dua kali (untuk membaca token masuk dan menulis logit keluar), gradiennya datang dari <b>dua arah</b> dan dijumlahkan.")
        + formula("dE<sub>output</sub> = dlogit&#7488; &times; xf &nbsp;&nbsp; dxf = dlogit &middot; E")
        + table(["", "d0", "d1", "d2", "d3"],
                [[f"dE[{tokname(TGT_TOK)}] sisi output"] + [f"{v:.5f}" for v in R["dE_out"][tg]],
                 ["dxf = dlogit&middot;E"] + [f"{v:.5f}" for v in R["dxf"]]], cls="cmp", heat=1.0)
    )
    # --- 7.3
    s73 = (
        human("Cara 'membatalkan' RMSNorm. Gradien harus memperhitungkan bahwa mengubah satu angka ikut mengubah rata-rata semua angka.")
        + formula("dx = rsqrt &middot; ( dy&middot;&gamma; &minus; normed &middot; mean(dy&middot;&gamma;&middot;normed) ) &nbsp;&nbsp; d&gamma; = dy &middot; normed")
        + table(["", "d0", "d1", "d2", "d3"],
                [["normed (xf / &gamma;)"] + [f"{v:.5f}" for v in R["nnf"]],
                 ["dy&middot;&gamma;&middot;normed"] + [f"{v:.5f}" for v in R["dxf"] * P['gF'] * R['nnf']],
                 ["dx1"] + [f"<b>{v:.5f}</b>" for v in R["dx1"]], ["d&gamma;_final"] + [f"{v:.5f}" for v in R["dgF"]]], cls="cmp", heat=0.2)
        + f"<p>rsqrt = {R['rsf']:.5f}; mean(dy&middot;&gamma;&middot;normed) = {R['meanf']:.6f}.</p>"
    )
    # --- 7.4
    s74 = (
        human("Mundur melewati FFN: gradien dibagi ke jalur <b>gate</b> dan jalur <b>up</b>. Untuk SiLU, turunannya "
              "<b>silu(u) + sigmoid(u) &times; (1 &minus; silu(u))</b>.")
        + formula("dW2 = ffᵀ &middot; dx1 &nbsp; dff = dx1 &middot; W2ᵀ &nbsp; dgate = dff &middot; silu(up) &nbsp; dup = dff &middot; gate &middot; silu&prime;(up)")
        + table(["k", "dff", "sigmoid(up)", "silu&prime;(up)", "dgate", "dup"],
                [[k, f"{R['dff'][k]:.5f}", f"{R['sg'][k]:.5f}", f"{R['dsilu'][k]:.5f}", f"{R['dgate'][k]:.5f}", f"{R['dup'][k]:.5f}"] for k in range(8)],
                cls="cmp", numeric_from=1)
        + fold("dW2 (8&times;4)", mat("dW2", R["dW2"], [str(i) for i in range(8)], ["d0", "d1", "d2", "d3"], nd=5), False)
        + fold("dW1 dan dW3 (4&times;8)", mat("dW1", R["dW1"], ["0", "1", "2", "3"], [f"g{j}" for j in range(8)])
               + mat("dW3", R["dW3"], ["0", "1", "2", "3"], [f"u{j}" for j in range(8)]), False)
        + f"<p>dn2 = dgate&middot;W1ᵀ + dup&middot;W3ᵀ = ({v5(R['dn2'])})</p>"
    )
    # --- 7.5 7.6
    s75 = (
        human("Gradien yang kembali dari RMSNorm kedua <b>ditambahkan</b> ke gradien jalur pintas (residual), karena x_mid dipakai di dua tempat.")
        + table(["", "d0", "d1", "d2", "d3"],
                [["lewat jalur pintas (= dx1)"] + [f"{v:.5f}" for v in R["dx1"]],
                 ["lewat RMSNorm kedua"] + [f"{v:.5f}" for v in R["dxm_n"]],
                 ["dx_mid = jumlah"] + [f"<b>{v:.5f}</b>" for v in R["dxm"]]], cls="cmp", heat=0.3)
        + f"<p>rsqrt2 = {R['rs2']:.5f}; mean(dn2&middot;&gamma;&middot;normed) = {R['mean2']:.6f}.</p>"
        + fold("dWo = attᵀ &middot; dx_mid", mat("dWo", R["dWo"], ["0", "1", "2", "3"], ["o0", "o1", "o2", "o3"]), False)
        + f"<p>datt = dx_mid&middot;Woᵀ = ({v5(R['datt'])}); head 0 menerima dO = ({v5(R['datt'][:2])}), head 1 menerima dO = ({v5(R['datt'][2:])}).</p>"
    )
    # --- 7.7 attention
    def head_back(h):
        H = R["heads"][h]
        rows = [[j, tokname(A[j]), f"{H['p'][j]:.5f}", f"{H['dp'][j]:.5f}", f"{H['ds'][j]:.5f}"] for j in range(pos + 1)]
        return (f"<h4>Head {h}: dO = ({v5(H['dO'])}), D = &Sigma;p&middot;dp = {H['D']:.6f}</h4>"
                + fold("p, dp, dscore tiap posisi", table(["j", "token", "p", "dp = dO&middot;v", "dscore = p&middot;(dp&minus;D)&middot;&frasl;&radic;2"], rows, cls="cmp", numeric_from=2))
                + f"<p>dQ head {h} = &Sigma; dscore&middot;K = ({v5(H['dq'])})</p>")
    rowsKV = [[j, tokname(A[j]), f"{R['dK'][j][0]:.5f}", f"{R['dK'][j][1]:.5f}", f"{R['dKp'][j][0]:.5f}", f"{R['dKp'][j][1]:.5f}",
               f"{R['dV'][j][0]:.5f}", f"{R['dV'][j][1]:.5f}"] for j in range(pos + 1)]
    s77 = (
        human("Gradien mengalir ke tiga tempat: ke Q token ini, ke K semua token sebelumnya, dan ke V semua token sebelumnya. "
              "Karena dua head Q memakai satu head K/V (GQA), gradien K dan V adalah <b>jumlah kontribusi kedua head</b>.")
        + formula("dp<sub>j</sub> = dO &middot; v<sub>j</sub> &nbsp; D = &Sigma;p<sub>j</sub>dp<sub>j</sub> &nbsp; dscore<sub>j</sub> = p<sub>j</sub>(dp<sub>j</sub> &minus; D)/&radic;d &nbsp; "
                  "dQ += dscore&middot;K &nbsp; dK<sub>j</sub> += dscore&middot;Q &nbsp; dV<sub>j</sub> += p<sub>j</sub>&middot;dO")
        + head_back(0) + head_back(1)
        + f"<p>dQ gabungan (ruang RoPE) = ({v5(R['dQ'])})</p>"
        + fold("dK dan dV tiap posisi",
               table(["j", "token", "dK k0", "dK k1", "dK sebelum RoPE k0", "k1", "dV v0", "v1"], rowsKV, cls="cmp", numeric_from=2), False)
        + dev("Ini adalah <code>gqa_attn_bwd_kernel</code>: dua pass per query (pass 1 menghitung D, pass 2 mengakumulasi dQ/dK/dV), dengan "
              "<code>atomicAdd</code> ke buffer FP32 karena banyak query menulis ke baris K/V yang sama.")
    )
    s78 = (
        human("RoPE diputar balik dengan sudut negatif, lalu gradien diteruskan ke bobot Q/K/V dan ke RMSNorm pertama.")
        + f"<p>dQ posisi {pos} sebelum RoPE balik = ({v5(R['dQ'])}) &rarr; sesudah putar &minus;{pos} rad = <b>({v5(R['dQp'])})</b>.</p>"
        + fold("dWq, dWk, dWv", mat("dWq (hanya posisi ini punya query bermask)", R["dWq"], ["0", "1", "2", "3"], ["q0", "q1", "q2", "q3"])
               + mat("dWk = &Sigma;<sub>j</sub> n1[j]ᵀ&middot;dK&prime;[j]", R["dWk"], ["0", "1", "2", "3"], ["k0", "k1"])
               + mat("dWv = &Sigma;<sub>j</sub> n1[j]ᵀ&middot;dV[j]", R["dWv"], ["0", "1", "2", "3"], ["v0", "v1"]))
        + fold("gradien ke baris embedding tiap posisi (setelah RMSNorm pertama)",
               table(["pos", "token", "baris E", "dx0 d0", "d1", "d2", "d3"],
                     [[j, tokname(A[j]), c["idx"][j]] + [f"{v:.5f}" for v in R["dx0"][j]] for j in range(pos + 1)], cls="cmp", numeric_from=3), False)
    )
    emb_tok = A[pos]
    r_in = R["dx0"][pos]
    r_out = R["dE_out"][m0.idx[emb_tok]]
    s79 = (
        human("Satu baris embedding menerima gradien dari <b>dua sisi</b>: saat dipakai sebagai input (lookup) dan saat dipakai sebagai output (logit). Totalnya dijumlahkan.")
        + table(["", "d0", "d1", "d2", "d3"],
                [[f"{tokname(emb_tok)}: sisi input"] + [f"{v:.5f}" for v in r_in], [f"{tokname(emb_tok)}: sisi output"] + [f"{v:.5f}" for v in r_out],
                 ["<b>total dE</b>"] + [f"<b>{v:.5f}</b>" for v in R["dE"][m0.idx[emb_tok]]],
                 [f"{tokname(TGT_TOK)}: sisi output (target)"] + [f"{v:.5f}" for v in R["dE_out"][tg]]], cls="cmp", heat=1.0)
        + dev("<code>embed_scatter_add_kernel</code> mengakumulasi sisi input ke buffer FP32 (<code>embed_accum</code>); sisi output (<code>d_embed_out</code>) "
              "dipakai sebagai benih (<code>launch_f16_to_f32_seed</code>) sebelum backward layer dijalankan.")
    )
    cv = "<br>".join(f"<code>{k}</code>: {v:.1e}" for k, v in chk.items())
    s7v = box("ok", "Verifikasi", f"Semua gradien yang dihitung di bagian 7 dibandingkan dengan implementasi backward penuh (<code>sim.loss_grad</code>) untuk posisi yang sama. "
              f"Selisih maksimum per tensor:<br>{cv}<br><br>Implementasi penuh sendiri dicocokkan dengan turunan numerik (selisih sentral, &epsilon;=10<sup>&minus;6</sup>) "
              f"di {len(mchk.P)*3} titik acak: selisih mutlak maksimum <b>{worst:.1e}</b>.")
    inner = (sub("7.1 Gradien logit", s71) + sub("7.2 Embedding sisi output dan dxf", s72) + sub("7.3 RMSNorm akhir mundur", s73)
             + sub("7.4 FFN mundur", s74) + sub("7.5 Residual, RMSNorm kedua, dan Wo", s75) + sub("7.6 Attention mundur", s77)
             + sub("7.7 RoPE mundur dan gradien Q/K/V", s78) + sub("7.8 Embedding menerima dua gradien", s79) + s7v)
    return sec(7, "Backward untuk satu posisi", inner, f"Posisi {POS} (&lt;|assistant|&gt; &rarr; {tokname(TGT_TOK)}) diisolasi, hanya posisi ini yang bermask.")


# ---------------------------------------------------------------------------
# 8. Clipping dan AdamW
# ---------------------------------------------------------------------------
def s8():
    a0, a1, a2 = ADAM
    norms = a0["norms"]
    tot2 = sum(v * v for v in norms.values())
    rows = [[k, f"{v:.5f}", f"{v*v:.6f}"] for k, v in norms.items()]
    rows.append(["<b>&Sigma; g&sup2;</b>", "", f"<b>{tot2:.6f}</b>"])
    s81 = (
        human("Sebelum bobot digeser, kalau gradiennya terlalu besar seluruhnya dikecilkan sama rata, supaya satu langkah tidak melompat terlalu jauh.")
        + formula("&Vert;g&Vert; = &radic;(&Sigma; semua g&sup2;) &nbsp;&nbsp; jika &Vert;g&Vert; &gt; 1: g &larr; g &middot; 1 / (&Vert;g&Vert; + 10<sup>&minus;6</sup>)")
        + table(["tensor", "norma &Vert;g&Vert;", "g&sup2;"], rows, cls="cmp", numeric_from=1)
        + f"<p>&Vert;g&Vert; = &radic;{tot2:.6f} = <b>{a0['gn']:.5f}</b> &gt; 1, jadi koefisien = 1/({a0['gn']:.5f}+10<sup>&minus;6</sup>) = <b>{a0['coef']:.5f}</b>.</p>"
        + dev("<code>clip_grads()</code> di <code>train_cuda_unified.cu</code>: menjumlahkan kuadrat semua tensor gradien lewat <code>launch_l2_norm_sq</code>, "
              "membaca satu angka ke CPU, lalu <code>launch_scale_fp16</code> jika melewati batas. Baris embedding mendominasi norma (sisi input diperbesar RMSNorm).")
    )
    lrrows = [[s, f"{sim.sched(s):.5f}"] for s in (0, 1, 2, 9, 19, 20, 100, 300, 599)]
    s82 = (
        human("Laju belajar (LR) naik pelan dulu (warmup), lalu turun mengikuti kurva cosine, tidak pernah di bawah 10% dari puncak.")
        + formula("step &lt; warmup: LR = 0,03 &middot; (step+1)/20 &nbsp;&nbsp; sesudahnya: LR = max( 0,03 &middot; &frac12;(1 + cos(&pi; &middot; p)), 0,003 ) &nbsp; dengan p = (step&minus;20)/580")
        + table(["step", "LR"], lrrows, cls="cmp")
        + dev("<code>sched_lr()</code>: berbasis langkah (beda dengan pretraining yang berbasis token), progress dijepit ke [0,1] supaya cosine tidak naik lagi.")
    )
    w = a0["Wk"]; w1 = a1["Wk"]; w2 = a2["Wk"]
    s83 = (
        human("Setiap bobot punya dua &lsquo;ingatan&rsquo;: <b>m</b> (arah gradien belakangan ini) dan <b>v</b> (besar gradien). Langkahnya kira-kira LR &times; arah/besar, "
              "ditambah sedikit penyusutan bobot (weight decay) yang dihitung terpisah dari gradien (itulah AdamW).")
        + formula("m &larr; 0,9m + 0,1g &nbsp; v &larr; 0,95v + 0,05g&sup2; &nbsp; m&#770; = m/(1&minus;0,9<sup>t</sup>) &nbsp; v&#770; = v/(1&minus;0,95<sup>t</sup>) &nbsp; "
                  "w &larr; w &minus; LR&middot;m&#770;/(&radic;v&#770; + 10<sup>&minus;8</sup>) &minus; LR&middot;0,01&middot;w")
        + f"<p>Bobot yang diikuti: <b>Wk[0][0]</b> (baris 0, kolom 0 dari Wk).</p>"
        + table(["", "langkah 0 (t=1)", "langkah 1 (t=2)", "langkah 2 (t=3)"],
                [["loss", f"{a0['L']:.5f}", f"{a1['L']:.5f}", f"{a2['L']:.5f}"],
                 ["&Vert;g&Vert; / koefisien clip", f"{a0['gn']:.5f} / {a0['coef']:.5f}", f"{a1['gn']:.5f} / {a1['coef']:.5f}", f"{a2['gn']:.5f} / {a2['coef']:.5f}"],
                 ["LR", f"{a0['lr']:.5f}", f"{a1['lr']:.5f}", f"{a2['lr']:.5f}"],
                 ["w sebelum", f"{w['w']:.6f}", f"{w1['w']:.6f}", f"{w2['w']:.6f}"],
                 ["g mentah", f"{w['g']:.6f}", f"{w1['g']:.6f}", f"{w2['g']:.6f}"],
                 ["g setelah clip", f"{w['gc']:.6f}", f"{w1['gc']:.6f}", f"{w2['gc']:.6f}"],
                 ["m", f"{w['m']:.7f}", f"{w1['m']:.7f}", f"{w2['m']:.7f}"], ["v", f"{w['v']:.9f}", f"{w1['v']:.9f}", f"{w2['v']:.9f}"],
                 ["m&#770;", f"{w['mh']:.6f}", f"{w1['mh']:.6f}", f"{w2['mh']:.6f}"], ["v&#770;", f"{w['vh']:.9f}", f"{w1['vh']:.9f}", f"{w2['vh']:.9f}"],
                 ["update = LR&middot;m&#770;/(&radic;v&#770;+&epsilon;)", f"{w['upd']:.7f}", f"{w1['upd']:.7f}", f"{w2['upd']:.7f}"],
                 ["decay = LR&middot;0,01&middot;w", f"{w['dec']:.7f}", f"{w1['dec']:.7f}", f"{w2['dec']:.7f}"],
                 ["<b>w sesudah</b>", f"<b>{a0['Wk_new']:.6f}</b>", f"<b>{a1['Wk_new']:.6f}</b>", f"<b>{a2['Wk_new']:.6f}</b>"]],
                cls="cmp", numeric_from=1)
        + f"<p>Langkah 0: update = {w['lr_upd'] if False else w['upd']:.7f} = LR &times; 1 persis ({a0['lr']:.4f}), karena pada langkah pertama m&#770;/&radic;v&#770; = g/|g| = &plusmn;1 berapa pun besar gradiennya. "
          f"Besar gradien baru berpengaruh mulai langkah berikutnya.</p>"
        + table(["dimensi &gamma;_attn", "w", "g setelah clip", "update langkah 0"],
                [[i, f"{a0['ga'][i][0]:.5f}", f"{a0['ga'][i][1]:.6f}", f"{a0['ga'][i][2]:.6f}"] for i in range(4)], cls="cmp", numeric_from=1)
        + dev("<code>adamw_all()</code>: &beta;1 = 0,9, &beta;2 = 0,95, &epsilon; = 1e-8; weight decay 0,01 untuk E, Wq, Wk, Wv, Wo, W1, W2, W3 dan 0 untuk gamma "
              "(<code>(i &lt; 7 ? wd : 0)</code>). Kernel <code>adamw_fused_kernel</code> menyimpan <code>master_param</code> FP32 dan menyalin hasilnya ke bobot FP16; "
              "elemen dengan gradien NaN/Inf dilewati.")
    )
    inner = sub("8.1 Gradient clipping (langkah 0)", s81) + sub("8.2 Jadwal learning rate", s82) + sub("8.3 AdamW untuk satu bobot", s83)
    return sec(8, "Clipping dan AdamW", inner)


# ---------------------------------------------------------------------------
# 9. Kurva
# ---------------------------------------------------------------------------
def s9():
    keys = (0, 1, 2, 5, 10, 19, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 599)
    rows = [[s, f"{HIST[s][1]:.5f}", f"{HIST[s][2]:.5f}", f"{HIST[s][4]:.4f}", f"{HIST[s][3]:.5f}"] for s in keys]
    nclip = sum(1 for h in HIST if h[2] > 1.0)
    ch1 = svg_chart([(h[0], h[1]) for h in HIST], logy=True, ylabel="log10(loss)", marks=[(20, "warmup selesai")])
    ch2 = svg_chart([(h[0], h[3]) for h in HIST], ylabel="learning rate", color="#a78bfa", marks=[(20, "warmup selesai")])
    inner = (
        human("Satu langkah = hitung loss &rarr; backward &rarr; clip &rarr; AdamW &rarr; bobot baru. Diulang 600 kali pada dua kalimat yang sama. "
              "Loss turun dari ~3,4 (menebak acak) sampai hampir 0 (hafal).")
        + ch1 + ch2
        + table(["langkah", "loss", "&Vert;g&Vert;", "koef. clip", "LR"], rows, cls="cmp", numeric_from=1)
        + f"<p>Clipping aktif (&Vert;g&Vert; &gt; 1) pada <b>{nclip}</b> dari 600 langkah. Loss akhir yang dievaluasi ulang: <b>{LFINAL:.2e}</b>.</p>"
        + box("note", "Catatan skala",
              f"LR puncak 0,03 adalah ~1000&times; default SFT repo (3e-5). Itu wajar untuk model {nparam} parameter dengan 2 contoh, tetapi <b>tidak boleh</b> dipakai di model asli. "
              "Hasil di sini juga bergantung pada seed acak; angka ini adalah satu run.")
    )
    return sec(9, "Kurva training", inner)


# ---------------------------------------------------------------------------
# 10. Setelah training
# ---------------------------------------------------------------------------
def s10():
    def tab(idx, seq, mk):
        info = INFF[idx]
        r = []
        for i in range(len(seq)):
            if mk[i]:
                tg = seq[i + 1] if i + 1 < len(seq) else sim.END
                r.append([i, tokname(seq[i]), tokname(tg), f"{info['p'][i][info['tg'][i]]:.6f}"])
        return r
    cA = fwd_trained(A)
    pL = cA["logits"][POS]
    order = np.argsort(-pL)[:6]
    top_rows = [[tokname(UNIQ[i]), f"{pL[i]:.3f}"] for i in order]
    b0 = np.linalg.norm(m0.P["E"]); b1 = np.linalg.norm(mT.P["E"])

    def cos_(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    pairs = [(4766, 21873, "cat", "猫"), (11099, 1940, "house", "家"), (4244, 14132, "my", "私 (dgn spasi)")]
    crow = []
    for a, b, na, nb in pairs:
        if a in m0.idx and b in m0.idx:
            crow.append([f"{na} &ndash; {nb}", f"{cos_(m0.P['E'][m0.idx[a]], m0.P['E'][m0.idx[b]]):.4f}", f"{cos_(mT.P['E'][mT.idx[a]], mT.P['E'][mT.idx[b]]):.4f}"])
    inner = (
        human("Setelah 600 langkah, model menebak token jawaban dengan keyakinan hampir 100%. Bobotnya (terutama embedding) sudah bergeser jauh dari angka acak awal.")
        + sub("10.1 Peluang token target", table(["pos", "input", "target", "p(target)"], tab(0, A, mA), cls="cmp", numeric_from=3, caption="Kalimat A (EN&rarr;JA)")
              + table(["pos", "input", "target", "p(target)"], tab(1, B, mB), cls="cmp", numeric_from=3, caption="Kalimat B (JA&rarr;EN)"))
        + sub(f"10.2 Logit di posisi {POS} (&lt;|assistant|&gt;) setelah training", table(["token", "logit"], top_rows, cls="cmp")
              + f"<p>Sebelum training semua logit berada di sekitar 0 (p &asymp; 1/{m0.V}). Sekarang token <b>{tokname(TGT_TOK)}</b> unggul jauh.</p>")
        + sub("10.3 Bobot berubah", f"<p>Norma Frobenius tabel embedding: <b>{b0:.3f}</b> &rarr; <b>{b1:.3f}</b> (naik {b1/b0:.0f}&times;).</p>")
        + sub("10.4 Apakah embedding kata yang bersesuaian mendekat?",
              table(["pasangan", "cosinus awal", "cosinus akhir"], crow, cls="cmp", numeric_from=1)
              + warn("Jangan dibaca sebagai &lsquo;model memahami makna&rsquo;",
                     "Embedding pasangan kata yang bersesuaian <b>tidak</b> mendekat secara konsisten pada model mini ini. Dengan hanya dua contoh model "
                     "<i>menghafal urutan</i>; ia tidak perlu membangun ruang makna bersama. Itu baru terjadi di model besar dengan jutaan pasangan."))
    )
    return sec(10, "Setelah training", inner)


def fwd_trained(seq):
    return sim.forward(mT, seq)


# ---------------------------------------------------------------------------
# 11. Inferensi
# ---------------------------------------------------------------------------
def s11():
    def trace_tbl(tr):
        rows = []
        for t in tr:
            top = "; ".join(f"{tokname(i)} {p:.6f}" for i, _, p in t["top"])
            rows.append([t["pos"], tokname(t["tok"]), f"{t['logit']:.3f}", f"{t['p']:.6f}", top])
        return table(["pos", "dipilih", "logit", "p", "3 teratas (token p)"], rows, cls="cmp", numeric_from=2)

    def dec(ids):
        return toks_to_text(ids)

    inner = (
        human("Saat dipakai, model menulis jawaban <b>satu token demi satu</b>. Prompt (soal + <code>&lt;|assistant|&gt;</code>) dibaca dulu; lalu tiap token baru dipilih dari "
              "peluang tertinggi (greedy), ditempel ke prompt, dan dipakai menebak token berikutnya, sampai muncul <code>&lt;|end|&gt;</code>.")
        + dev("<code>Transformer::forward_one_token()</code>: K (sesudah RoPE) dan V tiap token disimpan di <b>KV-cache</b>. Token baru hanya menghitung Q miliknya, "
              "lalu membaca K/V semua posisi sebelumnya dari cache, jadi biaya per token O(panjang) bukan O(panjang&sup2;). Pada GQA nyata cache memuat "
              "5 head KV (bukan 20): 2&times;24 layer&times;5&times;64&times;2 byte = 30.720 byte/token (vs 122.880 byte tanpa GQA).")
        + sub("11.1 EN&rarr;JA", f"<p>Prompt ({len(pa)} token, termasuk BOS): {chips(pa)}</p>" + trace_tbl(tr_a)
              + box("ok", "Hasil", f"ID: {chips(out_a)}<br>Setelah <code>decode</code> (byte digabung lalu UTF-8): <b class='ja'>{dec(out_a)}</b> &nbsp;&mdash;&nbsp; "
                    f"{'sama persis dengan jawaban training' if out_a == ans_a else 'BERBEDA dari jawaban training'}"))
        + sub("11.2 JA&rarr;EN", f"<p>Prompt ({len(pb)} token, termasuk BOS): {chips(pb)}</p>" + trace_tbl(tr_b)
              + box("ok", "Hasil", f"ID: {chips(out_b)}<br>Setelah <code>decode</code>: <b>{dec(out_b)}</b> &nbsp;&mdash;&nbsp; "
                    f"{'sama persis dengan jawaban training' if out_b == ans_b else 'BERBEDA dari jawaban training'}"))
        + sub("11.3 Menjalankan agar sama dengan training",
              f'<pre class="raw">./build_infer/infer --model &lt;ckpt&gt; --config &lt;config&gt; --vocab vocab_9 \\\n'
              f'  --system "You are a multilingual translation assistant." \\\n'
              f'  --prompt "日本語に翻訳: I have a cat in my house"</pre>'
              + "<p>Setelah perbaikan di branch <code>fix/sft-mask-bos-eval</code>, <code>infer</code> menambahkan BOS otomatis (<code>--no_bos</code> untuk mematikan) dan "
                "hard-block repetisi tidak lagi aktif kecuali diminta (<code>--hard_block N</code>).</p>")
    )
    return sec(11, "Inferensi", inner)


# ---------------------------------------------------------------------------
# 12. Eksperimen kontrol
# ---------------------------------------------------------------------------
def s12():
    def agg(tag, key, sub_):
        return sum(1 for _, _, r in EXP[tag] if r[key][sub_])
    n = len(EXP["old"])
    rows = [["Mask LAMA (token pertama tidak dilatih)", f"{agg('old','a',0)} / {n}", f"{agg('old','b',0)} / {n}", f"{agg('old','a',1)} / {n}", f"{agg('old','b',1)} / {n}"],
            ["Mask BARU (token pertama dilatih)", f"<b>{agg('new','a',0)} / {n}</b>", f"<b>{agg('new','b',0)} / {n}</b>", f"{agg('new','a',1)} / {n}", f"{agg('new','b',1)} / {n}"]]
    # contoh keluaran mask lama
    _, ro = run_exp(1, 8, sim.mask_repo_old)
    ex_old_a, ex_old_b = ro["a"][2], ro["b"][2]
    s121 = (
        human("Dua model identik dilatih dengan data yang sama; satu-satunya beda: apakah kata pertama jawaban ikut dilatih. "
              f"Percobaan diulang pada {n} kombinasi (seed 1&ndash;8 &times; dua ukuran FFN: 4 dan 8). Kolom kanan: prompt <b>tanpa BOS</b>.")
        + table(["", "EN&rarr;JA benar", "JA&rarr;EN benar", "EN&rarr;JA tanpa BOS", "JA&rarr;EN tanpa BOS"], rows, cls="cmp", numeric_from=99)
        + f"<p><b>Contoh keluaran mask lama</b> (seed 1, FFN 8): EN&rarr;JA = <b class='ja'>{toks_to_text(ex_old_a)}</b>, JA&rarr;EN = <b>{toks_to_text(ex_old_b)}</b>. "
          f"Kata pertama jawaban ({tokname(ans_a[0])} dan {tokname(ans_b[0])}) tidak pernah diajarkan, dan keluarannya sudah salah sejak token pertama.</p>"
        + warn("Batas kesimpulan",
               "Ini hasil model mini yang menghafal 2 kalimat. Pada model besar dengan jutaan contoh, token pertama bisa terbantu generalisasi dari posisi lain "
               "(dan dari pretraining), jadi dampaknya <b>belum tentu</b> sebesar ini. Yang dibuktikan: kode lama dan komentarnya bertentangan, dan pada kasus minimal itu mematahkan terjemahan.")
    )
    s122 = (
        human("Pelatihan memakai <code>&lt;|endoftext|&gt;</code> (BOS) di awal setiap sampel, sedangkan <code>infer.cpp</code> versi lama tidak menambahkannya. "
              "Tabel di atas (dua kolom kanan) menunjukkan akibatnya pada model mini: tanpa BOS sebagian run masih benar dan sebagian rusak. "
              "Itu sebabnya inferensi sekarang menambahkan BOS secara default.")
    )
    rowsC = [["Tanpa penalti (rep=1,0), tanpa hard-block", toks_to_text(out_c0), "benar" if out_c0 == ans_c else "salah"],
             ["rep_penalty 1,3 saja", toks_to_text(out_c13), "benar" if out_c13 == ans_c else "salah"],
             ["hard-block N=3 (perilaku lama)", toks_to_text(out_c3), "benar" if out_c3 == ans_c else "salah"],
             ["hard-block N=3 + rep 1,3 (default lama)", toks_to_text(out_c3r), "benar" if out_c3r == ans_c else "salah"]]
    s123 = (
        human(f"Kalimat yang memang berisi pengulangan: <b>ha ha ha</b> &rarr; <b>ha ha ha</b>. Jawaban benarnya {chips(ans_c)}. "
              "Hard-block lama melarang token yang baru saja keluar, sehingga model dipaksa keluar jalur.")
        + table(["pengaturan", "keluaran", "status"], rowsC, cls="cmp", numeric_from=99)
        + dev("Hard-block lama ada di <code>infer.cpp</code> (<code>HARD_BLOCK_N=3</code>): token non-spesial yang muncul &ge;3&times; di 64 token terakhir diblokir, dan "
              "token yang baru dihasilkan <b>selalu</b> diblokir, terlepas dari <code>--rep_penalty</code>. Sekarang opsional lewat <code>--hard_block N</code>.")
    )
    prow = [[a, toks_to_text(o) or "(kosong)"] for a, o in PERT]
    s124 = (
        human("Apa yang terjadi kalau sumbernya diubah sedikit? Jika model benar-benar menerjemahkan, keluarannya ikut berubah dengan wajar.")
        + table(["perubahan pada prompt", "keluaran"], prow, cls="cmp", numeric_from=99)
        + warn("Ini bukan terjemahan",
               "Keluarannya kacau atau bocor ke bahasa lain. Model mini hanya menghafal dua urutan token dan tidak punya cara menerjemahkan kalimat baru.")
    )
    inner = (sub("12.1 Mask lama vs mask baru", s121) + sub("12.2 BOS", s122) + sub("12.3 Hard-block repetisi", s123) + sub("12.4 Perubahan input", s124))
    return sec(12, "Eksperimen kontrol pada model mini", inner)


# ---------------------------------------------------------------------------
# 13. Model nyata
# ---------------------------------------------------------------------------
def s13():
    rows = [
        ["Parameter", f"{nparam}", "491.582.720"],
        ["Data SFT", "2 kalimat", "1.796.000 baris (<code>sft_translation_enja/train.jsonl</code>)"],
        ["Pretraining sebelumnya", "tidak ada (bobot acak)", "ratusan juta&ndash;miliaran token EN/JA; checkpoint Anda 2,67 miliar token"],
        ["Presisi", "float64", "FP16 + master FP32, tanpa loss-scale di SFT"],
        ["Head attention", "2 Q / 1 KV", "20 Q / 5 KV"],
        ["RoPE", "1 frekuensi", "32 frekuensi/head"],
        ["Layer", "1", "24"],
        ["Hasil", "menghafal 2 urutan", "belajar pola terjemahan umum (bila data cukup)"],
    ]
    inner = (
        human("Mekanismenya sama: teks &rarr; token &rarr; tebakan token berikutnya &rarr; loss pada jawaban &rarr; gradien &rarr; bobot digeser. "
              "Bedanya skala, dan skala itulah yang mengubah &lsquo;menghafal&rsquo; menjadi &lsquo;menerjemahkan&rsquo;.")
        + table(["", "Model mini", "Model nyata"], rows, cls="cmp", numeric_from=99)
        + sub("Mengapa data besar mengubah hafalan menjadi terjemahan",
              "<p>Dengan jutaan pasangan, menghafal tidak lagi menurunkan loss pada data baru. Satu-satunya cara menurunkan loss pada kalimat yang belum pernah dilihat adalah "
              "membangun mekanisme umum: sebagian head attention belajar menghubungkan kata yang sedang ditulis dengan kata sumber yang bersesuaian, dan FFN menyimpan "
              "pemetaan kosakata serta perubahan urutan (SOV Jepang &rarr; SVO Inggris). Dua arah memakai satu set bobot yang sama; instruksi "
              "(<code>日本語に翻訳:</code> atau <code>Translate to English:</code>) hanyalah konteks yang menentukan arah.</p>")
        + warn("Yang belum dibuktikan halaman ini",
               "Halaman ini membuktikan bahwa <b>alur mekanis</b> benar (tokenisasi &rarr; mask &rarr; forward &rarr; backward &rarr; optimizer &rarr; inferensi), bukan bahwa "
               "model 492M akan menerjemahkan dengan baik. Itu hanya bisa dibuktikan dengan menjalankan SFT lalu mengukur pada data uji terpisah "
               "(<code>sft_data/sft_translation_enja/eval.jsonl</code>).")
        + box("note", "Langkah berikutnya untuk model nyata",
              "<pre class='raw'>./build/train_sft_cuda --stage 2 \\\n  --base_model &lt;ckpt_pretrain.safetensors&gt; \\\n  --vocab_dir vocab_9 --bpe bpe_v9.bin \\\n"
              "  --sft_dir sft_data/sft_translation_enja \\\n  --checkpoint_dir &lt;folder berisi train_state.txt&gt;</pre>"
              "Guard fingerprint tokenizer membaca <code>train_state.txt</code> dari <code>--checkpoint_dir</code>; fingerprint checkpoint Anda "
              "<code>17261940542271645870</code> cocok dengan <code>vocab_9</code> / <code>bpe_v9.bin</code>.")
    )
    return sec(13, "Dari model mini ke model nyata", inner)
