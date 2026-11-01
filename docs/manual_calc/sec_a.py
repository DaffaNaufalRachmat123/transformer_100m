"""sec_a.py - bagian 0..6 manual (batasan, data, tokenisasi, mask, bobot awal, forward, loss)."""
import math
import re

import numpy as np


def setup(ctx):
    globals().update(ctx)


# ---------------------------------------------------------------------------
# helper khusus bagian ini
# ---------------------------------------------------------------------------
def kind_of(tid):
    t = sim.TOKS[tid]
    if t.startswith("<|"):
        return "sp"
    b = sim.glyph_bytes(t)
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "by"
    return "ja" if any(ord(ch) > 0x2E80 for ch in s) else "en"


def chip(tid, show_id=True):
    idtxt = f'<i>{tid}</i>' if show_id else ""
    return f'<span class="chip {kind_of(tid)}">{tokname(tid)}{idtxt}</span>'


def chunk_trace(word):
    """Salinan BPEReference._encode_chunk yang mencatat tiap tahap."""
    e = sim.enc
    wb = word.encode("utf-8")
    seq, pos, n = [], 0, len(wb)
    while pos < n:
        hit = False
        for ln in range(min(n - pos, 8), 0, -1):
            tid = e.bytes2id.get(wb[pos:pos + ln])
            if tid is not None:
                seq.append(tid); pos += ln; hit = True
                break
        if not hit:
            pos += 1
    passes = [list(seq)]
    changed = True
    while changed and len(seq) > 1:
        changed = False
        merged, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq):
                m = e.merges.get((seq[i], seq[i + 1]))
                if m is not None:
                    merged.append(m); i += 2; changed = True
                    continue
            merged.append(seq[i]); i += 1
        seq = merged
        passes.append(list(seq))
    return passes


def chips(ids):
    return " ".join(chip(i) for i in ids)


def raw_hex(glyphs):
    return " ".join(f"{sim.INV_BYTE[c]:02X}" for c in glyphs)


def cells(vec, W, cols, outname, nd=5):
    """Tabel perkalian vektor x matriks: tiap suku ditulis (a x w = t)."""
    vec = np.ravel(vec)
    n = len(vec)
    head = ["keluaran"] + [f"suku {i+1}" for i in range(n)] + ["jumlah"]
    rows = []
    for j in cols:
        terms = [vec[i] * W[i][j] for i in range(n)]
        r = [f"<b>{outname}[{j}]</b>"]
        for i in range(n):
            r.append(f'<span class="dim">{vec[i]:.{nd}f}&times;{W[i][j]:.{nd}f}</span><br>{terms[i]:.{nd}f}')
        r.append(f"<b>{sum(terms):.{nd}f}</b>")
        rows.append(r)
    return table(head, rows, cls="cells")


def lab_pos(seq, upto=None):
    upto = len(seq) if upto is None else upto
    return [f"{j}&nbsp;{tokname(seq[j])}" for j in range(upto)]


# ---------------------------------------------------------------------------
# 0. Batasan
# ---------------------------------------------------------------------------
def s0():
    rows = [
        ["d_model", "4", "1280"], ["Head Q / head KV (GQA)", "2 / 1", "20 / 5"],
        ["d_head", "2", "64"], ["ffn_dim", "8", "3584"], ["Layer", "1", "24"],
        ["Vocab (baris embedding)", f"{m0.V}", "49.152"],
        ["Total parameter", f"{nparam}", "491.582.720"],
        ["Frekuensi RoPE per head", "1", "32"],
    ]
    t1 = table(["Parameter", "Model mini (halaman ini)", "Config 492M (config_pretrain.json)"], rows, cls="cmp")
    hyper = [
        ["LR puncak", "3e-2", "3e-5 (default SFT)"], ["Warmup", "20 langkah", "100 langkah"],
        ["Total langkah", "600", "tergantung data"], ["Weight decay", "0,01", "0,01"],
        ["Gradient clip (norma global)", "1,0", "1,0"], ["AdamW &beta;1 / &beta;2 / &epsilon;", "0,9 / 0,95 / 1e-8", "sama"],
        ["Presisi", "float64 (numpy)", "FP16 + master weight FP32"],
    ]
    t2 = table(["Hyperparameter SFT", "Model mini", "Repo"], hyper, cls="cmp")
    inner = (
        human("Model asli (492 juta parameter) tidak mungkin dihitung dengan tangan, jadi di sini dipakai <b>model mini</b> "
              "dengan <b>rumus yang sama persis</b>, hanya ukurannya diperkecil. Semua angka di halaman ini dihitung ulang oleh "
              "skrip, bukan ditulis tangan.")
        + t1 + t2
        + dev("Yang disimulasikan mengikuti kode repo: tokenizer = <code>scripts/bpe_reference.py</code> (dicocokkan dengan "
              "<code>bpe/bpe_tokenizer.cpp</code>), mask = <code>build_assistant_mask()</code> di <code>transformer/train_cuda_unified.cu</code>, "
              "forward = <code>train_step.cuh</code> + <code>cuda/*.cu</code>, jadwal LR = <code>sched_lr()</code>, clipping = <code>clip_grads()</code>, "
              "AdamW = <code>adamw_all()</code>. Yang <b>bukan</b> disimulasikan: kernel CUDA itu sendiri dan aritmatika FP16.")
        + warn("KOREKSI atas penjelasan di percakapan sebelumnya",
               "Di percakapan sebelumnya saya sempat menampilkan tokenisasi <b>私の家には猫がいます</b> dengan token yang memotong tengah "
               "karakter (22921 dan 831). Itu salah: replika Python saya waktu itu memakai jendela longest-match 8 <i>karakter glyph</i>, padahal "
               "kode C++ memakai 8 <i>byte</i> dari string glyph (glyph non-ASCII = 2 byte). Halaman ini memakai tokenizer yang sudah "
               "dicocokkan dengan C++ (3 dari 3 teks identik, plus 6 dari 6 pada uji sebelumnya). Karena itu angka token, ukuran vocab mini "
               f"({m0.V} baris), dan semua angka turunannya berbeda dari yang saya tulis dulu.")
        + box("note", "Cara mereproduksi",
              "<code>python3 docs/manual_calc/build_manual.py</code> &mdash; menghasilkan ulang halaman ini. Backward manual satu posisi "
              f"dicocokkan dengan implementasi penuh (selisih maksimum {max(chk.values()):.1e}) dan dengan turunan numerik "
              f"(selisih maksimum {worst:.1e} pada {len(mchk.P)*3} titik uji).")
    )
    return sec(0, "Batasan simulasi", inner, "Baca ini dulu: apa yang dihitung, apa yang tidak.")


# ---------------------------------------------------------------------------
# 1. Dua kalimat -> teks training
# ---------------------------------------------------------------------------
def s1():
    import json
    combos = {}
    p = ROOT / "sft_data/sft_translation_enja/train.jsonl"
    nrows = 0
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                nrows += 1
                if nrows > 20000:
                    break
                t = json.loads(line)["text"]
                sy = t.split("<|user|>")[0].replace("<|system|>", "")
                us = t.split("<|user|>")[1].split("<|assistant|>")[0]
                ins = us.split(":")[0] + ":"
                combos[(sy, ins)] = combos.get((sy, ins), 0) + 1
    ex = combos.get((sim.SYS, "日本語に翻訳:"), 0)
    ex2 = combos.get((sim.SYS, "Translate to English:"), 0)
    top = sorted(combos.items(), key=lambda kv: -kv[1])[:6]
    rows = [[esc(k[0]), esc(k[1]), v] for k, v in top]
    inner = (
        human("Satu pasangan terjemahan dijadikan <b>dua baris training</b>, satu untuk tiap arah. Modelnya satu, "
              "arahnya ditentukan oleh kalimat perintah di bagian <code>&lt;|user|&gt;</code>.")
        + f'<div class="pair"><div class="p1"><span class="tag en">EN</span>{esc(sim.EN)}</div>'
          f'<div class="p2"><span class="tag ja">JA</span>{esc(sim.JA)}</div></div>'
        + sub("Baris training EN&rarr;JA", f'<pre class="raw">{esc(sim.TEXT_EN2JA)}</pre>')
        + sub("Baris training JA&rarr;EN", f'<pre class="raw">{esc(sim.TEXT_JA2EN)}</pre>')
        + dev("Format baris mengikuti <code>sft_data/sft_translation_enja/train.jsonl</code> "
              "(<code>{\"text\": \"...\", \"lang\": \"en-ja\", ...}</code>). Parser SFT hanya membaca field <code>text</code>.")
        + sub(f"Kombinasi system prompt + instruksi di data asli ({min(nrows, 20000):,} baris pertama)",
              table(["System prompt", "Instruksi", "Jumlah"], rows, cls="cmp")
              + f"<p class='dim'>Dalam sampel itu, system &lsquo;multilingual&rsquo; + &lsquo;日本語に翻訳:&rsquo; (kombinasi yang dipakai contoh di atas) muncul "
                f"<b>{ex}</b> kali, sedangkan system &lsquo;multilingual&rsquo; + &lsquo;Translate to English:&rsquo; muncul <b>{ex2}</b> kali. "
                f"Di data asli, &lsquo;日本語に翻訳:&rsquo; berpasangan dengan system &lsquo;helpful translation engine&rsquo;. "
                f"Contoh di halaman ini sengaja menyatukan keduanya supaya kedua arah memakai system prompt yang sama; "
                f"bagian-bagiannya ada di data, kombinasinya tidak.</p>")
    )
    return sec(1, "Dua kalimat menjadi teks training", inner)


# ---------------------------------------------------------------------------
# 2. Tokenisasi
# ---------------------------------------------------------------------------
def s2():
    full_a = sim.enc.encode(sim.TEXT_EN2JA, add_bos=True)
    full_b = sim.enc.encode(sim.TEXT_JA2EN, add_bos=True)

    # tabel byte untuk kalimat Jepang
    rows = []
    for ch in sim.JA:
        b = ch.encode("utf-8")
        gl = "".join(sim.BYTE_ENCODER[x] for x in b)
        rows.append([f'<span class="ja big">{ch}</span>', f"U+{ord(ch):04X}",
                     " ".join(f"{x:02X}" for x in b), f'<code>{esc(gl)}</code>'])
    bytes_tbl = table(["Karakter", "Codepoint", "Byte UTF-8", "Glyph byte-encoder"], rows, cls="cmp")

    # trace pretoken: bagian user + jawaban untuk kedua arah
    def trace_segment(seg):
        out = []
        for w in sim.enc.pretokenize(seg.encode("utf-8")):
            passes = chunk_trace(w)
            raw = raw_hex(w)
            seed = passes[0]
            final = passes[-1]
            extra = ""
            if len(passes) > 2:
                extra = " &rarr; ".join(chips(p) for p in passes[1:-1])
            out.append([f"<code>{esc(w)}</code>", f'<span class="dim">{raw}</span>', chips(seed),
                        (extra + (" &rarr; " if extra else "") if len(passes) > 2 else "") + (chips(final) if len(passes) > 1 else '<span class="dim">tidak berubah</span>')])
        return out

    hdr = ["Pretoken (glyph)", "Byte asli", "Seed longest-match", "Setelah pass merge"]
    seg_a_user = "日本語に翻訳: I have a cat in my house"
    seg_b_user = "Translate to English: 私の家には猫がいます"
    ta = table(hdr, trace_segment(seg_a_user), cls="cmp tok")
    tb = table(hdr, trace_segment(seg_b_user), cls="cmp tok")
    tans_a = table(hdr, trace_segment(sim.JA), cls="cmp tok")
    tans_b = table(hdr, trace_segment(sim.EN), cls="cmp tok")
    tsys = table(hdr, trace_segment(sim.SYS), cls="cmp tok")

    def seq_table(ids):
        sysend = ids.index(2)
        asst = ids.index(3)
        rows = []
        for i, t in enumerate(ids):
            role = "spesial" if i in (0, 1, sysend, asst, len(ids) - 1) else ("system" if i < sysend else ("user" if i < asst else "jawaban"))
            rows.append([i, t, tokname(t), role])
        return table(["pos", "ID", "teks", "peran"], rows, cls="seq", numeric_from=99)

    inner = (
        human("Teks diubah jadi angka dalam tiga langkah: (1) setiap karakter dipecah jadi <b>byte</b>, (2) teks dipotong jadi <b>pretoken</b> "
              "di spasi, (3) tiap pretoken dipecah jadi potongan terpanjang yang ada di kamus <code>vocab_9</code> (49.152 token), lalu "
              "potongan yang punya aturan gabung (<code>merges.txt</code>) digabung.")
        + sub("2a. Karakter Jepang menjadi byte", bytes_tbl
              + dev("Satu huruf Jepang = 3 byte UTF-8. <code>BYTE_ENCODER</code> (gaya GPT-2) memetakan setiap byte 0&ndash;255 ke satu karakter tampak; "
                    "spasi <code>0x20</code> menjadi <code>&Gdot;</code>. Total jawaban Jepang: 10 karakter &times; 3 = 30 byte."))
        + sub("2b. Algoritma encoder",
              formula("<b>1.</b> pecah di spasi (satu spasi menempel ke kata berikutnya)<br>"
                      "<b>2.</b> <b>longest-match</b>: dari posisi sekarang coba potongan sepanjang 8 byte, lalu 7, ... sampai ketemu di vocab<br>"
                      "<b>3.</b> <b>pass merge</b>: pindai kiri&rarr;kanan, gabungkan pasangan yang ada di tabel merge; ulangi sampai tak ada perubahan")
              + dev("Panjang jendela 8 adalah <b>8 byte dari string glyph</b> (<code>std::string</code> di C++), bukan 8 karakter. Glyph non-ASCII memakai 2 byte, "
                    "jadi untuk teks non-ASCII jendela efektifnya 4 byte asli. Gabungan yang lebih panjang (mis. <b>翻訳</b>, 6 byte asli) baru terbentuk "
                    "di pass merge. Encoder ini bukan BPE murni: tidak memakai peringkat merge."))
        + sub("2c. System prompt (sama di kedua arah)", tsys)
        + sub("2d. Bagian user, EN&rarr;JA", ta)
        + sub("2e. Jawaban Jepang", tans_a)
        + sub("2f. Bagian user, JA&rarr;EN", tb)
        + sub("2g. Jawaban Inggris", tans_b)
        + box("note", "Dua hal yang perlu diperhatikan",
              "<b>Spasi ikut kata.</b> Di sumber JA&rarr;EN, spasi setelah titik dua menempel ke 私 sehingga token-nya <code>&sp;私</code> (ID 14132), "
              "sedangkan 私 di jawaban EN&rarr;JA tanpa spasi adalah ID 3402. Huruf yang sama, dua ID berbeda.<br>"
              "<b>Token byte parsial.</b> <code>日本語に翻訳</code> menghasilkan <code>[E6 97 A5 E6]</code> dan <code>[9C AC]</code>, token yang memotong "
              "karakter 本 di tengah. Itu sah (decoder menyambung byte), tetapi model harus belajar menggabungkannya.")
        + sub(f"2h. Urutan lengkap EN&rarr;JA ({len(full_a)} token)", fold("tampilkan tabel", seq_table(full_a), False))
        + sub(f"2i. Urutan lengkap JA&rarr;EN ({len(full_b)} token)", fold("tampilkan tabel", seq_table(full_b), False))
        + dev("<code>encode(text, add_bos=true, add_eos=false)</code>: BOS (<code>&lt;|endoftext|&gt;</code>, ID 0) ditambahkan di depan. Teks sudah berakhir "
              "<code>&lt;|end|&gt;</code>, jadi EOS <b>tidak</b> ditambahkan lagi (perbaikan di branch <code>fix/sft-mask-bos-eval</code>; sebelumnya muncul "
              "dua <code>&lt;|end|&gt;</code> berturut-turut).")
        + sub("2j. Versi yang disimulasikan",
              f"<p>Agar angka muat dicetak, simulasi membuang system prompt (posisi 1&ndash;11) dan memakai urutan:</p>"
              f'<div class="seqline"><b>A (EN&rarr;JA, {len(A)} token)</b> {chips(A)}</div>'
              f'<div class="seqline"><b>B (JA&rarr;EN, {len(B)} token)</b> {chips(B)}</div>'
              f"<p class='dim'>Gabungan {len(UNIQ)} ID unik dipetakan ke baris 0&ndash;{len(UNIQ)-1} tabel embedding mini (bagian 4).</p>")
    )
    return sec(2, "Tokenisasi", inner, "Dari teks ke ID, persis seperti <code>BPEEncoder::encode</code>.")


# ---------------------------------------------------------------------------
# 3. Input, target, mask
# ---------------------------------------------------------------------------
def s3():
    def rows_for(seq, mo, mn):
        r = []
        for i, t in enumerate(seq):
            tgt = seq[i + 1] if i + 1 < len(seq) else sim.END
            diff = mo[i] != mn[i]
            tr = f'<tr class="hl">' if diff else "<tr>"
            r.append(f'{tr}<td class="r">{i}</td><td>{chip(t)}</td><td>{chip(tgt)}</td>'
                     f'<td class="r">{mo[i]}</td><td class="r"><b>{mn[i]}</b></td></tr>')
        return "".join(r)

    def mk(seq, mo, mn, title):
        return (f'<div class="tw"><table class="mask"><caption>{title}</caption><thead><tr><th>pos</th><th>input</th><th>target (token berikutnya)</th>'
                f'<th>mask LAMA</th><th>mask BARU</th></tr></thead><tbody>{rows_for(seq, mo, mn)}</tbody></table></div>')

    inner = (
        human("Pada tiap posisi, model diminta menebak <b>token berikutnya</b>. <b>Mask</b> menentukan tebakan mana yang dihitung kesalahannya: "
              "hanya tebakan di bagian <b>jawaban</b> yang dihitung. Soal, system prompt, dan instruksi cuma jadi konteks.")
        + formula("input[t] = token[t] &nbsp;&nbsp; target[t] = token[t+1] &nbsp;&nbsp; (target posisi terakhir = <code>&lt;|end|&gt;</code>)<br>"
                  "loss = &Sigma;<sub>t</sub> mask[t] &middot; (&minus;ln p(target[t])) &nbsp;/&nbsp; &Sigma;<sub>t</sub> mask[t]")
        + dev("Mask diindeks pada <b>posisi input</b>. Posisi <code>&lt;|assistant|&gt;</code> punya target = token pertama jawaban. "
              "Kode lama memberi mask 0 di posisi itu, jadi <b>kata pertama jawaban tidak pernah dilatih</b>, sementara posisi "
              "<code>&lt;|end|&gt;</code> terakhir (target-nya hanya cadangan <code>TOK_EOS</code>) justru dihitung. "
              "<code>build_assistant_mask()</code> yang baru memperbaikinya. Baris yang berbeda diberi sorotan di tabel.")
        + mk(A, mA_old, mA, "Kalimat A &mdash; EN&rarr;JA")
        + mk(B, mB_old, mB, "Kalimat B &mdash; JA&rarr;EN")
        + table(["", "Mask lama", "Mask baru"],
                [["Posisi bermask, A", sum(mA_old), sum(mA)], ["Posisi bermask, B", sum(mB_old), sum(mB)],
                 ["Total (pembagi loss)", TOTOLD, TOT0],
                 ["Kata pertama jawaban dilatih?", "tidak", "<b>ya</b>"],
                 ["Loss langkah 0", f"{LOLD:.5f}", f"{L0:.5f}"]], cls="cmp", numeric_from=99)
        + dev("Batch builder (<code>train_cuda_unified.cu</code>) menyusun <code>batch_tokens[t]=tokens[t]</code>, "
              "<code>batch_targets[t]=tokens[t+1]</code> (atau <code>TOK_EOS</code> di akhir), <code>batch_mask[t]=loss_mask[t]</code>; "
              "sisa panjang diisi <code>TOK_PAD</code> dengan mask 0.")
    )
    return sec(3, "Input, target, dan mask loss", inner)


# ---------------------------------------------------------------------------
# 4. Bobot awal
# ---------------------------------------------------------------------------
def s4():
    labs = [f"{i}&nbsp;&middot;&nbsp;<span class='dim'>{uid}</span>&nbsp;{tokname(uid)}" for i, uid in enumerate(UNIQ)]
    P = P0
    inner = (
        human("Sebelum belajar, semua bobot diisi angka acak kecil. Setiap token punya satu baris di tabel embedding (4 angka). "
              "Model baru &lsquo;tahu&rsquo; sesuatu setelah angka-angka ini digeser oleh training.")
        + dev("Inisialisasi mengikuti <code>init_params()</code> di <code>train_cuda.cu</code>: embedding ~ N(0, 0,02&sup2;); Wq, Wk, Wv, Wo, W1, W3 ~ N(0, 1/D); "
              "W2 ~ N(0, 1/FFN); semua gamma RMSNorm = 1. Generator acak simulasi ini adalah <code>numpy</code> seed 42, bukan <code>mt19937</code> repo, "
              "jadi nilainya tidak identik dengan yang dihasilkan repo; yang sama hanya distribusinya.")
        + sub(f"Embedding E ({m0.V}&times;4) &mdash; N(0, 0,02&sup2;)",
              fold("tampilkan 31 baris", mat("E", P["E"], labs, ["d0", "d1", "d2", "d3"], heat=0.04, nd=5)))
        + sub("Proyeksi attention",
              fold("Wq, Wk, Wv, Wo",
                   mat("Wq (D&times;D)", P["Wq"], ["0", "1", "2", "3"], ["q0", "q1", "q2", "q3"])
                   + mat("Wk (D&times;KV)", P["Wk"], ["0", "1", "2", "3"], ["k0", "k1"])
                   + mat("Wv (D&times;KV)", P["Wv"], ["0", "1", "2", "3"], ["v0", "v1"])
                   + mat("Wo (D&times;D)", P["Wo"], ["0", "1", "2", "3"], ["o0", "o1", "o2", "o3"])))
        + sub("Proyeksi FFN (SwiGLU)",
              fold("W1 (gate), W3 (up), W2 (down)",
                   mat("W1 gate (4&times;8)", P["W1"], ["0", "1", "2", "3"], [f"g{j}" for j in range(8)])
                   + mat("W3 up (4&times;8)", P["W3"], ["0", "1", "2", "3"], [f"u{j}" for j in range(8)])
                   + mat("W2 down (8&times;4)", P["W2"], [str(i) for i in range(8)], ["d0", "d1", "d2", "d3"])))
        + p_gamma()
        + table(["Tensor", "Bentuk", "Jumlah"],
                [["E", f"{m0.V}&times;4", m0.V * 4], ["Wq", "4&times;4", 16], ["Wk", "4&times;2", 8], ["Wv", "4&times;2", 8], ["Wo", "4&times;4", 16],
                 ["W1", "4&times;8", 32], ["W2", "8&times;4", 32], ["W3", "4&times;8", 32], ["gamma attn/ffn/final", "3&times;4", 12],
                 ["<b>Total</b>", "", f"<b>{nparam}</b>"]], cls="cmp", numeric_from=99)
    )
    return sec(4, "Bobot awal model mini", inner)


def p_gamma():
    return f'<div class="vec"><span class="vn">gamma_attn = gamma_ffn = gamma_final</span> = (1, 1, 1, 1)</div>'


# ---------------------------------------------------------------------------
# 5. Forward satu posisi
# ---------------------------------------------------------------------------
def s5():
    c = c0
    P = m0.P
    pos = POS
    token = A[pos]
    labs = lab_pos(A, pos + 1)
    S1 = []
    x0 = c["x0"][pos]
    sq = x0 * x0
    ms = sq.mean()
    rs = 1 / math.sqrt(ms + 1e-6)
    n1 = c["n1"][pos]
    cos_p, sin_p = math.cos(pos), math.sin(pos)
    Qpre = c["qpre"][pos]
    Q = c["Q"][pos]
    Kpre = c["kpre"][pos]
    K = c["K"][pos]

    s51 = (
        human(f"Posisi yang dibedah: token <b>{tokname(token)}</b> (<code>&lt;|assistant|&gt;</code>) di posisi {pos}. Tugasnya: menebak token berikutnya, "
              f"yaitu <b>{tokname(TGT_TOK)}</b>. Ini persis tebakan yang dulu tidak pernah dilatih. Karena attention hanya boleh melihat ke belakang, "
              f"posisi ini melihat posisi 0&ndash;{pos} ({pos+1} token).")
        + formula("x&#8320; = E[baris token] &nbsp;&nbsp;&nbsp; rsqrt = 1 / &radic;( mean(x&sup2;) + 10<sup>&minus;6</sup> ) &nbsp;&nbsp;&nbsp; n1 = x&#8320; &middot; rsqrt &middot; &gamma;")
        + table(["", "d0", "d1", "d2", "d3"],
                [["x&#8320; (embedding)"] + [f"{v:.5f}" for v in x0],
                 ["x&#8320;&sup2;"] + [f"{v:.8f}" for v in sq],
                 ["n1 = x&#8320;&middot;rsqrt"] + [f"{v:.5f}" for v in n1]], cls="cmp", heat=2.0)
        + f"<p>&Sigma;x&sup2; = {sq.sum():.8f}; mean = {ms:.8f}; + &epsilon; = {ms+1e-6:.8f}; rsqrt = 1/&radic;{ms+1e-6:.8f} = <b>{rs:.5f}</b>. "
          "Embedding yang bernilai ~0,01 diperbesar ~80&times; jadi bernilai ~1.</p>"
    )
    s52 = (
        human("Setiap token membuat tiga vektor dari <code>n1</code>: <b>Q</b> (yang dicari), <b>K</b> (label yang ditawarkan), <b>V</b> (isi yang dibawa). "
              "Kedua head Q berbagi satu head K/V (GQA 2:1).")
        + cells(n1, P["Wq"], range(4), "Q") + cells(n1, P["Wk"], range(2), "K") + cells(n1, P["Wv"], range(2), "V")
        + f"<p>Q dibagi dua head: head 0 = ({Qpre[0]:.5f}, {Qpre[1]:.5f}), head 1 = ({Qpre[2]:.5f}, {Qpre[3]:.5f}).</p>"
    )
    s53 = (
        human("Posisi tidak ditambahkan sebagai angka; vektor Q dan K <b>diputar</b> sebesar sudut yang bergantung posisi. Karena itu skor Q&middot;K "
              "hanya bergantung pada <b>jarak</b> antar token.")
        + formula("&theta; = pos / 500000<sup>2j/d_head</sup> &nbsp;&nbsp; x&prime;<sub>2j</sub> = x<sub>2j</sub>cos&theta; &minus; x<sub>2j+1</sub>sin&theta; &nbsp;&nbsp; "
                  "x&prime;<sub>2j+1</sub> = x<sub>2j</sub>sin&theta; + x<sub>2j+1</sub>cos&theta;")
        + f"<p>d_head = 2 hanya punya satu pasangan (j = 0), frekuensinya 500000<sup>0</sup> = 1, jadi &theta; = posisi = <b>{pos}</b> radian: "
          f"cos {pos} = <b>{cos_p:.5f}</b>, sin {pos} = <b>{sin_p:.5f}</b>.</p>"
        + table(["", "sebelum RoPE", "perhitungan", "sesudah"],
                [["Q head 0", f"({Qpre[0]:.5f}, {Qpre[1]:.5f})",
                  f"x0&prime; = {Qpre[0]:.5f}&times;({cos_p:.5f}) &minus; {Qpre[1]:.5f}&times;({sin_p:.5f}) = {Q[0]:.5f}<br>x1&prime; = {Qpre[0]:.5f}&times;({sin_p:.5f}) + {Qpre[1]:.5f}&times;({cos_p:.5f}) = {Q[1]:.5f}",
                  f"<b>({Q[0]:.5f}, {Q[1]:.5f})</b>"],
                 ["Q head 1", f"({Qpre[2]:.5f}, {Qpre[3]:.5f})",
                  f"x0&prime; = {Qpre[2]:.5f}&times;({cos_p:.5f}) &minus; {Qpre[3]:.5f}&times;({sin_p:.5f}) = {Q[2]:.5f}<br>x1&prime; = {Qpre[2]:.5f}&times;({sin_p:.5f}) + {Qpre[3]:.5f}&times;({cos_p:.5f}) = {Q[3]:.5f}",
                  f"<b>({Q[2]:.5f}, {Q[3]:.5f})</b>"],
                 ["K", f"({Kpre[0]:.5f}, {Kpre[1]:.5f})",
                  f"x0&prime; = {Kpre[0]:.5f}&times;({cos_p:.5f}) &minus; {Kpre[1]:.5f}&times;({sin_p:.5f}) = {K[0]:.5f}<br>x1&prime; = {Kpre[0]:.5f}&times;({sin_p:.5f}) + {Kpre[1]:.5f}&times;({cos_p:.5f}) = {K[1]:.5f}",
                  f"<b>({K[0]:.5f}, {K[1]:.5f})</b>"]], cls="cmp", numeric_from=99)
        + dev("V tidak diputar. Di model asli ada 32 pasangan dimensi per head dengan frekuensi 500000<sup>&minus;2j/64</sup>: j=0 berputar cepat (kata bersebelahan), "
              "j=31 hampir diam (jarak jauh).")
    )
    # K/V cache semua posisi
    rows = []
    for j in range(pos + 1):
        rows.append([f"{j}", tokname(A[j])]
                    + [f"{v:.4f}" for v in c["n1"][j]]
                    + [f"{v:.4f}" for v in c["K"][j]] + [f"{v:.4f}" for v in c["V"][j]])
    kv_tbl = table(["pos", "token", "n1 d0", "d1", "d2", "d3", "K&prime; k0", "k1", "V v0", "v1"], rows, cls="cmp", numeric_from=2)
    s54 = (
        human("Posisi 0&ndash;14 dihitung dengan cara yang sama seperti 5.1&ndash;5.3 (token masing-masing, sudut = posisi). Hasilnya adalah &lsquo;KV-cache&rsquo; yang dilihat posisi 15.")
        + fold("tampilkan K, V semua posisi", kv_tbl, False)
    )

    def head_block(h):
        q = c["Q"][pos][h * 2:h * 2 + 2]
        s = c["S"][h][pos][:pos + 1]
        sm = s.max()
        e = np.exp(s - sm)
        Z = e.sum()
        w = e / Z
        rows = []
        for j in range(pos + 1):
            k = c["K"][j]
            rows.append([j, tokname(A[j]), f"{k[0]:.5f}", f"{k[1]:.5f}", f"{q[0]*k[0]:.5f}", f"{q[1]*k[1]:.5f}",
                         f"{q[0]*k[0]+q[1]*k[1]:.5f}", f"{s[j]:.5f}", f"{e[j]:.5f}", f"<b>{w[j]:.5f}</b>"])
        t1 = table(["j", "token", "k0", "k1", "q0&middot;k0", "q1&middot;k1", "q&middot;k", "skor = (q&middot;k)/&radic;2", f"e^(skor&minus;{sm:.4f})", "w = e/Z"],
                   rows, cls="cmp", numeric_from=2, heat=None)
        V = c["V"]
        rows2 = []
        for j in range(pos + 1):
            rows2.append([j, tokname(A[j]), f"{w[j]:.5f}", f"{V[j][0]:.5f}", f"{w[j]*V[j][0]:.5f}", f"{V[j][1]:.5f}", f"{w[j]*V[j][1]:.5f}"])
        o = (w[:, None] * V[:pos + 1]).sum(0)
        rows2.append(["", "<b>jumlah</b>", f"{w.sum():.5f}", "", f"<b>{o[0]:.5f}</b>", "", f"<b>{o[1]:.5f}</b>"])
        t2 = table(["j", "token", "w", "V v0", "w&middot;v0", "V v1", "w&middot;v1"], rows2, cls="cmp", numeric_from=2)
        return (f"<h4>Head {h}: q = ({q[0]:.5f}, {q[1]:.5f})</h4>"
                f"<p>skor maksimum = {sm:.5f} di posisi {int(s.argmax())}; &Sigma; e = Z = <b>{Z:.5f}</b>; bobot dijumlahkan = {w.sum():.5f}.</p>"
                + fold("skor dan bobot softmax", t1) + fold("jumlah berbobot nilai V", t2))

    s55 = (
        human("Q posisi 15 dicocokkan dengan K semua token sebelumnya (dan dirinya sendiri) lewat perkalian titik. Skor tinggi = lebih relevan. "
              "Softmax mengubah skor jadi bobot yang jumlahnya 1, lalu nilai V dijumlahkan sesuai bobot itu.")
        + formula("skor<sub>j</sub> = (q &middot; k<sub>j</sub>) / &radic;d_head &nbsp;&nbsp; w<sub>j</sub> = e<sup>skor<sub>j</sub>&minus;max</sup> / &Sigma;e<sup>skor&minus;max</sup> &nbsp;&nbsp; out = &Sigma; w<sub>j</sub>&middot;v<sub>j</sub>")
        + head_block(0) + head_block(1)
        + dev("Kernel (<code>attn_kernels.cuh</code>) memakai <b>online softmax</b>: satu pass, menjaga max <code>m</code>, penyebut <code>l</code>, dan akumulator, tanpa menyimpan matriks S&times;S. "
              "Hasilnya sama dengan softmax langsung seperti di atas. Penyebut disimpan sebagai LSE = m + ln l untuk backward.")
    )
    att = c["att"][pos]
    o = c["o"][pos]
    xm = c["xm"][pos]
    s56 = (
        human("Hasil dua head digabung (4 angka), dicampur oleh <code>Wo</code>, lalu <b>ditambahkan</b> ke vektor awal (jalur pintas / residual) supaya informasi lama tidak hilang.")
        + vec("att", att) + cells(att, P["Wo"], range(4), "o")
        + table(["", "d0", "d1", "d2", "d3"],
                [["x&#8320;"] + [f"{v:.5f}" for v in x0], ["o = att&middot;Wo"] + [f"{v:.5f}" for v in o],
                 ["x_mid = x&#8320; + o"] + [f"<b>{v:.5f}</b>" for v in xm]], cls="cmp", heat=1.0)
    )
    sq2 = xm * xm
    ms2 = sq2.mean()
    rs2_ = 1 / math.sqrt(ms2 + 1e-6)
    n2 = c["n2"][pos]
    s57 = (
        human("RMSNorm kedua, sebelum masuk jaringan makanan (FFN).")
        + table(["", "d0", "d1", "d2", "d3"],
                [["x_mid&sup2;"] + [f"{v:.6f}" for v in sq2], ["n2 = x_mid&middot;rsqrt"] + [f"{v:.5f}" for v in n2]], cls="cmp")
        + f"<p>&Sigma; = {sq2.sum():.6f}; mean = {ms2:.6f}; rsqrt = 1/&radic;({ms2:.6f}+10<sup>&minus;6</sup>) = <b>{rs2_:.5f}</b>.</p>"
    )
    gate, up, su, ff, down = c["gate"][pos], c["up"][pos], c["su"][pos], c["ff"][pos], c["down"][pos]
    rows = []
    for k in range(8):
        e_ = math.exp(-up[k])
        rows.append([k, f"{gate[k]:.5f}", f"{up[k]:.5f}", f"{e_:.5f}", f"{up[k]:.5f}/{1+e_:.5f} = {su[k]:.5f}", f"<b>{ff[k]:.5f}</b>"])
    s58 = (
        human("Tiap token diolah sendirian (tanpa melihat tetangga) oleh dua jalur: <b>gate</b> dan <b>up</b>. Jalur up dihaluskan fungsi SiLU, lalu dikalikan dengan gate.")
        + formula("gate = n2&middot;W1 &nbsp;&nbsp; up = n2&middot;W3 &nbsp;&nbsp; silu(u) = u / (1 + e<sup>&minus;u</sup>) &nbsp;&nbsp; ff = gate &middot; silu(up)")
        + fold("gate = n2&middot;W1 (8 keluaran)", cells(n2, P["W1"], range(8), "gate"))
        + fold("up = n2&middot;W3 (8 keluaran)", cells(n2, P["W3"], range(8), "up"))
        + table(["k", "gate", "up", "e^(&minus;up)", "silu(up) = up/(1+e^&minus;up)", "ff = gate&middot;silu"], rows, cls="cmp", numeric_from=1)
        + dev("Dalam kode, nama <code>gate</code> dan <code>up</code> tertukar dibanding konvensi umum (SiLU dikenakan pada <code>up</code>, bukan pada <code>gate</code>). "
              "Secara matematis setara karena keduanya dipelajari; yang penting konsisten antara training dan inferensi.")
    )
    x1 = c["x1"][pos]
    xf = c["xf"][pos]
    sq3 = x1 * x1
    ms3 = sq3.mean()
    rs3 = 1 / math.sqrt(ms3 + 1e-6)
    s59 = (
        human("Hasil FFN dikembalikan ke 4 dimensi oleh <code>W2</code>, ditambahkan ke jalur pintas, lalu dinormalisasi sekali lagi (RMSNorm terakhir).")
        + cells(ff, P["W2"], range(4), "down")
        + table(["", "d0", "d1", "d2", "d3"],
                [["x_mid"] + [f"{v:.5f}" for v in xm], ["down = ff&middot;W2"] + [f"{v:.5f}" for v in down],
                 ["x1 = x_mid + down"] + [f"{v:.5f}" for v in x1], ["x1&sup2;"] + [f"{v:.6f}" for v in sq3],
                 ["xf = x1&middot;rsqrt"] + [f"<b>{v:.5f}</b>" for v in xf]], cls="cmp", heat=1.0)
        + f"<p>mean(x1&sup2;) = {ms3:.6f}; rsqrt = <b>{rs3:.5f}</b>.</p>"
    )
    lg = c["logits"][pos]
    mx = lg.max()
    ex = np.exp(lg - mx)
    Z = ex.sum()
    pr = ex / Z
    tgi = m0.idx[TGT_TOK]
    terms = [xf[i] * P["E"][tgi][i] for i in range(4)]
    rows = []
    for i, uid in enumerate(UNIQ):
        mark = " &larr; target" if i == tgi else ""
        rows.append([i, tokname(uid) + mark, f"{lg[i]:.5f}", f"{ex[i]:.5f}", f"{pr[i]:.5f}"])
    s510 = (
        human("Untuk tiap token di kamus, model menghitung satu skor (logit): seberapa cocok token itu jadi kata berikutnya. "
              "Tabel embedding dipakai lagi di sini (weight tying), jadi tidak ada matriks output terpisah.")
        + formula("logit[v] = xf &middot; E[v] &nbsp;&nbsp; p[v] = e<sup>logit[v]&minus;max</sup> / Z &nbsp;&nbsp; loss = &minus;ln p[target]")
        + f"<p>Contoh untuk target <b>{tokname(TGT_TOK)}</b> (baris {tgi}): " +
          " + ".join(f"({xf[i]:.5f}&times;{P['E'][tgi][i]:.5f})" for i in range(4)) +
          " = " + " + ".join(f"{t:.5f}" for t in terms) + f" = <b>{sum(terms):.5f}</b></p>"
        + fold(f"semua {m0.V} logit", table(["baris", "token", "logit", "e^(logit&minus;max)", "p"], rows, cls="cmp", numeric_from=2))
        + f"<p>max = {mx:.5f}; Z = &Sigma; = <b>{Z:.5f}</b>; ln Z = {math.log(Z):.5f}.</p>"
        + f"<p><b>loss</b> = &minus;(logit_target &minus; max &minus; ln Z) = &minus;({lg[tgi]:.5f} &minus; {mx:.5f} &minus; {math.log(Z):.5f}) = <b>{-(lg[tgi]-mx-math.log(Z)):.5f}</b> "
          f"(cek: &minus;ln {pr[tgi]:.6f}). Model acak dengan {m0.V} kemungkinan punya loss ln {m0.V} = {math.log(m0.V):.5f}.</p>"
    )
    inner = (
        sub("5.1 Embedding dan RMSNorm pertama", s51) + sub("5.2 Proyeksi Q, K, V", s52) + sub("5.3 RoPE", s53)
        + sub("5.4 K dan V semua posisi", s54) + sub("5.5 Attention, dua head", s55)
        + sub("5.6 Wo dan residual pertama", s56) + sub("5.7 RMSNorm kedua", s57) + sub("5.8 FFN SwiGLU", s58)
        + sub("5.9 W2, residual kedua, RMSNorm akhir", s59) + sub("5.10 Logit, softmax, loss", s510)
    )
    return sec(5, "Forward pass satu posisi, dihitung penuh", inner,
               f"Posisi {POS} kalimat A: input <b>&lt;|assistant|&gt;</b>, target <b>{tokname(TGT_TOK)}</b>.")


# ---------------------------------------------------------------------------
# 6. Loss seluruh batch
# ---------------------------------------------------------------------------
def s6():
    rowsA, rowsB = [], []
    sA = sB = 0.0
    for nm, idx, seq, mk, store in (("A", 0, A, mA, rowsA), ("B", 1, B, mB, rowsB)):
        info = INF0[idx]
        tot = 0.0
        for i in range(len(seq)):
            if mk[i]:
                tg = seq[i + 1] if i + 1 < len(seq) else sim.END
                p = info["p"][i][info["tg"][i]]
                store.append([i, tokname(seq[i]), tokname(tg), f"{p:.5f}", f"{info['loss'][i]:.5f}"])
                tot += info["loss"][i]
        store.append(["", "", "<b>jumlah</b>", "", f"<b>{tot:.5f}</b>"])
        if nm == "A": sA = tot
        else: sB = tot
    hdr = ["pos", "input", "target", "p(target)", "loss = &minus;ln p"]
    inner = (
        human("Semua posisi bermask dari kedua kalimat dijumlahkan, lalu dibagi jumlah posisi bermask. Itulah angka tunggal yang ingin diturunkan training.")
        + sub("Kalimat A (EN&rarr;JA)", table(hdr, rowsA, cls="cmp", numeric_from=3))
        + sub("Kalimat B (JA&rarr;EN)", table(hdr, rowsB, cls="cmp", numeric_from=3))
        + formula(f"loss batch = (&Sigma;A + &Sigma;B) / jumlah posisi = ({sA:.5f} + {sB:.5f}) / {TOT0} = <b>{L0:.5f}</b>")
        + f"<p>Pembanding model acak: ln {m0.V} = {math.log(m0.V):.5f}. Loss awal hampir sama, artinya model belum tahu apa-apa. "
          f"(Dengan mask lama loss awal {LOLD:.5f}.)</p>"
    )
    return sec(6, "Loss seluruh batch (langkah 0)", inner)
