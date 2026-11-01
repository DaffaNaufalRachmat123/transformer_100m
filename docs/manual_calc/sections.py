"""sections.py - merakit halaman lengkap (CSS + navigasi + bagian 0..13)."""
import sec_a
import sec_b

CSS = r"""
:root{
  --bg:#0b1020;--bg2:#111833;--panel:#151d3b;--panel2:#1b2548;--line:#27335f;
  --tx:#e8ecff;--dim:#8f9bc7;--accent:#38bdf8;--accent2:#a78bfa;--pink:#fb7185;--green:#34d399;--amber:#fbbf24;
  --ja:#fda4af;--en:#7dd3fc;--by:#fcd34d;--sp:#c4b5fd;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1200px 600px at 10% -10%,#1a2350 0,transparent 60%),
 radial-gradient(900px 500px at 100% 0,#2a1a50 0,transparent 55%),var(--bg);
 color:var(--tx);font-family:'Inter',system-ui,sans-serif;font-size:15.5px;line-height:1.62}
code,pre,.n,.vec,table.cells td,table.mat td{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace}
code{background:#0f1630;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:.88em;color:#cfe3ff}
a{color:var(--accent)}
.wrap{display:grid;grid-template-columns:260px minmax(0,1fr);gap:0;max-width:1480px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:28px 18px;border-right:1px solid var(--line);background:rgba(11,16,32,.6);backdrop-filter:blur(6px)}
nav .brand{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:15px;letter-spacing:.08em;color:var(--accent);text-transform:uppercase;margin-bottom:16px}
nav a{display:flex;gap:10px;align-items:baseline;padding:7px 10px;border-radius:9px;color:var(--dim);text-decoration:none;font-size:13.5px}
nav a:hover{background:var(--panel);color:var(--tx)}
nav a b{font-family:'JetBrains Mono',monospace;color:var(--accent2);min-width:20px}
main{padding:0 44px 120px;min-width:0}
header.hero{padding:70px 0 34px}
.kicker{font-family:'JetBrains Mono',monospace;font-size:12.5px;letter-spacing:.18em;color:var(--accent);text-transform:uppercase}
h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(34px,5vw,60px);line-height:1.04;margin:10px 0 14px;
 background:linear-gradient(92deg,#fff 0,#7dd3fc 45%,#c4b5fd 80%);-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--dim);max-width:820px;font-size:17px}
.pairs{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:26px 0 8px}
.card{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.card .t{font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:.16em;color:var(--dim);text-transform:uppercase}
.card .s{font-size:22px;margin-top:6px;font-weight:600}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.stat b{font-family:'Space Grotesk',sans-serif;font-size:26px;color:var(--accent);display:block}
.stat span{font-size:12.5px;color:var(--dim)}
section{margin-top:76px;scroll-margin-top:20px}
h2{font-family:'Space Grotesk',sans-serif;font-size:32px;margin:0 0 6px;display:flex;align-items:center;gap:14px;letter-spacing:-.01em}
h2 .sn{display:inline-grid;place-items:center;width:46px;height:46px;border-radius:13px;background:linear-gradient(140deg,var(--accent),var(--accent2));color:#07101f;font-size:22px;font-weight:700}
h3{font-family:'Space Grotesk',sans-serif;font-size:20px;margin:34px 0 10px;color:#dfe6ff;padding-left:12px;border-left:3px solid var(--accent2)}
h4{margin:20px 0 6px;color:var(--amber);font-family:'Space Grotesk',sans-serif}
.lead{color:var(--dim);margin:0 0 18px;font-size:16.5px}
.box{border-radius:14px;border:1px solid var(--line);margin:14px 0;overflow:hidden;background:var(--panel)}
.box .bt{font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:.16em;padding:7px 14px;font-weight:700}
.box .bb{padding:12px 16px}
.human{border-color:#1f6f5a}.human .bt{background:#0f3d33;color:var(--green)}
.dev{border-color:#4b3a8f}.dev .bt{background:#2a2160;color:#c4b5fd}
.formula{border-color:#6b5b1f;background:#1a1a0e}.formula .bt{background:#3b3310;color:var(--amber)}
.formula .bb{font-size:16.5px;font-family:'Space Grotesk',sans-serif}
.warn{border-color:#8f3a46;background:#221018}.warn .bt{background:#4a1522;color:#fda4af}
.note{border-color:#2a5f8f}.note .bt{background:#10304f;color:#7dd3fc}
.ok{border-color:#1f6f5a;background:#0d2a24}.ok .bt{background:#0f3d33;color:var(--green)}
.tw{overflow:auto;margin:12px 0;border:1px solid var(--line);border-radius:12px;background:#0e1530}
table{border-collapse:collapse;width:100%;font-size:13.8px}
caption{caption-side:top;text-align:left;padding:9px 14px;font-family:'Space Grotesk',sans-serif;font-weight:600;color:#cfd8ff;background:#121a38}
th{background:#16204a;color:#b9c6ff;font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:.06em;text-transform:none;text-align:left;padding:8px 11px;position:sticky;top:0;white-space:nowrap}
td{padding:6px 11px;border-top:1px solid #1d2850;vertical-align:top}
td.r{text-align:right;font-family:'JetBrains Mono',monospace;white-space:nowrap}
tbody tr:hover{background:#18224a}
tr.hl td{background:rgba(251,191,36,.14)!important;box-shadow:inset 3px 0 0 var(--amber)}
table.mat th.rl{background:#16204a;white-space:nowrap}
table.cells td{font-size:12.5px;line-height:1.35}
table.cells .dim{font-size:11px}
.dim{color:var(--dim)}
.chip{display:inline-flex;align-items:baseline;gap:6px;padding:2px 9px 2px 9px;border-radius:999px;font-family:'JetBrains Mono',monospace;font-size:12.5px;border:1px solid;margin:2px 2px;white-space:nowrap}
.chip i{font-style:normal;font-size:10.5px;opacity:.65}
.chip.en{color:var(--en);border-color:#1f5f8a;background:#0b2036}
.chip.ja{color:var(--ja);border-color:#8a3b4a;background:#2a0f1a}
.chip.by{color:var(--by);border-color:#8a6a1f;background:#2a210b}
.chip.sp{color:var(--sp);border-color:#5b4aa0;background:#1d1640}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}
.pair>div{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;font-size:20px;font-weight:600}
.tag{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.14em;padding:2px 8px;border-radius:6px;margin-right:10px;vertical-align:middle}
.tag.en{background:#0b2036;color:var(--en);border:1px solid #1f5f8a}.tag.ja{background:#2a0f1a;color:var(--ja);border:1px solid #8a3b4a}
.ja{color:var(--ja);font-family:'Noto Sans JP','Hiragino Sans','Yu Gothic',sans-serif}
.ja.big{font-size:20px}
pre.raw{background:#0a1028;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:14px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:13px;color:#d6e2ff}
.seqline{margin:10px 0;line-height:2.1}
.vec{background:#0a1028;border:1px dashed var(--line);border-radius:9px;padding:6px 12px;margin:8px 0;font-size:13px;color:#cfe3ff}
.vec .vn{color:var(--amber);font-weight:700}
details{margin:10px 0;border:1px solid var(--line);border-radius:12px;background:rgba(21,29,59,.5)}
summary{cursor:pointer;padding:9px 14px;font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--accent);list-style:none}
summary::before{content:"\25B8  ";color:var(--accent2)}
details[open]>summary::before{content:"\25BE  "}
details>.tw,details>p,details>table{margin:0 12px 12px}
.chart{width:100%;max-width:760px;display:block;margin:14px 0;background:#0a1028;border:1px solid var(--line);border-radius:12px}
.chart .gl{stroke:#1d2850;stroke-width:1}.chart .ax{fill:#8f9bc7;font-size:11px;font-family:'JetBrains Mono',monospace}.chart .mk{stroke:#fbbf24;stroke-dasharray:4 4}
.toc-note{color:var(--dim);font-size:13px}
footer{margin-top:90px;padding-top:24px;border-top:1px solid var(--line);color:var(--dim);font-size:13px}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}nav{position:static;height:auto;border:0}main{padding:0 18px 80px}.pairs,.pair{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
@media print{nav{display:none}.wrap{display:block}body{background:#fff;color:#000}}
"""

NAV = [
    (0, "Batasan simulasi"), (1, "Dua kalimat jadi teks training"), (2, "Tokenisasi"), (3, "Input, target, mask"),
    (4, "Bobot awal"), (5, "Forward satu posisi"), (6, "Loss batch"), (7, "Backward"), (8, "Clipping &amp; AdamW"),
    (9, "Kurva training"), (10, "Setelah training"), (11, "Inferensi"), (12, "Eksperimen kontrol"), (13, "Model mini &rarr; nyata"),
]


def build_all(ctx):
    sec_a.setup(ctx)
    sec_b.setup(ctx)
    g = ctx
    fonts = g["font_css"]()
    nav = "".join(f'<a href="#s{n}"><b>{n}</b>{t}</a>' for n, t in NAV)
    body = [sec_a.s0(), sec_a.s1(), sec_a.s2(), sec_a.s3(), sec_a.s4(), sec_a.s5(), sec_a.s6(),
            sec_b.s7(), sec_b.s8(), sec_b.s9(), sec_b.s10(), sec_b.s11(), sec_b.s12(), sec_b.s13()]
    sim = g["sim"]
    stats = (
        f'<div class="stat"><b>{g["nparam"]}</b><span>parameter model mini</span></div>'
        f'<div class="stat"><b>{g["L0"]:.3f}</b><span>loss langkah 0 (acak = ln {g["m0"].V} = {__import__("math").log(g["m0"].V):.3f})</span></div>'
        f'<div class="stat"><b>{g["LFINAL"]:.1e}</b><span>loss setelah 600 langkah</span></div>'
        f'<div class="stat"><b>{"2/2" if (g["out_a"] == g["ans_a"] and g["out_b"] == g["ans_b"]) else "gagal"}</b><span>arah terjemahan benar (EN&rarr;JA, JA&rarr;EN)</span></div>'
    )
    hero = f"""
<header class="hero">
  <div class="kicker">Manual perhitungan &middot; EN &harr; JA &middot; branch fix/sft-mask-bos-eval</div>
  <h1>Dari dua kalimat<br>sampai model bisa menerjemahkan</h1>
  <p class="sub">Setiap langkah dihitung penuh: tokenisasi, mask, forward, loss, backward, AdamW, sampai inferensi.
  Rumus sama dengan kode repo; ukuran model diperkecil supaya semua angka bisa dibaca.</p>
  <div class="pairs">
    <div class="card"><div class="t">English</div><div class="s">{sim.EN}</div></div>
    <div class="card"><div class="t">&#26085;&#26412;&#35486;</div><div class="s ja">{sim.JA}</div></div>
  </div>
  <div class="stats">{stats}</div>
  <p class="toc-note">Warna token: <span class="chip en">Inggris</span> <span class="chip ja">Jepang</span> <span class="chip by">byte parsial</span> <span class="chip sp">token spesial</span> &nbsp;
  Sel tabel: biru = positif, merah = negatif (makin pekat = makin besar).</p>
</header>"""
    foot = ('<footer>Dihasilkan oleh <code>docs/manual_calc/build_manual.py</code>. Semua angka dihitung ulang saat halaman dibuat. '
            'Aritmatika float64; kernel CUDA dan FP16 tidak disimulasikan.</footer>')
    return (f'<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Manual perhitungan EN-JA</title><style>{fonts}{CSS}</style></head><body><div class="wrap">'
            f'<nav><div class="brand">Manual EN&harr;JA</div>{nav}</nav><main>{hero}{"".join(body)}{foot}</main></div></body></html>')
