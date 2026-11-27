#!/usr/bin/env python3
"""summarize_speed.py <folder>  - ringkas raw/train.log + raw/nvidia_smi.csv menjadi
tok_per_s.csv dan summary.txt. Dipanggil oleh ./deploy.sh pull."""
import csv, os, re, statistics as st, sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
raw = os.path.join(d, "raw")
rx = re.compile(r"step=(\d+)\s+seq=(\d+).*?tok=([\d.]+)([MB])/.*?loss=([\d.]+) ema=([\d.]+) gnorm=([\d.]+).*?lr=([\d.e+-]+) "
                r"([\d.]+)K tok/s ([\d.]+)s/step elapsed=(\S+).*?VRAM ([\d.]+)GB GPU\(proc\) (\d+)%")
rows, run_id, marks = [], 0, []
try:
    lines = open(os.path.join(raw, "train.log"), errors="replace").read().splitlines()
except FileNotFoundError:
    sys.exit(f"tidak ada {raw}/train.log (jalankan ./deploy.sh pull setelah training dimulai)")

for ln in lines:
    if ln.startswith("=== START") or ln.startswith("=== RESTART"):
        run_id += 1
        marks.append((run_id, ln.strip("= ").strip()))
        continue
    m = rx.search(ln)
    if m:
        step, seq, tok, unit, loss, ema, gn, lr, tps, sps, el, vram, gpu = m.groups()
        tokb = float(tok) / 1000 if unit == "M" else float(tok)
        rows.append([run_id, int(step), int(seq), tokb, float(loss), float(ema), float(gn), lr,
                     float(tps) * 1000, float(sps), el, float(vram), int(gpu)])

with open(os.path.join(d, "tok_per_s.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", "seq", "tokens_B", "loss", "ema", "gnorm", "lr", "tok_per_s", "s_per_step", "elapsed", "vram_gb", "gpu_util_pct"])
    w.writerows(rows)

out = []
if rows:
    last = max(r[0] for r in rows)
    cur = [r for r in rows if r[0] == last]
    steady = [r[8] for r in cur[3:]] or [r[8] for r in cur]
    left_tok = max(0.0, 10.0 - cur[-1][3]) * 1e9
    out.append(f"baris log dibaca    : {len(rows)}  (run terakhir: #{last}, {len(cur)} baris)")
    out.append(f"tok/s run terakhir  : rata-rata {st.mean(steady):.0f} | median {st.median(steady):.0f} | min {min(steady):.0f} | maks {max(steady):.0f}")
    out.append(f"token per hari      : {st.mean(steady) * 86400 / 1e9:.2f} B")
    out.append(f"sisa ke 10B token   : {left_tok / 1e9:.2f} B  ~ {left_tok / st.mean(steady) / 3600:.1f} jam")
    out.append(f"langkah/token akhir : {cur[-1][1]} / {cur[-1][3]:.3f} B | gnorm {cur[-1][6]:.2f} | ema loss {cur[-1][5]:.3f}")
    if len({r[0] for r in rows}) > 1:
        out.append("--- per run ---")
        for rid, label in marks:
            rr = [r for r in rows if r[0] == rid]
            if rr:
                s = [r[8] for r in rr[3:]] or [r[8] for r in rr]
                out.append(f"  run {rid}: {st.mean(s):.0f} tok/s ({len(rr)} baris)  [{label[:70]}]")
smi_path = os.path.join(raw, "nvidia_smi.csv")
if os.path.exists(smi_path):
    smi = list(csv.DictReader(open(smi_path), skipinitialspace=True))
    def col(k):
        v = []
        for r in smi:
            try: v.append(float(r[k]))
            except (KeyError, ValueError, TypeError): pass
        return v
    out.append("--- nvidia-smi (seluruh sampel) ---")
    for k, n in (("util_gpu", "GPU util %"), ("power_w", "daya W"), ("sm_clock_mhz", "clock SM MHz"), ("temp_c", "suhu C"), ("mem_used_mib", "VRAM MiB")):
        v = col(k)
        if v:
            out.append(f"  {n:<14}: rata-rata {st.mean(v):.0f} | maks {max(v):.0f}")
open(os.path.join(d, "summary.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\nfile: {d}/summary.txt  {d}/tok_per_s.csv")
