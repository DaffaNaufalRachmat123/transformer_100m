#!/usr/bin/env python3
"""
gpu_info.py - tabel GPU Vast.ai -> arsitektur CUDA (sm_XX) untuk build.

  python3 gpu_info.py list [filter]     tabel GPU (filter: potongan nama, mis. 4090)
  python3 gpu_info.py show <nama>       detail satu GPU + baris .env + link pencarian Vast
  python3 gpu_info.py arch <nama>       cetak angka arsitektur saja (mis. 120) - dipakai deploy.sh
  python3 gpu_info.py refresh           ambil ulang daftar dari Vast.ai (butuh internet, ~1-2 menit)

SUMBER DATA
  gpu_table.tsv dibuat oleh 'refresh' dari API pencarian publik Vast.ai
  (https://console.vast.ai/api/v0/bundles/), TANPA API key. Kolom `sm` berasal dari
  field `compute_cap` tiap penawaran (mis. 890 -> sm_89), jadi mengikuti data Vast,
  bukan tebakan. Kolom `vram_gb` = nilai VRAM paling sering muncul pada penawaran
  GPU utuh (bukan pecahan). `cuda_min` dan `muat` adalah penilaian tambahan, bukan dari Vast.
"""
import argparse, collections, json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "gpu_table.tsv")
API = "https://console.vast.ai/api/v0/bundles/"
CAPS = [520, 530, 600, 610, 700, 720, 750, 800, 860, 870, 890, 900, 1000, 1010, 1030, 1100, 1200, 1210]
COLS = ["gpu_name", "sm", "vram_gb", "vram_max_gb", "offers", "min_usd_gpu_hr", "slug"]

# GPU yang benar-benar sudah dijalankan untuk proyek ini (model ini, bukan sekadar arsitektur sama)
TESTED = {"Tesla T4": "TERUJI (Kaggle 2xT4, batch 7)", "RTX 4090": "TERUJI (Vast, batch 8-14)"}


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def cuda_min(sm):
    """Toolkit CUDA minimum untuk mengompilasi sm_XX (dokumentasi NVIDIA; bukan data Vast)."""
    if sm >= 121: return "13.0"
    if sm >= 120: return "12.8"
    if sm >= 103: return "12.9"
    if sm >= 100: return "12.8"
    if sm >= 89:  return "11.8"
    if sm >= 86:  return "11.1"
    if sm >= 80:  return "11.0"
    if sm >= 75:  return "10.0"
    return "9.0"


def cuda_max(sm):
    """CUDA 13.0 menghapus kompilasi untuk sm_50..sm_70 (terlihat juga di `nvcc --list-gpu-code` instance Anda)."""
    return "12.x (CUDA 13 tidak mendukung)" if sm < 75 else "-"


def fit(vram_gb):
    """PERKIRAAN saya (bukan data Vast). Dasar: T4 15 GB berhasil batch 7 (log Kaggle), 4090 24 GB berhasil
    batch 14. Bobot+optimizer+gradien ~8.5 GB (dari ukuran checkpoint), sisanya aktivasi + arena trainer.
    Di bawah 12 GB hampir pasti tidak muat; 12-14 GB belum pernah diuji."""
    if vram_gb >= 22: return "ok"
    if vram_gb >= 15: return "ok (batch kecil)"
    if vram_gb >= 12: return "sempit (tak diuji)"
    return "kemungkinan tidak"


def load():
    if not os.path.exists(TABLE):
        sys.exit(f"{TABLE} belum ada. Jalankan:  ./deploy.sh gpus refresh")
    rows, header = [], None
    for ln in open(TABLE, encoding="utf-8"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if header is None:
            header = parts
            continue
        r = dict(zip(header, parts))
        r["sm"] = int(r["sm"]); r["vram_gb"] = int(r["vram_gb"]); r["vram_max_gb"] = int(r["vram_max_gb"])
        r["offers"] = int(r["offers"]); r["min_usd_gpu_hr"] = float(r["min_usd_gpu_hr"])
        rows.append(r)
    return rows


def find(rows, name):
    n = norm(name)
    exact = [r for r in rows if norm(r["gpu_name"]) == n]
    if exact:
        return exact[0]
    cand = [r for r in rows if n in norm(r["gpu_name"])]
    if len(cand) == 1:
        return cand[0]
    if not cand:
        sys.exit(f"GPU '{name}' tidak ada di tabel. Lihat daftar:  ./deploy.sh gpus   (atau perbarui: ./deploy.sh gpus refresh)")
    sys.exit(f"'{name}' ambigu, pilih salah satu: " + ", ".join(r["gpu_name"] for r in cand))


# Catatan tambahan (penilaian saya, BUKAN dari Vast)
NOTES = {
    "GB10": "nilai sm dari Vast kemungkinan kurang akurat untuk chip ini (belum saya verifikasi); build tetap memeriksa GPU asli",
    "GB200": "CPU ARM (Grace)", "GB300": "CPU ARM (Grace)",
    "CMP 170HX": "GPU mining, tidak diuji",
}


def note(r):
    out = []
    if r["gpu_name"] in TESTED: out.append(TESTED[r["gpu_name"]])
    if r["gpu_name"] in NOTES: out.append(NOTES[r["gpu_name"]])
    if r["sm"] < 75: out.append("butuh image CUDA 12.x")
    return "; ".join(out)


def cmd_list(a):
    rows = load()
    if a.filter:
        f = norm(a.filter)
        rows = [r for r in rows if f in norm(r["gpu_name"]) or f == f"sm{r['sm']}" or f == str(r["sm"])]
    if not rows:
        sys.exit("tidak ada GPU yang cocok")
    rows.sort(key=lambda r: (r["sm"], r["gpu_name"]))
    w = max(len(r["gpu_name"]) for r in rows)
    print(f"{'GPU (nama di Vast)':<{w}}  {'arsitektur':<10} {'VRAM':>5} {'tawaran':>7} {'min $/jam':>9}  {'muat 492M?':<19} catatan")
    print("-" * (w + 70))
    for r in rows:
        print(f"{r['gpu_name']:<{w}}  sm_{r['sm']:<7} {r['vram_gb']:>3}GB {r['offers']:>7} {r['min_usd_gpu_hr']:>9.2f}  {fit(r['vram_gb']):<19} {note(r)}")
    print(f"\n{len(rows)} GPU. Pilih satu di .env:  GPU_MODEL=\"<nama persis di kolom pertama>\"   |   detail: ./deploy.sh gpus show <nama>")
    print("'min $/jam' = penawaran termurah per GPU saat tabel dibuat (bukan harga tetap).")


def cmd_show(a):
    r = find(load(), a.name)
    sm = r["sm"]
    vmin = max(12.4, float(cuda_min(sm)))
    url = ("https://cloud.vast.ai/?gpu_option=" + urllib.parse.quote_plus(r["gpu_name"].lower())
           + "&gpuModelNames=" + r["slug"] + "&instanceDiskSizeMin=40&machinePortsOpenMin=8"
           + "&offerGpuNumMax=1&offerGpuNumMin=1&versionCudaMin=" + f"{vmin:g}" + "&volumeDiskSizeMin=10")
    print(f"GPU            : {r['gpu_name']}")
    print(f"arsitektur     : sm_{sm}  (compute capability {sm // 10}.{sm % 10}, dari data Vast)")
    print(f"VRAM           : {r['vram_gb']} GB (maks pada penawaran: {r['vram_max_gb']} GB)")
    print(f"muat model 492M: {fit(r['vram_gb'])}")
    print(f"CUDA toolkit   : butuh >= {cuda_min(sm)}" + (f", dan <= {cuda_max(sm)}" if sm < 75 else ""))
    print(f"penawaran      : {r['offers']} saat tabel dibuat, termurah ${r['min_usd_gpu_hr']:.2f}/GPU/jam")
    if note(r): print(f"catatan        : {note(r)}")
    print("catatan        : kolom 'muat' = perkiraan saya, bukan data Vast")
    if r["gpu_name"] not in TESTED:
        print("peringatan     : model GPU ini BELUM pernah diuji untuk proyek ini. Arsitektur dan kemampuan memori dari data Vast,")
        print("                 kecepatan/batch terbaik harus Anda ukur sendiri (coba beberapa BATCH_SIZE ~1 menit).")
    print("\nIsi di .env:")
    print(f'  GPU_MODEL="{r["gpu_name"]}"')
    print("  CUDA_ARCH=auto            # arsitektur otomatis dari GPU_MODEL; tervalidasi dengan GPU asli di instance")
    print("\nPencarian di Vast (pola URL dari contoh Anda; slug model multi-kata belum saya verifikasi):")
    print("  " + url)
    print(f"  Pilih image CUDA >= {max(12.4, float(cuda_min(sm))):g}, disk >= 40 GB, GPU = 1.")


def cmd_arch(a):
    r = find(load(), a.name)
    print(r["sm"])


def fetch(args):
    cap, typ = args
    q = {"compute_cap": {"eq": cap}, "order": [["dph_total", "asc"]], "limit": 1000, "type": typ}
    url = API + "?q=" + urllib.parse.quote(json.dumps(q))
    err = ""
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            return args, json.load(urllib.request.urlopen(req, timeout=45))
        except Exception as e:
            err = str(e); time.sleep(1.5)
    return args, {"error": err}


def cmd_refresh(a):
    agg = collections.defaultdict(lambda: {"caps": collections.Counter(), "ram_full": collections.Counter(),
                                           "ram_all": [], "n": 0, "min_pg": 1e9})
    errs = []
    print(f"mengambil data dari {API} ({len(CAPS) * 2} permintaan)...", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for (cap, typ), d in ex.map(fetch, [(c, t) for c in CAPS for t in ("on-demand", "bid")]):
            if "error" in d:
                errs.append(f"cap={cap} {typ}: {d['error']}"); continue
            for o in d.get("offers", []):
                g = o.get("gpu_name")
                if not g: continue
                s = agg[g]; s["n"] += 1; s["caps"][o.get("compute_cap")] += 1
                ram = o.get("gpu_ram") or 0
                s["ram_all"].append(ram)
                if (o.get("gpu_frac") in (None, 1, 1.0)) and (o.get("num_gpus") or 1) == 1:
                    s["ram_full"][round(ram / 1024)] += 1
                s["min_pg"] = min(s["min_pg"], (o.get("dph_total") or 0) / max(o.get("num_gpus") or 1, 1))
    if errs:
        print("PERINGATAN, sebagian permintaan gagal:\n  " + "\n  ".join(errs), file=sys.stderr)
    if not agg:
        sys.exit("tidak ada data diterima; tabel lama TIDAK diubah")
    out = []
    for g, s in agg.items():
        cap = s["caps"].most_common(1)[0][0]
        vram = (s["ram_full"].most_common(1)[0][0] if s["ram_full"] else round(max(s["ram_all"]) / 1024))
        out.append([g, cap // 10, vram, round(max(s["ram_all"]) / 1024), s["n"], f"{s['min_pg']:.2f}", norm(g)])
    out.sort(key=lambda r: (r[1], r[0]))
    tmp = TABLE + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"# Dibuat {time.strftime('%Y-%m-%d %H:%M:%S')} oleh gpu_info.py refresh\n")
        f.write(f"# Sumber: {API} (publik, tanpa API key). sm = compute_cap/10 dari tiap penawaran.\n")
        f.write("# Jangan diedit manual; jalankan  ./deploy.sh gpus refresh  untuk memperbarui.\n")
        f.write("\t".join(COLS) + "\n")
        for r in out:
            f.write("\t".join(str(x) for x in r) + "\n")
    os.replace(tmp, TABLE)
    print(f"{len(out)} tipe GPU ditulis ke {TABLE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list"); p.add_argument("filter", nargs="?", default=""); p.set_defaults(fn=cmd_list)
    p = sp.add_parser("show"); p.add_argument("name"); p.set_defaults(fn=cmd_show)
    p = sp.add_parser("arch"); p.add_argument("name"); p.set_defaults(fn=cmd_arch)
    p = sp.add_parser("refresh"); p.set_defaults(fn=cmd_refresh)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
