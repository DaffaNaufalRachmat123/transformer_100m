#!/usr/bin/env python3
"""market_scan.py - pantau pasar offer on-demand Vast selama ~N menit.

Mengambil beberapa 'round' snapshot (default 5 round @ 120 detik = ~10 menit),
menyaring host layak (1 GPU utuh, rentable, non-CN, reliab>=0.95, up & down
>=100Mbps, disk>=75GB, VRAM memadai, driver CUDA cukup), lalu mencetak
LEADERBOARD diurutkan token training NET per $5 (setelah potong overhead
upload korpus + build + checkpoint ~1.25 jam).

Pakai:
  cd vast_deploy && set -a; . ./.env; set +a
  python3 tools/market_scan.py --rounds 5 --interval 120
"""
import argparse
import json
import os
import sys
import time
import urllib.request

API = "https://console.vast.ai/api/v0"
UA = {"User-Agent": "curl/8", "Content-Type": "application/json"}

# (TFLOPS FP16-tensor FP32acc, bandwidth GB/s, VRAM minimum GB).
# Jangkar TERUKUR proyek ini: RTX 4090 = 12.8K tok/s (batch 8).
SPECS = {
    "RTX 4090":    (165.2, 1008, 22),
    "RTX 4090D":   (132.2, 1008, 22),
    "RTX 5090":    (209.5, 1792, 30),
    "RTX 3090":    (71.3,   936, 22),
    "RTX 3090 Ti": (79.6,  1008, 22),
    "Tesla V100S": (125.0, 1134, 22),
    "Tesla V100":  (125.0,  900, 22),
    "A100 SXM4":   (312.0, 2039, 38),
    "A100 PCIE":   (312.0, 1555, 38),
    "RTX A6000":   (77.4,   768, 22),
    "L40S":        (181.3,  864, 22),
    "RTX 5080":    (112.6,  960, 15),
    "RTX 5070 Ti": (89.0,   896, 15),
}
A_TF, A_BW, A_TPS = 165.2, 1008, 12800.0
DISK_GB = 70
OVERHEAD_H = 1.25   # potongan setup: upload korpus 25GB + build + checkpoint


def est_tps(gpu):
    """Estimasi tok/s (model Amdahl: 60% matmul skala TFLOPS, 40% elemenwise skala BW)."""
    tf, bw, vram = SPECS[gpu]
    t = A_TPS / (0.6 / (tf / A_TF) + 0.4 / (bw / A_BW))
    return t * 0.85 if vram < 22 else t   # VRAM sempit -> batch diturunkan


def api_key():
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        for ln in open(env):
            if ln.startswith("VAST_API_KEY="):
                key = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("VAST_API_KEY tidak ditemukan (env atau vast_deploy/.env)")
    return key


def asks(key, gpu):
    body = {"q": {"gpu_name": {"eq": gpu}, "order": [["dph_total", "asc"]],
                  "type": "on-demand", "limit": 200}}
    req = urllib.request.Request(API + "/search/asks/", method="PUT",
                                 data=json.dumps(body).encode(),
                                 headers={**UA, "Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=60)).get("offers", [])


def filter_ok(gpu, o):
    """None kalau host tidak layak; dict ringkas kalau layak."""
    if (o.get("gpu_frac") or 1) != 1 or (o.get("num_gpus") or 1) != 1:
        return None
    if not o.get("rentable") or o.get("rented"):
        return None                      # ghost listing / sudah disewa
    loc = (o.get("geolocation") or "?")
    if loc.split(",")[-1].strip() == "CN":
        return None                      # Docker Hub sering nyangkut di CN
    rel = o.get("reliability2") or o.get("reliability") or 0
    up, down = o.get("inet_up") or 0, o.get("inet_down") or 0
    disk, vram = o.get("disk_space") or 0, (o.get("gpu_ram") or 0) / 1024
    cuda = float(o.get("cuda_max_good") or 0)
    if rel < 0.95 or up < 100 or down < 100 or disk < DISK_GB + 5:
        return None
    if vram and vram < SPECS[gpu][2]:
        return None
    if cuda and cuda < 12.4:
        return None
    if gpu in ("RTX 5090", "RTX 5080", "RTX 5070 Ti") and cuda and cuda < 12.8:
        return None                      # Blackwell butuh image CUDA >= 12.8
    eff = (o.get("dph_base") or o.get("dph_total", 0)) + \
        (o.get("storage_cost") or 0.10) / 730 * DISK_GB
    return {"id": o.get("id"), "eff": eff, "rel": rel, "up": up, "down": down,
            "disk": disk, "vram": vram, "cuda": cuda, "loc": loc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--budget", type=float, default=5.0)
    a = ap.parse_args()
    key = api_key()

    hist = {}   # (gpu, id) -> {"first": row, "prices": [eff per round]}
    for rnd in range(1, a.rounds + 1):
        print("[round %d/%d  %s]" % (rnd, a.rounds, time.strftime("%H:%M:%S")), flush=True)
        for gpu in SPECS:
            try:
                rows = [r for r in (filter_ok(gpu, o) for o in asks(key, gpu)) if r]
            except Exception as e:
                print("  %-12s ERROR %s" % (gpu, e), flush=True)
                continue
            for r in rows:
                h = hist.setdefault((gpu, r["id"]), {"first": r, "prices": []})
                h["prices"].append(r["eff"])
            if rows:
                b = min(rows, key=lambda x: x["eff"])
                print("  %-12s id=%-9s $%-6.3f reliab=%.3f up=%4.0fMbps down=%4.0fMbps "
                      "vram=%2.0fGB disk=%4.0fGB  %s" % (
                          gpu, b["id"], b["eff"], b["rel"], b["up"], b["down"],
                          b["vram"], b["disk"], b["loc"]), flush=True)
            else:
                print("  %-12s (tidak ada offer layak)" % gpu, flush=True)
        if rnd < a.rounds:
            time.sleep(a.interval)

    need = max(2, a.rounds // 2 + 1)
    board = []
    for (gpu, oid), h in hist.items():
        if len(h["prices"]) < need:
            continue                     # tidak stabil: muncul lalu hilang
        eff = max(h["prices"])           # konservatif: harga tertinggi yang terlihat
        t = est_tps(gpu)
        hrs = a.budget / eff
        net = t * 3600 * max(0.1, hrs - OVERHEAD_H) / 1e9
        board.append((net, t, hrs, gpu, oid, h))
    print("\n=== LEADERBOARD: token NET per $%.2f (potong %.2f jam setup; hanya offer "
          "stabil >= %d/%d round) ===" % (a.budget, OVERHEAD_H, need, a.rounds), flush=True)
    for net, t, hrs, gpu, oid, h in sorted(board, reverse=True)[:12]:
        r = h["first"]
        lo, hi = min(h["prices"]), max(h["prices"])
        print("  %-12s id=%-9s $%-6.3f (rng $%.3f-%.3f, %d/%d)  ~%4.1fK tok/s  "
              "NET=%.2fB jam=%.1f  reliab=%.3f up=%d/%dMbps vram=%.0fGB disk=%.0fGB  %s" % (
                  gpu, oid, hi, lo, hi, len(h["prices"]), a.rounds, t / 1000, net, hrs,
                  r["rel"], r["up"], r["down"], r["vram"], r["disk"], r["loc"]), flush=True)
    if not board:
        print("  (tidak ada offer stabil yang layak - coba jalankan lagi nanti)", flush=True)


if __name__ == "__main__":
    main()
