#!/usr/bin/env python3
"""vast_rent.py — sewa instance Vast on-demand otomatis (dipanggil deploy.sh cmd_rent).

Alur: cari offer on-demand terbaik utk GPU -> saring host jelek -> daftarkan key SSH
Mac ke akun Vast -> PUT /asks/{id}/ -> tunggu SSH siap (host:port terbuka) ->
tulis VAST_INSTANCE_ID / SSH_HOST / SSH_PORT / GPU_MODEL ke .env (backup otomatis).

Aman diuji: --dry-run tidak menyewa apa pun, hanya menampilkan rencana.
Baca VAST_API_KEY dari --env-file (default: .env di folder vast_deploy).
"""
import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"   # /v0/instances/ sudah deprecated (respons error -> polling buta)
UA = {"User-Agent": "curl/8", "Content-Type": "application/json"}

# ── Estimasi kecepatan training (492M, batch 8) ──────────────────────────────
# (TFLOPS FP16-tensor akumulasi-FP32 dense, bandwidth GB/s, sm)
# Jangkar TERUKUR proyek ini: RTX 4090 batch 8 = 12.8K tok/s.
# Model Amdahl: 60% waktu matmul (skala TFLOPS tensor), 40% elemenwise (skala bandwidth).
SPECS = {
    "RTX 3090":    (71.3,   936, 86),
    "RTX 3090 Ti": (79.6,  1008, 86),
    "RTX 4090":    (165.2, 1008, 89),
    "RTX 4090D":   (132.2, 1008, 89),
    "RTX 5090":    (209.5, 1792, 120),
    "RTX 5080":    (112.6,  960, 120),
    "Tesla V100S": (125.0, 1134, 70),
    "Tesla V100":  (125.0,  900, 70),
    "L4":        (60.5,   300, 89),
    "L40S":      (181.3,  864, 89),
    "RTX A6000": (77.4,   768, 86),
    "A100 PCIE": (312,   1555, 80),
    "A100 SXM4": (312,   2039, 80),
    "H100 SXM":  (989.4, 3350, 90),
    "H100 NVL":  (756.0, 3350, 90),
    "H200":      (989.4, 4800, 90),
    "H200 NVL":  (1978.8, 4800, 90),
}
ANCHOR_GPU, ANCHOR_TPS = "RTX 4090", 12800.0
F_MAT, F_MEM = 0.6, 0.4

# Image: harus punya nvcc (devel) + cuDNN (CMake REQUIRED). PyTorch cuDNN-devel memenuhi.
IMAGE_SM120 = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"   # Blackwell (5090): CUDA >= 12.8
IMAGE_STD   = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"   # sm 80..90

# onstart instance baru: remote.sh belum ada -> wrapper tidak error, jalan penuh setelah ./deploy.sh source
ONSTART = ("bash -lc 'if [ -f /workspace/remote.sh ]; then bash /workspace/remote.sh onstart; "
           "else echo \"[onstart] remote.sh belum ada (instance baru) - lewati\"; fi'")


def die(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def http(method, path, key, body=None, timeout=60):
    url = path if path.startswith("http") else API + path
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={**UA, "Authorization": "Bearer " + key})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except Exception:
            return {"success": False, "msg": "HTTP %s" % e.code}
    except Exception as e:
        return {"success": False, "msg": str(e)}


def read_env(path):
    env = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def write_env(path, updates):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    bak = path + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(txt)
    lines = txt.splitlines()
    for key, val in updates.items():
        for i, ln in enumerate(lines):
            if ln.startswith(key + "="):
                lines[i] = "%s=%s" % (key, val)
                break
        else:
            lines.append("%s=%s" % (key, val))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return bak


def est_tps(gpu):
    spec = SPECS.get(gpu)
    if not spec:
        return None
    a = SPECS[ANCHOR_GPU]
    r_tf, r_bw = spec[0] / a[0], spec[1] / a[1]
    return ANCHOR_TPS / (F_MAT / r_tf + F_MEM / r_bw)


def fmt_tok(n):
    if n >= 1e9:
        return "%.2f miliar" % (n / 1e9)
    if n >= 1e6:
        return "%.0f juta" % (n / 1e6)
    return "{:,.0f}".format(n)


def fetch_offers(key, gpu):
    # PUT /search/asks/ = sumber ask id yang bisa langsung di-PUT /asks/{id}/.
    # (GET /bundles/ menampilkan listing yang ternyata "ghost": rentable=False -> no_such_ask.)
    body = {"q": {"gpu_name": {"eq": gpu}, "order": [["dph_total", "asc"]],
                  "type": "on-demand", "limit": 200}}
    req = urllib.request.Request(API + "/search/asks/", method="PUT",
                                 data=json.dumps(body).encode(),
                                 headers={**UA, "Authorization": "Bearer " + key})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:
        die("tidak bisa menghubungi API Vast: %s" % e)
    rows = []
    for o in d.get("offers", []):
        if (o.get("gpu_frac") or 1) != 1 or (o.get("num_gpus") or 1) != 1:
            continue
        if o.get("rented"):
            continue
        if not o.get("rentable"):
            continue     # ghost listing: muncul di pencarian tapi tidak bisa disewa (no_such_ask)
        rows.append(o)
    return rows


def offer_view(o):
    rel = o.get("reliability2") or o.get("reliability") or 0.0
    return {
        "id": o.get("id"),
        "dph_total": o.get("dph_total", 0),
        "dph_base": o.get("dph_base", o.get("dph_total", 0)),
        "storage_cost": o.get("storage_cost") or 0.10,   # $/GB/bulan (default konservatif)
        "reliab": rel,
        "inet_up": o.get("inet_up") or 0,
        "inet_down": o.get("inet_down") or 0,
        "disk_space": o.get("disk_space") or 0,
        "cuda_max": float(o.get("cuda_max_good") or 0),
        "vram": o.get("gpu_ram"),
        "loc": o.get("geolocation") or "?",
    }


def tcp_ok(host, port, timeout=6):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Sewa instance Vast on-demand otomatis")
    ap.add_argument("--env-file", default=None, help="path .env (default: <script>/../.env)")
    ap.add_argument("--gpu", default=None, help="nama GPU persis Vast (default: GPU_MODEL dari .env)")
    ap.add_argument("--budget", type=float, default=5.0, help="anggaran $ untuk estimasi token (default 5)")
    ap.add_argument("--disk", type=int, default=70, help="disk GB yang dialokasikan (default 70)")
    ap.add_argument("--max-price", type=float, default=None, help="batas $/jam termasuk storage")
    ap.add_argument("--min-reliab", type=float, default=0.95, help="batas bawah reliability host (default 0.95)")
    ap.add_argument("--min-up", type=float, default=100.0, help="batas bawah uplink Mbps utk upload Drive (default 100)")
    ap.add_argument("--image", default=None, help="override image docker (default dipilih otomatis)")
    ap.add_argument("--label", default="492m-pretrain", help="label instance di Vast")
    ap.add_argument("--allow-cn", action="store_true", help="bolehkan host di Tiongkok (default: tidak, image Docker Hub sering nyangkut)")
    ap.add_argument("--dry-run", action="store_true", help="tampilkan rencana tanpa menyewa")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    env_path = args.env_file or os.path.join(here, "..", ".env")
    if not os.path.exists(env_path):
        die("file .env tidak ditemukan: %s" % env_path)
    env = read_env(env_path)

    key = env.get("VAST_API_KEY") or os.environ.get("VAST_API_KEY") or ""
    if not key:
        die("VAST_API_KEY tidak ada di %s (buat di https://cloud.vast.ai/cli/)" % env_path)
    gpu = args.gpu or env.get("GPU_MODEL") or ""
    if not gpu:
        die('GPU belum ditentukan. Isi GPU_MODEL di .env atau tambahkan --gpu "RTX 5090"')
    sm = SPECS.get(gpu, (0, 0, None))[2]
    if sm is None:
        print("PERINGATAN: %s belum ada di tabel estimasi SPECS -> tok/s tidak bisa diestimasi "
              "(sewa tetap bisa jalan)." % gpu)

    # ── saldo ────────────────────────────────────────────────────────────────
    u = http("GET", "/users/current/", key)
    u = u.get("user") or u
    credit = u.get("credit")
    if credit is None:
        credit = u.get("balance")
    print("saldo Vast : %s" % ("$%.2f" % credit if credit is not None else "(tidak terbaca)"))
    if credit is not None and credit < 1.0:
        print("PERINGATAN: saldo di bawah $1 — top-up dulu sebelum menyewa (on-demand butuh saldo).")

    # ── offer ────────────────────────────────────────────────────────────────
    offers = [offer_view(o) for o in fetch_offers(key, gpu)]
    if not offers:
        die('tidak ada offer on-demand 1-GPU-utuh untuk "%s" sekarang. Coba nama lain: ./deploy.sh gpus' % gpu)

    need_cuda = 12.8 if sm == 120 else 12.4
    good, rejected = [], []
    for o in offers:
        eff = o["dph_base"] + o["storage_cost"] / 730.0 * args.disk
        o["eff"] = eff
        why = []
        if o["reliab"] < args.min_reliab:
            why.append("reliab %.3f" % o["reliab"])
        if o["inet_up"] < args.min_up:
            why.append("uplink %.0fMbps" % o["inet_up"])
        if o["disk_space"] < args.disk + 10:
            why.append("disk host %.0fGB" % o["disk_space"])
        if o["cuda_max"] and o["cuda_max"] < need_cuda:
            why.append("driver CUDA max %.1f < %.1f" % (o["cuda_max"], need_cuda))
        if args.max_price and eff > args.max_price:
            why.append("harga eff $%.3f > $%.3f" % (eff, args.max_price))
        if not args.allow_cn and (o["loc"] or "").split(",")[-1].strip() == "CN":
            why.append("host di Tiongkok (tarik image Docker Hub sering nyangkut -> 'Creating' selamanya)")
        if why:
            rejected.append((o, why))
        else:
            good.append(o)

    print("\n== Offer on-demand %s (termurah dulu; 'eff' = GPU + storage %dGB) ==" % (gpu, args.disk))
    shown = good[:5] + [o for o, _ in rejected[:2]]
    for o in sorted(shown, key=lambda x: x["eff"]):
        tag = "OK " if o in good else "X  "
        print("  %s id=%-9s $%.3f/jam eff  reliab=%.3f  up=%4.0fMbps  disk=%5.0fGB  cuda<=%.1f  %s"
              % (tag, o["id"], o["eff"], o["reliab"], o["inet_up"], o["disk_space"], o["cuda_max"], o["loc"]))
    if not good:
        die("tidak ada offer yang lolos saringan (reliab>=%.2f, up>=%.0fMbps, disk>=%dGB, CUDA>=%.1f). "
            "Longgarkan dengan --min-reliab / --min-up, atau ganti GPU."
            % (args.min_reliab, args.min_up, args.disk + 10, need_cuda))

    best = sorted(good, key=lambda x: x["eff"])[0]
    image = args.image or (IMAGE_SM120 if sm == 120 else IMAGE_STD)

    # ── estimasi token utk budget ───────────────────────────────────────────
    tps = est_tps(gpu)
    hours = args.budget / best["eff"] if best["eff"] > 0 else 0
    print("\n== Rencana ==")
    print("  offer      : id %s (%s)" % (best["id"], best["loc"]))
    print("  biaya      : $%.3f/jam (GPU $%.3f + storage %dGB)" % (best["eff"], best["dph_base"], args.disk))
    print("  image      : %s" % image)
    print("  disk/label : %dGB / %s" % (args.disk, args.label))
    print("  onstart    : %s" % ONSTART[:80] + "...")
    if tps:
        print("  estimasi   : ~%.1fK tok/s -> $%g = ~%.1f jam = %s token"
              % (tps / 1000.0, args.budget, hours, fmt_tok(tps * 3600 * hours)))
        print("               (di luar ~1-2 jam pertama utk upload korpus 25GB + build)")

    if args.dry_run:
        print("\nDRY-RUN: tidak menyewa. Jalankan tanpa --dry-run untuk menyewa instance ini.")
        return

    # ── cek instance lama dgn label sama ────────────────────────────────────
    insts = http("GET", API_V1 + "/instances/", key).get("instances") or []
    for inst in insts:
        if inst.get("label") == args.label and inst.get("status") in ("creating", "loading", "running", "paused"):
            die("sudah ada instance berlabel '%s' (id %s, status %s, ssh %s:%s). "
                "Pakai itu, atau destroy dulu di https://cloud.vast.ai/instances/"
                % (args.label, inst.get("id"), inst.get("status"),
                   inst.get("ssh_host"), inst.get("ssh_port")))

    # ── daftarkan pubkey Mac ke akun Vast (instance baru otomatis dapat key) ─
    ssh_key_path = os.path.expanduser(env.get("SSH_KEY") or "~/.ssh/id_ed25519")
    pub_path = ssh_key_path + ".pub"
    if os.path.exists(pub_path):
        with open(pub_path) as f:
            pub = f.read().strip()
        r = http("POST", "/ssh/", key, {"ssh_key": pub})
        if r.get("success") or "exist" in str(r.get("msg", "")).lower():
            print("\nkey SSH %s terdaftar di akun Vast (instance baru otomatis bisa di-SSH)." % os.path.basename(pub_path))
        else:
            print("PERINGATAN: gagal daftar key ke akun (%s). SSH manual nanti: ./deploy.sh key" % r.get("msg"))
    else:
        print("PERINGATAN: %s tidak ada — ./deploy.sh key nanti setelah instance jadi." % pub_path)

    # ── sewa (coba 5 kandidat; offer rentable bisa berubah status tiap menit) ──
    new_id = None
    for cand in sorted(good, key=lambda x: x["eff"])[:5]:
        body = {
            "client_id": "me",
            "image": image,
            "disk": args.disk,
            "label": args.label,
            "onstart": ONSTART,
            "runtype": "ssh_direc ssh_proxy",
            "cancel_unavail": True,
        }
        r = http("PUT", "/asks/%s/" % cand["id"], key, body)
        if r.get("success"):
            new_id = r.get("new_contract")
            print("\ninstance DIBUAT: id %s (offer %s, %s, $%.3f/jam)"
                  % (new_id, cand["id"], cand["loc"], cand["eff"]))
            break
        print("offer %s gagal (%s) — coba offer berikutnya..." % (cand["id"], r.get("msg")))
    if new_id is None:
        die("semua kandidat gagal diambil orang lain. Ulangi perintah ini (harga offer berubah terus).")

    # ── tunggu SSH siap ─────────────────────────────────────────────────────
    print("menunggu instance siap (tarik image %s ± beberapa menit)..." % image.split(":")[1])
    host = port = None
    seen = False
    misses = 0
    t0 = time.time()
    while time.time() - t0 < 900:
        insts = http("GET", API_V1 + "/instances/", key).get("instances") or []
        inst = next((i for i in insts if i.get("id") == new_id), None)
        if inst:
            seen = True
            misses = 0
            host, port = inst.get("ssh_host"), inst.get("ssh_port")
            st = inst.get("status") or "?"
            if host and port:
                if tcp_ok(host, port):
                    print("SSH siap dalam %d detik: %s:%s (status %s)"
                          % (time.time() - t0, host, port, st))
                    break
                sys.stdout.write("\r  status %-8s ssh %s:%s belum terbuka... " % (st, host, port))
                sys.stdout.flush()
            else:
                sys.stdout.write("\r  status %-8s (menunggu host:port)... " % st)
                sys.stdout.flush()
        else:
            misses += 1
            if seen and misses >= 5:
                die("instance %s menghilang saat menunggu (di-destroy host/gagal jadwal). "
                    "Jalankan ulang perintah ini untuk offer lain." % new_id)
            sys.stdout.write("\r  instance belum muncul di daftar... ")
            sys.stdout.flush()
        time.sleep(8)
    print()

    updates = {"VAST_INSTANCE_ID": str(new_id), "GPU_MODEL": '"%s"' % gpu}
    if host and port:
        updates["SSH_HOST"] = str(host)
        updates["SSH_PORT"] = str(port)
    bak = write_env(env_path, updates)
    print(".env diperbarui (backup: %s):" % os.path.basename(bak))
    for k in ("VAST_INSTANCE_ID", "SSH_HOST", "SSH_PORT", "GPU_MODEL"):
        print("  %s=%s" % (k, updates.get(k, "(belum ada - cek di UI Vast)")))

    print("\nLangkah berikutnya:")
    print("  1. ./deploy.sh check          (cek koneksi/GPU; tunggu 1-2 menit kalau baru saja)")
    print("  2. ./deploy.sh corpus         (25GB, ±1-2 jam)")
    if os.environ.get("RENT_TELL_CHECKPOINT", "") != "1":
        print("  3. taruh checkpoint_latest.zip di folder vast_deploy/ lalu ./deploy.sh checkpoint")
        print("     (atau lewati — training akan auto-recovery dari Google Drive saat start)")
    print("  4. ./deploy.sh source && ./deploy.sh build && ./deploy.sh start")
    print("  5. pantau: ./deploy.sh panel  |  saldo & status: ./deploy.sh vast")
    if host and port and not tcp_ok(host, port):
        print("\nCatatan: host:port belum terbuka saat script selesai — image mungkin masih ditarik. "
              "Ulangi ./deploy.sh check dalam 1-2 menit.")


if __name__ == "__main__":
    main()
