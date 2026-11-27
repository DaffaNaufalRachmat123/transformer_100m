#!/usr/bin/env python3
"""
patch_source.py - bangun source_code.zip untuk dikirim ke instance GPU.

Mengambil kode yang dibutuhkan dari folder proyek (BUKAN korpus/checkpoint/dataset
besar), menambahkan credential Google Drive, dan menerapkan patch kecil pada SALINAN
di dalam zip. File asli di proyek Anda tidak diubah.

Patch:
  1. train_cuda.cu    : porsi VRAM arena (ARENA_FRACTION)
  2. upload_gdrive.py : snapshot checkpoint pakai HARD LINK (tanpa salin 4.9 GB, dan
                        pasangan bobot+optimizer tetap konsisten walau trainer menyimpan
                        checkpoint baru saat zip sedang dibuat), zip ditulis ke disk
  3. upload_gdrive.py : GDRIVE_FOLDER_ID dari .env (kalau diisi)
"""
import argparse, glob, os, sys, time, zipfile

REQUIRED_DIRS = ["bpe", "cuda", "scripts", "transformer"]
OPTIONAL_DIRS = ["tools", "infer_cpp", "eval_data", "rl_data/ceiling_enja"]
EXCLUDE_NAMES = {".DS_Store", "__pycache__", ".git", ".vscode"}
EXCLUDE_SUFFIX = (".pyc", ".o", ".d", ".safetensors", ".zip")


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sub_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        die(f"patch '{label}': teks target ditemukan {n}x (harus 1x). Kode sumber berubah? "
            f"Perbarui tools/patch_source.py.")
    return text.replace(old, new)


# Banner trainer asli menulis "sm_75 / T4" dan "T4 x2" secara permanen (hanya tampilan, bukan logika).
# Diganti dengan nama + arsitektur GPU yang sebenarnya, supaya log live tidak menyesatkan.
BANNER_OLD_TITLE = (
    'printf("\u2551  Transformer \u2014 CUDA Pre-training (sm_75 / T4)           \u2551\\n");'
)
BANNER_NEW_TITLE = (
    '{ cudaDeviceProp _bp; cudaGetDeviceProperties(&_bp, 0); '
    'printf("\u2551  Transformer \u2014 CUDA Pre-training (sm_%d%d / %-24.24s)\u2551\\n", _bp.major, _bp.minor, _bp.name); }'
)
BANNER_OLD_GPU = (
    'printf("\u2551  GPU: %s | FP16 Mixed Precision | AdamW       \u2551\\n",\n'
    '           g_use_ddp ? "T4 x2 (DDP)" : "Single T4   ");'
)
BANNER_NEW_GPU = (
    'printf("\u2551  GPU: %s x%d | FP16 Mixed Precision | AdamW       \u2551\\n",\n'
    '           g_use_ddp ? "DDP   " : "Single", g_use_ddp ? g_num_gpus : 1);'
)


def patch_train_cuda(text, arena):
    notes = []
    if abs(arena - 0.75) > 1e-9:
        text = sub_once(text, "cuda::arena.init_auto(0.75f);", f"cuda::arena.init_auto({arena:g}f);", "arena")
        notes.append(f"arena VRAM 0.75 -> {arena:g}")
    else:
        notes.append("arena tetap 0.75 (tidak diubah)")
    text = sub_once(text, BANNER_OLD_TITLE, BANNER_NEW_TITLE, "banner-title")
    text = sub_once(text, BANNER_OLD_GPU, BANNER_NEW_GPU, "banner-gpu")
    notes.append("banner trainer: nama GPU/arsitektur asli (bukan 'T4' tetap)")
    return text, notes


SNAPSHOT_FN = '''def _snapshot(src, dst):
    """Hard link (nol byte tambahan, tahan terhadap rename atomik trainer); salin kalau beda filesystem."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

'''


GUARD_FN = '''    if use_zip:
        # ─── GUARD anti-tumpuk + pembersih sisa crash ─────────────────────
        # Trainer meluncurkan upload tiap N langkah tanpa menunggu upload
        # sebelumnya selesai. Upload yang melambat (pernah 1.3 MB/s) membuat
        # dua zip 4.5 GB bersamaan memenuhi disk -> kompresi gagal -> crash
        # -> snap & zip bocor -> /dev/shm ikut penuh -> penyimpanan checkpoint
        # gagal (insiden 03:15 & 09:35 UTC). Guard: buang sisa crash berumur
        # > 75 menit, lewati siklus kalau upload lain masih berjalan, dan
        # pastikan disk cukup sebelum mulai.
        import glob as _glob
        _zipbase = os.environ.get('CKPT_UPLOAD_TMP') or tempfile.gettempdir()
        _now = time.time()
        for _d in (_glob.glob(os.path.join(_zipbase, 'ckpt_zip_*')) +
                   _glob.glob(os.path.join(ckpt_dir, '.ckpt_snap_*'))):
            try:
                if _now - os.path.getmtime(_d) > 4500:
                    shutil.rmtree(_d, ignore_errors=True)
                    print("[UPLOAD] buang sisa crash: %s" % _d, flush=True)
            except OSError:
                pass
        if _glob.glob(os.path.join(_zipbase, 'ckpt_zip_*')):
            print("[UPLOAD] SKIP: upload lain masih berjalan", flush=True)
            return True
        if shutil.disk_usage(_zipbase).free < 6e9:
            print("[UPLOAD] SKIP: disk kurang (%.1f GB bebas)" % (shutil.disk_usage(_zipbase).free / 1e9), flush=True)
            return True

'''


def patch_upload(text, folder_id):
    notes = []
    text = sub_once(text, "def upload(ckpt_dir, step, tokens,", SNAPSHOT_FN + "def upload(ckpt_dir, step, tokens,", "snapshot-fn")
    text = sub_once(text, "    if use_zip:\n", GUARD_FN, "upload-guard")
    notes.append("upload guard: anti-tumpuk + bersih-bersih sisa crash")
    text = sub_once(
        text,
        "tmp_dir = tempfile.mkdtemp(prefix='ckpt_upload_')",
        "tmp_dir = tempfile.mkdtemp(prefix='.ckpt_snap_', dir=ckpt_dir)  # hard link, filesystem sama dgn checkpoint\n"
        "        out_dir = tempfile.mkdtemp(prefix='ckpt_zip_', dir=os.environ.get('CKPT_UPLOAD_TMP') or None)  # zip di disk",
        "tmp-dir")
    text = sub_once(text, "shutil.copy2(src, os.path.join(tmp_dir, fname))",
                    "_snapshot(src, os.path.join(tmp_dir, fname))", "snapshot-call")
    text = sub_once(text, "zip_path = os.path.join(tmp_dir, zip_name)",
                    "zip_path = os.path.join(out_dir, zip_name)", "zip-path")
    old = "shutil.rmtree(tmp_dir, ignore_errors=True)"
    if text.count(old) != 2:
        die(f"patch 'rmtree': ditemukan {text.count(old)}x (harus 2x)")
    text = text.replace(old, "shutil.rmtree(tmp_dir, ignore_errors=True); shutil.rmtree(out_dir, ignore_errors=True)")
    # Lepas snapshot SEGERA setelah zip jadi (bukan setelah upload selesai):
    # hard-link menahan inode checkpoint LAMA tetap hidup selama trainer me-rename
    # file baru. Dua snapshot basi + tmp pernah memenuhi /dev/shm dan menggagalkan
    # penyimpanan checkpoint (insiden 03:15 UTC, run gagal di step 471.200).
    # Ditempatkan SETELAH blok rmtree di atas supaya pemeriksaan jumlahnya tidak terganggu.
    text = sub_once(text, "\n        # Upload to GDrive — always overwrite same filename",
                    "\n        shutil.rmtree(tmp_dir, ignore_errors=True)  # lepas snapshot segera (inode lama bebas)\n\n        # Upload to GDrive — always overwrite same filename",
                    "snap-release")
    notes.append("upload Drive: snapshot hard-link + zip di disk")
    notes.append("snapshot dilepas segera setelah zip jadi (cegah /dev/shm penuh)")
    if folder_id:
        old_id = "GDRIVE_FOLDER_ID = '1tVYqa2W5lxloMEE6yXQhc_vpj6q6O18Y'"
        text = sub_once(text, old_id, f"GDRIVE_FOLDER_ID = '{folder_id}'", "folder-id")
        notes.append(f"GDRIVE_FOLDER_ID -> {folder_id}")
    return text, notes


def iter_tree(root, rel):
    base = os.path.join(root, rel)
    for dp, dns, fns in os.walk(base):
        dns[:] = sorted(d for d in dns if d not in EXCLUDE_NAMES)
        for fn in sorted(fns):
            if fn in EXCLUDE_NAMES or fn.endswith(EXCLUDE_SUFFIX):
                continue
            full = os.path.join(dp, fn)
            if os.path.islink(full) and not os.path.exists(full):
                continue
            yield full, os.path.relpath(full, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder proyek")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-name", required=True)
    ap.add_argument("--vocab-dir", default="vocab_9")
    ap.add_argument("--bpe", default="bpe_v9.bin")
    ap.add_argument("--arena", type=float, default=0.75)
    ap.add_argument("--folder-id", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--client-secret", default="")
    ap.add_argument("--service-account", default="")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    top = a.project_name
    entries = []  # (full, arcname)

    for d in REQUIRED_DIRS + [a.vocab_dir]:
        if not os.path.isdir(os.path.join(root, d)):
            die(f"folder wajib tidak ada: {d}")
    for d in REQUIRED_DIRS + OPTIONAL_DIRS + [a.vocab_dir]:
        if os.path.isdir(os.path.join(root, d)):
            entries += [(f, r) for f, r in iter_tree(root, d)]
        else:
            print(f"  (lewati folder opsional yang tidak ada: {d})")
    for f in ["CMakeLists.txt", "config.ini", a.bpe] + sorted(glob.glob(os.path.join(root, "config_*.json"))):
        full = f if os.path.isabs(f) else os.path.join(root, f)
        if os.path.isfile(full):
            entries.append((full, os.path.relpath(full, root)))
        elif f in ("CMakeLists.txt", a.bpe):
            die(f"file wajib tidak ada: {f}")

    # Credential Google Drive (nama file tujuan ditentukan kode upload_gdrive.py)
    creds = []
    for path, arc, label in [(a.token, "gdrive_token.json", "token OAuth"),
                             (a.client_secret, None, "client secret"),
                             (a.service_account, None, "service account")]:
        if not path:
            continue
        p = os.path.abspath(path)
        if os.path.isfile(p):
            creds.append((p, arc or os.path.basename(p)))
        else:
            print(f"  PERINGATAN: {label} tidak ditemukan: {path}  (upload ke Drive bisa gagal)", file=sys.stderr)
    if not any(arc == "gdrive_token.json" for _, arc in creds):
        print("  PERINGATAN: gdrive_token.json tidak masuk zip -> upload checkpoint ke Drive tidak akan jalan", file=sys.stderr)

    patched = {"transformer/train_cuda.cu": lambda t: patch_train_cuda(t, a.arena),
               "scripts/upload_gdrive.py": lambda t: patch_upload(t, a.folder_id)}

    out = os.path.abspath(a.out)
    tmp = out + ".part"
    n_files, applied = 0, []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        seen = set()
        for full, rel in entries:
            rel = rel.replace(os.sep, "/")
            if rel in seen:
                continue
            seen.add(rel)
            arc = f"{top}/{rel}"
            if rel in patched:
                text, note = patched[rel](open(full, encoding="utf-8").read())
                applied += [note] if isinstance(note, str) else note
                zi = zipfile.ZipInfo(arc, time.localtime(os.path.getmtime(full))[:6])
                zi.external_attr = (os.stat(full).st_mode & 0o777) << 16
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, text.encode("utf-8"))
            else:
                zf.write(full, arc)
            n_files += 1
        for full, arc in creds:
            zf.write(full, f"{top}/{arc}")
            n_files += 1
    os.replace(tmp, out)
    print(f"  {n_files} file -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    for n in applied:
        print(f"  patch: {n}")
    print("  credential: " + (", ".join(arc for _, arc in creds) if creds else "(tidak ada)"))


if __name__ == "__main__":
    main()
