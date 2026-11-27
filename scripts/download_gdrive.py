#!/usr/bin/env python3
"""
download_gdrive.py - pulihkan checkpoint dari Google Drive (untuk resume).

    python3 scripts/download_gdrive.py --ckpt_dir /workspace/checkpoints

Dipakai otomatis oleh remote.sh (mode interruptible Vast) saat instance
dinyalakan ulang dan checkpoint lokal tidak ada / hilang: zip
`checkpoint_latest.zip` (atau `checkpoint_{stage}.zip`) diunduh dari folder
Drive yang sama dengan upload_gdrive.py, diverifikasi, diekstrak ke --ckpt_dir,
lalu training melanjutkan dari step terakhir.

Menggunakan token/kredensial yang sama dengan upload_gdrive.py, jadi tidak
perlu setup tambahan. Folder bisa dioverride lewat env GDRIVE_FOLDER_ID atau
--folder_id (default: konstanta GDRIVE_FOLDER_ID di upload_gdrive.py, yang
sudah di-patch oleh tools/patch_source.py sesuai .env saat packaging).
"""
import argparse, os, shutil, sys, time, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FILES = {
    "pretrain": ["ckpt_pretrain.safetensors", "ckpt_optim.safetensors", "train_state.txt"],
}


def die(msg):
    print(f"[DOWNLOAD] GAGAL: {msg}", file=sys.stderr)
    sys.exit(1)


def connect():
    try:
        import upload_gdrive
    except ImportError:
        die("upload_gdrive.py tidak ditemukan di folder yang sama")
    token = upload_gdrive.find_token_file()
    if token is None:
        die("token Google Drive (gdrive_token.json) tidak ditemukan")
    creds = upload_gdrive.get_credentials()
    if creds is None:
        die("kredensial Google Drive tidak valid / gagal refresh")
    from googleapiclient.discovery import build
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return upload_gdrive, service


def find_remote_zip(service, folder_id, zip_name):
    q = f"name='{zip_name}' and '{folder_id}' in parents and trashed=false"
    res = service.files().list(q=q, fields="files(id,name,size,modifiedTime)",
                               spaces="drive", supportsAllDrives=True,
                               includeItemsFromAllDrives=True).execute()
    files = res.get("files", [])
    if not files:
        return None
    return files[0]


def download(service, file_id, zip_path):
    from googleapiclient.http import MediaIoBaseDownload
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    size_mb = float("nan")
    t0 = time.time()
    with open(zip_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=32 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                got_mb = status.resumable_progress / 1e6
                if status.total_size:
                    size_mb = status.total_size / 1e6
                    spd = got_mb / max(time.time() - t0, 1e-6)
                    bar = "=" * (pct // 5) + ">" + "." * (19 - pct // 5)
                    print(f"\r[DOWNLOAD] [{bar}] {pct}% | {got_mb:.0f}/{size_mb:.0f} MB | {spd:.1f} MB/s",
                          end="", flush=True)
    print(f"\r[DOWNLOAD] selesai: {size_mb:.0f} MB dalam {time.time()-t0:.0f}s{' '*30}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt_dir", default=None, help="tujuan ekstrak (mis. /workspace/checkpoints)")
    ap.add_argument("--out", default=None, help="simpan zip ke path ini TANPA mengekstrak (mis. ambil checkpoint_latest.zip untuk diunggah manual)")
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "sft", "dpo"])
    ap.add_argument("--folder_id", default=None, help="override folder Drive (default: env GDRIVE_FOLDER_ID / upload_gdrive)")
    ap.add_argument("--tmp_dir", default=None, help="lokasi zip sementara (default: env CKPT_UPLOAD_TMP / sebelah ckpt_dir)")
    ap.add_argument("--no-verify", action="store_true", help="lewati pemeriksaan CRC zip (lebih cepat, kurang aman)")
    args = ap.parse_args()
    if not args.out and not args.ckpt_dir:
        die("--ckpt_dir wajib (atau pakai --out untuk menyimpan zip saja)")

    upload_gdrive, service = connect()
    folder_id = args.folder_id or os.environ.get("GDRIVE_FOLDER_ID") or upload_gdrive.GDRIVE_FOLDER_ID

    zip_name = "checkpoint_latest.zip" if args.stage == "pretrain" else f"checkpoint_{args.stage}.zip"
    print(f"[DOWNLOAD] mencari '{zip_name}' di folder Drive {folder_id} ...", flush=True)
    info = find_remote_zip(service, folder_id, zip_name)
    if info is None:
        die(f"'{zip_name}' tidak ada di folder Drive (belum pernah diupload?)")

    size_mb = int(info.get("size") or 0) / 1e6
    print(f"[DOWNLOAD] ditemukan: {info['name']} ({size_mb:.0f} MB, diubah {info.get('modifiedTime', '?')})", flush=True)

    tmp_dir = args.tmp_dir or os.environ.get("CKPT_UPLOAD_TMP") or os.path.dirname(os.path.abspath(args.ckpt_dir or args.out or "."))
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, ".recover_" + zip_name)

    download(service, info["id"], zip_path)

    if not args.no_verify:
        print("[DOWNLOAD] verifikasi CRC zip ...", flush=True)
        t0 = time.time()
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
        if bad is not None:
            os.remove(zip_path)
            die(f"CRC rusak di dalam zip: {bad} (unduh ulang)")
        print(f"[DOWNLOAD] CRC OK ({time.time()-t0:.0f}s)", flush=True)

    if args.out:
        # Mode arsip: simpan zip apa adanya (untuk diunggah manual / dipindah),
        # tanpa mengekstrak. crc sudah diverifikasi di atas.
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.move(zip_path, out)
        print(f"[DOWNLOAD] zip disimpan: {out} (tidak diekstrak)")
        print("[DOWNLOAD] OK - siap dipakai, mis. ditaruh di vast_deploy/ untuk ./deploy.sh checkpoint")
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"[DOWNLOAD] ekstrak ke {args.ckpt_dir} ...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.ckpt_dir)
    os.remove(zip_path)
    missing = [f for f in REQUIRED_FILES.get(args.stage, []) if not os.path.isfile(os.path.join(args.ckpt_dir, f))]
    if missing:
        die("isi zip tidak lengkap, hilang: " + ", ".join(missing))

    state = os.path.join(args.ckpt_dir, REQUIRED_FILES["pretrain"][2])
    if os.path.isfile(state):
        with open(state) as fh:
            head = [next(fh, "").strip() for _ in range(2)]
        print("[DOWNLOAD] checkpoint dipulihkan: " + " | ".join(head))
    print("[DOWNLOAD] OK - training bisa resume dari checkpoint ini")


if __name__ == "__main__":
    main()
