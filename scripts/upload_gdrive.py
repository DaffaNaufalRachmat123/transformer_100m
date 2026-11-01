#!/usr/bin/env python3
"""
Upload checkpoint files to Google Drive via OAuth2.
Works on Kaggle (no mount needed) and Colab.

Usage (called automatically by train_pretrain_cuda):
  python3 scripts/upload_gdrive.py --ckpt_dir /kaggle/working/checkpoints --step 5000 --tokens 192000000

Setup (one-time, run locally):
  1. Go to https://console.cloud.google.com/ → project suberes-414115
  2. APIs & Services → Credentials → Create OAuth Client ID → Desktop App
  3. Download the client_secret JSON
  4. Run: python3 scripts/upload_gdrive.py --auth (follow browser prompt)
  5. This saves a refresh token to gdrive_token.json — include in your dataset
"""
import argparse, json, os, sys, time, zipfile, shutil, tempfile

# ─── HARDCODED CONFIG ─────────────────────────────────────────────────────────
GDRIVE_FOLDER_ID = '1tVYqa2W5lxloMEE6yXQhc_vpj6q6O18Y'
# OAuth2 credentials (from Google Cloud Console → OAuth 2.0 Client ID → Desktop)
OAUTH_CLIENT_ID     = '117015903667205033756'
OAUTH_CLIENT_SECRET = ''  # Fill after creating OAuth Desktop client
TOKEN_FILENAME      = 'gdrive_token.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# ──────────────────────────────────────────────────────────────────────────────

def find_token_file():
    """Find OAuth2 token file in common locations."""
    candidates = [
        TOKEN_FILENAME,
        os.path.join('/kaggle/working/transformer_100m_cuda', TOKEN_FILENAME),
        os.path.join('/kaggle/working', TOKEN_FILENAME),
        os.path.join(os.path.dirname(__file__), '..', TOKEN_FILENAME),
    ]
    # Also search kaggle input datasets
    kaggle_input = '/kaggle/input'
    if os.path.isdir(kaggle_input):
        for ds in os.listdir(kaggle_input):
            candidates.append(os.path.join(kaggle_input, ds, TOKEN_FILENAME))
            candidates.append(os.path.join(kaggle_input, ds, 'transformer_100m_cuda', TOKEN_FILENAME))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def get_credentials():
    """Get OAuth2 credentials from saved token file (refresh token flow)."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_path = find_token_file()
    if token_path is None:
        print(f"[UPLOAD] Token file '{TOKEN_FILENAME}' not found!", file=sys.stderr)
        print(f"[UPLOAD] Run: python3 scripts/upload_gdrive.py --auth", file=sys.stderr)
        return None

    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save refreshed token
        with open(token_path, 'w') as f:
            f.write(creds.to_json())
    return creds

def do_auth_flow():
    """Interactive OAuth2 flow — run this ONCE locally to generate token file."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Install: pip install google-auth-oauthlib")
        sys.exit(1)

    # Look for client_secret file
    client_secret = None
    for name in os.listdir('.'):
        if name.startswith('client_secret') and name.endswith('.json'):
            client_secret = name
            break
    if not client_secret:
        print("ERROR: Download OAuth Desktop client JSON from Google Cloud Console")
        print("       (APIs & Services → Credentials → OAuth 2.0 Client IDs → Download)")
        print("       Place it in this directory as client_secret_*.json")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
    # Force the Google account picker: without prompt=select_account, the
    # browser silently reuses whatever account is already logged in, so a
    # re-auth "for the new Gmail" would silently land on the old one.
    creds = flow.run_local_server(port=0, prompt="select_account")

    token_path = TOKEN_FILENAME
    with open(token_path, 'w') as f:
        f.write(creds.to_json())
    print(f"[AUTH] Token saved to {token_path}")
    print(f"[AUTH] Include this file in your Kaggle dataset!")
    return creds

def upload_file_to_drive(service, filepath, folder_id, filename=None):
    """Upload or update a single file in a Google Drive folder with progress."""
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    fname = filename or os.path.basename(filepath)
    file_size = os.path.getsize(filepath)
    size_mb = file_size / 1e6

    # Use 25MB chunks for large files (fewer API calls, faster upload)
    chunk_size = 25 * 1024 * 1024
    media = MediaFileUpload(filepath, resumable=True, chunksize=chunk_size)

    # Check if file already exists in folder → update instead of duplicate
    try:
        results = service.files().list(
            q=f"name='{fname}' and '{folder_id}' in parents and trashed=false",
            fields="files(id,name)", spaces='drive',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
    except HttpError as e:
        if e.resp.status == 404:
            print(f"[UPLOAD] ERROR: Folder ID '{folder_id}' not accessible!", file=sys.stderr)
            print(f"[UPLOAD] → Share the folder with your service account email (Editor role)", file=sys.stderr)
            raise
        raise

    existing = results.get('files', [])

    if existing:
        file_id = existing[0]['id']
        request = service.files().update(fileId=file_id, media_body=media,
                                         supportsAllDrives=True)
        action = 'updated'
    else:
        meta = {'name': fname, 'parents': [folder_id]}
        request = service.files().create(body=meta, media_body=media, fields='id',
                                         supportsAllDrives=True)
        action = 'created'

    # Execute with progress
    t0 = time.time()
    print(f"[UPLOAD] {fname} ({size_mb:.1f} MB) uploading...", flush=True)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            uploaded_mb = status.resumable_progress / 1e6
            elapsed = time.time() - t0
            speed_mbs = uploaded_mb / elapsed if elapsed > 0 else 0
            eta_s = (size_mb - uploaded_mb) / speed_mbs if speed_mbs > 0 else 0
            bar_filled = pct // 5
            bar = '=' * bar_filled + '>' + '.' * (20 - bar_filled - 1)
            print(f"\r[UPLOAD] {fname} [{bar}] {pct}% | {uploaded_mb:.0f}/{size_mb:.0f} MB | {speed_mbs:.1f} MB/s | ETA {int(eta_s)}s", end='', flush=True)

    total_time = time.time() - t0
    avg_speed = size_mb / total_time if total_time > 0 else 0
    file_id = response.get('id', existing[0]['id'] if existing else '?')
    print(f"\r[UPLOAD] {fname} [====================] 100% | {size_mb:.0f} MB | {avg_speed:.1f} MB/s | {action} in {total_time:.0f}s ✓{'':20}")
    return file_id, action

def upload(ckpt_dir, step, tokens, folder_id=None, use_zip=False, prepare_only=False, stage="pretrain"):
    # ─── File list per stage ─────────────────────────────────────────────────
    _STAGE_STATE_FILE = {
        "pretrain": "train_state.txt",
        "sft":      "train_state_sft.txt",
        "dpo":      "train_state_dpo.txt",
    }
    _STAGE_OPTIM_FILE = {
        "pretrain": "ckpt_optim.safetensors",
        "sft":      "ckpt_sft_optim.safetensors",
        "dpo":      "ckpt_dpo_optim.safetensors",
    }

    state_file = _STAGE_STATE_FILE.get(stage, "train_state.txt")
    optim_file = _STAGE_OPTIM_FILE.get(stage, "ckpt_optim.safetensors")

    # Read last_ckpt from train_state to find the actual weights file name
    # (during training it's ckpt_sft_step200.safetensors, at final it's ckpt_sft_final.safetensors)
    state_path = os.path.join(ckpt_dir, state_file)
    weights_file = f"ckpt_{stage}.safetensors" if stage == "pretrain" else None
    if not weights_file and os.path.isfile(state_path):
        with open(state_path) as f:
            for line in f:
                if line.startswith("last_ckpt="):
                    weights_file = line.split("=", 1)[1].strip()
                    break
    if not weights_file:
        # Fallback: search for latest stage checkpoint
        import glob
        pattern = os.path.join(ckpt_dir, f"ckpt_{stage}_*.safetensors")
        candidates = glob.glob(pattern)
        if candidates:
            weights_file = os.path.basename(max(candidates, key=os.path.getmtime))
        else:
            weights_file = f"ckpt_{stage}_final.safetensors"

    files = [weights_file, optim_file, state_file]
    if stage == "pretrain":
        files = ["ckpt_pretrain.safetensors", "ckpt_optim.safetensors", "train_state.txt"]

    # prepare_only is deprecated (kept for CLI compat, does nothing)
    if prepare_only:
        print("[PREP] prepare-only mode deprecated, skipping")
        return True

    key_path = find_token_file()
    if key_path is None:
        print(f"[UPLOAD] Token file '{TOKEN_FILENAME}' not found — skipping", file=sys.stderr)
        print(f"[UPLOAD] Run: python3 scripts/upload_gdrive.py --auth", file=sys.stderr)
        return False

    # Authenticate via OAuth2 refresh token
    try:
        from googleapiclient.discovery import build
        creds = get_credentials()
        if creds is None:
            return False
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f"[UPLOAD] Auth failed: {e}", file=sys.stderr)
        return False

    if folder_id is None:
        folder_id = GDRIVE_FOLDER_ID

    if use_zip:
        # ─── ZIP mode: snapshot to unique temp dir + compress + upload ────
        tmp_dir = tempfile.mkdtemp(prefix='ckpt_upload_')
        print(f"[UPLOAD] Snapshotting checkpoint...", flush=True)
        t0 = time.time()
        for fname in files:
            src = os.path.join(ckpt_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))
        print(f"[UPLOAD] Snapshot done in {time.time()-t0:.1f}s")

        zip_name = f"ckpt_step{step}_tok{tokens}.zip"
        zip_path = os.path.join(tmp_dir, zip_name)
        total_bytes = sum(os.path.getsize(os.path.join(tmp_dir, f))
                         for f in files if os.path.isfile(os.path.join(tmp_dir, f)))
        total_mb = total_bytes / 1e6
        print(f"[UPLOAD] Compressing {total_mb:.0f} MB...", flush=True)
        t0 = time.time()
        compressed_bytes = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for fname in files:
                fpath = os.path.join(tmp_dir, fname)
                if os.path.isfile(fpath):
                    fsize = os.path.getsize(fpath)
                    zf.write(fpath, fname)
                    compressed_bytes += fsize
                    pct = int(compressed_bytes / total_bytes * 100) if total_bytes > 0 else 100
                    bar_filled = pct // 5
                    bar = '=' * bar_filled + '>' + '.' * (20 - bar_filled - 1)
                    print(f"\r[UPLOAD] Compress [{bar}] {pct}% | {fname} ({fsize/1e6:.0f} MB)", end='', flush=True)
        zip_size = os.path.getsize(zip_path) / 1e6
        print(f"\r[UPLOAD] Compress [====================] 100% | {zip_size:.0f} MB in {time.time()-t0:.0f}s ✓{'':30}")

        # Upload to GDrive — always overwrite same filename
        GDRIVE_ZIP_NAME = 'checkpoint_latest.zip' if stage == 'pretrain' else f'checkpoint_{stage}.zip'
        MAX_RETRIES = 2
        for attempt in range(MAX_RETRIES + 1):
            try:
                fid, action = upload_file_to_drive(service, zip_path, folder_id,
                                                   filename=GDRIVE_ZIP_NAME)
                print(f"[UPLOAD] Done: {GDRIVE_ZIP_NAME} {action} on Google Drive (step={step})")
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    print(f"\n[UPLOAD] Retry {attempt+1}/{MAX_RETRIES} after error: {e}")
                    time.sleep(3)
                else:
                    print(f"\n[UPLOAD] Failed after {MAX_RETRIES+1} attempts: {e}", file=sys.stderr)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return False

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True

    else:
        # ─── Individual file mode ─────────────────────────────────────────
        uploaded = 0
        for fname in files:
            src = os.path.join(ckpt_dir, fname)
            if not os.path.isfile(src):
                continue
            t0 = time.time()
            try:
                fid, action = upload_file_to_drive(service, src, folder_id)
                dt = time.time() - t0
                uploaded += 1
            except Exception as e:
                print(f"[UPLOAD] Failed {fname}: {e}", file=sys.stderr)

        # Upload metadata
        meta_path = os.path.join(ckpt_dir, '_upload_meta.txt')
        with open(meta_path, 'w') as f:
            f.write(f"step={step}\ntokens={tokens}\n")
            f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        try:
            upload_file_to_drive(service, meta_path, folder_id)
        except Exception:
            pass

        print(f"[UPLOAD] Done: {uploaded}/{len(files)} files (step={step}, tok={tokens})")
        return uploaded > 0

def upload_eval_log(path, step, tokens, folder_id=None):
    """Upload the eval report (a small .txt) to Drive.

    Kept completely separate from upload(), whose file list is hardcoded for
    checkpoints. Reuses find_token_file/get_credentials/upload_file_to_drive.

    This is BEST-EFFORT by design: it is invoked from run_eval() during
    training, so it must never raise and never be fatal. Every failure path
    returns False after printing a warning.
    """
    try:
        if not os.path.isfile(path):
            print(f"[EVAL-UPLOAD] File not found: {path} — skipping", file=sys.stderr)
            return False

        key_path = find_token_file()
        if key_path is None:
            print(f"[EVAL-UPLOAD] Token '{TOKEN_FILENAME}' not found — skipping",
                  file=sys.stderr)
            return False

        from googleapiclient.discovery import build
        creds = get_credentials()
        if creds is None:
            print("[EVAL-UPLOAD] No credentials — skipping", file=sys.stderr)
            return False
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        if folder_id is None:
            folder_id = GDRIVE_FOLDER_ID

        # Fixed remote name -> upload_file_to_drive() updates the existing file
        # instead of creating duplicates, so Drive keeps one growing history.
        size_kb = os.path.getsize(path) / 1024.0
        upload_file_to_drive(service, path, folder_id, filename='eval_log.txt')
        print(f"[EVAL-UPLOAD] eval_log.txt ({size_kb:.1f} KB) "
              f"synced (step={step}, tok={tokens})")
        return True

    except Exception as e:
        # Swallow everything: a failed eval upload must not stop training.
        print(f"[EVAL-UPLOAD] Failed: {e} — continuing", file=sys.stderr)
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--auth', action='store_true', help='Run OAuth2 flow (once, locally)')
    p.add_argument('--ckpt_dir', default='.')
    p.add_argument('--step', type=int, default=0)
    p.add_argument('--tokens', type=int, default=0)
    p.add_argument('--folder_id', default=None, help='Override GDRIVE_FOLDER_ID')
    p.add_argument('--zip', action='store_true', help='Compress to zip before upload')
    p.add_argument('--stage', default='pretrain', choices=['pretrain','sft','dpo'],
                   help='Stage tag for file selection [pretrain]')
    p.add_argument('--prepare-only', action='store_true', dest='prepare_only',
                   help='Only snapshot+zip, do not upload')
    p.add_argument('--eval_log', default=None, metavar='PATH',
                   help='Upload an eval report .txt instead of a checkpoint')
    args = p.parse_args()

    if args.auth:
        do_auth_flow()
        sys.exit(0)

    if args.eval_log:
        # Always exit 0: run_eval() treats a non-zero status as a warning, and
        # a failed eval upload is never a reason to interrupt training.
        try:
            upload_eval_log(args.eval_log, args.step, args.tokens, args.folder_id)
        except Exception as e:
            print(f"[EVAL-UPLOAD] Unexpected error: {e}", file=sys.stderr)
        sys.exit(0)

    ok = upload(args.ckpt_dir, args.step, args.tokens, args.folder_id,
                use_zip=args.zip, prepare_only=args.prepare_only, stage=args.stage)
    sys.exit(0 if ok else 1)
