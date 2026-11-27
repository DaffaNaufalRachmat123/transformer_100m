#!/usr/bin/env bash
# =============================================================================
#  aws_launch.sh  -  SATU PERINTAH untuk pre-training di AWS:
#  1. Cek kuota GPU (G/VT) di ap-southeast-2
#  2. Luncurkan / nyalakan instance g6.xlarge (L4 24GB, sm_89) + 100 GB EBS
#  3. Tunggu SSH + CUDA + alat siap (user-data ~2-4 menit)
#  4. Cek token Google Drive masih bisa refresh (untuk upload checkpoint)
#  5. Tulis IP instance ke .env lalu jalankan ./deploy.sh all
#     (check -> package -> source -> korpus 25 GB -> checkpoint 4.5 GB -> build -> start)
#
#  Jalankan dari folder ini:   ./aws_launch.sh
#  Aman diulang: instance 'pretrain-g6' yang sudah ada dipakai lagi (IP di-refresh).
#
#  Syarat:
#    - Kuota "Running On-Demand G and VT instances" >= 4 vCPU sudah disetujui
#      (request L-DB2E81BA sudah diajukan; skrip berhenti dengan pesan jelas
#       kalau belum disetujui - tunggu lalu ulangi).
#    - AWS CLI di Mac masih login (masa aktif 12 jam): aws login --profile default
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

REGION=ap-southeast-2
AMI=ami-0e47a1871d7ef7d7d   # Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) 20261117 - G6 ok
ITYPE=g6.xlarge             # NVIDIA L4 24GB (sm_89)
KEY=daffan-ed25519-mac
SG=sg-02d80c61dd6cb7f8b     # vast-deploy-sg: SSH 22 dari IP Mac saja
SUBNETS="subnet-077463be0b293ccf5 subnet-08147007a9942d8aa subnet-09869d88c48b63d38"  # AZ a, b, c
TAG=pretrain-g6
DISK_GB=100
AWS="aws --region $REGION"

c_cya=$'\033[36m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_red=$'\033[31m'; c_off=$'\033[0m'
say()  { printf '%s>>%s %s\n' "$c_cya" "$c_off" "$*"; }
ok()   { printf '%s OK%s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s !!%s %s\n' "$c_ylw" "$c_off" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

# ─── 1. Kuota GPU ────────────────────────────────────────────────────────────
say "cek kuota G/VT di $REGION ..."
QUOTA=$($AWS service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
        --query 'Quota.Value' --output text 2>/dev/null || echo 0)
if awk "BEGIN{exit !($QUOTA < 4)}"; then
  die "kuota G/VT masih $QUOTA vCPU (butuh >= 4 untuk g6.xlarge).
   Request kenaikan sudah diajukan (status CASE_OPENED, minta 8 vCPU).
   Cek: console > Service Quotas > AWS services > EC2 > Running On-Demand G and VT instances.
   Setelah disetujui, jalankan lagi: ./aws_launch.sh"
fi
ok "kuota G/VT: $QUOTA vCPU"

# ─── 2. Instance: pakai yang ada / luncurkan baru ────────────────────────────
say "cari instance '$TAG' ..."
IID=$($AWS ec2 describe-instances \
        --filters "Name=tag:Name,Values=$TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text | head -1 || true)
if [ -n "$IID" ]; then
  ok "instance sudah ada: $IID"
  STATE=$($AWS ec2 describe-instances --instance-ids "$IID" \
          --query 'Reservations[0].Instances[0].State.Name' --output text)
  if [ "$STATE" = "stopped" ] || [ "$STATE" = "stopping" ]; then
    say "instance dalam kondisi stopped - menyalakan ..."
    $AWS ec2 start-instances --instance-ids "$IID" >/dev/null
  fi
else
  say "meluncurkan $ITYPE (disk $DISK_GB GB gp3, AMI CUDA) ..."
  for SUBNET in $SUBNETS; do
    IID=$($AWS ec2 run-instances --image-id "$AMI" --count 1 --instance-type "$ITYPE" \
          --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$SUBNET" \
          --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
          --user-data file://"$ROOT/aws_userdata.sh" \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
          --query 'Instances[0].InstanceId' --output text 2>/dev/null || true)
    [ -n "$IID" ] && break
    say "AZ $SUBNET tidak punya kapasitas $ITYPE - coba AZ berikutnya ..."
  done
  [ -n "$IID" ] || die "gagal meluncurkan instance. Cek manual:
   $AWS ec2 run-instances ... (error kuota/kapasitas/koneksi?)"
  ok "instance diluncurkan: $IID"
fi

say "menunggu status 'running' (boot 1-2 menit) ..."
$AWS ec2 wait instance-running --instance-ids "$IID"

say "menunggu IP publik ..."
IP=""
for i in $(seq 1 20); do
  IP=$($AWS ec2 describe-instances --instance-ids "$IID" \
       --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  { [ -n "$IP" ] && [ "$IP" != "None" ]; } && break
  sleep 5
done
{ [ -n "$IP" ] && [ "$IP" != "None" ]; } || die "IP publik tidak muncul - cek console EC2"
ok "IP publik: $IP"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new
          -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519")

say "menunggu SSH root + CUDA + alat siap (user-data ~2-4 menit; log: /var/log/aws_prep.log) ..."
READY=""
for i in $(seq 1 90); do
  if ssh "${SSH_OPTS[@]}" root@"$IP" 'test -f /workspace/.ready' 2>/dev/null; then READY=1; break; fi
  sleep 5
done
[ -n "$READY" ] || die "instance belum siap setelah ~7 menit. Diagnosis manual:
   ssh ${SSH_OPTS[*]} ubuntu@$IP 'tail -50 /var/log/aws_prep.log'"
ok "instance siap (SSH root, CUDA, nvcc, alat, library Drive)"

# ─── 3. Pre-flight token Google Drive (refresh) ─────────────────────────────
say "cek token Google Drive bisa di-refresh (untuk upload checkpoint) ..."
if python3 - <<'PYEOF' 2>/dev/null
import json, urllib.request, urllib.parse
tok = json.load(open('../gdrive_token.json'))
cs = json.load(open('../client_secret_658498396853-ptrpq91qrncr01b1iqpt2j018175qoil.apps.googleusercontent.com.json'))['installed']
data = urllib.parse.urlencode({'client_id': cs['client_id'], 'client_secret': cs['client_secret'],
                               'refresh_token': tok['refresh_token'], 'grant_type': 'refresh_token'}).encode()
urllib.request.urlopen(urllib.request.Request('https://oauth2.googleapis.com/token', data=data), timeout=20)
PYEOF
then
  ok "token Google Drive valid (auto-refresh jalan)"
else
  warn "refresh token Drive GAGAL. Upload checkpoint ke Drive TIDAK akan jalan sampai:
   1. cd .. && python3 scripts/upload_gdrive.py --auth   (ikuti browser, pilih akun)
   2. ulangi ./aws_launch.sh
   (training tetap jalan; hanya upload Drive yang mati)"
fi

# ─── 4. Tulis .env ───────────────────────────────────────────────────────────
say "perbarui .env: SSH_HOST=$IP SSH_PORT=22 SSH_USER=root ..."
sed -i '' -e "s|^SSH_HOST=.*|SSH_HOST=$IP|" \
          -e "s|^SSH_PORT=.*|SSH_PORT=22|" \
          -e "s|^SSH_USER=.*|SSH_USER=root|" .env
ok ".env siap"

# ─── 5. Jalankan semuanya ────────────────────────────────────────────────────
say "GO: ./deploy.sh all"
say "  check -> package -> source -> korpus 25 GB -> checkpoint 4,5 GB -> build sm_89 -> start"
say "  (korpus + checkpoint pakai rsync/4-stream paralel, resume otomatis kalau putus)"
exec ./deploy.sh all
