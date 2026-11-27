#!/bin/bash
# =============================================================================
#  aws_userdata.sh - bootstrap instance AWS saat pertama kali menyala.
#  Dijalankan AWS sebagai root secara otomatis (cloud-init).
#  Log: /var/log/aws_prep.log  |  Marker selesai: /workspace/.ready
# =============================================================================
exec >> /var/log/aws_prep.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive

# 1) Alat yang dibutuhkan deploy.sh / remote.sh (unzip rsync tmux cmake g++ make pip)
apt-get update -qq
apt-get install -y -qq unzip rsync tmux cmake build-essential python3-pip htop

# 2) /workspace seperti di Vast.ai
mkdir -p /workspace

# 3) Akses SSH root dengan key yang sama seperti user ubuntu
#    (pipeline deploy.sh/remote.sh berjalan sebagai root, sama seperti di Vast.ai:
#     apt-get tanpa sudo, tmux, kepemilikan /workspace)
install -d -m 700 /root/.ssh
cp /home/ubuntu/.ssh/authorized_keys /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
printf 'PermitRootLogin prohibit-password\n' > /etc/ssh/sshd_config.d/99-root-key.conf
systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true

# 4) nvcc harus terlihat di PATH default (remote.sh cek: command -v nvcc).
#    DLAMI menaruh nvcc di /usr/local/cuda/bin yang TIDAK masuk PATH shell non-login SSH.
for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
  if [ -x "$c" ]; then ln -sf "$c" /usr/local/bin/nvcc; break; fi
done
# symlink /usr/local/cuda untuk deteksi CMake (find_package(CUDAToolkit))
if [ ! -e /usr/local/cuda ]; then
  latest=$(ls -d /usr/local/cuda-* 2>/dev/null | grep -v '/bin' | sort -V | tail -1)
  [ -n "$latest" ] && ln -sfn "$latest" /usr/local/cuda
fi

# 5) Library Google Drive untuk upload checkpoint dari instance
pip3 install -q --break-system-packages google-api-python-client google-auth-oauthlib \
  || pip3 install -q google-api-python-client google-auth-oauthlib || true

# 6) Selesai
touch /workspace/.ready
