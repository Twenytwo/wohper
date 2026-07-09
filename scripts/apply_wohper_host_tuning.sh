#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo bash scripts/apply_wohper_host_tuning.sh" >&2
  exit 1
fi

echo "[wohper] applying runtime sysctl"
sysctl -w vm.swappiness=1

if [ -r /proc/sys/kernel/io_uring_disabled ]; then
  sysctl -w kernel.io_uring_disabled=0
fi

echo "[wohper] writing persistent sysctl config"
cat >/etc/sysctl.d/99-zc-infer.conf <<'EOF'
vm.swappiness=1
kernel.io_uring_disabled=0
EOF

echo "[wohper] writing memlock limits"
cat >/etc/security/limits.d/99-zc-infer.conf <<'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF

echo "[wohper] applied. Log out/in or restart services for PAM memlock limits to affect new sessions."
