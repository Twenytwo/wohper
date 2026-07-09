#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${SUDO_USER:-${USER}}"
MEMLOCK_KB="${MEMLOCK_KB:-unlimited}"

if [ "$(id -u)" -ne 0 ]; then
  SUDO=sudo
else
  SUDO=
fi

echo "[1/5] Installing native build and benchmark dependencies"
${SUDO} apt-get update
${SUDO} apt-get install -y \
  build-essential \
  ca-certificates \
  clang \
  curl \
  fio \
  git \
  liburing-dev \
  linux-libc-dev \
  lld \
  llvm \
  pkg-config \
  python3 \
  python3-pip \
  python3-venv \
  sysstat \
  util-linux

echo "[2/5] Installing Rust toolchain if missing"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  echo "Rust installed. Restart the shell or source ~/.cargo/env before running cargo."
else
  cargo --version
fi

echo "[3/5] Configuring memlock limits for ${USER_NAME}"
LIMITS_FILE="/etc/security/limits.d/99-zc-infer.conf"
${SUDO} tee "${LIMITS_FILE}" >/dev/null <<EOF_LIMITS
# Wohper local inference: allow fixed O_DIRECT/io_uring buffers to be mlocked.
${USER_NAME} soft memlock ${MEMLOCK_KB}
${USER_NAME} hard memlock ${MEMLOCK_KB}
EOF_LIMITS

echo "[4/5] Applying best-effort sysctl tuning"
SYSCTL_FILE="/etc/sysctl.d/99-zc-infer.conf"
${SUDO} tee "${SYSCTL_FILE}" >/dev/null <<'EOF_SYSCTL'
# Keep the OS from eagerly swapping benchmark pages.
vm.swappiness = 1

# Reduce dirty writeback pressure during read-only I/O benchmarks.
vm.dirty_background_ratio = 3
vm.dirty_ratio = 10

# Some hardened environments expose this knob. 0 means io_uring enabled.
kernel.io_uring_disabled = 0
EOF_SYSCTL

${SUDO} sysctl --system >/dev/null || true

echo "[5/5] Environment summary"
uname -a
python3 --version
if command -v cargo >/dev/null 2>&1; then cargo --version; fi
if command -v cc >/dev/null 2>&1; then cc --version | head -1; fi
if command -v fio >/dev/null 2>&1; then fio --version; fi
ulimit -l || true

cat <<EOF_DONE

Setup complete.

Important:
- Log out and back in for /etc/security/limits.d memlock changes to apply to new sessions.
- For systemd services, also set: LimitMEMLOCK=infinity
- For Docker containers, run with: --ulimit memlock=-1:-1 --cap-add IPC_LOCK

Next:
  cd /home/deploy/hermes-agent/loop_workspace
  scripts/linux_bench_zc_core.sh
EOF_DONE
