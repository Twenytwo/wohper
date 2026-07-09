# Linux Environment Setup

This is the final pre-deploy checklist for the Wohper Core I/O benchmark.

## System Packages

Install native compiler, Rust support dependencies, `liburing-dev`, Python, and benchmark tools:

```bash
sudo apt-get update
sudo apt-get install -y \
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
```

Install Rust if missing:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

## memlock

The engine uses fixed aligned buffers and calls `mlock` best-effort. For serious tests, raise memlock:

```bash
sudo tee /etc/security/limits.d/99-zc-infer.conf >/dev/null <<'EOF'
deploy soft memlock unlimited
deploy hard memlock unlimited
EOF
```

Then log out and back in.

Check:

```bash
ulimit -l
```

For a systemd service:

```ini
LimitMEMLOCK=infinity
```

For Docker:

```bash
docker run --ulimit memlock=-1:-1 --cap-add IPC_LOCK ...
```

## sysctl

Best-effort benchmark tuning:

```bash
sudo tee /etc/sysctl.d/99-zc-infer.conf >/dev/null <<'EOF'
vm.swappiness = 1
vm.dirty_background_ratio = 3
vm.dirty_ratio = 10
kernel.io_uring_disabled = 0
EOF

sudo sysctl --system
```

## Automated Setup

Run:

```bash
scripts/setup_linux_zc_core_env.sh
```

## Docker Dev Container

Optional:

```bash
docker build -f engine/zc_infer_core/Dockerfile.dev -t zc-infer-dev .
docker run --rm -it --ulimit memlock=-1:-1 --cap-add IPC_LOCK -v "$PWD:/workspace" zc-infer-dev bash
```

Docker is useful for compile checks, but bare-metal Linux is preferred for real NVMe/io_uring benchmarks.
