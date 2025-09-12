#!/bin/bash

# chmod +x setup_plm.sh
# ./setup_plm.sh

# Strict mode for safer scripting
set -euo pipefail

echo "Setting up Python virtual environment for PLM training..."

# Configurable variables (can be overridden via environment)
: "${VENV_DIR:=$HOME/plm_venv}"
: "${PYTORCH_CUDA_URL:=https://download.pytorch.org/whl/cu128}"

# Nuke any existing venv at target path to ensure a clean setup
echo "Removing existing venv at $VENV_DIR (if any)..."
if [ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" = "${VENV_DIR}" ]; then
    deactivate || true
fi
rm -rf "$VENV_DIR"

# Create a fresh virtual environment
python3 -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Update pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install torch and torchvision (CUDA wheel index can be overridden)
echo "Installing torch and torchvision from: $PYTORCH_CUDA_URL"
pip install --force-reinstall torch torchvision --index-url "$PYTORCH_CUDA_URL"

# Install project requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Ensure ninja is available for Triton/Inductor builds
python - <<'PY' >/dev/null 2>&1 || true
import importlib
exit(0 if importlib.util.find_spec('ninja') else 1)
PY
if [ "$?" -ne 0 ]; then
    echo "Installing ninja..."
    pip install --upgrade ninja
fi

# Check for system build deps (Python.h, gcc) and optionally install if permitted
echo "Checking system build dependencies..."
PY_VER=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
PY_INCLUDE_DIR=$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"]) 
PY
)

if ! command -v gcc >/dev/null 2>&1; then
    echo "Warning: gcc is not installed. torch.compile may fail to build extensions."
    echo "Install a compiler toolchain (e.g., build-essential on Debian/Ubuntu)."
fi

if [ ! -f "$PY_INCLUDE_DIR/Python.h" ]; then
    echo "Warning: Python.h not found at: $PY_INCLUDE_DIR"
    echo "torch.compile may fail to build small helper extensions."
    if [ "${INSTALL_SYSTEM_DEPS:-0}" = "1" ]; then
        echo "Attempting to install Python development headers (requires sudo)..."
        if command -v apt-get >/dev/null 2>&1; then
            sudo -n apt-get update || true
            sudo -n apt-get install -y "python${PY_VER}-dev" python3-dev build-essential || true
        elif command -v dnf >/dev/null 2>&1; then
            sudo -n dnf groupinstall -y "Development Tools" || true
            sudo -n dnf install -y python3-devel || true
        elif command -v yum >/dev/null 2>&1; then
            sudo -n yum groupinstall -y "Development Tools" || true
            sudo -n yum install -y python3-devel || true
        elif command -v zypper >/dev/null 2>&1; then
            sudo -n zypper install -y python3-devel gcc gcc-c++ make || true
        elif command -v pacman >/dev/null 2>&1; then
            sudo -n pacman -Sy --noconfirm base-devel python || true
        fi
    else
        echo "To install headers:"
        echo "- Debian/Ubuntu: sudo apt-get install -y python3-dev python${PY_VER}-dev build-essential"
        echo "- Fedora/RHEL:   sudo dnf install -y python3-devel @development-tools"
        echo "- CentOS:        sudo yum install -y python3-devel 'Development Tools'"
        echo "- openSUSE:      sudo zypper install -y python3-devel gcc gcc-c++ make"
        echo "- Arch:          sudo pacman -Sy --noconfirm base-devel python"
        echo "Then re-run this script. You can also set INSTALL_SYSTEM_DEPS=1 to let the script attempt installation."
    fi
fi

# Detect CUDA toolkit (if present) to help dynamic linker
CUDA_HOME=""
if [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
else
    # Pick the highest versioned CUDA directory if multiple exist
    latest_cuda_dir=$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V | tail -n1 || true)
    if [ -n "${latest_cuda_dir}" ] && [ -d "${latest_cuda_dir}" ]; then
        CUDA_HOME="${latest_cuda_dir}"
    fi
fi

# Locate torch's bundled shared libs directory
TORCH_LIB_DIR=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)

# Export runtime library paths for this session
if [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    export CUDA_HOME
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# Persist environment exports inside the venv activate script (idempotent)
ACTIVATE_FILE="$VENV_DIR/bin/activate"
MARKER="# === PLM_SETUP CUDA/Torch dynamic libs ==="

# Remove any previously inserted PLM setup block (handles old end marker too)
if grep -q "$MARKER" "$ACTIVATE_FILE"; then
    awk -v start="$MARKER" -v end1="# ============================================" -v end2="# === END PLM_SETUP ===" '
        BEGIN{skip=0}
        $0 ~ start {skip=1; next}
        skip==1 && ($0 ~ end1 || $0 ~ end2) {skip=0; next}
        skip==0 {print}
    ' "$ACTIVATE_FILE" > "$ACTIVATE_FILE.tmp" && mv "$ACTIVATE_FILE.tmp" "$ACTIVATE_FILE"
fi

# Append a safe, single-line Python invocation version of the block
cat >> "$ACTIVATE_FILE" <<'EOF'
# === PLM_SETUP CUDA/Torch dynamic libs ===
# Add torch's bundled libs to runtime path for torch.compile/triton
export LD_LIBRARY_PATH="$(python -c 'import os, torch, sys; sys.stdout.write(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
# Optionally add CUDA toolkit if present
if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; fi
if [ -z "${CUDA_HOME:-}" ]; then latest=$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V | tail -n1 || true); if [ -n "$latest" ]; then export CUDA_HOME="$latest"; fi; fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"; export PATH="$CUDA_HOME/bin:$PATH"; fi
# === END PLM_SETUP ===
EOF


# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

# Quick diagnostics
echo -e "\nDiagnostics:"
python - <<'PY'
import os, torch, sysconfig
print('torch_version:', torch.__version__)
print('torch_cuda_version:', torch.version.cuda)
print('cuda_is_available:', torch.cuda.is_available())
print('torch_lib_dir:', os.path.join(os.path.dirname(torch.__file__), 'lib'))
inc = sysconfig.get_paths().get('include')
print('python_include_dir:', inc)
print('python_h_exists:', os.path.exists(os.path.join(inc or '', 'Python.h')))
try:
    import triton  # noqa: F401
    print('triton_import: ok')
except Exception as e:
    print('triton_import: fail ->', e)
if torch.cuda.is_available() and inc and os.path.exists(os.path.join(inc, 'Python.h')):
    try:
        f = torch.compile(lambda t: t + 1)
        x = torch.randn(16, device='cuda')
        y = f(x)
        print('torch.compile_smoke: ok (y_cuda:', y.is_cuda, ')')
    except Exception as e:
        print('torch.compile_smoke: fail ->', e)
else:
    reason = []
    if not torch.cuda.is_available():
        reason.append('no CUDA device')
    if not (inc and os.path.exists(os.path.join(inc, 'Python.h'))):
        reason.append('no Python.h')
    print('torch.compile_smoke: skipped (' + ', '.join(reason) + ')')
PY

# Instructions for future use
echo -e "\n======================="
echo "Setup complete!"
echo "======================="
echo "To activate this environment in the future, run:"
echo "    source \"$VENV_DIR/bin/activate\""
echo ""
echo "To deactivate the environment, simply run:"
echo "    deactivate"
echo ""
echo "Your virtual environment is located at: $VENV_DIR"
echo "======================="

