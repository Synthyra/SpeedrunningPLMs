#!/bin/bash

# chmod +x setup_plm.sh
# ./setup_plm.sh

# Strict mode for safer scripting
set -euo pipefail

echo "Setting up Python virtual environment for PLM training..."

# Configurable variables (can be overridden via environment)
: "${VENV_DIR:=$HOME/plm_venv}"
: "${PYTORCH_CUDA_URL:=https://download.pytorch.org/whl/cu128}"

# Create virtual environment if missing
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

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
if ! grep -q "$MARKER" "$ACTIVATE_FILE"; then
    {
        echo "$MARKER"
        echo "# Add torch's bundled libs to runtime path for torch.compile/triton"
        echo "export LD_LIBRARY_PATH=\"\$(python - <<'PY'\nimport os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\nPY\n):\${LD_LIBRARY_PATH:-}\""
        echo "# Optionally add CUDA toolkit if present"
        echo "if [ -d /usr/local/cuda ]; then export CUDA_HOME=/usr/local/cuda; fi"
        echo "if [ -z \"\${CUDA_HOME:-}\" ]; then latest=\$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V | tail -n1 || true); if [ -n \"\$latest\" ]; then export CUDA_HOME=\"\$latest\"; fi; fi"
        echo "if [ -n \"\${CUDA_HOME:-}\" ] && [ -d \"\$CUDA_HOME/lib64\" ]; then export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\"; export PATH=\"\$CUDA_HOME/bin:\$PATH\"; fi"
        echo "# ============================================"
    } >> "$ACTIVATE_FILE"
fi

# Quick diagnostics
echo -e "\nDiagnostics:"
python - <<'PY'
import os, torch
print('torch_version:', torch.__version__)
print('torch_cuda_version:', torch.version.cuda)
print('cuda_is_available:', torch.cuda.is_available())
print('torch_lib_dir:', os.path.join(os.path.dirname(torch.__file__), 'lib'))
try:
    import triton  # noqa: F401
    print('triton_import: ok')
except Exception as e:
    print('triton_import: fail ->', e)
if torch.cuda.is_available():
    try:
        f = torch.compile(lambda t: t + 1)
        x = torch.randn(16, device='cuda')
        y = f(x)
        print('torch.compile_smoke: ok (y_cuda:', y.is_cuda, ')')
    except Exception as e:
        print('torch.compile_smoke: fail ->', e)
else:
    print('torch.compile_smoke: skipped (no CUDA device)')
PY

# List installed packages for verification
echo -e "\nInstalled packages:"
pip list

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

