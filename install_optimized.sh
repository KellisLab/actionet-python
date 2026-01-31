#!/usr/bin/env bash
set -euo pipefail

# Optimized (non-portable) build for local x86_64 Linux hosts.
# Uses native CPU tuning, IPO/LTO, and optional OpenMP runtime override.
# Usage: ./install_optimized.sh [pip args...]

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

os="$(uname -s)"
arch="$(uname -m)"

if [[ "$os" == "Darwin" ]]; then
    echo "[INFO] macOS detected; the default portable build is already optimized for Apple targets."
    echo "[INFO] Run 'pip install .' directly or set CMAKE_ARGS manually if you want to experiment."
    python -m pip install "$@"
    exit 0
fi

if [[ "$arch" != "x86_64" && "$arch" != "amd64" ]]; then
    echo "[WARN] Non-x86_64 architecture ($arch) detected; skipping native flags. Running portable build."
    python -m pip install "$@"
    exit 0
fi

native_cxxflags="-march=native -mtune=native -O3 -ffp-contract=fast -funroll-loops -fomit-frame-pointer -fno-strict-aliasing"
runtime="${ACTIONET_OPENMP_RUNTIME:-AUTO}"  # set to INTEL/GNU/LLVM to force, otherwise auto

cmake_args="-DCMAKE_CXX_FLAGS='${native_cxxflags}' -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
if [[ "$runtime" != "AUTO" ]]; then
    cmake_args+=" -DLIBACTIONET_OPENMP_RUNTIME=${runtime}"
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} ${cmake_args}"

echo "[INFO] Using CMAKE_ARGS=${CMAKE_ARGS}"
echo "[INFO] Invoking pip install with optimized flags..."
python -m pip install "$@"
