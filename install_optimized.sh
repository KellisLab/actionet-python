#!/usr/bin/env bash
set -euo pipefail

# Optimized (non-portable) build helper.
# - Linux x86_64: native tuning enabled by default.
# - macOS: portable by default; opt into native tuning with --native-macos
#   or ACTIONET_MACOS_NATIVE=1.
# Usage:
#   ./install_optimized.sh [--native-macos] [pip args...]
#   ACTIONET_MACOS_NATIVE=1 ./install_optimized.sh [pip args...]

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

os="$(uname -s)"
arch="$(uname -m)"
runtime="${ACTIONET_OPENMP_RUNTIME:-AUTO}"  # set to INTEL/GNU/LLVM to force, otherwise auto

enable_macos_native="${ACTIONET_MACOS_NATIVE:-0}"
declare -a pip_args=()
for arg in "$@"; do
    if [[ "$arg" == "--native-macos" ]]; then
        enable_macos_native=1
        continue
    fi
    pip_args+=("$arg")
done

run_pip_install() {
    if [[ "${#pip_args[@]}" -eq 0 ]]; then
        python -m pip install .
    else
        python -m pip install "${pip_args[@]}"
    fi
}

if [[ "$os" == "Darwin" ]]; then
    if [[ "$enable_macos_native" != "1" ]]; then
        echo "[INFO] macOS detected; running portable build."
        echo "[INFO] Use --native-macos (or ACTIONET_MACOS_NATIVE=1) to enable non-portable native CPU tuning."
        run_pip_install
        exit 0
    fi

    macos_common_flags="-O3 -ffp-contract=fast -fno-strict-aliasing"
    if [[ "$arch" == "arm64" || "$arch" == "aarch64" ]]; then
        native_cxxflags="-mcpu=native ${macos_common_flags}"
    elif [[ "$arch" == "x86_64" || "$arch" == "amd64" ]]; then
        native_cxxflags="-march=native -mtune=native ${macos_common_flags}"
    else
        echo "[WARN] Unrecognized macOS arch ($arch); using generic optimization flags without native CPU target."
        native_cxxflags="${macos_common_flags}"
    fi

    cmake_args="-DCMAKE_CXX_FLAGS='${native_cxxflags}' -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
    if [[ "$runtime" != "AUTO" ]]; then
        cmake_args+=" -DLIBACTIONET_OPENMP_RUNTIME=${runtime}"
    fi

    export CMAKE_ARGS="${CMAKE_ARGS:-} ${cmake_args}"
    echo "[INFO] Using CMAKE_ARGS=${CMAKE_ARGS}"
    echo "[INFO] Invoking pip install with macOS native optimized flags..."
    run_pip_install
    exit 0
fi

if [[ "$arch" != "x86_64" && "$arch" != "amd64" ]]; then
    echo "[WARN] Non-x86_64 architecture ($arch) detected; skipping native flags. Running portable build."
    run_pip_install
    exit 0
fi

native_cxxflags="-march=native -mtune=native -O3 -ffp-contract=fast -funroll-loops -fomit-frame-pointer -fno-strict-aliasing"

cmake_args="-DCMAKE_CXX_FLAGS='${native_cxxflags}' -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
if [[ "$runtime" != "AUTO" ]]; then
    cmake_args+=" -DLIBACTIONET_OPENMP_RUNTIME=${runtime}"
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} ${cmake_args}"

echo "[INFO] Using CMAKE_ARGS=${CMAKE_ARGS}"
echo "[INFO] Invoking pip install with optimized flags..."
run_pip_install
