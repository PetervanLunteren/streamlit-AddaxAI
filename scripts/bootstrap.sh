#!/usr/bin/env bash
set -euo pipefail

# Tested on Fedora Linux
# Not tested on macOS

echo "Installing AddaxAI dependencies and configuring local environment"

OS="$(uname | tr '[:upper:]' '[:lower:]')"
ARCH=$(uname -m)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case "$OS" in
    linux)
        case "$ARCH" in
            x86_64) MICROMAMBA_OS_ARCH="linux-64" ;;
            aarch64|arm64) MICROMAMBA_OS_ARCH="linux-aarch64" ;;
            ppc64le) MICROMAMBA_OS_ARCH="linux-ppc64le" ;;
            *) echo "Unsupported Linux architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    darwin)
        case "$ARCH" in
            x86_64) MICROMAMBA_OS_ARCH="osx-64" ;;
            arm64) MICROMAMBA_OS_ARCH="osx-arm64" ;;
            *) echo "Unsupported macOS architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS"; exit 1
        ;;
esac

BIN_DIR="$PROJECT_ROOT/bin/$OS"
mkdir -p $BIN_DIR

MICROMAMBA_BIN="$BIN_DIR/micromamba"

# This is the official way to install micromamba
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
if [[ ! -f "$MICROMAMBA_BIN" ]]; then
    echo "Downloading micromamba..."
    curl -Ls "https://micro.mamba.pm/api/micromamba/$MICROMAMBA_OS_ARCH/latest" | tar -xvj -C "$BIN_DIR" --strip-components=1 bin/micromamba
else
    echo "Micromamba already installed at $MICROMAMBA_BIN"
fi

# Setting up python env may exceed /tmp on Linux, so use a local temp folder
TMP_LOCAL="$PROJECT_ROOT/tmp"
mkdir -p $TMP_LOCAL
export TMPDIR=$TMP_LOCAL
export PIP_CACHE_DIR="$TMP_LOCAL/pip_cache"

echo "Creating Python environment..."
ENV_PREFIX="$PROJECT_ROOT/envs/env-addaxai-base"
$MICROMAMBA_BIN env create -f "$PROJECT_ROOT/envs/ymls/addaxai-base/macos/environment.yml" --prefix $ENV_PREFIX -y

SPECIESNET="speciesnet==5.0.2"

echo "Installing $SPECIESNET into $ENV_PREFIX..."

if [[ "$OS" == "darwin" ]]; then
    "$MICROMAMBA_BIN" run -p "$ENV_PREFIX" pip install --use-pep517 "$SPECIESNET"
else
    "$MICROMAMBA_BIN" run -p "$ENV_PREFIX" pip install "$SPECIESNET"
fi

echo "Done. You can now run AddaxAI with the following command:"
echo "$MICROMAMBA_BIN run -p $ENV_PREFIX streamlit run main.py"
