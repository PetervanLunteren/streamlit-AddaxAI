#!/usr/bin/env bash
set -euo pipefail

echo "Tearing down AddaxAI local environment..."

OS="$(uname | tr '[:upper:]' '[:lower:]')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BIN_DIR="$PROJECT_ROOT/bin/$OS"
MICROMAMBA_BIN="$BIN_DIR/micromamba"
ENV_PREFIX="$PROJECT_ROOT/envs/env-addaxai-base"
TMP_LOCAL="$PROJECT_ROOT/tmp"

echo "Removing micromamba binary (if any)..."
rm -f "$MICROMAMBA_BIN"

echo "Removing environment..."
rm -rf "$ENV_PREFIX"

echo "Removing temp directories..."
rm -rf "$TMP_LOCAL"

echo "Teardown complete."
