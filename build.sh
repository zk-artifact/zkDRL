#!/usr/bin/env bash
# Build the six approved guest profiles and five static host variants.
# Requires: Rust toolchain, SP1 6.9.0 (cargo-prove), protoc, libclang, clang.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
if test -f "$HERE/env.sh"; then
  # shellcheck disable=SC1091
  source "$HERE/env.sh"
fi
OUT="${ZKDRL_OUT:-$HERE/runs/full-023-unified}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HERE/.build/target}"
export SP1_CIRCUIT_MODE="${SP1_CIRCUIT_MODE:-release}"
mkdir -p "$OUT" "$HERE/bin"

(cd "$HERE/a2c" && cargo prove build --locked --output-directory elf) > "$OUT/build-a2c.log" 2>&1
for row in 'dqn-h16-c64|default|elf' 'dqn-h32-c64|h32|elf-h32' 'dqn-h64-c64|h64|elf-h64' 'dqn-h16-c128|c128|elf-c128' 'dqn-h16-c256|c256|elf-c256'; do
  IFS='|' read -r name features dest <<< "$row"
  args=()
  if test "$features" != default; then args=(--features "$features"); fi
  (cd "$HERE/dqn" && cargo prove build --locked "${args[@]}" --output-directory "$dest") > "$OUT/build-guest-$name.log" 2>&1
  cargo build --release --offline --locked --manifest-path "$HERE/host/Cargo.toml" "${args[@]}" > "$OUT/build-host-$name.log" 2>&1
  cp "$CARGO_TARGET_DIR/release/zkdrl-ctc-host" "$HERE/bin/host-$name"
  cargo test --release --offline --locked --manifest-path "$HERE/host/Cargo.toml" "${args[@]}" > "$OUT/tests-host-$name.log" 2>&1
  printf 'Built and unit-tested %s\n' "$name"
done
printf 'All six approved program profiles and five host binaries built.\n'
