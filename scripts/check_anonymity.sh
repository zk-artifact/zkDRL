#!/usr/bin/env bash
# Best-effort anonymity/privacy scan for the review package.
# Exits nonzero on a hit or a scanner error.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
fail=0

scan() {
  local label=$1 pattern=$2 hits status
  hits=$(rg -n --no-heading --hidden --no-ignore -i -e "$pattern" \
    --glob '!Cargo.lock' --glob '!*.npz' --glob '!*.pyc' \
    --glob '!MANIFEST.sha256' --glob '!dist/**' \
    --glob '!scripts/check_anonymity.sh' .)
  status=$?
  if test "$status" -eq 2; then
    printf '\n[scanner error: %s]\n' "$label" >&2
    fail=1
    return
  fi
  if test -n "$hits"; then
    printf '\n[%s]\n%s\n' "$label" "$hits"
    fail=1
  fi
}

scan "absolute workspace path" '/workspace/|/home/|/root/|/Users/'
scan "development campaign label" 'replacement-[0-9]+|scaling-[0-9]+|pilot-[0-9]+|research_A'
# Single backslash: rg sees \. and matches a literal dot.
scan "email address" '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|edu|net|cn|io|ai|gov)'
scan "cloud/provider credentials" 'VAST_API_KEY|vast\.ai|ssh-ed25519|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}'
scan "GPU/instance identifier" 'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|instance_id'
scan "author/institution metadata" 'Anonymous Institution|University of|Department of|Corresponding author'

# Binary metadata: NPZ, SVG, and any PDF that is later regenerated.
while IFS= read -r f; do
  test -n "$f" || continue
  hits=$(strings -a "$f" | rg -n -i -e '/workspace/|/home/|/root/|/Users/|replacement-[0-9]+|scaling-[0-9]+|VAST_API_KEY|ssh-ed25519|AKIA[0-9A-Z]{16}|GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' || true)
  if test -n "$hits"; then
    printf '\n[binary metadata: %s]\n%s\n' "$f" "$hits"
    fail=1
  fi
done < <(find . -type f \( -name '*.npz' -o -name '*.pdf' -o -name '*.svg' \) -not -path './dist/*' -print)

# Archive structure, if the package has already been built.
if test -f dist/zkdrl-anonymous.zip; then
  if ! python3 - <<'PY'
import sys, zipfile
bad = []
with zipfile.ZipFile('dist/zkdrl-anonymous.zip') as z:
    for n in z.namelist():
        parts = n.split('/')
        if n.startswith('/') or '..' in parts or n.endswith('/'):
            bad.append(('unsafe path', n))
        if any(p in ('.git', '__pycache__', 'runs', 'cloud') for p in parts):
            bad.append(('excluded directory', n))
        if n.endswith(('.pyc', '.sqlite', '.log', '.key', '.pem')):
            bad.append(('excluded file', n))
if bad:
    for kind, name in bad:
        print(f'[{kind}] {name}')
    sys.exit(1)
PY
  then
    fail=1
  fi
fi

if test -d .git; then
  echo "[git history] .git directory is present" >&2
  fail=1
fi
if test "$fail" -eq 0; then
  echo "anonymity scan: clean"
else
  echo "anonymity scan: findings above (review false positives manually)" >&2
fi
exit "$fail"
