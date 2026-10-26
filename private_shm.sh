#!/usr/bin/env bash
# A private mount/IPC namespace; never remount the host/container /dev/shm.
set -euo pipefail
# A dedicated rental may already provide sufficient shared memory while denying
# user namespaces. Reuse that mount; never remount its parent or change settings.
if python3 -c 'import shutil,sys; sys.exit(0 if shutil.disk_usage("/dev/shm").total >= 8*1024**3 else 1)'; then
  exec "$@"
fi
exec unshare --user --map-root-user --mount --ipc bash -c '
  set -euo pipefail
  mount --make-rprivate /
  mount -t tmpfs -o size=8G,mode=1777 tmpfs /dev/shm
  exec "$@"
' zkrl-private-shm "$@"
