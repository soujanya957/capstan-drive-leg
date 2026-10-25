#!/bin/sh
# Keep the code in rl/ and rl2/ the same on this machine and on cake (the training box). Code only: policies, logs,
# checkpoints and films stay on cake. Run from anywhere:  sh rl2/sync.sh
set -e
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-cake:dev/capstan-drive-leg/CapstanPedal/$(basename "$PWD")}
for d in rl rl2; do
    rsync -a --exclude policies --exclude logs --exclude __pycache__ --exclude '*.mp4' "$d/" "$REMOTE/$d/"
done
echo "code synced to $REMOTE"
