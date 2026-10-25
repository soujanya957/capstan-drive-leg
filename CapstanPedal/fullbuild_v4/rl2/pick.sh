#!/bin/sh
# Choose the deployable snapshot of finished runs on the CPU (pick_best.py), then evaluate it on 200 held-out
# episodes:  sh pick.sh handstand2 terrain_uniform_s0 ...   -> policies/<name>_best.npz, logs/<name>.pick
cd "$(dirname "$0")"
PY=${PY:-~/miniforge3/envs/capstan-gpu/bin/python}
export JAX_PLATFORMS=cpu
for n in "$@"; do
  { echo "== $n"; $PY pick_best.py $n --k=4 --n=120 2>&1 | grep -v -i warn
    $PY play.py eval ${n}_best --n=200 2>&1 | grep -v -i warn | head -1; } > logs/$n.pick &
done
wait
