#!/bin/sh
# Train every run in a batch list, in order, on cake:  sh run_queue.sh experiments/queue4.list
# One run per line: <name> <train.py options>, or <name> <script.py> <its options> (distill.py, train_sac.py). A run whose policies/<name>.npz exists is skipped; lines added while it
# runs are picked up. After each run, on the CPU while the GPU trains the next: progress film, final film, held-out
# evaluation on the full model, stress audit, and the best snapshot (policies/<name>_best.npz). Logs: logs/<name>.*
cd "$(dirname "$0")"
LIST=${1:?usage: sh run_queue.sh experiments/<batch>.list}
PY=${PY:-~/miniforge3/envs/capstan-gpu/bin/python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
i=1
while :; do
  line=$(sed -n "${i}p" "$LIST")
  [ -z "$line" ] && { sleep 60; line=$(sed -n "${i}p" "$LIST"); [ -z "$line" ] && break; }
  name=${line%% *}; args=${line#* }
  if [ ! -f policies/$name.npz ]; then
    script=train.py
    case "$args" in *.py\ *) script=${args%% *}; args=${args#* } ;; esac
    echo "=== $name $script $args $(date)"; $PY $script --name=$name $args > logs/$name.out 2>&1; tail -2 logs/$name.out
    case "$args" in *--teacher=1*) i=$((i+1)); continue ;; esac     # a teacher cannot run on the robot: no films
    ( export MUJOCO_GL=egl JAX_PLATFORMS=cpu; $PY progress.py $name; $PY play.py film $name; $PY play.py eval $name --n=200
      $PY audit.py $name --n=10 > logs/$name.audit 2>&1
      sh pick.sh $name ) > logs/$name.post 2>&1 &
  fi
  i=$((i+1))
done
wait
echo "=== $LIST done $(date)"
