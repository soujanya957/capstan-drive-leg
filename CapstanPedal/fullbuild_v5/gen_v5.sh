#!/bin/sh
# v5 on cake: body + head meshes, the v4 pods' meshes reused, checks
set -e
cd ~/dev/capstan-drive-leg/CapstanPedal
PY=~/miniforge3/envs/capstan-cad/bin/python
V4=fullbuild_v4/mujoco/meshes; V5=fullbuild_v5/mujoco/meshes
mkdir -p $V5
for f in $V4/*.stl; do b=$(basename $f); case $b in Q_*) ;; *) cp -n $f $V5/ ;; esac; done
cd fullbuild_v5/src
echo "=== generate $(date)"
$PY generate_quad.py ../mujoco/meshes 2>&1 | grep -v -i warn
$PY - <<'PYEOF'
import json, os
v5 = json.load(open("../mujoco/meshes/quad.json")); v4 = json.load(open("../../fullbuild_v4/mujoco/meshes/quad.json"))
meta = {}
for k, v in v5["meta"].items():
    v["file"] = os.path.basename(v["file"]); meta[k] = v
for k, v in v4["meta"].items():
    if not k.startswith("Q_"):
        meta[k] = v
v5["meta"] = meta
json.dump(v5, open("../mujoco/meshes/quad.json", "w"), indent=1)
print("quad.json:", len(meta), "items")
PYEOF
echo "=== generated $(date)"
