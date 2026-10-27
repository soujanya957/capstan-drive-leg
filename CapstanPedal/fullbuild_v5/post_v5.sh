#!/bin/sh
# v5 on cake, after gen_v5.sh: quad.json paths made absolute, clash/head/assembly checks, stress, URDF, gait films, build film
cd ~/dev/capstan-drive-leg/CapstanPedal/fullbuild_v5
PY=~/miniforge3/envs/capstan-cad/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$PY - <<'PYEOF'
import json, os
p = os.path.abspath("mujoco/meshes"); q = json.load(open(p + "/quad.json"))
for v in q["meta"].values():
    if not v["file"].startswith("/"):
        v["file"] = os.path.join(p, v["file"])
json.dump(q, open(p + "/quad.json", "w"), indent=1)
PYEOF
mkdir -p logs video
( cd src && $PY check_quad.py > ../logs/check_quad.log 2>&1; echo "check_quad exit $?" >> ../logs/check_quad.log ) &
( cd src && $PY check_head.py > ../logs/check_head.log 2>&1; echo "check_head exit $?" >> ../logs/check_head.log ) &
( cd src && $PY stress_test.py > ../logs/stress.log 2>&1; echo "stress exit $?" >> ../logs/stress.log ) &
( cd src && $PY check_assembly_quad.py ../mujoco/meshes > ../logs/check_assembly.log 2>&1; echo "check_assembly exit $?" >> ../logs/check_assembly.log ) &
( cd urdf && PYTHONPATH=../mujoco $PY build_urdf.py > ../logs/urdf.log 2>&1; echo "urdf exit $?" >> ../logs/urdf.log
  cd ../mujoco && $PY model.py >> ../logs/urdf.log 2>&1
  $PY make_videos.py > ../logs/videos.log 2>&1; echo "videos exit $?" >> ../logs/videos.log
  $PY make_videos.py --notext trot turn spin height camera >> ../logs/videos.log 2>&1 ) &
( cd src && $PY make_quad_film.py ../mujoco/meshes > ../logs/film.log 2>&1; echo "film exit $?" >> ../logs/film.log ) &
wait
echo "=== post done $(date)"
