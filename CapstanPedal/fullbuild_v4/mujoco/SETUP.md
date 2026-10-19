# Setting up the simulator

One conda environment, `capstan-sim`, runs everything in this folder: the teleop, the videos, the assembly
viewer and the URDF builder. Tested on macOS (Apple silicon) with Miniconda; Linux works the same way.
Then see [RUN.md](RUN.md).

## 1. Get the repo

The meshes and the URDF's meshes are stored with Git LFS, so install it before cloning:

```
brew install git-lfs          # or: conda install -c conda-forge git-lfs
git lfs install
git clone --recurse-submodules https://github.com/soujanya957/capstan-drive-leg.git
cd capstan-drive-leg/CapstanPedal/fullbuild_v4/mujoco
```

Already cloned without LFS? Run `git lfs pull` in the repo. If `meshes/*.stl` or `../urdf/meshes/*.obj` are
a few hundred bytes of text, LFS hasn't fetched them, and MuJoCo will fail to load the model.

## 2. Create the environment

From `fullbuild_v4/mujoco`:

```
conda env create -f environment.yml
conda activate capstan-sim
```

`environment.yml` installs Python 3.12 and ffmpeg from conda-forge, then `requirements.txt` with pip:

| Package | Used by |
|---|---|
| `mujoco` 3.16 | the physics and rendering |
| `glfw` | the teleop and assembly windows |
| `numpy`, `pillow` | everything; captions in the videos |
| `build123d`, `fast-simplification` | only `watch_assembly.py` and `../urdf/build_urdf.py` (they load the CAD code) |
| `ffmpeg` (conda) | only `make_videos.py` |

To update an existing environment after `requirements.txt` changes:

```
conda env update -f environment.yml --prune
```

## 3. Check it

```
python -c "import model as M, mujoco; m = M.build(); print(m.nbody, 'bodies,', m.nu, 'motors')"
```

It should print the body and motor counts (12 motors) after a few seconds of loading meshes. Then
`python teleop.py` should open a window with the robot standing.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Error opening file ... .obj` / `.stl`, or a mesh "is not a mesh" | LFS files not fetched: `git lfs pull` |
| `glfw.GLFWError` / no window | Needs a display. Over SSH use `make_videos.py` (offscreen) instead, with `MUJOCO_GL=egl` on a Linux GPU box |
| `ffmpeg: not found` | `conda activate capstan-sim` first (or `brew install ffmpeg`) |
| Teleop runs slowly | Shadows are off by default; press G to check they're off. Close other GPU-heavy apps |
| `ModuleNotFoundError: build123d` | Only the assembly viewer and URDF builder need it; it's in the env, so check the env is active |

To remove the environment: `conda env remove -n capstan-sim`.
