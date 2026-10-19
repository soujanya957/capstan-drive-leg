# capstan-drive-leg

A capstan-driven robot leg and the quadruped built from four of them.

| Folder | What |
|---|---|
| `CapstanPedal/v9_1_buildable/` | The leg module: 9:1 coaxial capstan drives, 4-bar leg, ab/ad drive. Parametric CAD (build123d), print files, BOM, checks, build films |
| `CapstanPedal/fullbuild_v3/` | The quadruped: four v9.1 pods in a screw-together body with an electronics cassette. CAD, BOM, checks, RL policies, the scripted jump, URDF and MuJoCo simulation |
| `CapstanPedal/ref_cad/` | Reference CAD used by the generators (EaglePower 8308 motor STEP) and local CAD notes |
| `CapstanPedal/Capstan-Drive`, `OpenQDD-V1`, `TOPS` | Upstream reference designs by aaedmusa (git submodules) |
| `report/` | Capstan drive report (LaTeX + PDF) |
| `biped/` | Biped notes: circuit, control, parts |

Large files (STL, STEP, videos, policies) are stored with Git LFS:

```
git lfs install
git clone --recurse-submodules https://github.com/soujanya957/capstan-drive-leg.git
```

Start with `CapstanPedal/fullbuild_v3/README.md`. Nothing here has been built or tested on hardware yet.
