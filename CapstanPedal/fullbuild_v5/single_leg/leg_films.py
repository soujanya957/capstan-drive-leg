"""Films of the single v5 leg (v9.1 leg_abad_gantry on its rail stand), rendered with v9.1's MuJoCo film code.

  build    the step-by-step assembly, in the build order (v9.1's author_variant)
  jump     squat, stepping with ab/ad, three hops (v9.1's leg_film, 3-DOF)
  joints   each joint alone with v5's limits: ab/ad +-12, hip swing, knee fold, then all three together
  gait     foot in the air, tracing a trot stride, then a side-step stride with ab/ad (a bench gait test)

Run (capstan-sim env):
  python ../../v9_1_buildable/src/export_variants.py <mesh_dir> leg_abad_gantry     (build123d, once)
  python leg_films.py <mesh_dir> [build jump joints gait] [--preview] [--at=t1,t2]
Writes video/single_leg_<clip>.mp4 (and _notext.mp4). CAPSTAN_VIDEO_SIZE=1280x720 for smaller files.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "v9_1_buildable" / "src"))
os.environ.setdefault("CAPSTAN_LEG_L", "200")
import make_videos as mv                                                    # noqa: E402
import mujoco                                                               # noqa: E402

VNAME = "leg_abad_gantry"
AB_LIM, KNEE_LIM, OUT_LIM = 12.0, 60.0, 90.0          # v5's software ab/ad limit; knee from rest; capstan travel
RATE = 120                                            # track samples per video second


def check_limits(r, phi, ab):
    qt, qs = mv.leg4_ik(r, phi)
    dt_, ds = qt - mv.L4_Q0["thigh"], qs - mv.L4_Q0["shin"]
    knee = (qs - qt) - (mv.L4_Q0["shin"] - mv.L4_Q0["thigh"])
    assert abs(dt_) <= OUT_LIM and abs(ds) <= OUT_LIM, (r, phi, dt_, ds)
    assert abs(knee) <= KNEE_LIM + 1e-6, (r, phi, knee)
    assert abs(ab) <= AB_LIM + 1e-6, ab


def motion_film(meta, verts, track, script):
    """A 3-DOF film from a track [(r, phi, hip height above the floor, ab/ad deg)] at RATE samples a second.
    script(F, T, look) adds the cameras and captions; T(i) is the time of sample i. Posing as v9.1's leg_film."""
    for r, p, _, a in track:
        check_limits(r, p, a)
    F = mv.Film(meta, verts)
    F.theme = "dark"
    L = lambda *n: mv.labels_of(meta, *n)
    F.hidden = set(L("Test_Arm", "Base_Mount", "Leg_", "Rail_Carriage", "REF_MGN12H", "Carriage_HW"))
    F.base = set(meta)
    F.world_cam = True
    tt = np.arange(len(track)) / RATE
    rr, pp, hh, ab = (np.array([x[i] for x in track]) for i in range(4))
    at = lambda arr, t: float(np.interp(t, tt, arr))
    F.duration = float(tt[-1])
    F.up_at = lambda t: (np.eye(3), np.array([0.0, 0.0, at(hh, t) - mv.OUT[1]]))
    pose = lambda t: mv.leg4_ik(at(rr, t), at(pp, t))
    F.angle_fn = lambda t: {"A": math.radians(pose(t)[1] - mv.L4_Q0["shin"]), "B": math.radians(pose(t)[0] - mv.L4_Q0["thigh"]),
                            "C": math.radians(at(ab, t))}
    F.abad_fn = lambda t: math.radians(at(ab, t))

    def turn(angle, pivot0, pivot):
        dp, q = mv.rot_xz(angle, np.array([pivot0[0], 0.0, pivot0[1]]))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3), dp + np.array([pivot[0] - pivot0[0], 0.0, pivot[1] - pivot0[1]])

    def shin4(t):
        qt, qs = pose(t)
        return turn(math.radians(qs - mv.L4_Q0["shin"]), mv.L4_K0, mv.leg4_points(qt, qs)[0])

    def coupler4(t):
        qt, qs = pose(t)
        return turn(math.radians(qt - mv.L4_Q0["thigh"]), mv.L4_C0, mv.leg4_points(qt, qs)[2])

    for k in L("Leg4_Shin", "Leg4_Foot", "Leg4_Shin_Plate"):
        F.custom[k] = shin4
    F.custom["Leg4_Coupler"] = coupler4

    def joint4(j):
        idx = {"knee": 0, "crank": 2, "lever": 3}[j]
        p0 = (mv.L4_K0, mv.L4_F0, mv.L4_C0, mv.L4_D0)[idx]
        return lambda t: (np.eye(3), np.array([mv.leg4_points(*pose(t))[idx][0] - p0[0], 0.0,
                                               mv.leg4_points(*pose(t))[idx][1] - p0[1]]))
    for k in L("Leg4_HW_"):
        j = meta[k].get("part", k)[len("Leg4_HW_"):].rsplit("_", 1)[0]
        F.custom[k] = shin4 if j in ("foot", "shin") else joint4(j)
    t_rest = mv.L4_R0 + mv.L4_FOOT_R - mv.OUT[1]
    for k in L("Stand_"):                                               # the stand stays put in the world
        F.custom[k] = lambda t: (np.eye(3), np.array([0.0, 0.0, t_rest]) - np.asarray(mv.UP_T, float))
    F.rope = [("A", -2.0, -1.0), ("B", -2.0, -1.0)]
    F.beads.append((0.0, F.duration))
    F.readouts.append((0.0, F.duration, ["A", "B", "C"]))
    script(F, lambda i: float(tt[min(i, len(tt) - 1)]), np.array([0.0, 40.0, 300.0]))
    return F


R0, FR = mv.L4_R0, mv.L4_FOOT_R
LIFT = 90.0                                           # hip raised on the rail so the foot swings clear of the floor
ease = mv.ease


def seg(track, T, f):
    n = int(T * RATE)
    for i in range(n):
        track.append(f((i + 1) / n))


def hold(track, T, r=R0, phi=-90.0, h=None, ab=0.0):
    seg(track, T, lambda u: (r, phi, R0 + FR if h is None else h, ab))


def joints_film(meta, verts):
    tr, marks = [], {}
    up = R0 + FR + LIFT
    hold(tr, 2.5)
    seg(tr, 1.5, lambda u: (R0, -90.0, R0 + FR + LIFT * ease(u), 0.0))        # lift the hip on the rail
    marks["ab"] = len(tr)
    seg(tr, 6.0, lambda u: (R0, -90.0, up, AB_LIM * math.sin(2 * math.pi * u)))
    hold(tr, 0.5, h=up)
    marks["hip"] = len(tr)
    seg(tr, 6.0, lambda u: (R0, -90.0 + 30.0 * math.sin(2 * math.pi * u) * math.sin(math.pi * u) ** 0.3, up, 0.0))
    hold(tr, 0.5, h=up)
    marks["knee"] = len(tr)
    r_fold = R0 - 120.0
    seg(tr, 6.0, lambda u: (R0 - (R0 - r_fold) * (1 - math.cos(2 * math.pi * u)) / 2, -90.0, up, 0.0))
    hold(tr, 0.5, h=up)
    marks["all"] = len(tr)
    env = lambda u: math.sin(math.pi * u) ** 0.5
    seg(tr, 8.0, lambda u: (R0 - 40.0 * env(u) * (1 - math.cos(2 * math.pi * 2 * u)),          # foot loop: r and swing
                            -90.0 + 22.0 * env(u) * math.sin(2 * math.pi * 2 * u), up,
                            10.0 * env(u) * math.sin(2 * math.pi * u)))
    seg(tr, 1.5, lambda u: (R0, -90.0, up - LIFT * ease(u), 0.0))
    marks["end"] = len(tr)
    hold(tr, 2.5)

    def script(F, T, look):
        hip_look = np.array([-120.0, 20.0, R0 + FR - mv.OUT[1] + 60.0])
        F.cam(0.0, look + [0, 0, 160], 1500, -128, -6)
        F.cards.append((0.2, 3.8, "Capstan v5  ·  one leg", "Each joint alone, at the quadruped's limits"))
        F.cam(T(marks["ab"]) + 0.5, hip_look + [0, 0, 60], 1200, -20, -8)
        F.captions.append((T(marks["ab"]) + 0.2, T(marks["hip"]) - 0.2, "C", "Ab/ad  ±12°",
                           "Motor C turns the whole module about the body axis through the third 9:1 capstan"))
        F.cam(T(marks["hip"]) - 0.3, hip_look + [0, 0, 60], 1200, -20, -8)
        F.cam(T(marks["hip"]) + 0.8, look + [0, 0, 120], 1450, -110, -4)
        F.captions.append((T(marks["hip"]) + 0.2, T(marks["knee"]) - 0.2, "B", "Hip swing  ±30°",
                           "Output B turns the thigh; the crank follows so the shin keeps its angle to the thigh"))
        F.cam(T(marks["knee"]) - 0.3, look + [0, 0, 120], 1450, -110, -4)
        F.cam(T(marks["knee"]) + 0.8, look + [0, 0, 120], 1400, -135, -4)
        F.captions.append((T(marks["knee"]) + 0.2, T(marks["all"]) - 0.2, "A", "Knee fold",
                           "Output A's crank folds the shin through the parallel coupler; the foot comes up 120 mm"))
        F.cam(T(marks["all"]) - 0.3, look + [0, 0, 120], 1400, -135, -4)
        F.cam(T(marks["end"]), look + [0, 0, 120], 1500, -55, -6)
        F.captions.append((T(marks["all"]) + 0.2, T(marks["end"]) - 0.2, "A·B·C", "All three",
                           "The foot traces a loop while the module rolls ±10°"))
        F.cam(F.duration, look + [0, 0, 120], 1500, -80, -6)
        F.cards.append((T(marks["end"]) + 0.2, F.duration, "Capstan v5  ·  one leg", "CAD animation  ·  kinematics only"))
    return motion_film(meta, verts, tr, script)


def stride(u, length=140.0, height=60.0, h=None, duty=0.55):
    """Foot (x forward, z up) relative to the hip over one trot cycle u in [0, 1): stance straight back, swing an arc."""
    h = h or R0 - 25.0
    if u < duty:                                                       # stance: front to back at a steady speed
        s = u / duty
        return length / 2 - length * s, -h
    s = (u - duty) / (1 - duty)                                         # swing: back to front, lifted
    return -length / 2 + length * (1 - math.cos(math.pi * s)) / 2, -h + height * math.sin(math.pi * s)


def to_polar(x, z):
    return math.hypot(x, z), math.degrees(math.atan2(z, x))


def gait_film(meta, verts):
    tr, marks = [], {}
    up = R0 + FR + LIFT
    hold(tr, 2.0)
    seg(tr, 1.5, lambda u: (R0, -90.0, R0 + FR + LIFT * ease(u), 0.0))
    r_a, p_a = to_polar(*stride(0.0))
    seg(tr, 1.0, lambda u: (R0 + (r_a - R0) * ease(u), -90.0 + (p_a + 90.0) * ease(u), up, 0.0))
    marks["trot"] = len(tr)
    period, n = 0.8, 6                                                 # v5's trot runs ~2 steps a second; shown at half speed
    seg(tr, period * n, lambda u: (*to_polar(*stride((u * n) % 1.0)), up, 0.0))
    marks["side"] = len(tr)
    def side(u):
        c = (u * n) % 1.0
        x, z = stride(c, length=60.0)
        a = 8.0 * math.sin(2 * math.pi * c) * min(1.0, u * n, (1 - u) * n)   # ab/ad swings the foot sideways each step
        return (*to_polar(x, z), up, a)
    seg(tr, period * n, side)
    seg(tr, 1.0, lambda u: (R0 + (to_polar(*stride(0.0, length=60.0))[0] - R0) * (1 - ease(u)),
                            -90.0 + (to_polar(*stride(0.0, length=60.0))[1] + 90.0) * (1 - ease(u)), up, 0.0))
    seg(tr, 1.5, lambda u: (R0, -90.0, up - LIFT * ease(u), 0.0))
    marks["end"] = len(tr)
    hold(tr, 2.0)

    def script(F, T, look):
        F.cam(0.0, look + [0, 0, 160], 1500, -100, -5)
        F.cards.append((0.2, 3.3, "Capstan v5  ·  one leg", "Gait trajectories in the air, as a bench test runs them"))
        F.cam(T(marks["trot"]) + 0.6, look + [0, 0, 140], 1350, -135, -4)
        F.captions.append((T(marks["trot"]) + 0.2, T(marks["side"]) - 0.2, "", "Trot stride (half speed)",
                           "140 mm stride, 60 mm swing height, 55 % stance: hip on B, knee on A"))
        F.cam(T(marks["side"]) - 0.3, look + [0, 0, 140], 1350, -135, -4)
        F.cam(T(marks["side"]) + 0.8, look + [0, 0, 140], 1350, -30, -6)
        F.captions.append((T(marks["side"]) + 0.2, T(marks["end"]) - 0.6, "", "Side-step (half speed)",
                           "Shorter stride with ab/ad ±8° swinging the foot sideways each step"))
        F.cam(F.duration, look + [0, 0, 140], 1450, -70, -5)
        F.cards.append((T(marks["end"]) + 0.2, F.duration, "Capstan v5  ·  one leg", "CAD animation  ·  kinematics only"))
    return motion_film(meta, verts, tr, script)


def main():
    mesh_dir = Path(sys.argv[1]).resolve()
    at = [float(x) for a in sys.argv if a.startswith("--at=") for x in a[5:].split(",")]
    preview = "--preview" in sys.argv or bool(at)
    jobs = [a for a in sys.argv[2:] if not a.startswith("--")] or ["joints", "gait", "jump", "build"]
    out = HERE / ("preview" if preview else "video")
    out.mkdir(exist_ok=True)
    meta = json.loads((mesh_dir / f"{VNAME}.json").read_text())
    models, verts = mv.load(mesh_dir, meta)
    for job in jobs:
        if job == "build":
            film = mv.author_variant(meta, verts, VNAME, mv.load_tris(models["light"], meta))
            title = "Capstan v5  ·  one leg (3 DOF, rail stand)  ·  build"
        elif job == "jump":
            film = mv.leg_film(meta, verts, variant="3dof")
            title = "Capstan v5  ·  one leg  ·  squat, step, hop"
        elif job == "joints":
            film = joints_film(meta, verts)
            title = "Capstan v5  ·  one leg  ·  joints"
        elif job == "gait":
            film = gait_film(meta, verts)
            title = "Capstan v5  ·  one leg  ·  gait"
        else:
            raise SystemExit(f"unknown clip {job}")
        print(f"{job}: {film.duration:.1f} s", flush=True)
        mv.render(film, models[film.theme], meta, out / f"single_leg_{job}.mp4", title, preview, at,
                  notext_path=None if preview else out / f"single_leg_{job}_notext.mp4")


if __name__ == "__main__":
    main()
