"""One v5 leg, standalone, for a bench build: the v9.1 `leg_abad_gantry` module (the pod the quadruped hangs four of)
with its own ab/ad cradle and the MGN12 rail stand in place of the body's bulkheads.

The quadruped (../src/generate_quad.py) imports v9.1's generate.py with CAPSTAN_LEG_L = 200 and changes no other
constant. It drops the cradle parts (Hip_Outer/Inner/Front_Plate, Hip_Spine, the rail blocks) because the bulkheads
replace them. For one leg on the bench those parts come back. The rotating pod is the same part for part, so this
leg is the robot's left-front (LF) pod: the right pods are its mirror image and the rear pods swap the end brackets.

Writes, next to this file:
  print/   one STL per printed part, named <qty>x_<part>.stl, in print orientation (copied from v9.1's print/)
  step/    the same parts as STEP
  assembly/leg_abad_gantry.step   the whole leg on its stand (v9.1's assembly export)
  print/0_Fit_Coupon.stl + step/0_Fit_Coupon.step   print this first: a 6806 pocket, a Ø30 journal, an M4 insert
           hole and a tensioner-nut slot, drawn with the same allowances as the leg
  parts.csv  the printed parts, quantities and source files

Run (capstan-sim env):  python make_single_leg.py [--regen]
  --regen  first rebuild v9.1's print/, step/ and assembly/ from its generate.py (several minutes)
"""
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.dirname(HERE)
V91 = os.path.normpath(os.path.join(V5, "..", "v9_1_buildable"))
LEG_L = 200.0                                    # the quadruped's thigh and shin (generate_quad.LEG_L)
os.environ["CAPSTAN_LEG_L"] = str(LEG_L)
sys.path.insert(0, os.path.join(V91, "src"))
import generate as g                                                    # noqa: E402
from build123d import Align, Box, Cylinder, Pos, export_step, export_stl  # noqa: E402

VARIANT = "leg_abad_gantry"
# assembly label -> v9.1 export (subfolder, file stem) where they differ
SRC = {"Hip_Posts": ("abad", "Hip_Post")}
SUB = {**{n: "leg_4bar" for n in ("Leg4_Coupler", "Leg4_Crank", "Leg4_Foot", "Leg4_Foot_Core_A", "Leg4_Foot_Core_B", "Leg4_Shin", "Leg4_Shin_Plate_Back",
                                  "Leg4_Shin_Plate_Front", "Leg4_Thigh")},
       **{n: "abad" for n in ("Abad_Drum", "Abad_Small_Drum", "End_Bracket_Front", "Hip_Front_Plate", "Hip_Inner_Plate",
                              "Hip_Outer_Plate", "Hip_Spine", "Hip_Trunnion")}}
# printed parts the quadruped replaces with its body (needed on the bench only)
BENCH_ONLY = ("Hip_Outer_Plate", "Hip_Inner_Plate", "Hip_Front_Plate", "Hip_Spine")


def printed_parts():
    """Printed part -> quantity in one leg_abad_gantry build, counted from v9.1's assembly."""
    c = Counter()
    for n, (solid, grp, color, kind) in g.assembly(VARIANT).items():
        if kind != "part":
            continue
        base = re.sub(r"_\d+$", "", n)
        c[base] += len(solid.solids()) if base == "Hip_Posts" else 1      # one compound holds both ab/ad posts
    return c


def fit_coupon():
    """60 x 40 x 12 block: a 6806 pocket (Ø42 + PRESS, 7 deep, through hole behind it), a Ø30 journal stub 8 tall
    (+ JOURNAL_CLR) on top, an M4 insert hole (Ø M4_INSERT, 8 deep) and the drums' M4 nylon-nut slot (open at the
    side, as in the drums). Pocket and journal take the same 6806, so one bearing tests both."""
    t = 12.0
    b = Box(60.0, 40.0, t, align=(Align.MIN, Align.MIN, Align.MIN))
    b -= Pos(20.0, 20.0, t - 7.0) * Cylinder((42.0 + g.PRESS) / 2, 7.01, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos(20.0, 20.0, -0.01) * Cylinder(17.0, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos(50.0, 30.0, t - 8.0) * Cylinder(g.M4_INSERT / 2, 8.01, align=(Align.CENTER, Align.CENTER, Align.MIN))
    af, h = g.PEG_NUT_AF + 0.3, g.PEG_NUT_H + 0.3                     # nut slot, as cut in the drums (peg_cuts)
    b -= Pos(50.0, 0.0, t / 2) * Box(af, 12.0 + af / 2, h, align=(Align.CENTER, Align.MIN, Align.CENTER))  # open at y 0
    b -= Pos(50.0, 12.0, -0.01) * Cylinder(g.M4_CLR / 2, t + 0.02, align=(Align.CENTER, Align.CENTER, Align.MIN))
    j = Pos(86.0, 20.0, 0) * (Cylinder(24.0, 3.0, align=(Align.CENTER, Align.CENTER, Align.MIN)) +
                              Pos(0, 0, 3.0) * Cylinder((30.0 + g.JOURNAL_CLR) / 2, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    return b + j


def main():
    if "--regen" in sys.argv:
        subprocess.run([sys.executable, "generate.py"], cwd=os.path.join(V91, "src"), check=True)
    assert abs(g.L4_LEN - LEG_L) < 1e-9, f"v9.1 built with {g.L4_LEN} mm links, v5 uses {LEG_L}"
    for d in ("print", "step", "assembly"):
        shutil.rmtree(os.path.join(HERE, d), ignore_errors=True)
        os.makedirs(os.path.join(HERE, d))
    rows = []
    for name, qty in sorted(printed_parts().items()):
        sub, stem = SRC.get(name, (SUB.get(name, ""), name))
        out = f"{qty}x_{name}"
        for ext, folder in (("stl", "print"), ("step", "step")):
            src = os.path.join(V91, folder, sub, f"{stem}.{ext}")
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src}: run with --regen")
            shutil.copy2(src, os.path.join(HERE, folder, f"{out}.{ext}"))
        rows.append((name, qty, "bench only (the robot's bulkheads replace it)" if name in BENCH_ONLY else "pod",
                     os.path.relpath(os.path.join(V91, "print", sub, f"{stem}.stl"), HERE)))
        print(f"  {qty} x {name}")
    c = fit_coupon()
    export_stl(c, os.path.join(HERE, "print", "0_Fit_Coupon.stl"), tolerance=0.02, angular_tolerance=0.15)
    export_step(c, os.path.join(HERE, "step", "0_Fit_Coupon.step"))
    shutil.copy2(os.path.join(V91, "assembly", f"{VARIANT}.step"), os.path.join(HERE, "assembly", f"{VARIANT}.step"))
    with open(os.path.join(HERE, "parts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "qty", "role", "source"])
        w.writerows(rows)
    print(f"{sum(r[1] for r in rows)} printed parts ({len(rows)} kinds) + fit coupon -> {HERE}")


if __name__ == "__main__":
    main()
