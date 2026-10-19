"""Hand-calculation strength check (beam formulas, lever rules and catalogue ratings; NOT FEA).

Load cases
  motor:  the largest foot force the leg can push with, 28 N*m at each output (8308 at ~40 A, 8:1),
          searched over force directions within 30 deg of straight up, at standing and crouched poses
  hop:    4 x the rig's weight at the foot (estimated mass from the CAD volumes)
The larger one is the design load. Printed parts are rated as PETG: 45 MPa along the layers, 25 MPa
across them (the weaker figure is used wherever the stress crosses layers in the part's print
orientation). Safety factor targets: >= 2 for printed parts, >= 1.5 for screws and inserts, >= 1 for
bearing static ratings (C0) and rail-block static ratings.
"""
import math

import numpy as np

import generate as g

TAU = g.MOTOR_TAU * g.RATIO          # N*m per output (v9: 31.5 at 9:1; v8: 28)
PETG_XY, PETG_Z = 45.0, 25.0   # MPa
PC_XY, PC_Z = 60.0, 35.0
PETG_COMP = 50.0              # MPa, compressive
C0 = {"6000": 1960.0, "6903": 2550.0, "626ZZ": 950.0, "636ZZ": 1370.0, "625ZZ": 670.0, "MR105ZZ": 170.0, "6902": 2260.0, "6808": 4000.0, "6806": 2900.0, "MGN12H": 3900.0}
M6_SHEAR = 0.6 * 640.0 * 17.9                     # 8.8 bolt, single shear on the minor diameter (N)
M4_PROOF = 580.0 * 8.78                           # N
INSERT_M4_PULL = 1000.0                           # brass M4 heat-set insert in PETG, conservative (N)
WOOD_SCREW_PULL = 800.0                           # 5 mm wood screw, 22 mm into particle board (N)
ROPE_BREAK = 1800.0                               # v8: Ø1.5 mm Dyneema SK99, conservative; check the datasheet (N)
rows, fails = [], []


def report(item, load, capacity, target, note=""):
    sf = capacity / load if load > 0 else float("inf")
    ok = sf >= target
    rows.append((item, load, capacity, sf, target, ok, note))
    if not ok:
        fails.append(item)


def unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def leg_statics(qt, qs, F):
    """4-bar: foot force F (N, model XZ, ground on foot). Returns coupler force, knee force, tau A, tau B."""
    k, f, c, d = (np.array(p, float) for p in g.leg4_points(qt, qs))
    o = np.array(g.OUT, float)
    u = unit(d - c)
    fc = -cross(f - k, F) / cross(d - k, u)            # coupler (two-force member) on the shin, along u
    fk = -(F + fc * u)                                # thigh on the shin at the knee
    tau_a = cross(c - o, fc * u)
    tau_b = cross(k - o, fk)
    return abs(fc), float(np.linalg.norm(fk)), tau_a, tau_b


def motor_limited(qt, qs):
    best = (0.0, None)
    for ang in np.arange(60.0, 121.0, 2.0):           # pushing up, within 30 deg of vertical
        Fu = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        _, _, ta, tb = leg_statics(qt, qs, Fu)
        s = TAU * 1000.0 / max(abs(ta), abs(tb))         # leg points in mm
        if s > best[0]:
            best = (s, Fu * s)
    return best


def mass(name):
    """Rough mass: printed parts at 1.27 g/cm3 x 0.6 (4 walls, 40 % infill), 8308 ~ 0.25 kg each (assumed),
    moteus 15 g, bearings and hardware ~ 0.35 kg."""
    asm = g.assembly(name)
    printed = sum(s.volume for s, grp, _, kind in asm.values() if kind == "part" and grp != "world") / 1000.0
    motors = 3 if g.VARIANTS[name]["abad"] else 2
    return printed * 1.27 * 0.6 / 1000.0 + motors * 0.265 + 0.35, printed


# ---------------------------------------------------------------------------------------------
print(__doc__.split("\n\n")[0])
m, vol = mass("leg_abad_gantry")
m2, _ = mass("leg_gantry")
print(f"estimated mass: 3-DOF leg {m:.2f} kg (moving printed parts {vol:.0f} cm3), 2-DOF leg {m2:.2f} kg")
hop = 4.0 * m * 9.81
poses = {"standing": g.leg4_ik(259.8, -90.0), "crouched": g.leg4_ik(200.0, -90.0)}
worst = None
shin_cases = []                                   # (pose, foot force vector) for the shin bending check
for tag, (qt, qs) in poses.items():
    s, Fv = motor_limited(qt, qs)
    fc, fk, ta, tb = leg_statics(qt, qs, Fv)
    fc_h, fk_h, ta_h, tb_h = leg_statics(qt, qs, np.array([0.0, hop]))
    shin_cases += [((qt, qs), Fv), ((qt, qs), np.array([0.0, hop]))]
    print(f"{tag:9s}: motor-limited foot force {s:6.0f} N (coupler {fc:5.0f} N, knee {fk:5.0f} N) | "
          f"hop {hop:4.0f} N needs A {abs(ta_h) / 1000:4.1f} N*m, B {abs(tb_h) / 1000:4.1f} N*m (coupler {fc_h:4.0f} N, knee {fk_h:4.0f} N)")
    cand = (max(s, hop), max(fc, fc_h), max(fk, fk_h))
    worst = cand if worst is None else tuple(max(a, b) for a, b in zip(worst, cand))
F, FC, FK = worst
print(f"design loads: foot {F:.0f} N, coupler / crank pin / lever pin {FC:.0f} N, knee pin {FK:.0f} N\n")

# --- leg pivots ---
report("Knee pivot 6000ZZ (C0)", FK, C0["6000"], 1.0, "v9: one bearing, Ø10 shoulder screw (v8: 636ZZ)")
report("Crank/lever pivot 636ZZ (C0)", FC, C0["636ZZ"], 1.5, "N; s0 >= 1.5 for shock (v9's 626ZZ: 1.07)")
rows.append(("  (v4b: 2x MR105ZZ at the knee)", FK, 2 * C0["MR105ZZ"], 2 * C0["MR105ZZ"] / FK, 1.0, 2 * C0["MR105ZZ"] / FK >= 1, "replaced"))
report("M6 pin, single shear (crank, lever)", FC, M6_SHEAR, 1.5)


def coupler_buckling(E=2000.0, n=200):
    """v9.1: out-of-plane Euler load of the coupler (pin-ended, its tapered 8 mm bar plus the back rib),
    beam FE with a geometric stiffness matrix. The coupler pulls when the foot pushes up; it is pushed only
    when output A drives the shin (31.5 N*m on the 55 mm crank at most)."""
    import scipy.linalg as sl
    L, (rw, rd, r0, r1) = g.L4_L1, g.COUPLER_RIB
    ps, pw = zip(*g.COUPLER_PROFILE)
    xs = np.linspace(0.0, L, n + 1)
    K, G = np.zeros((2 * n + 2,) * 2), np.zeros((2 * n + 2,) * 2)
    for i in range(n):
        h, s = xs[i + 1] - xs[i], 0.5 * (xs[i] + xs[i + 1])
        parts = [(float(np.interp(s, ps, pw)), 8.0, 4.0)] + ([(rw, rd, -rd / 2)] if r0 < s < r1 else [])
        A = sum(b * t for b, t, _ in parts)
        yb = sum(b * t * y for b, t, y in parts) / A
        I = sum(b * t ** 3 / 12 + b * t * (y - yb) ** 2 for b, t, y in parts)
        K[2 * i:2 * i + 4, 2 * i:2 * i + 4] += E * I / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                                                       [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
        G[2 * i:2 * i + 4, 2 * i:2 * i + 4] += np.array([[36, 3 * h, -36, 3 * h], [3 * h, 4 * h * h, -3 * h, -h * h],
                                                         [-36, -3 * h, 36, -3 * h], [3 * h, -h * h, -3 * h, 4 * h * h]]) / (30 * h)
    keep = [j for j in range(2 * n + 2) if j not in (0, 2 * n)]
    lam = sl.eigh(K[np.ix_(keep, keep)], G[np.ix_(keep, keep)], eigvals_only=True)
    return lam[lam > 0].min()


F_w = FC                                                    # the coupler's pull (crouched, motor-limited)
w_w = 26.0
report("Coupler waist (offset sideways), bending + tension", F_w * abs(g.COUPLER_WAIST_DX) / (8.0 * w_w ** 2 / 6) + F_w / (8.0 * w_w),
       PETG_XY, 2.0, f"MPa; {abs(g.COUPLER_WAIST_DX):.1f} mm off the pin line, 26 x 8 mm, prints flat")
P_cr = coupler_buckling()
report(f"Coupler buckling, output A pushing at its {g.A_PUSH_TAU:.0f} N*m limit", g.A_PUSH_TAU / (g.L4_C / 1000.0), P_cr, 2.0,
       f"N; pin-ended, E 2000 MPa (at the full {TAU:.1f} N*m: SF {P_cr / (TAU / (g.L4_C / 1000.0)):.1f}); windowed bar: ~380 N")
report("Ø10 knee shoulder screw, single shear", FK, M6_SHEAR * (10.0 / 6.0) ** 2, 1.5)
report("M6 hole bearing in an 8 mm fixed link", FC / (6.0 * 8.0), PETG_XY, 2.0, "MPa; crank / coupler")
report("Ø10 knee hole bearing in the 8 mm thigh", FK / (10.0 * 8.0), PETG_XY, 2.0, "MPa")


# --- shin bending (v6 sandwich): below a section only the foot force acts, so M(s) = |(f - p(s)) x F|
def shin_moment(s_mm):
    m = 0.0
    for (qt, qs), Fv in shin_cases:
        k, f, _, _ = (np.array(p, float) for p in g.leg4_points(qt, qs))
        p = k + (f - k) * s_mm / g.L4_L2
        m = max(m, abs(cross(f - p, Fv)))
    return m


def shin_width(s_mm):
    """Core width (in-plane) at s mm below the knee along the knee-foot line: from the chain's radii."""
    pts = [(0.0, 0.0, g.SHIN_CORE_R[0]), (10.0, -70.0, g.SHIN_CORE_R[1]),
           (4.0, -g.L4_L2 + g.L4_CLAW_R[1] + 2.0, g.SHIN_CORE_R[2])]
    for (x0, y0, r0), (x1, y1, r1) in zip(pts, pts[1:]):
        if -y1 >= s_mm:
            return 2 * (r0 + (r1 - r0) * (s_mm + y0) / (y0 - y1))
    return 2 * pts[-1][2]


s_top = -g.SHIN_PLATE_TOP[False][1] + g.SHIN_PLATE_TOP[False][2] * 0.5   # where both plates are full width (back plate starts lower)
for s_mm, plates in ((18.0, False), (s_top - 4.0, False), (s_top + 6.0, True), (84.0, True)):
    w = shin_width(s_mm)
    t = 8.0 + (2 * g.SHIN_PLATE_T if plates else 0.0)
    holes = g.M3_CLR if plates else 0.0
    z_sec = t * (w ** 3 - holes ** 3) / 12.0 / (w / 2)
    report(f"Shin bending at {s_mm:.0f} mm below the knee ({'sandwich' if plates else 'core only'})",
           shin_moment(s_mm) / z_sec, PETG_XY, 2.0, f"MPa; {w:.0f} x {t:.0f} mm, prints flat")

# --- output support: crank load on A's nose, with and without the gantry ---
y_crank = g.L4_Y["Leg4_Crank"] + 4.0
y_b2 = g.B2_Y + g.B30[2] / 2
y_g = g.GANTRY_Y[0] + g.GANTRY_BRG[2] / 2
z_nose = math.pi * (2 * g.A_NOSE_R) ** 3 / 32                         # solid Ø22 nose
m_free = FC * (y_crank - y_b2)
r_g = FC * (y_crank - y_b2) / (y_g - y_b2)
m_prop = (FC - r_g) * (y_crank - y_b2)
report("A's nose bending, NO gantry", m_free / z_nose, PETG_Z, 2.0, "MPa; drum A prints axis-up")
report("A's nose bending, with gantry", m_prop / z_nose, PETG_Z, 2.0, "MPa")
y_leg = 125.0
wall_y = sum(g.WALL_BRG_Y) / 2
share_g = (y_leg - wall_y) / (y_g - wall_y)
print(f"leg-plane load split (lever rule, y {y_leg:.0f} between wall {wall_y:.1f} and gantry {y_g:.1f}): "
      f"gantry {share_g:.0%}, wall {1 - share_g:.0%}")
Fg = max(r_g, share_g * F)
report("Gantry 6903 (C0)", Fg, C0["6903"], 1.0)
report("Wall 6808, drum B (C0)", max(FK, (1 - share_g) * F) + FC, C0["6808"], 1.0)
z_arm = 10.0 * 24.0 ** 2 / 6                                             # in-plane, 24 mm deep x 10 thick
report("Gantry plate, in-plane bending", (Fg / 2 * 62.0) / z_arm, PETG_XY, 2.0, "MPa")
D, d = 2 * g.GANTRY_POST_R, g.M4_CLR
z_post = math.pi * (D ** 4 - d ** 4) / (32 * D)
L_post = g.GANTRY_Y[0] - g.FRONT_Y1
report("Gantry posts Ø20, bending (guided)", (Fg / 2 * L_post / 2) / z_post, PETG_Z, 2.0, "MPa; posts print upright")
report("Gantry M4 x 65 bolts, tension", Fg / 2, M4_PROOF, 1.5)
arm = y_g - (g.L4_Y["Leg4_Crank"] + 8.0)
z_sh = math.pi * (2 * g.CRANK_SHOULDER_R) ** 3 / 32
report("Crank stub Ø21/Ø17, bending (PETG)", Fg * arm / z_sh, PETG_Z, 2.0, "MPa; v9 shoulder; stub prints upright")
report("Crank stub Ø21/Ø17, bending (PC)", Fg * arm / z_sh, PC_Z, 2.0, "MPa; print the crank in PC, as Stanley does")

# --- capstans ---
T = TAU / (g.BIG_PITCH_R / 1000.0)
report(f"Rope tension at {TAU:.1f} N*m", T, ROPE_BREAK, 4.0, "N; Ø1.5 rope rating assumed")
# winding peg: the rope pulls across the M4 shank in the well; the shank bends between the head's
# bore (above the well) and the nut (below it), and bears on the plastic either side of the well
span = g.PEG_WELL_H + g.PEG_NUT_GAP                               # well plus the gap down to the nut
z_root = math.pi * 3.24 ** 3 / 32                                  # M4 minor diameter
report("Tensioner peg M4, bending across the well", T * span / 4 / z_root, 640.0, 2.0, "MPa; 8.8 proof, simply supported")
report("Tensioner peg M4, net section at the rope's Ø2 cross-hole", T / ((3.24 - 2.0) * 3.24), 0.6 * 640.0, 2.0, "MPa; shear")
report("Tensioner peg, bearing on the drum (2 x 4 mm each side of the well)", T / 2 / (4.0 * 4.0), PETG_XY, 2.0, "MPa")
report("Tensioner peg nut slot, wall toward the end face, shear", T / (2 * (g.PEG_X - g.PEG_NUT_AF / 2) * g.PEG_NUT_H), 0.55 * PETG_XY, 2.0, "MPa")

# --- ab/ad hub (3-DOF builds) ---
x_in = sum(g.AB_X["inner"]) / 2
x_out = sum(g.AB_X["outer"]) / 2
lever = 0.0 - x_in                                                         # foot is near x 0
M_hub = F * lever                                                          # N*mm
z_hub = math.pi * (40.0 ** 4 - 24.0 ** 4) / (32 * 40.0)
report("Ab/ad hub Ø40/Ø24, bending", M_hub / z_hub, PETG_Z, 2.0, "MPa")
r_out = M_hub / (x_in - x_out)
report("Inner 6808 (C0)", F + r_out, C0["6808"], 1.0)
report("Outer 6808 (C0)", r_out, C0["6808"], 1.0)
bolt = M_hub * g.AB_HUB_R / (g.AB_HUB_N * g.AB_HUB_R ** 2 / 2)
report("Hub M4 screw, tension", bolt, M4_PROOF, 1.5)
report("Hub M4 heat-set insert, pull-out", bolt, 1000.0, 2.0, "N; v9.1: 8 inserts in the hub's end (1000 N, as the other M4s)")
w_web, t_web = 118.0, g.END_X[1] - g.END_X[0]
report("End bracket web, bending", M_hub / (w_web * t_web ** 2 / 6), PETG_XY, 2.0, "MPa; web prints flat")
tau_ab = F * (g.L4_Y["Leg4_Shin"] - g.AB_Y) / 1000.0
report("Ab/ad holding torque for the foot force", tau_ab, TAU, 1.0, "N*m; motor C at 28 N*m")
print("(the hub items above assume no front support at all: the v6 trunnion only lowers them)")

# --- v6 cradle: the module also rides on the front trunnion. Lever rule between the inner 6808 and the
# front 6806 for the share at the front (the foot is near x 0; the rear pair takes the rest and the moment)
x_f = g.HF_X[1] - g.B30[2] / 2
R_f = F * (0.0 - x_in) / (x_f - x_in)
F_lat = TAU * 1000.0 / (g.AB_Z - (g.OUT[1] - 259.8))              # sideways foot force motor C can hold
print(f"cradle: front trunnion share {R_f:.0f} N of {F:.0f} N (bearings at x {x_in:.0f} and {x_f:.1f}); "
      f"sideways foot force up to {F_lat:.0f} N")
report("Front 6806 (C0)", R_f, C0["6806"], 1.0)
D, d = g.TRUNNION_D, 16.0
arm_t = x_f - g.TRUNNION_X[1]
report("Trunnion Ø29.9/Ø16, bending", R_f * arm_t / (math.pi * (D ** 4 - d ** 4) / (32 * D)), PETG_Z, 2.0,
       "MPa; prints flange-down, stress across layers")
report("Trunnion 4x M4 x 10, shear", R_f / 4, 0.6 * M4_PROOF, 1.5, "N per screw")
report("Front bracket web (8 mm), in-plane bending", R_f * 105.0 / (8.0 * 118.0 ** 2 / 6), PETG_XY, 2.0,
       "MPa; trunnion to the ears at the posts")
L_fp = g.AB_Y - (g.AB_RAIL_Y0 + g.SPINE_T + 10.0)
d_fp = g.FRONT_TAB_ZR[1] - g.FRONT_TAB_ZR[0]
report("Cradle front plate, in-plane bending", R_f * L_fp / (8.0 * d_fp ** 2 / 6), PETG_XY, 2.0, "MPa; at the tab")
report("Cradle front tab M4 inserts, pull-out", R_f * L_fp / (g.FRONT_TAB_Z[1] - g.FRONT_TAB_Z[0]), INSERT_M4_PULL, 2.0, "N")
span = g.HF_X[0] + 7.0 - g.AB_RAIL_X
xs_ = g.HF_X[0] - 40.0                                                    # through the front-most window
top = g.spine_top(xs_) - 16.0
bands = [(22.0 * g.SPINE_T, g.SPINE_Z[0] + 11.0), (12.0 * 8.0, g.SPINE_Z[0] + 4.0), (16.0 * g.SPINE_T, top + 8.0)]
zc = sum(a * z for a, z in bands) / sum(a for a, _ in bands)
I_sp = sum(a * (z - zc) ** 2 for a, z in bands)
report("Spine, vertical bending (front share)", R_f * span / (I_sp / max(abs(z - zc) + 11.0 for _, z in bands)), PETG_XY, 2.0,
       "MPa; blocks to the front tab, through a window")
z_out = (g.spine_top(xs_) - g.SPINE_Z[0] - 2 * 12.0) * g.SPINE_T ** 2 / 6 + 12.0 * 8.0 * (g.SPINE_T + 6.0)
report("Spine, sideways bending (front share)", F_lat / 2 * span / z_out, PETG_XY, 2.0, "MPa; prints flat")

# --- rail blocks and mounts ---
for name, x_rail, s, y0r in (("2-DOF leg", 0.0, g.RAIL5_BLOCKS_Z[1] - g.RAIL5_BLOCKS_Z[0], g.CARRIAGE5_Y0),
                             ("3-DOF leg", g.AB_RAIL_X, g.AB_BLOCKS_Z[1] - g.AB_BLOCKS_Z[0], g.AB_RAIL_Y0)):
    y_rail = y0r - 14.0
    mx, my = F * (y_leg - y_rail), F * (0.0 - x_rail)
    mtot = math.hypot(mx, my)
    report(f"MGN12H pair, {name} ({s:.0f} mm apart)", mtot / s, C0["MGN12H"], 1.0,
           f"single block would see {mtot / 1000:.0f} N*m (catalogue moment ratings ~20-30 N*m)")
# 2-DOF leg: rail carriage upright at the lower block, with its two gussets
y_up = g.CARRIAGE5_Y0 + 4.0
M_up = F * (y_leg - y_up)
zc = g.RAIL5_BLOCKS_Z[0]
depth = 70.0 * (1 - (zc - (g.RAIL_TOP + g.DESK_T)) / 90.0)
I = 2 * 8.0 * depth ** 3 / 12 + 70.0 * 8.0 ** 3 / 12 + 70.0 * 8.0 * (depth / 2) ** 2
report("Rail carriage upright + gussets, bending", M_up / (I / (depth / 2 + 4.0)), PETG_Z, 2.0, "MPa; upright crosses layers")
pair = F * (y_leg - sum(g.RAIL_INSERT_Y) / 2) / (g.RAIL_INSERT_Y[1] - g.RAIL_INSERT_Y[0]) / 2
report("Top-rail M4 inserts, pull-out (2-DOF leg)", pair, INSERT_M4_PULL, 2.0, "N per insert")
# desk builds: only the test arms' torques and ab/ad torque reach the desk
M_desk = 2 * TAU * 1000.0
report("Desk base wood screws, pull-out", M_desk / (2 * 125.0) / 2, WOOD_SCREW_PULL, 2.0, "N; both outputs at 28 N*m")
xs = [x for x, _ in g.desk_angle_screws()]
ys = [y for _, y in g.desk_angle_screws()]
report("Desk angle wood screws, pull-out", (TAU * 1000.0 / (max(ys) - min(ys)) + M_desk / (max(xs) - min(xs))) / 2,
       WOOD_SCREW_PULL, 2.0, "N; ab/ad + outputs at 28 N*m")
dg = 50.0 * (40.0 - (g.DESK_Z_AB - 110.0)) / 110.0
I = 2 * 8.0 * dg ** 3 / 12 + 72.5 * 8.0 ** 3 / 12 + 72.5 * 8.0 * (dg / 2) ** 2
report("Desk angle upright + gussets, bending", TAU * 1000.0 / (I / (dg / 2 + 4.0)), PETG_Z, 2.0, "MPa")

print(f"\n{'item':48s} {'load':>9s} {'capacity':>9s} {'SF':>6s}  target")
for item, load, cap, sf, target, ok, note in rows:
    print(f"{'ok  ' if ok else 'LOW '}{item:44s} {load:9.1f} {cap:9.1f} {sf:6.1f}  >= {target:<4} {note}")
print("\nHand calculations only (fea_check.py runs FEA on the lowest): no fatigue, no impact. Nothing has been built or load-tested.")
nogantry = [f for f in fails if "NO gantry" in f]
sf_free = PETG_Z / (m_free / z_nose)
if sf_free < 2.0:
    print(f"without the gantry: limit output A to {TAU * sf_free / 2.0:.0f} N*m (current limit ~{40 * sf_free / 2.0:.0f} A) "
          f"for SF 2 on drum A's nose")
assert not [f for f in fails if f not in nogantry], f"below target: {fails}"
print("pass: every item meets its target" + (f" except without the gantry: {nogantry}" if nogantry else ""))
