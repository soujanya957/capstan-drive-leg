"""Capstan quadruped v4 (full build; v3 narrowed by 28 mm, full X-truss plates), after Stanley: four v6.2 leg pods hung in a carbon-and-printed body
frame that carries the hip ab/ad drives.

Body frame: +X forward, +Y left, +Z up (Z = the pods' own Z).
- Each pod is the v6.2 3-DOF module's turning part: motor plate, posts, front wall, both capstans, the
  4-bar leg and the gantry, with its two end brackets. It hangs on two bearings: its ab/ad hub (the
  ab/ad drum's Ø40 tube in two 6808s) at the end toward the middle of the body, and its Ø30 trunnion
  (in a 6806) at the outer end.
- The module's fixed cradle (outer plate with motor C, inner plate, front plate, spine) is no longer
  per module: left and right copies merge into six body bulkheads (Stanley's frame plates):
    End frames (front, rear)        both trunnions' 6806s; stadium plates with a hexagon window
    Inner bulkheads (front, rear)   the hubs' inner 6808s and motor C's tip bearings
    Centre bulkheads (front, rear)  the hubs' outer 6808s and motors C (moteus C on the bay side)
  The ab/ad drives of all four legs sit together round the centre bay, as on Stanley.
- Legs all one way: the left and right pods are mirror images (the right side's parts print mirrored);
  the front and rear pods face the same way, and the rear pods have their two end brackets swapped
  (hub bracket in front, trunnion bracket behind) so their ab/ad end is also toward the middle.
- Top and bottom carbon plates with X-truss windows tie the bulkheads: tabs on the bulkheads' edges
  key into slots in the plates, M3 countersunk screws into inserts in the bulkheads' edge bosses.
- Electronics sled (KIWI-style slide-in and clip): Raspberry Pi 5 + mjbots pi3hat on its floor, two
  upright 3S packs (6S in series) on a shelf above, an mjbots power_dist board on top. It slides into
  the centre bay along two rails on the bottom plate; an X-plate clip across its open end screws into
  the rails' ends.

Run:  python generate_quad.py [mesh_dir [leg_mesh_dir]]
"""
import json
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V62 = os.path.normpath(os.path.join(ROOT, "..", "v9_1_buildable", "src"))
sys.path.insert(0, V62)
LEG_L = 200.0                          # the quadruped's thigh and shin (v8 default; v6.2: 150)
os.environ.setdefault("CAPSTAN_LEG_L", str(LEG_L))
import generate as g                                                # noqa: E402
from build123d import (Align, Axis, Box, Circle, Cone, Cylinder, ExportDXF, Face, Kind, Location, Plane, Polygon, Pos,   # noqa: E402
                       RegularPolygon, Rot, Unit, export_step, export_stl, extrude, make_hull, mirror, offset)

MIN = (Align.CENTER, Align.CENTER, Align.MIN)
BOX0 = (Align.MIN, Align.MIN, Align.MIN)

# ------------------------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------------------------
Y_AX = 86.0                            # pods' ab/ad axes at Y = +-Y_AX. v4: 14 mm in from v3 (100): the pods' inboard faces
                                       # (motor plate, moteus boards, post screws) clear each other by ~6 mm at +-AB_LIMIT
BAY = 70.0                             # centre bay (electronics) between the centre bulkheads
AB_LIMIT = 12.0                        # ab/ad used on the quadruped (deg), a software limit (v3: 20, with the pods 28 mm
                                       # further apart); the bottom plate clears the legs to here
DY = Y_AX - g.AB_Y                     # pod-local -> body Y (left pods)
DX = BAY / 2 - g.AB_X["outer"][0]      # front pods: pod-local -> body X (rear pods: -DX)
Z_TOP = 75.0                           # top plate underside: the pods sweep to Z 71.9 over +-34 deg
Z_BOT = -140.0                         # bottom plate's top face: the pods' bottoms reach -123.5; at ab/ad -20 drum A and its peg reach Z -133 just inboard of the notches (-130, tried in v3, clashes)
PLATE_T = 3.0                          # carbon top / bottom plates (laser-cut build)
PRINT_PLATE_T = 3.0                    # printed top / bottom plates (printable build: 3 pieces each)
LASER_T = 4.0                          # laser build: each bulkhead is two 4 mm 6061 plates (seat + lip)
LASER_T_THIN = 3.0                     # ... or 4 + 3 mm where the bulkhead is thinner than 8 (the 7.5 mm centre ones)
BULK_W = 126.0                         # bulkheads' half width (the cradle plates' post lobes reach further)
DRUM_NOTCH = (96.0, 34.0)              # bottom plate: notch under each pod's drums (length along X, depth into the edge)
TOP_W, BOT_W = 116.0, 96.0             # plates' half widths (v3: 130, 110; everything moves in with the pods)
BODY_R = 44.0                          # plates' and bulkheads' corner radius
PODS = ("LF", "LR", "RF", "RR")
# pod parts mirrored in X on the rear pods: the ab/ad end and the two end brackets swap ends
END_PREFIX = ("End_Bracket", "Hip_Trunnion", "Hip_HW_trunnion", "Abad_", "Hip_Bearing_front", "Hip_Posts", "Hip_HW_post")
# cradle parts the body replaces
DROP_PREFIX = ("Hip_Outer_Plate", "Hip_Inner_Plate", "Hip_Front_Plate", "Hip_Spine", "Hip_REF_MGN12H", "Hip_HW_block_screws",
               "Hip_HW_rail_screws", "Stand_", "REF_MGN12H", "Rail_Carriage", "Mount_HW", "Carriage_HW")


def pod_tf(pod, end=False):
    """(M, t): pod-local -> body. M is a mirror (diagonal +-1): Y for the right pods, X for the ab/ad end
    of the rear pods; otherwise the pod is only moved."""
    mx = -1.0 if (pod[1] == "R" and end) else 1.0
    my = -1.0 if pod[0] == "R" else 1.0
    dx = DX if pod[1] == "F" else -DX
    return np.diag([mx, my, 1.0]), np.array([dx, my * DY, 0.0])


def place(shape, pod, end=False):
    """A pod-local solid in the body frame (mirrored where the pod is)."""
    M, t = pod_tf(pod, end)
    if M[0, 0] < 0:
        shape = mirror(shape, about=Plane.YZ)
    if M[1, 1] < 0:
        shape = mirror(shape, about=Plane.XZ)
    return Pos(*t) * shape


# ------------------------------------------------------------------------------------------------
# 2D helpers (profiles drawn in a plate's own plane)
# ------------------------------------------------------------------------------------------------
YZ = lambda x0: Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))   # (u, v) = (y, z), normal +X


def rounded_poly(pts, r):
    """Polygon with its corners rounded to r."""
    f = Polygon(*pts, align=None)
    return offset(offset(f, -r, kind=Kind.ARC), r, kind=Kind.ARC)


def rrect(u0, u1, v0, v1, r):
    return rounded_poly([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], r)


def x_truss(u0, u1, w, rib, r=4.0):
    """Stanley's X-truss: the rectangle u0..u1 x -w..w split by its two diagonals into four triangles, each
    window inset by half a rib from the rectangle's edges and the diagonals, corners rounded to r. v4: all
    four windows (v3 built them corner-first and dropped the two that failed: half the X was missing)."""
    um = (u0 + u1) / 2
    c = (um, 0.0)
    corners = [(u0, -w), (u1, -w), (u1, w), (u0, w)]
    out = []
    for i in range(4):
        tri = Polygon(corners[i], corners[(i + 1) % 4], c, align=None)
        try:
            win = offset(tri, -rib / 2, kind=Kind.INTERSECTION)
            win = offset(offset(win, -r, kind=Kind.ARC), r, kind=Kind.ARC)
            if win.area > 4 * r * r:
                out.append(win)
        except Exception as e:
            raise RuntimeError(f"x_truss window {i} in {u0:.0f}..{u1:.0f} x +-{w:.0f}: {e}")
    return out


def yz_slab(face, x0, x1):
    """A (y, z) face extruded along body X from x0 to x1."""
    return YZ(x0) * extrude(face, x1 - x0)


def outline_faces(solid, xm):
    """Outer outlines (holes filled) of a plate perpendicular to X, as (y, z) faces."""
    out = []
    for f in (solid & Pos(xm, 0, 0) * Box(0.02, 2000, 2000)).faces():
        if abs(f.normal_at().X) > 0.9:
            out.append(YZ(f.center().X).inverse() * Face(f.outer_wire()))
    return out


# ------------------------------------------------------------------------------------------------
# Bulkheads
# ------------------------------------------------------------------------------------------------
BOSS_YS = (-86.0, 0.0, 86.0)           # M3 insert bosses on each bulkhead's top and bottom edges (plates)
TAB_DY = 22.0                          # a tab beside each boss keys into the plate's slot


def tab_dy(y):
    """The tab sits inboard of its boss (the middle one: to the left)."""
    return -TAB_DY if y > 0 else TAB_DY


def cradle_plates():
    _, fx = g.ab_parts()
    return {"outer": fx["Hip_Outer_Plate"][0], "inner": fx["Hip_Inner_Plate"][0], "front": fx["Hip_Front_Plate"][0]}


def bulkhead_x(kind, front):
    x0, x1 = {"outer": g.AB_X["outer"], "inner": g.AB_X["inner"], "front": g.HF_X}[kind]
    return (x0 + DX, x1 + DX) if front else (-x1 - DX, -x0 - DX)


def bulkhead(kind, front, plates):
    """A frame from the bottom plate to the top plate across the body, windowed (hexagon in the middle,
    rounded ones round it), merged with the left and right pods' cradle plates (bearing housings, motor C's
    mount, the post lobes)."""
    pods = ("LF", "RF") if front else ("LR", "RR")
    xa, xb = bulkhead_x(kind, front)
    xm = (xa + xb) / 2
    wide = BULK_W if kind != "front" else Y_AX + 30.0
    b = yz_slab(rrect(-wide, wide, Z_BOT, Z_TOP, BODY_R if kind != "front" else 52.0), xa, xb)
    # each pod's cradle plate kept to its own side of the centreline: v4's pods are close enough that one
    # plate's lobes reach across and would fill the other pod's bearing bore and drum clearance
    crad = [place(plates[kind], p, end=True) & box(-2000.0, 2000.0, 0.0 if p[0] == "L" else -2000.0,
                                                   2000.0 if p[0] == "L" else 0.0, -2000.0, 2000.0) for p in pods]
    outl = [f for c in crad for f in outline_faces(c, xm)]
    for f in outl:
        b -= yz_slab(f, xa - 0.01, xb + 0.01)
    for c in crad:
        b += c
    if kind == "outer":                                            # motor C's phase-wire exit: keep v6.2's open bay
        motor = g.ab_parts()[1]["Abad_Motor"][0]
        for p in pods:
            for s in place(motor, p, end=True).solids():
                x = s & b
                if x is not None and x.volume > 0.1:
                    k = s.bounding_box()
                    b -= box(xa - 1, xb + 1, k.min.Y - 1.5, k.max.Y + 1.5, k.min.Z - 1.5, k.max.Z + 1.5)
    # windows: everything inside a 14 mm border except Stanley's X of ribs and 8 mm round the cradle
    # outlines (bearing housings, motor C's mount), corners rounded
    win = rrect(-wide + 14.0, wide - 14.0, Z_BOT + 14.0, Z_TOP - 14.0, 10.0)
    zc = g.AB_Z
    ribs = [rrect(-8.0, 8.0, Z_BOT, Z_TOP, 1.0), rrect(-wide, wide, zc - 8.0, zc + 8.0, 1.0)]
    for (ya, za), (yb, zb) in (((-wide, Z_TOP), (wide, Z_BOT)), ((-wide, Z_BOT), (wide, Z_TOP))):
        L = math.hypot(yb - ya, zb - za)
        ny, nz = -(zb - za) / L * 7.0, (yb - ya) / L * 7.0
        ribs.append(Polygon((ya + ny, za + nz), (yb + ny, zb + nz), (yb - ny, zb - nz), (ya - ny, za - nz), align=None))
    for y in BOSS_YS:                                              # round the edge bosses and tabs
        ya, yb = min(y - 12.0, y + tab_dy(y) - 9.0), max(y + 12.0, y + tab_dy(y) + 9.0)
        ribs.append(rrect(ya, yb, Z_TOP - 22.0, Z_TOP, 1.0))
        ribs.append(rrect(ya, yb, Z_BOT, Z_BOT + 22.0, 1.0))
    for k in outl:
        win = win - offset(k, 8.0, kind=Kind.ARC)
    for r_ in ribs:
        win = win - r_
    try:
        win = offset(offset(win, -5.0, kind=Kind.ARC), 5.0, kind=Kind.ARC)
    except Exception:
        pass
    for f in win.faces():
        if f.area > 150.0:
            b -= yz_slab(f, xa - 1, xb + 1)
    if kind == "outer":                                            # v3: the lid's thumb screws, into the centre bulkheads' tops
        for _, y in LID_PTS[:2]:
            b += Pos(xm, y, Z_TOP - 12.0) * Cylinder(5.5, 12.0, align=MIN)
            b -= Pos(xm, y, Z_TOP - 6.0) * Cylinder(g.M3_INSERT / 2, 6.01, align=MIN)
    for y in BOSS_YS:                                              # edge bosses (M3 inserts) and tabs
        for z0, top in ((Z_TOP, True), (Z_BOT, False)):
            b += Pos(xm, y, z0 - 12.0 if top else z0) * Cylinder(5.5, 12.0, align=MIN)
            b -= Pos(xm, y, z0 - 6.0 if top else z0 - 0.01) * Cylinder(g.M3_INSERT / 2, 6.01, align=MIN)
            if top or abs(y) + 6 <= BOT_W:
                b += Pos(xm, y + tab_dy(y), z0 if top else z0 - PLATE_T) * Box(xb - xa, 14.0, PLATE_T, align=MIN)
    return b


# ------------------------------------------------------------------------------------------------
# Plates
# ------------------------------------------------------------------------------------------------
def bulkhead_xs():
    return sorted(bulkhead_x(k, fr) for fr in (True, False) for k in ("front", "inner", "outer"))


def laser_ranges(xa, xb):
    """X ranges of a bulkhead's two laser plates (A at low X, then B). Two 4 mm plates; where the bulkhead
    is thinner than 8 mm, a 4 mm and a 3 mm plate flush with its outboard face (motor C seats there) and
    the 0.5 mm left over on the bay side."""
    if xb - xa >= 2 * LASER_T - 0.01:
        return (xa, xa + LASER_T), (xa + LASER_T, xa + 2 * LASER_T)
    t2 = LASER_T_THIN
    if abs(xb) > abs(xa):
        return (xb - t2 - LASER_T, xb - t2), (xb - t2, xb)
    return (xa, xa + t2), (xa + t2, xa + t2 + LASER_T)


def screw_x(xa, xb, laser):
    """X of the plate screws into a bulkhead: its middle (printed: inserts), or the middle of its first
    plate A (laser: T-slot nuts in A)."""
    return sum(laser_ranges(xa, xb)[0]) / 2 if laser else (xa + xb) / 2


def window_x(xa, xb):
    """Laser build: X of the middle of plate B (the windows over the T-slot nuts)."""
    return sum(laser_ranges(xa, xb)[1]) / 2


def plate(top, laser=True):
    PLATE_T = globals()["PLATE_T"] if laser else PRINT_PLATE_T
    xs = bulkhead_xs()
    x_end = max(x1 for _, x1 in xs) + 4.0
    w = TOP_W if top else BOT_W
    p = extrude(rrect(-x_end, x_end, -w, w, BODY_R), PLATE_T)
    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):                      # X-truss windows between the bulkheads
        u0, u1 = a1 + 6.0, b0 - 6.0
        if u1 - u0 < 60.0:                                          # the short bays between bulkheads: solid
            continue
        wt = w - 14.0 if top else w - DRUM_NOTCH[1] - 8.0              # bottom: clear of the drum notches
        for wnd in x_truss(u0, u1, wt, 10.0):
            p -= extrude(wnd, PLATE_T)
    for xa, xb in xs:                                              # slots for the tabs, holes for the screws
        xm = screw_x(xa, xb, laser)
        for y in BOSS_YS:
            if abs(y) + 6 > w:
                continue
            (r0, _), (_, r1) = laser_ranges(xa, xb) if laser else ((xa, 0), (0, xb))
            p -= Pos((r0 + r1) / 2, y + tab_dy(y), 0) * Box(r1 - r0 + 0.3, 14.3, PLATE_T, align=MIN)
            p -= Pos(xm, y, 0) * Cylinder(g.M3_CLR / 2, PLATE_T, align=MIN)
            cs = Cone(1.7, 3.2, 1.7, align=MIN)
            p -= (Pos(xm, y, PLATE_T - 1.7) * cs) if top else (Pos(xm, y, 1.7) * Rot(180, 0, 0) * cs)
    if not top:                                                    # v8: notches under each pod's drums, whose rims sweep
        for xc in (DX, -DX):                                       # a full circle (+-90 deg) and dip below the plate's
            for sy in (-1, 1):                                     # top at 20 deg of ab/ad inward
                y0, y1 = sorted((sy * (w - DRUM_NOTCH[1]), sy * (w + 20.0)))
                p -= extrude(rrect(xc - DRUM_NOTCH[0] / 2, xc + DRUM_NOTCH[0] / 2, y0, y1, 12.0), PLATE_T)
        for x, y in DOCK_SCREWS:                                   # v3: the cassette dock, from below
            p -= Pos(x, y, 0) * Cylinder(g.M3_CLR / 2, PLATE_T, align=MIN)
            p -= Pos(x, y, 1.7) * Rot(180, 0, 0) * Cone(1.7, 3.2, 1.7, align=MIN)
    else:                                                          # v3: the cassette's opening and the lid's screws
        p -= extrude(rrect(-OPEN[0], OPEN[0], -OPEN[1], OPEN[1], 6.0), PLATE_T)
        for x, y in LID_PTS:
            p -= Pos(x, y, 0) * Cylinder(g.M3_CLR / 2, PLATE_T, align=MIN)
        p -= Pos(*ESTOP, 0) * Cylinder(22.3 / 2, PLATE_T, align=MIN)          # e-stop, fixed (not on the lid)
    return Pos(0, 0, Z_TOP) * p if top else Pos(0, 0, Z_BOT - PLATE_T) * p


def bulkhead_plane_faces(bulk, x):
    """(y, z) faces of a bulkhead's section at x (holes kept)."""
    out = []
    for f in (bulk & Pos(x, 0, 0) * Box(0.02, 2000, 2000)).faces():
        if abs(f.normal_at().X) > 0.9:
            out.append(YZ(f.center().X).inverse() * f)
    return out


LASER_BOLTS_Z = (Z_BOT + 7.0, g.AB_Z, Z_TOP - 7.0)                 # M3 bolts holding each pair of plates together


def laser_pair(kind, front, bulk):
    """Laser build of a bulkhead: two 4 mm plates cut from the printed bulkhead's sections, the first
    (A, at the low-X face) through its bearing seats, the second (B) through the lips; together they
    are the printed part's 8 mm. A has a T-slot (screw slot + nut pocket) at every plate screw on its top
    and bottom edges; B a window over each nut. Six M3 bolts hold A to B."""
    xa, xb = bulkhead_x(kind, front)
    out = []
    for i, (r0, r1) in enumerate(laser_ranges(xa, xb)):
        sl = None                                                  # material at every depth of the plate (pockets
        for x in (r0 + 0.15, (r0 + r1) / 2, r1 - 0.15):              # and recesses cut through)
            cur = None
            for f in bulkhead_plane_faces(bulk, x):
                e = YZ(r0) * extrude(f, r1 - r0, dir=(0, 0, 1))
                cur = e if cur is None else cur + e
            sl = cur if sl is None else sl & cur
        xs, xw = screw_x(xa, xb, True), window_x(xa, xb)
        for y in BOSS_YS:
            for z0, sg in ((Z_TOP, -1.0), (Z_BOT, 1.0)):
                if sg > 0 and abs(y) + 6 > BOT_W:
                    continue
                zn = z0 + sg * 6.0
                if i == 0:
                    sl -= Pos(xs, y, min(z0, z0 + sg * 12.0)) * Box(LASER_T + 1, 3.4, 12.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                sl -= Pos(xs if i == 0 else xw, y, zn - 1.3) * Box(LASER_T + 1, 5.8, 2.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
        wide = BULK_W if kind != "front" else Y_AX + 30.0
        for y in (-wide + 7.0, 0.0, wide - 7.0):
            for z in LASER_BOLTS_Z:
                if y == 0.0 and z == g.AB_Z:
                    continue
                sl -= Pos(xa - 1, y, z) * Rot(0, 90, 0) * Cylinder(g.M3_CLR / 2, 2 * LASER_T + 2, align=MIN)
        out.append(sl)
    return out


def printed_plate_pieces(top):
    """Printable build: the top (bottom) plate in three pieces that fit the bed, half-lapped 20 mm over
    the inner bulkheads, where the plate screws pass through both laps."""
    p = plate(top, laser=False)
    t = PRINT_PLATE_T
    z0 = Z_TOP if top else Z_BOT - t
    cuts = sorted((xa + xb) / 2 for xa, xb in (bulkhead_x("inner", True), bulkhead_x("inner", False)))
    big = 2000.0
    pieces = []
    edges = [-big] + cuts + [big]
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        keep = box(lo + (10.0 if i > 0 else 0.0), hi - (10.0 if i < 2 else 0.0), -big, big, z0 - 1, z0 + t + 1)
        if i > 0:                                                  # upper half of the lap on its low side
            keep += box(lo - 10.0, lo + 10.0, -big, big, z0 + t / 2, z0 + t + 1)
        if i < 2:                                                  # lower half on its high side
            keep += box(hi - 10.0, hi + 10.0, -big, big, z0 - 1, z0 + t / 2)
        pieces.append(p & keep)
    return pieces


LAP_Z = (Z_BOT + 20.0, g.AB_Z, Z_TOP - 20.0)   # v3: the halves' joint screws (clear of the edge bosses' inserts)


def bay_face(kind, front):
    """x of a bulkhead's face toward the middle of the body, and its outward normal's sign."""
    xa, xb = bulkhead_x(kind, front)
    return (xa, -1.0) if front else (xb, 1.0)


def split_halves(bulk, kind, front):
    """Printable build: a bulkhead too wide for the bed (~386 mm with its bearing lobes), in left and right
    halves half-lapped 24 mm over its centre rib: 3x M3 x 12 countersunk, flush on the bay face (the
    cassette passes 2 mm away), into M3 nylon nuts on the outboard face, each in an 11 mm pocket for the
    socket, on the bench (the laps are 4 mm: too thin for inserts)."""
    xa, xb = bulkhead_x(kind, front)
    xm = (xa + xb) / 2
    xf, sx = bay_face(kind, front)
    xo = xb if xf == xa else xa                                                       # the outboard face
    big = 2000.0
    halves = []
    for s_ in (-1, 1):
        keep = box(-big, big, min(s_ * 12.0, s_ * big), max(s_ * 12.0, s_ * big), -big, big)
        lap = box(xa - 10 if s_ < 0 else xm, xm if s_ < 0 else xb + 10, -12.0, 12.0, -big, big)
        h = bulk & (keep + lap)
        for z in LAP_Z:
            h -= Pos(xa - 12, 0, z) * Rot(0, 90, 0) * Cylinder(g.M3_CLR / 2, xb - xa + 24, align=MIN)
            h -= Location(Plane(origin=(xf, 0, z), x_dir=(0, 1, 0), z_dir=(-sx, 0, 0))) * Cone(3.2, 1.7, 1.7, align=MIN)
            # the nut's side: a socket's pocket down to the lap's outboard face (the centre bulkheads carry
            # bosses up to 6.5 mm proud of it there; v3 first cut put the nut inside them)
            h -= Location(Plane(origin=(xo, 0, z), x_dir=(0, 1, 0), z_dir=(-sx, 0, 0))) * Cylinder(5.5, 20.0, align=MIN)
        halves.append(h)
    return halves


def plate_dxf(solid, path, normal="Z"):
    """2D outline (holes included) of a flat plate for the laser cutter."""
    bb = solid.bounding_box()
    if normal == "Z":
        zc = (bb.min.Z + bb.max.Z) / 2
        faces = [f for f in (solid & Pos(0, 0, zc) * Box(4000, 4000, 0.02)).faces() if abs(f.normal_at().Z) > 0.9]
        faces = [Location((0, 0, -f.center().Z)) * f for f in faces]
    elif normal == "X":
        xc = (bb.min.X + bb.max.X) / 2
        faces = [YZ(f.center().X).inverse() * f for f in (solid & Pos(xc, 0, 0) * Box(0.02, 4000, 4000)).faces()
                 if abs(f.normal_at().X) > 0.9]
    else:
        yc = (bb.min.Y + bb.max.Y) / 2
        faces = [Location(Plane(origin=(0, f.center().Y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))).inverse() * f
                 for f in (solid & Pos(0, yc, 0) * Box(4000, 0.02, 4000)).faces() if abs(f.normal_at().Y) > 0.9]
    seen = set()
    dxf = ExportDXF(unit=Unit.MM)
    for f in faces:
        key = round(f.area, 1)
        if key in seen:
            continue
        seen.add(key)
        dxf.add_shape(f)
    dxf.write(path)


# ------------------------------------------------------------------------------------------------
# v3 electronics: a top-loading cassette in the centre bay. mjbots power_dist on its floor (XT30s to the
# legs low down), Pi 5 + pi3hat on the middle shelf, the 6S battery in a tray at the top, right under the
# lid: battery swap = lid off (4 thumb screws), unplug, strap off, lift (no tools, the rack stays in).
# The rack drops straight down through the middle of the bay (|y| < 58: the moteus C boards stand at
# |y| 62..127), sits in a dock screwed to the bottom plate, and the lid holds it down. Between the hat and
# the tray, ~100 mm of open height: the service loop of the leg harness, and the air path (bottom plate
# windows -> open rack -> lid vents). A latching e-stop is fixed in the top plate beside the lid.
# ------------------------------------------------------------------------------------------------
CAS_X, CAS_Y = 30.0, 58.0              # cassette half sizes (the printed centre bulkheads' bay faces at +-33.2)
CAS_Z0 = Z_BOT + 4.0                   # its floor sits on the dock's floor
CAS_TOP = Z_TOP - 8.0                  # top of its pillars, under the lid's ribs
PACK6 = (46.0, 108.0, 40.0)            # 6S pack envelope it takes, x y z (e.g. 6S 1800 mAh 100C+, about 35 x 45 x 105)
POST_R = 4.5                           # board posts: 2.5 mm of plastic round an M2.5 insert (v3 first cut: 3.0, 1.2 mm)
PD = (54.0, 62.0)                      # mjbots power_dist footprint (approx.), x y
PD_Z = CAS_Z0 + 3.0 + 6.0              # on the floor's 6 mm posts
PD_DY = 3.0                            # power board moved 3 mm to +Y: its posts' access holes clear the Pi's posts
PD_POSTS = [(sx * 23.0, sy * 26.0 + PD_DY) for sx in (-1, 1) for sy in (-1, 1)]
PD_ACCESS = (3.5, 3.0)                 # radii of the access holes over those posts: Pi shelf (soldering iron), tray (hex key)
PI_SHELF = PD_Z + 1.6 + 14.0           # middle shelf's underside: over the power board's XT90
PI_Z = PI_SHELF + 3.0 + 3.0            # Pi underside: on the shelf's 3 mm posts
PI_DY = -10.0                          # Pi + hat moved toward the cassette's middle: ports clear of the corner pillars
PI_HOLES = [(x, y + PI_DY) for x in (-24.5, 24.5) for y in (-29.0, 29.0)]
TRAY_Z = CAS_TOP - 2.0 - PACK6[2] - 3.0  # battery tray's underside: the pack's top 2 mm under the pillars' tops
LID = (44.0, 72.0, 4.0)                # printed lid: half x, half y, thickness
LID_PTS = [(sx * 38.75, sy * 52.0) for sx in (-1, 1) for sy in (-1, 1)]   # thumb screws into the centre bulkheads' tops
OPEN = (32.5, 60.0)                    # top plate's opening for the cassette, half x / half y
ESTOP = (0.0, 92.0)                    # 22 mm latching e-stop in the top plate (x, y), over the bay's centre strip
DOCK_SCREWS = [(sx * 20.0, sy * 48.0) for sx in (-1, 1) for sy in (-1, 1)]


def box(xa, xb, ya, yb, za, zb):
    return Pos(xa, ya, za) * Box(xb - xa, yb - ya, zb - za, align=BOX0)


def dock():
    """Printed tray on the bottom plate: a 4 mm floor and 10 mm walls 0.5 mm round the cassette's foot;
    4x M3 up through the plate into its inserts."""
    w, t = 2.0, 4.0
    d = box(-CAS_X - 0.5 - w, CAS_X + 0.5 + w, -CAS_Y - 0.5 - w, CAS_Y + 0.5 + w, Z_BOT, Z_BOT + t + 10.0)
    d -= box(-CAS_X - 0.5, CAS_X + 0.5, -CAS_Y - 0.5, CAS_Y + 0.5, Z_BOT + t, Z_BOT + t + 11.0)
    d -= box(-14.0, 14.0, -40.0, 40.0, Z_BOT - 1, Z_BOT + t + 1)                  # wiring / lightening window
    for x, y in DOCK_SCREWS:
        d -= Pos(x, y, Z_BOT - 0.01) * Cylinder(g.M3_INSERT / 2, t + 0.02, align=MIN)
    return d


def cassette():
    """Printed open rack: floor with the power board's posts, four corner pillars, the Pi shelf, and the
    battery tray at the top (end stops with finger slots: the rack's handles), strap slots in the tray.
    Every board on M2.5 heat-set inserts in 4.5 mm posts."""
    c = box(-CAS_X, CAS_X, -CAS_Y, CAS_Y, CAS_Z0, CAS_Z0 + 3.0)                     # floor
    c -= box(-16.0, 16.0, -42.0, 42.0, CAS_Z0 - 1, CAS_Z0 + 4)                       # floor window (air in)
    for x, y in PD_POSTS:                                                           # power board posts (6 mm)
        c += Pos(x, y, CAS_Z0 + 3.0) * Cylinder(POST_R, 6.0, align=MIN)
        c -= Pos(x, y, PD_Z - 5.0) * Cylinder(g.M2_5_INSERT / 2, 5.01, align=MIN)
    for sx in (-1, 1):                                                              # corner pillars
        for sy in (-1, 1):
            c += Pos(sx * (CAS_X - 3.0), sy * (CAS_Y - 3.0), CAS_Z0) * Box(6.0, 6.0, CAS_TOP - CAS_Z0, align=MIN)
    sh = box(-CAS_X, CAS_X, -CAS_Y, CAS_Y, PI_SHELF, PI_SHELF + 3.0)                 # Pi shelf
    sh -= box(-CAS_X + 9.0, CAS_X - 9.0, -CAS_Y + 9.0, -45.0, PI_SHELF - 1, PI_SHELF + 4)     # wiring windows
    sh -= box(-CAS_X + 9.0, CAS_X - 9.0, 25.0, CAS_Y - 9.0, PI_SHELF - 1, PI_SHELF + 4)
    sh -= box(-15.0, 15.0, -30.0, 12.0, PI_SHELF - 1, PI_SHELF + 4)                  # air up past the Pi
    for x, y in PD_POSTS:                                                           # the power board's inserts and screws
        sh -= Pos(x, y, PI_SHELF - 1) * Cylinder(PD_ACCESS[0], 5.0, align=MIN)          # go in from above, through
    c += sh                                                                         # the shelf and the tray
    for x, y in PI_HOLES:                                                           # Pi posts (3 mm) with inserts
        c += Pos(x, y, PI_SHELF + 3.0) * Cylinder(POST_R, 3.0, align=MIN)
        c -= Pos(x, y, PI_Z - 6.0) * Cylinder(g.M2_5_INSERT / 2, 6.01, align=MIN)
    t = box(-CAS_X, CAS_X, -CAS_Y, CAS_Y, TRAY_Z, TRAY_Z + 3.0)                      # battery tray
    t -= box(-14.0, 14.0, -40.0, 40.0, TRAY_Z - 1, TRAY_Z + 4)                       # air / plugs through
    for x, y in PD_POSTS:
        t -= Pos(x, y, TRAY_Z - 1) * Cylinder(PD_ACCESS[1], 5.0, align=MIN)
    for x in (-CAS_X + 3.0, CAS_X - 6.0):
        t -= box(x, x + 3.0, -12.0, 12.0, TRAY_Z - 1, TRAY_Z + 4)                     # hook-and-loop strap
    for sy in (-1, 1):                                                              # end stops = handles
        a, b = sorted((sy * (PACK6[1] / 2 + 0.5), sy * (PACK6[1] / 2 + 3.5)))
        e = box(-CAS_X + 6.0, CAS_X - 6.0, a, b, TRAY_Z + 3.0, TRAY_Z + 3.0 + 26.0)
        e -= box(-13.0, 13.0, a - 1, b + 1, TRAY_Z + 3.0 + 12.0, TRAY_Z + 3.0 + 21.0)   # finger slot
        t += e
    return c + t


def lid():
    """Printed lid over the top plate's opening; 4 thumb screws into inserts in the centre bulkheads'
    top edges; two ribs under it hold the cassette's pillars down; vent slots over the pack's sides."""
    lx, ly, t = LID
    L = Pos(0, 0, Z_TOP + PLATE_T) * extrude(rrect(-lx, lx, -ly, ly, 10.0), t)
    for sy in (-1, 1):                                                              # ribs press the pillars down
        a, b = sorted((sy * (CAS_Y - 6.0), sy * CAS_Y))
        L += box(-CAS_X, CAS_X, a, b, CAS_TOP + 0.5, Z_TOP + PLATE_T)
    for sx in (-1, 1):                                                              # vents over the pack's sides
        for y0 in (-46.0, -14.0, 18.0):
            L -= Pos(sx * 26.5, y0 + 14.0, Z_TOP + PLATE_T - 1) * extrude(rrect(-2.5, 2.5, -14.0, 14.0, 2.4), LID[2] + 2)
    for x, y in LID_PTS:
        L -= Pos(x, y, Z_TOP) * Cylinder(g.M3_CLR / 2, PLATE_T + t + 1, align=MIN)
    return L


def estop():
    """22 mm latching mushroom e-stop (envelope): Ø40 head over the top plate, its ring nut under it, the
    contact block below (wired to the power board's switch input; a plug at the cassette)."""
    x, y = ESTOP
    z = Z_TOP + PLATE_T
    e = Pos(x, y, z) * (Cylinder(14.0, 8.0, align=MIN) + Pos(0, 0, 8.0) * Cylinder(20.0, 12.0, align=MIN))
    e += Pos(x, y, Z_TOP - 4.0) * Cylinder(11.0, 4.0 + PLATE_T, align=MIN)
    e += Pos(x, y, Z_TOP - 4.0) * Box(30.0, 26.0, 36.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
    return e


def pack6():
    b = box(-PACK6[0] / 2, PACK6[0] / 2, -PACK6[1] / 2, PACK6[1] / 2, TRAY_Z + 3.0, TRAY_Z + 3.0 + PACK6[2])
    try:
        b = b.fillet(3.0, b.edges())
    except Exception:
        pass
    return b


def pi5():
    """Raspberry Pi 5 flat on the middle shelf; ports at the +Y short edge."""
    b = Pos(0, PI_DY, PI_Z) * extrude(rrect(-28.0, 28.0, -32.5, 52.5, 3.0), 1.6)
    for x, y in PI_HOLES:
        b -= Pos(x, y, PI_Z - 0.1) * Cylinder(1.4, 2.0, align=MIN)
    z = PI_Z + 1.6
    ports = Pos(0, PI_DY, 0) * (box(-26.0, -13.0, 36.0, 54.5, z, z + 16.0) + box(-9.0, 4.0, 36.0, 54.5, z, z + 16.0)
                                + box(8.0, 24.0, 33.0, 54.5, z, z + 13.5))
    cooler = Pos(0, PI_DY, 0) * box(-14.0, 18.0, -16.0, 16.0, z, z + 9.0)
    return b, ports, cooler


def pi3hat():
    z0 = PI_Z + 1.6 + 12.0
    b = Pos(0, PI_DY, z0) * extrude(rrect(-28.0, 28.0, -32.5, 32.5, 3.0), 1.6)
    for x, y in PI_HOLES:
        b -= Pos(x, y, z0 - 0.1) * Cylinder(1.4, 2.0, align=MIN)
    jst = None
    for k in range(5):                                                                          # 4 CAN-FD + 1
        c = box(18.0, 24.0, -26.0 + 11.0 * k + PI_DY, -18.0 + 11.0 * k + PI_DY, z0 + 1.6, z0 + 7.6)
        jst = c if jst is None else jst + c
    return b, jst


def power_dist():
    b = Pos(0, PD_DY, PD_Z) * extrude(rrect(-PD[0] / 2, PD[0] / 2, -PD[1] / 2, PD[1] / 2, 3.0), 1.6)
    for x, y in PD_POSTS:
        b -= Pos(x, y, PD_Z - 0.1) * Cylinder(1.4, 2.0, align=MIN)
    xt = None
    for k in range(4):                                                                          # XT30 outputs
        c = box(-PD[0] / 2 + 2, -PD[0] / 2 + 12, -22.0 + 12.0 * k, -14.0 + 12.0 * k, PD_Z + 1.6, PD_Z + 9.6)
        xt = c if xt is None else xt + c
    xt += box(PD[0] / 2 - 18, PD[0] / 2 - 2, -12.0, 12.0, PD_Z + 1.6, PD_Z + 11.6)            # XT90 in
    return b, Pos(0, PD_DY, 0) * xt


def cs_screw(d, length):
    """Countersunk along +Z, head top at z 0."""
    return Pos(0, 0, -d * 0.6) * Cone(d / 2, d, d * 0.6, align=MIN) + Pos(0, 0, -length) * Cylinder(d / 2, length - d * 0.6, align=MIN)


def hardware(laser=False):
    """v3 (one build): every plate screw into a heat-set insert in a printed bulkhead's edge boss; the
    dock, the boards and the lid on inserts too. (label, solid, colour, fit kind, direction, length)."""
    hw = []
    up, down = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
    t = PRINT_PLATE_T
    for fr in (True, False):
        for kind in ("front", "inner", "outer"):
            xa, xb = bulkhead_x(kind, fr)
            xm = (xa + xb) / 2
            tag = f"{'F' if fr else 'R'}_{kind}"
            for i, y in enumerate(BOSS_YS):
                for top in (True, False):
                    if not top and abs(y) + 6 > BOT_W:
                        continue
                    w = "top" if top else "bot"
                    ins = Pos(xm, y, Z_TOP - 6.0) * g.insert(4.6, 2.6, 5.7) if top else Pos(xm, y, Z_BOT) * g.insert(4.6, 2.6, 5.7)
                    hw.append((f"ins_{w}_{tag}_{i}", ins, "brass", "insert", up if top else down, 5.7))
                    L = t + 5.0
                    scr = Pos(xm, y, Z_TOP + t) * cs_screw(3.0, L) if top else Pos(xm, y, Z_BOT - t) * Rot(180, 0, 0) * cs_screw(3.0, L)
                    hw.append((f"scr_{w}_{tag}_{i}", scr, "screw", "screw", up if top else down, L))
    for fr in (True, False):                                                                  # bulkhead halves' joints
        for kind in ("inner", "outer"):
            xa, xb = bulkhead_x(kind, fr)
            xf, sx = bay_face(kind, fr)
            xo = xb if xf == xa else xa                                                       # the outboard face
            tag = f"{'F' if fr else 'R'}_{kind}"
            nut = g.hex_prism_z(5.5, 4.0) - Cylinder(1.5, 4.0, align=MIN)
            for i, z in enumerate(LAP_Z):
                scr = Location(Plane(origin=(xf, 0, z), x_dir=(0, 1, 0), z_dir=(sx, 0, 0))) * cs_screw(3.0, 12.0)
                hw.append((f"lap_scr_{tag}_{i}", scr, "screw", "screw", (sx, 0.0, 0.0), 12.0))
                hw.append((f"lap_nut_{tag}_{i}", Location(Plane(origin=(xo, 0, z), x_dir=(0, 1, 0), z_dir=(-sx, 0, 0))) * nut,
                           "steel", "nut", (-sx, 0.0, 0.0), 4.0))
    for k, (x, y) in enumerate(DOCK_SCREWS):                                                  # dock, from below
        hw.append((f"dock_ins_{k}", Pos(x, y, Z_BOT) * g.insert(4.6, 2.6, 4.0), "brass", "insert", down, 4.0))
        hw.append((f"dock_scr_{k}", Pos(x, y, Z_BOT - t) * Rot(180, 0, 0) * cs_screw(3.0, t + 3.5), "screw", "screw", down, t + 3.5))
    for k, (x, y) in enumerate(PI_HOLES):                                                     # Pi 5 + pi3hat
        hw.append((f"pi_ins_{k}", Pos(x, y, PI_Z - 4.0) * g.insert(3.9, 2.1, 4.0), "brass", "insert", up, 4.0))   # flush
        so = g.hex_prism_z(5.0, 12.0) - Cylinder(1.05, 12.0, align=MIN) + Pos(0, 0, -5.0) * Cylinder(1.2, 5.0, align=MIN)
        hw.append((f"pi_so_{k}", Pos(x, y, PI_Z + 1.6) * so, "brass", "screw", up, 17.0))
        hw.append((f"hat_scr_{k}", Pos(x, y, PI_Z + 1.6 + 12.0 + 1.6) * g.shcs(2.5, 6.0, 4.5, 2.5), "screw", "screw", up, 6.0))
    for k, (x, y) in enumerate(PD_POSTS):                                                     # power board
        hw.append((f"pd_ins_{k}", Pos(x, y, PD_Z - 4.0) * g.insert(3.9, 2.1, 4.0), "brass", "insert", up, 4.0))   # flush
        hw.append((f"pd_scr_{k}", Pos(x, y, PD_Z + 1.6) * g.shcs(2.5, 6.0, 4.5, 2.5), "screw", "screw", up, 6.0))
    for k, (x, y) in enumerate(LID_PTS):                                                      # lid
        hw.append((f"lid_ins_{k}", Pos(x, y, Z_TOP - 6.0) * g.insert(4.6, 2.6, 5.7), "brass", "insert", up, 5.7))
        thumb = Cylinder(1.5, 12.0 + LID[2] + PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MAX)) + Cylinder(6.0, 6.0, align=MIN)
        hw.append((f"lid_scr_{k}", Pos(x, y, Z_TOP + PLATE_T + LID[2]) * thumb, "screw", "screw", up, 12.0 + LID[2] + PLATE_T))
    return hw


BULK_NAMES = (("front", "End_Frame"), ("inner", "Inner_Bulkhead"), ("outer", "Centre_Bulkhead"))


def parts(laser=False, bulks=None):
    """v3 body (one build), label -> (solid, colour, kind): printed bulkheads (the end frames whole, the
    inner and centre ones in halves) with inserts in their edge bosses, laser-cut 3 mm top and bottom
    plates screwed into them, the cassette dock, cassette and lid, and the electronics."""
    pl = cradle_plates()
    out = {}
    bulks = bulks or {}
    for fr in (True, False):
        for kind, name in BULK_NAMES:
            key = f"{name}_{'F' if fr else 'R'}"
            if key not in bulks:
                bulks[key] = bulkhead(kind, fr, pl)
            if kind == "front":
                out[key] = (bulks[key], "black", "part")
            else:
                l_, r_ = split_halves(bulks[key], kind, fr)
                out[f"{key}_Right"], out[f"{key}_Left"] = (l_, "black", "part"), (r_, "black", "part")
    out["Body_Plate_Top"] = (plate(True, laser=False), "cf", "part")
    out["Body_Plate_Bottom"] = (plate(False, laser=False), "cf", "part")
    out["Cassette_Dock"] = (dock(), "red", "part")
    out["Cassette"] = (cassette(), "red", "part")
    out["Lid"] = (lid(), "black", "part")
    out["REF_Pack_6S"] = (pack6(), "blue", "ref")
    out["REF_EStop"] = (estop(), "red", "ref")
    b, ports, cooler = pi5()
    out["REF_Pi5"], out["REF_Pi5_ports"], out["REF_Pi5_cooler"] = (b, "pcb", "ref"), (ports, "silver", "ref"), (cooler, "black", "ref")
    b, jst = pi3hat()
    out["REF_pi3hat"], out["REF_pi3hat_ports"] = (b, "purple", "ref"), (jst, "white", "ref")
    b, xt = power_dist()
    out["REF_power_dist"], out["REF_power_dist_xt"] = (b, "purple", "ref"), (xt, "yellow", "ref")
    return out


# ------------------------------------------------------------------------------------------------
# Film scene: body meshes, and per pod the leg module's meshes (mirrored copies where the pod is)
# ------------------------------------------------------------------------------------------------
def mirror_stl(src, dst, diag):
    d = open(src, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    rec = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])).copy()
    v = rec["v"].reshape(-1, 3, 3) * np.asarray(diag, np.float32)
    rec["v"] = v[:, [0, 2, 1], :].reshape(-1, 9)                     # a mirror image: reverse the winding
    rec["n"] = rec["n"] * np.asarray(diag, np.float32)
    with open(dst, "wb") as f:
        f.write(d[:84])
        f.write(rec.tobytes())


def pod_scene(mesh_dir, leg_dir):
    leg_meta = json.loads(open(os.path.join(leg_dir, "leg_abad_gantry.json")).read())
    meta = {}
    for pod in PODS:
        for k, v in leg_meta.items():
            if k.startswith(DROP_PREFIX):
                continue
            end = k.startswith(END_PREFIX)
            M, t = pod_tf(pod, end)
            diag = [float(M[0, 0]), float(M[1, 1]), 1.0]
            src = v.get("file", k)
            tag = ("x" if diag[0] < 0 else "") + ("y" if diag[1] < 0 else "")
            if tag:
                name = f"{src}__m{tag}"
                if not os.path.exists(os.path.join(mesh_dir, name + ".stl")):
                    mirror_stl(os.path.join(leg_dir, src + ".stl"), os.path.join(mesh_dir, name + ".stl"), diag)
                fpath = os.path.join(mesh_dir, name)
            else:
                fpath = os.path.join(leg_dir, src)
            meta[f"{pod}__{k}"] = dict(v, file=fpath, pod=pod, base=k, mirror=diag, shift=[float(x) for x in t],
                                       body_fixed=v.get("abad") == "fixed")
    return meta


def export_builds():
    """v3 (one build): printed/ (STL + STEP of every printed body part: bulkheads, dock, cassette, lid),
    laser_cut/ (DXF + STEP of the top and bottom plates), legs/ (the v9.1 4-bar links). Returns the parts."""
    for d in ("laser_cut/dxf", "laser_cut/step", "printed/stl", "printed/step", "legs/stl"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    ps = parts()
    for n, (s_, c, k) in ps.items():
        if k != "part":
            continue
        if n.startswith("Body_Plate"):
            export_step(s_, os.path.join(ROOT, "laser_cut", "step", f"{n}.step"))
            plate_dxf(s_, os.path.join(ROOT, "laser_cut", "dxf", f"{n}.dxf"), "Z")
            print(f"  laser  {n:26s} solids {len(s_.solids())}", flush=True)
            continue
        export_stl(s_, os.path.join(ROOT, "printed", "stl", f"{n}.stl"), tolerance=0.02)
        export_step(s_, os.path.join(ROOT, "printed", "step", f"{n}.step"))
        bb = s_.bounding_box()
        print(f"  print  {n:26s} solids {len(s_.solids())}  {s_.volume / 1000:6.1f} cm3  {bb.size.X:5.0f} x {bb.size.Y:5.0f} x {bb.size.Z:5.0f}",
              flush=True)
    for n, (solid, _, _) in g.leg4_parts(gantry=True).items():
        export_stl(solid, os.path.join(ROOT, "legs", "stl", f"{n}.stl"), tolerance=0.02)
    return ps


def main(mesh_dir=None, leg_dir=None):
    ps = export_builds()
    hw = hardware()
    print(f"quadruped: {len(ps)} body parts, {len(hw)} body hardware items", flush=True)
    if mesh_dir:
        os.makedirs(mesh_dir, exist_ok=True)
        meta = {}
        for n, (s, color, kind) in ps.items():
            export_stl(s, os.path.join(mesh_dir, f"Q_{n}.stl"), tolerance=0.05, angular_tolerance=0.2)
            meta[f"Q_{n}"] = {"file": os.path.join(mesh_dir, f"Q_{n}"), "color": color, "kind": kind}
        for n, s, color, kind, d, length in hw:
            export_stl(s, os.path.join(mesh_dir, f"Q_{n}.stl"), tolerance=0.05, angular_tolerance=0.2)
            meta[f"Q_{n}"] = {"file": os.path.join(mesh_dir, f"Q_{n}"), "color": color, "kind": "hw",
                              "fit": {"kind": kind, "dir": [float(v) for v in d], "len": float(length)}}
        export_stl(Pos(0, 0, Z_BOT - PLATE_T) * Box(260.0, 160.0, 600.0, align=(Align.CENTER, Align.CENTER, Align.MAX)),
                   os.path.join(mesh_dir, "Q_Build_Stand.stl"), tolerance=0.1)
        meta["Q_Build_Stand"] = {"file": os.path.join(mesh_dir, "Q_Build_Stand"), "color": "dimgray", "kind": "film"}
        if leg_dir:
            meta.update(pod_scene(mesh_dir, leg_dir))
        layout = {"y_ax": Y_AX, "dx": DX, "z_top": Z_TOP, "z_bot": Z_BOT, "plate_t": PLATE_T, "bay": BAY,
                  "cassette": [CAS_X, CAS_Y], "ab_limit": AB_LIMIT, "drop": list(DROP_PREFIX), "end": list(END_PREFIX)}
        with open(os.path.join(mesh_dir, "quad.json"), "w") as f:
            json.dump({"meta": meta, "layout": layout}, f, indent=1)
    return ps, hw


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, sys.argv[2] if len(sys.argv) > 2 else None)
