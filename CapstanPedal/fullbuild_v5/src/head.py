"""Capstan quadruped v5: the head. A printed box on the face end frame (the robot walks toward -X, so the face is
the -X end frame) carrying a vision computer and a stereo camera on a tilting cradle.

  Head_Mount    one printed part: a back plate on the end frame (4x M3 into inserts in new pads on the frame), two
                side walls that rise into the cradle's cheeks, a top web. Open at the bottom (ports, cables, air in)
                and at the front (the face plate). Inside: the Jetson Orin Nano dev kit's carrier board on four posts
                (its 86 x 58 corner holes, M2.5), fan forward, ports down; a 12 V buck converter on the left wall.
  Head_Face     a vented plate over the Jetson's fan, on four posts from the back plate.
  Camera_Cradle a shelf and back plate between the cheeks, on two M3 pivots. A lock screw per side goes through one
                of seven detent holes in each cheek: 0 to 90 deg down in 15 deg steps. The back plate is drilled for
                three cameras' rear threads: ZED Mini (2x M2, 56 mm, 6.5 mm up), RealSense D435/D435i (2x M3, 45 mm,
                12.5 up), D455 (2x M4, 95 mm, 13 up); a 1/4-20 slot in the shelf takes anything with a tripod thread.
                The cameras' ends stay open (USB-C at either end), and the cradle tilts in front of the face, so a
                camera looking straight down sees the ground ahead of the fore feet.

Built in a design frame at +X (x = distance from the body's centre, the face end frame's outer face at FACE) and
mirrored onto the -X end frame by `body()`.
"""
import math

import generate_quad as q
from build123d import Align, Axis, Box, Circle, Cone, Cylinder, Location, Plane, Pos, Rectangle, Rot, extrude, make_hull, mirror

g = q.g
MIN, box, rrect = q.MIN, q.box, q.rrect

FACE = -q.bulkhead_x("front", False)[0]      # 340.5: the face end frame's outer face (design frame: +X)
BP_T = 6.0                                    # back plate
X_IN = FACE + BP_T                            # its front face (the bay's back)
WALL_Y = (70.5, 76.5)                         # side walls / cheeks, |y|
Z0, Z1 = -30.0, 72.0                          # head bottom (the trunnions' 6806s reach z -34) and top (top plate: 75)
WEB_Z = 66.0                                  # top web's underside
X_FACE = (388.0, 391.0)                       # face plate (the Jetson's fan reaches x 383)
FACE_Y = 61.5                                 # face plate's half width (the cradle's tabs pass outside it)
PIVOT = (397.0, 40.0)                         # tilt axis (x, z), along Y: 6 mm in front of the face
R_LOCK = 24.0                                 # lock screw's radius from the pivot (15 deg apart: 2.9 mm between holes)
TILTS = (0, 15, 30, 45, 60, 75, 90)           # detents, deg down
TILT = 30.0                                   # the CAD's (and the simulation's) setting
TAB_Y = (62.5, 70.0)                          # cradle tabs, |y| (cheeks' inner faces at 70.5)
MOUNT_PTS = [(sy * 62.0, 58.0) for sy in (-1, 1)] + [(sy * 40.0, -24.0) for sy in (-1, 1)]    # (y, z) into the end frame
BOARD_X = X_IN + 8.0                          # carrier board's back face, on 8 mm posts
BOARD_Y, BOARD_Z = (-50.0, 50.0), (-18.0, 61.0)   # 100 x 79, ports along the bottom edge
JET_HOLES = [(sy * 43.0, 21.5 + sz * 29.0) for sy in (-1, 1) for sz in (-1, 1)]               # 86 x 58
FACE_POSTS = [(sy * 58.0, z) for sy in (-1, 1) for z in (-24.0, 30.0)]
BUCK = (350.0, 383.0, 60.5, 70.0, -8.0, 17.4)   # 12 V buck (33 x 25.4 x ~9 incl. parts) on the left wall, x0 x1 y0 y1 z0 z1
TIE_X = (356.0, 377.0)                         # its two cable ties, through slot pairs in the wall above and below it
# camera rear threads in the cradle's back plate (y, z above the shelf, hole d): from the makers' drawings / CAD
CAM_HOLES = {"ZED Mini": [(sy * 28.0, 6.5, 2.4) for sy in (-1, 1)],              # M2 x 0.4, 2 deep (Stereolabs ZEDM.step)
             "D435": [(sy * 22.5, 12.5, 3.4) for sy in (-1, 1)],                 # M3, 3 deep (D400 datasheet, fig. 10-9)
             "D455": [(sy * 47.5, 13.0, 4.5) for sy in (-1, 1)]}                 # M4, 4 deep (fig. 10-11)
ZED = (124.5, 30.25, 26.5, 63.0)
ZED_CB = 1.7                                    # counterbore of the ZED's screws in the cradle's 4 mm back plate                # ZED Mini: length, height (feet included), depth, baseline


def XZ(y0):
    """Location whose plane is (x, z) at y0, extruding toward -Y."""
    return Location(Plane(origin=(0, y0, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)))


def xz_slab(face, y0, y1):
    """An (x, z) face extruded along Y from y0 to y1."""
    a, b = sorted((y0, y1))
    return XZ(b) * extrude(face, b - a)


def hull(*faces):
    return make_hull([e for f in faces for e in f.edges()])


def along_x(x0, y, z, r, length):
    return Pos(x0, y, z) * Rot(0, 90, 0) * Cylinder(r, length, align=MIN)


def along_y(x, y0, z, r, length, sy):
    """Cylinder from y0 outward (sy = +-1) along Y."""
    return Pos(x, y0, z) * Rot(-90 if sy > 0 else 90, 0, 0) * Cylinder(r, length, align=MIN)


def detent(th):
    """Cheek hole for tilt th (deg down): where the cradle's lock insert comes to."""
    a = math.radians(th)
    return PIVOT[0] - R_LOCK * math.cos(a), PIVOT[1] + R_LOCK * math.sin(a)


def cheek_face():
    return hull(Pos(*PIVOT) * Circle(8.0), Pos(PIVOT[0], PIVOT[1] + R_LOCK) * Circle(6.0),
                Pos(385.0, 45.0) * Rectangle(12.0, 38.0))


# ------------------------------------------------------------------------------------------------
# Printed parts (design frame)
# ------------------------------------------------------------------------------------------------
def head_mount():
    w = WALL_Y[1]
    m = box(FACE, X_IN, -w, w, Z0, Z1)                                        # back plate
    m -= box(FACE - 1, FACE + 2.0, -6.5, 6.5, 61.0, Z1 + 1)                   # clear of the frame's top edge boss (1.5 proud)
    m += box(X_IN, X_FACE[1], -w, w, WEB_Z, Z1)                               # top web
    for sy in (-1, 1):                                                        # slots: the cradle's lock bosses swing up
        m -= box(PIVOT[0] - R_LOCK - 8.0, X_FACE[1] + 1, *sorted((sy * (TAB_Y[0] - 0.5), sy * (WALL_Y[0] + 0.01))), WEB_Z - 1, Z1 + 1)
    for sy in (-1, 1):
        m += box(X_IN, X_FACE[1], *sorted((sy * WALL_Y[0], sy * w)), Z0, Z1)    # side wall
        m += xz_slab(cheek_face(), sy * WALL_Y[0], sy * w)                     # cheek
        for th in TILTS:
            x, z = detent(th)
            m -= along_y(x, sy * (WALL_Y[0] - 1), z, g.M3_CLR / 2, 8.0, sy)
        m -= along_y(PIVOT[0], sy * (WALL_Y[0] - 1), PIVOT[1], g.M3_CLR / 2, 8.0, sy)
    m -= Pos(FACE - 1, 0, 0) * Rot(0, 90, 0) * extrude(rrect(-42.0, -4.0, -28.0, 28.0, 6.0), BP_T + 2)   # window behind the board
    for y, z in JET_HOLES:                                                    # the carrier board's posts
        m += along_x(X_IN - 0.01, y, z, 3.5, BOARD_X - X_IN + 0.01)
        m -= along_x(BOARD_X - 5.0, y, z, g.M2_5_INSERT / 2, 5.01)
    for y, z in FACE_POSTS:                                                   # the face plate's posts
        m += along_x(X_IN - 0.01, y, z, 4.0, X_FACE[0] - X_IN + 0.01)
        m -= along_x(X_FACE[0] - 6.0, y, z, g.M3_INSERT / 2, 6.01)
    for y, z in MOUNT_PTS:                                                    # to the end frame
        m -= along_x(FACE - 1, y, z, g.M3_CLR / 2, BP_T + 2)
    x0, x1, y0, y1, z0, z1 = BUCK                                             # tie slots for the buck converter
    for x in TIE_X:
        for zs in (z1 + 1.0, z0 - 3.0):
            m -= box(x - 2.0, x + 2.0, WALL_Y[0] - 1, WALL_Y[1] + 1, zs, zs + 2.0)
    for sy in (-1, 1):                                                        # strain relief: the camera's lead down the
        for zs in (-26.0, -16.0):                                             # outside of a wall, the Jetson's leads inside
            m -= box(384.0, 388.0, *sorted((sy * (WALL_Y[0] - 1), sy * (WALL_Y[1] + 1))), zs, zs + 2.0)
    return m


def head_face():
    f = box(X_FACE[0], X_FACE[1], -FACE_Y, FACE_Y, Z0, WEB_Z - 1.0)
    for k in range(-6, 7):                                                    # vents over the fan
        f -= box(X_FACE[0] - 1, X_FACE[1] + 1, k * 6.5 - 1.5, k * 6.5 + 1.5, 2.0, 58.0)
    for y, z in FACE_POSTS:                                                   # countersunk: the cradle's shelf passes 2 mm
        f -= along_x(X_FACE[0] - 1, y, z, g.M3_CLR / 2, 6.0)                  # in front at 90 deg
        f -= Pos(X_FACE[1] - 1.7, y, z) * Rot(0, 90, 0) * Cone(1.7, 3.2, 1.71, align=MIN)
    return f


def cradle_local():
    """The cradle in its own frame: the pivot at the origin, x forward (the camera's view at tilt 0), z up."""
    w = TAB_Y[0]
    c = box(0.0, 4.0, -w, w, -4.0, 26.0)                                      # back plate
    c += box(0.0, 26.0, -w, w, -4.0, 0.0)                                     # shelf
    tab = hull(Circle(7.0), Pos(-R_LOCK, 0) * Circle(6.0), Pos(2.0, 11.0) * Rectangle(4.0, 30.0))
    tab = tab & Pos(-10.0, 18.0) * Rectangle(60.0, 44.0)                     # flat with the shelf's underside: prints on it
    for sy in (-1, 1):
        c += xz_slab(tab, sy * (TAB_Y[0] - 0.01), sy * TAB_Y[1])
        for x in (0.0, -R_LOCK):                                              # pivot and lock inserts, from the outside
            c -= along_y(x, sy * (TAB_Y[1] - 6.0), 0.0, g.M3_INSERT / 2, 6.01, sy)
    for holes in CAM_HOLES.values():
        for y, z, d in holes:
            c -= along_x(-1.0, y, z, d / 2, 6.0)
    for y, z, d in CAM_HOLES["ZED Mini"]:                                     # its threads are 2 deep: M2 x 4 heads sit
        c -= along_x(-1.0, y, z, 2.1, 1.0 + ZED_CB)                           # 1.7 into the plate (1.7 of thread)
    c -= Pos(12.5, 0, -2.0) * Box(6.6, 50.0 + 6.6, 6.0)                       # 1/4-20 slot
    return c


def zed_local():
    """ZED Mini on the shelf (envelope from Stereolabs' ZEDM.step): body and its two lenses."""
    L, H, D, base = ZED
    b = Pos(4.0 + D / 2, 0, H / 2) * Box(D, L, H)
    b = b.fillet(8.0, b.edges().filter_by(Axis.X))                          # rounded ends, seen from the front
    for y, z, d in CAM_HOLES["ZED Mini"]:                                     # its rear M2 threads
        b -= along_x(4.0 - 0.01, y, z, 0.8, 2.5)
    lens = None
    for sy in (-1, 1):
        c = Pos(4.0 + D - 0.5, sy * base / 2, H / 2) * Rot(0, 90, 0) * Cylinder(6.0, 1.5, align=MIN)
        lens = c if lens is None else lens + c
    return b, lens


def to_cradle(shape, tilt=TILT):
    """Cradle frame -> design frame at a tilt (deg down)."""
    return Pos(PIVOT[0], 0, PIVOT[1]) * Rot(0, tilt, 0) * shape


def cradle_dir(v, tilt=TILT):
    a = math.radians(tilt)
    return (v[0] * math.cos(a) + v[2] * math.sin(a), v[1], -v[0] * math.sin(a) + v[2] * math.cos(a))


# ------------------------------------------------------------------------------------------------
# Reference parts (design frame)
# ------------------------------------------------------------------------------------------------
def jetson():
    """Jetson Orin Nano dev kit, its base removed: the carrier board (100 x 79), the module with its heatsink
    and fan (28.6 above the board's back), the port row along the bottom edge (NVIDIA SP-11324-001, fig. 4-1, 4-2)."""
    b = box(BOARD_X, BOARD_X + 1.6, *BOARD_Y, *BOARD_Z)
    for y, z in JET_HOLES:
        b -= along_x(BOARD_X - 0.1, y, z, 1.4, 2.0)
    x1 = BOARD_X + 1.6
    module = box(x1, BOARD_X + 28.6, -36.8, 33.4, 21.6, 60.4)
    ports = None
    for ya, yb, h in ((-50.0, -46.5, 11.0), (-38.0, -24.0, 13.0), (-20.0, -6.0, 15.5), (-4.0, 10.0, 15.5), (12.0, 28.0, 13.5),
                      (32.0, 40.0, 4.0)):
        p = box(x1, x1 + h, ya, yb, BOARD_Z[0] - 2.0, BOARD_Z[0] + 14.0)
        ports = p if ports is None else ports + p
    return b, module, ports


def buck():
    x0, x1, y0, y1, z0, z1 = BUCK
    return box(x0, x1, y0 + 6.0, y1, z0, z1) + box(x0 + 3.0, x0 + 14.0, y0, y0 + 6.0, z0 + 3.0, z1 - 3.0)


# ------------------------------------------------------------------------------------------------
# Body frame
# ------------------------------------------------------------------------------------------------
def body(shape):
    """Design frame (+X) -> body frame (the -X end)."""
    return mirror(shape, about=Plane.YZ)


def bdir(v):
    return (-v[0], v[1], v[2])


def parts(tilt=TILT):
    """label -> (solid, colour, kind), body frame."""
    out = {"Head_Mount": (body(head_mount()), "black", "part"), "Head_Face": (body(head_face()), "red", "part"),
           "Camera_Cradle": (body(to_cradle(cradle_local(), tilt)), "red", "part")}
    zb, zl = zed_local()
    out["REF_ZED_Mini"] = (body(to_cradle(zb, tilt)), "zed", "ref")
    out["REF_ZED_Mini_lenses"] = (body(to_cradle(zl, tilt)), "black", "ref")
    b, mod, ports = jetson()
    out["REF_Jetson_board"] = (body(b), "pcb", "ref")
    out["REF_Jetson_module"] = (body(mod), "jetson", "ref")
    out["REF_Jetson_ports"] = (body(ports), "port", "ref")
    out["REF_Buck_12V"] = (body(buck()), "buck", "ref")
    return out


def hardware(tilt=TILT):
    """(label, solid, colour, fit kind, approach direction, length), body frame (as generate_quad.hardware)."""
    hw = []
    px = (1.0, 0.0, 0.0)
    ins3 = g.insert(4.6, 2.6, 5.7)
    for i, (y, z) in enumerate(MOUNT_PTS):
        hw.append((f"hd_ins_{i}", Pos(FACE - 5.7, y, z) * Rot(0, 90, 0) * ins3, "brass", "insert", px, 5.7))
        hw.append((f"hd_scr_{i}", Pos(X_IN, y, z) * Rot(0, 90, 0) * g.shcs(3.0, 10.0), "screw", "screw", px, 10.0))
    for i, (y, z) in enumerate(JET_HOLES):
        hw.append((f"jt_ins_{i}", Pos(BOARD_X - 4.0, y, z) * Rot(0, 90, 0) * g.insert(3.9, 2.1, 4.0), "brass", "insert", px, 4.0))
        hw.append((f"jt_scr_{i}", Pos(BOARD_X + 1.6, y, z) * Rot(0, 90, 0) * g.shcs(2.5, 6.0, 4.5, 2.5), "screw", "screw", px, 6.0))
    for i, (y, z) in enumerate(FACE_POSTS):
        hw.append((f"fp_ins_{i}", Pos(X_FACE[0] - 5.7, y, z) * Rot(0, 90, 0) * ins3, "brass", "insert", px, 5.7))
        hw.append((f"fp_scr_{i}", Pos(X_FACE[1], y, z) * Rot(0, 90, 0) * q.cs_screw(3.0, 8.0), "screw", "screw", px, 8.0))
    for i, sy in enumerate((-1, 1)):
        rot = Rot(-90 if sy > 0 else 90, 0, 0)                                # local +z -> outward along Y
        for j, (x, z) in enumerate(((0.0, 0.0), (-R_LOCK, 0.0))):
            ins = Pos(x, sy * (TAB_Y[1] - 5.7), z) * rot * ins3
            hw.append((f"cr_ins_{i}{j}", to_cradle(ins, tilt), "brass", "insert", (0.0, sy, 0.0), 5.7))
        for nm, (x, z) in (("piv", PIVOT), ("lock", detent(tilt))):
            hw.append((f"cr_{nm}_{i}", Pos(x, sy * WALL_Y[1], z) * rot * g.shcs(3.0, 12.0), "screw", "screw", (0.0, sy, 0.0), 12.0))
    back = Rot(0, -90, 0)                                                     # local +z -> -x: heads behind the back plate
    for i, (y, z, _) in enumerate(CAM_HOLES["ZED Mini"]):
        s = Pos(ZED_CB, y, z) * back * g.shcs(2.0, 4.0)
        hw.append((f"cam_scr_{i}", to_cradle(s, tilt), "screw", "screw", cradle_dir((-1.0, 0.0, 0.0), tilt), 4.0))
    return [(n, body(s), c, k, bdir(d), L) for n, s, c, k, d, L in hw]


def frame_pads(b, xa, xb):
    """The face end frame (x xa..xb, outer face xa): solid pads with the head's four M3 inserts."""
    for y, z in MOUNT_PTS:
        b += Pos(xa, y, z) * Rot(0, 90, 0) * Cylinder(6.5, xb - xa, align=MIN)
        b -= Pos(xa - 0.01, y, z) * Rot(0, 90, 0) * Cylinder(g.M3_INSERT / 2, 6.01, align=MIN)
    return b


def layout(tilt=TILT):
    """What the simulation needs: the left lens (m, body frame), the tilt, the head's box for collisions."""
    L, H, D, base = ZED
    a = math.radians(tilt)
    x_, z_ = 4.0 + D, H / 2                                                   # lens centre, cradle frame
    lens = (PIVOT[0] + x_ * math.cos(a) + z_ * math.sin(a), -base / 2, PIVOT[1] - x_ * math.sin(a) + z_ * math.cos(a))
    return {"tilt": tilt, "tilts": list(TILTS), "pivot": [-PIVOT[0] / 1000.0, 0.0, PIVOT[1] / 1000.0],
            "lens": [-lens[0] / 1000.0, lens[1] / 1000.0, lens[2] / 1000.0],
            "box": [[-(PIVOT[0] + 42.0) / 1000.0, -WALL_Y[1] / 1000.0, Z0 / 1000.0], [-FACE / 1000.0, WALL_Y[1] / 1000.0, Z1 / 1000.0]]}
