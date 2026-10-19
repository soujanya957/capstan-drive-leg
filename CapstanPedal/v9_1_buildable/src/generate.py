"""
Coaxial capstan actuator, v6.2: the v6 family (eight builds) with a reinforced front wall, wider
sandwich shin, and 636ZZ knee bearing in the 4-bar leg.

Two EaglePower 8308 motors side by side, each with a moteus r4.11 behind it
(encoder on the motor axis, same stand-off as the v0 assembly), each driving
its own capstan stage onto a common output axis:

  motor A -> Small_Drum_A -> Big_Drum_A -> inner output (Output_Flange_A)
  motor B -> Small_Drum_B -> Big_Drum_B -> removable outer output flange

Global frame (mm):  X right, Y forward (motor axes / output axis), Z up.
Motor stator mounting face = Y 0 (front face of Back_Plate).

Run:  python generate.py        (needs: pip install build123d)
"""
import math
import os
from build123d import (
    Align, Axis, Box, Circle, CenterArc, Line, Rectangle, make_face, make_hull, Compound, Cone, Cylinder, Helix, Plane, Polygon,
    Pos, RegularPolygon, Rot, Solid, extrude, sweep, export_step, export_stl,
    import_step, Location, Color, Sphere, Kind, offset,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.normpath(os.path.join(ROOT, ".."))  # CapstanPedal/

MIN = (Align.CENTER, Align.CENTER, Align.MIN)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Motors (EaglePower 8308) - axes along Y
MOTOR_PITCH = 93.5                    # motor axis spacing (same as v0)
MA = (-MOTOR_PITCH / 2, 0.0)          # (x, z) of motor A axis
MB = (+MOTOR_PITCH / 2, 0.0)          # (x, z) of motor B axis
STATOR_BC = 40.0                      # stator mount: 4x M4 on PCD 40 (45 deg)
ROTOR_FACE_Y = 32.0                   # rotor output face
ROTOR_BOSS_D, ROTOR_BOSS_H = 16.5, 7.0
ROTOR_BC = 31.0                       # rotor: 4x M3 on PCD 31
MAGNET_HOLDER_CLEAR_D = 30.0          # hole in back plate for the magnet holder

# moteus r4.11 (46 x 53, 4x M2.5 on 40 x 38, encoder 15 mm below top holes)
PCB_Y = -12.5                         # motor-facing PCB surface (v0 value)
MOTEUS_HOLES = [(hx, hz) for hx in (-20, 20) for hz in (-15, 23)]   # 40 x 38 pattern about the encoder
# The phase wires are soldered straight from the 8308 to the board and are short. Each board is
# flipped 180 deg and then turned 90 deg about the encoder so its connector end points outboard,
# the same way as the motor's 3-wire exit (supplier orientation, straight outboard). The two holes
# nearest the encoder sit 5.9 mm from stator screws, where a printed boss would block the countersunk
# heads; those two take brass M-F standoffs in flush inserts, fitted after the stator screws.
MOTOR_TURN = {"A": 0.0, "B": 0.0}     # XZ-plane turn of each 8308 about its axis
MOTEUS_TURN = {"A": -90.0, "B": 90.0} # XZ-plane turn of each board after the flip: connectors outboard
STATOR_PHASE = 45.0                   # stator screws, PCD 40
MOTEUS_STANDOFF_L = 8.0               # plate back face to PCB
# v6: the stator screws' countersinks start 1 mm below the plate's back face, so the two standoffs
# beside them (5.97 mm off, hex 2.9 mm to the corners vs a 4 mm head) sit on plastic, clear of the heads
STATOR_RECESS = 1.0
# v6: the phase wires (soldered to the board before fitting) leave the motor's back at r 28.5 on the
# connector side; the plate is cut away there, between the two outboard standoffs, so the motor
# overhangs an open bay and goes on square to the plate (no slot to thread the wires into)
WIRE_BAY_R, WIRE_BAY_HALF_W = 26.0, 14.0


def moteus_holes(which, kind="all"):
    """Global (x, z) of a board's M2.5 holes: "boss" (printed), "standoff" (brass) or "all"."""
    mx, mz = MA if which == "A" else MB
    holes = [(-hx, -hz) for hx, hz in MOTEUS_HOLES]            # board flipped 180 deg
    # v5: all four on M-F standoffs and flush inserts (no printed bosses: the plates' back faces stay flat to print)
    pick = {"boss": lambda hz: False, "standoff": lambda hz: True, "all": lambda hz: True}[kind]
    c, s_ = math.cos(math.radians(MOTEUS_TURN[which])), math.sin(math.radians(MOTEUS_TURN[which]))
    return [(mx + hx * c - hz * s_, mz + hx * s_ + hz * c) for hx, hz in holes if pick(hz)]

# Capstan
ROPE_D = 1.5                          # v8: Ø1.5 Dyneema (Ø2 before): ten turns fit the small drum at 1.9 pitch
SMALL_PITCH_R = 70.0 / 9.0            # v9: rope-centre radius on the small drum, 9:1 (v8: 8.75, 8:1); D/d 10.4 for the Ø1.5 rope
BIG_PITCH_R = 70.0                    # rope-centre radius on big drum
RATIO = BIG_PITCH_R / SMALL_PITCH_R   # v9: 9.0 : 1 (v8: 8)
MOTOR_TAU = 3.5                       # 8308 at the moteus 40 A limit, N*m (output torque = MOTOR_TAU * RATIO)
CENTER_DIST = 84.0                    # ~v0 (83.5)
GROOVE_PITCH = 1.9                    # v8 (2.5 before)
TRAVEL_HALF = 90.0                    # v8: output travel +-90 deg (v6.2: +-34.2)
END_BLOCK_DEG = 21.4                  # the sector's end blocks reach this far past the rope hole
OUT = (0.0, -math.sqrt(CENTER_DIST**2 - (MOTOR_PITCH / 2) ** 2))  # (x, z)

# Axial layout (Y)
BACK_Y0, BACK_Y1 = -4.5, 3.0          # motor plate, 7.5 thick like v0; stator seats on Y 3
FRONT_Y0, FRONT_Y1 = 90.0, 98.0       # front wall: both small-drum tip bearings
# v8: both slabs 22.5 wide (18 before), the gap between them 2 mm (11): the small drums' ten turns of
# rope and the strands' +-3.8 mm walk fit their rims; everything in front of drum B stays where it was
A_Y0, A_Y1 = 41.0, 63.5               # big drum A slab
B_Y0, B_Y1 = 65.5, 88.0               # big drum B slab
AB_STAGE_Y = (70.0, 88.0)             # the ab/ad capstan keeps v6.2's drum-B stage (18 wide, +-34 deg)
LIP_T = 1.0                           # v8 (1.5 before)
# Drum A runs on a Ø30 shaft in two 6806 (30x42x7, as in v0) held 28.5 mm apart in a plate housing:
# A1 sits in the plate from the rear, A2 in the housing's front end. A shoulder on the shaft bears on
# A2's inner race; a printed cap and an M5 screw into the shaft end clamp A1's inner race from behind.
A_R1_Y, A_R2_Y = -4.5, 31.0           # rear faces of A1 and A2
A_HOUSE_R = 26.0
A_BUTTRESS = (33.0, 28.0)             # buttress radius at the plate's front face, and the y where it meets the housing
A_SHAFT_Y0 = A_R1_Y + 0.5             # shaft end sits just inside A1 so the cap clamps the race

# Plates: Stanley-style outline hugging the motors + output, flat top for the mount
MOTOR_BOSS_R = 34.0                   # v6: just past the moteus standoffs; the Ø92 motors overhang the plates
Z_TOP = 55.0
TOP_HALF_W = 90.0
RIB_T, RIB_H = 8.0, 8.0               # stiffening rib along each plate's top edge (M4 inserts)
POST_R = 8.0
PLATE_STRUT_R = 14.0                  # v6 skeleton plates: strut half-width at the motor / hub end
PLATE_FILLET = 6.0                    # v6: inside corners where a strut meets the body (no notch)
# v4 (Stanley-style, compact): four corner posts, clear of the drums' sweep (83.8 mm from the
# output axis vs 70.5 + 8) and of the motors; no top strip, ribs or base mount
POSTS = [(-80.0, 45.0), (80.0, 45.0), (-80.0, -45.0), (80.0, -45.0)]
OUT_BOSS_R = 34.0                     # plates around the output axis
# front gantry: a plate carrying a 6902 on output A's stub, on two posts. v8: with +-90 deg of travel the
# crank sweeps -50..130 deg (66 mm out), the thigh -150..30 deg and the coupler most of the rest; the posts
# go where none of them reaches over the knee range, between the end brackets (check_travel.py): 154 deg
# 66 mm out and 104 deg 104 mm out. The plate moves in front of the coupler (v6.2: in its plane).
GANTRY_POSTS = [(0.0 + r * math.cos(math.radians(a)), -math.sqrt(84.0 ** 2 - 46.75 ** 2) + r * math.sin(math.radians(a)))
                for r, a in ((66.0, 154.0), (104.0, 104.0))]
GANTRY_Y = (146.0, 156.0)            # v8 (v6.2: 137-147, in the coupler's plane)
GANTRY_POST_R = 10.0                  # Ø20 posts (v4b: Ø16)
END_X = (-112.0, -98.0)               # end bracket web, 14 mm (the ab/ad hub bolts to it with 6x M4)

# Bearings
B15 = (15.0, 28.0, 7.0)
GANTRY_BRG = (17.0, 30.0, 7.0)        # v9: 6903 on output A's stub (v8: 6902, Ø15): the stub's section +46 %
B30 = (30.0, 42.0, 7.0)
B40 = (40.0, 52.0, 7.0)               # 6808: drum B's journal in the front wall
B2_Y = 81.0                           # drum B's front 6806 on A's tube (both pressed in from the rear)
WALL_BRG_Y = (90.0, 97.0)             # 6808 pressed into the wall from the rear, against a front lip
WALL_BOSS_R, WALL_BOSS_H = 34.0, 6.0   # raised ring round the 6808 on the wall's front face (v9.1: 6, FEA SF 1.9 at 4)
WALL_OUT_R, WALL_LOWER_STRUT_R = 43.0, 22.0  # v6.2 bearing load path to the lower posts (v9.1: 43, was 40)
NOSE_HEX_AF, NOSE_HEX_L = 18.0, 8.0    # A's nose ends in a double-D (flats 18 apart) that drives flange A; the M3s only clamp


def double_d(r, af, h):
    """Ø2r with two flats af apart (normal to local x), local z 0..h."""
    return Cylinder(r, h, align=MIN) & Box(af, 2 * r + 1, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
A_NOSE_R = 11.0                       # A's tube steps down to Ø22 where it passes through B's journal
A_NOSE_SCREW_R = 7.0                  # flange A -> 4x M3 inserts in A's nose (PCD 14, 2 mm wall outside)
A_ARM_SPIGOT_R = 14.0                 # arm A's locating ring on flange A, outside the nose screws
B_HUB_SCREW_R = 16.0                  # flange B -> M3 inserts in the end of B's journal
PRESS = 0.1                           # added to bearing OD pockets
JOURNAL_CLR = 0.0                     # v9.1: diametral clearance of every printed journal in a bearing bore: nominal (prints
                                      # ~0.05-0.15 over -> a light press; v9: 0.05-0.10 under, a slip fit the race could creep in)

# Hardware holes
M2_5_INSERT, M3_TAP, M3_CLR, M4_CLR, M4_INSERT, M3_INSERT = 3.6, 2.5, 3.4, 4.5, 5.6, 4.0
M2_5_FLUSH = 3.3                      # short M2.5 insert beside a stator countersink (knife-edge at the surface only)

# Base mount: the module stands on it (assembly is exported upright); the rear flat clamps to the desk
MOUNT_Y0, MOUNT_Y1, MOUNT_T = -120.0, 106.0, 8.0
UPRIGHT = Pos(0, 0, Z_TOP + MOUNT_T) * Rot(0, 180, 0)   # model frame -> desk frame (base underside at Z 0)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def cyl_y(r, y0, y1, x=0.0, z=0.0):
    """Cylinder along +Y from y0 to y1 centred on (x, z)."""
    return Pos(x, y0, z) * Rot(-90, 0, 0) * Cylinder(r, y1 - y0, align=MIN)


def ring_y(ro, ri, y0, y1, x=0.0, z=0.0):
    return cyl_y(ro, y0, y1, x, z) - cyl_y(ri, y0 - 1, y1 + 1, x, z)


def to_y(local, y0, x=0.0, z=0.0):
    """Map a part modelled about local +Z (z=0 start) onto the Y axis.
    local (x, y, z) -> global (x, z, -y): local angle phi == global XZ angle -phi."""
    return Pos(x, y0, z) * Rot(-90, 0, 0) * local


def pie(r, a0, a1, h):
    """Exact circular sector: an analytic arc exports cleanly to STEP."""
    p0, p1 = polar(r, a0), polar(r, a1)
    face = make_face([Line((0, 0), p0), CenterArc((0, 0), r, a0, a1 - a0), Line(p1, (0, 0))])
    return extrude(face, h)


def ring_z(ro, ri, z0, z1):
    return Pos(0, 0, z0) * (Cylinder(ro, z1 - z0, align=MIN) - Cylinder(ri, z1 - z0, align=MIN))


def hex_prism_z(af, h):
    return extrude(RegularPolygon(af / 2, 6, major_radius=False), h)


def polar(r, deg):
    return r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))


def dir_deg(frm, to):
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


# Crossed (figure-8) rope loop: the strands cross in the gap, so they meet the big
# drum close to the small drum (+-TANGENT_DEG) and a short v0-like sector still
# gives useful travel. The big drum turns opposite to the small drum.
BETA = math.degrees(math.asin((BIG_PITCH_R + SMALL_PITCH_R) / CENTER_DIST))
TANGENT_DEG = 90.0 - BETA
ANCHOR_DEG = 20.0
ROPE_HOLE_DEG = TRAVEL_HALF + TANGENT_DEG + 2.0        # rope enters the tensioner here (v6.2: 56.6 for +-34.2)
SECTOR_SPAN = 2 * (ROPE_HOLE_DEG + END_BLOCK_DEG)      # v8: ~268 deg (v6.2: 156)
SECTOR_HALF = SECTOR_SPAN / 2
OUT_RANGE = 2 * (ROPE_HOLE_DEG - TANGENT_DEG - 2.0)    # usable output rotation, degrees
AB_ROPE_HOLE_DEG, AB_SECTOR_SPAN = 56.6, 156.0         # the ab/ad capstan's sector, as v6.2 (+-34 deg)


# ----------------------------------------------------------------------------
# Frame
# ----------------------------------------------------------------------------
def plate_outline(y0, y1):
    """Compact outline: hull of the two motor bosses, the output boss and the corner-post lobes."""
    shapes = [Pos(*MA) * Circle(MOTOR_BOSS_R), Pos(*MB) * Circle(MOTOR_BOSS_R), Pos(*OUT) * Circle(OUT_BOSS_R)]
    shapes += [Pos(x, z) * Circle(POST_R + 3.0) for x, z in POSTS]
    face = make_hull([e for s_ in shapes for e in s_.edges()])
    return Pos(0, y1, 0) * Rot(90, 0, 0) * extrude(face, y1 - y0)   # local y -> global Z


def hull2d(cs):
    """Planar hull of (u, v, r) circles."""
    return make_hull([e for (u, v, r) in cs for e in (Pos(u, v) * Circle(r)).edges()])


def fillet2d(faces, r):
    """Union of planar faces with every inside corner rounded to r (closing: grow by r, shrink by r)."""
    u = faces[0]
    for f in faces[1:]:
        u = u + f
    return offset(offset(u, r, kind=Kind.ARC), -r, kind=Kind.ARC)


def plate_skeleton(y0, y1, extra=(), strut_r=PLATE_STRUT_R, out_r=OUT_BOSS_R,
                   low_strut_r=None):
    """Motor plate (v6): a body joining the two motors and the output housing, and a tapered strut from
    each motor (and the output, for the lower ones) to each corner post. The Ø92 motors overhang it."""
    if low_strut_r is None:
        low_strut_r = strut_r
    faces = [hull2d([(*MA, MOTOR_BOSS_R), (*MB, MOTOR_BOSS_R), (*OUT, out_r)])]
    for (px, pz), m in zip(POSTS, (MA, MB, MA, MB)):
        faces.append(hull2d([(*m, strut_r), (px, pz, POST_R + 3.0)]))
        if pz < 0:
            faces.append(hull2d([(*OUT, low_strut_r), (px, pz, POST_R + 3.0)]))
    faces += [Pos(x, z) * Circle(r) for x, z, r in extra]
    return Pos(0, y1, 0) * Rot(90, 0, 0) * extrude(fillet2d(faces, PLATE_FILLET), y1 - y0)


def plate_blank(y0, y1, rib_side, skeleton=False, extra=(), strut_r=PLATE_STRUT_R,
                out_r=OUT_BOSS_R, low_strut_r=None):
    p = plate_skeleton(y0, y1, extra, strut_r, out_r, low_strut_r) if skeleton else plate_outline(y0, y1)
    for (x, z) in POSTS:
        p -= cyl_y(M4_CLR / 2, y0 - 20, y1 + 20, x, z)
    return p


def stator_cuts(mx, mz):
    """4x M4 countersunk from the plate's back (Y BACK_Y0), heads STATOR_RECESS below the face."""
    out = []
    for k in range(4):
        dx, dz = polar(STATOR_BC / 2, STATOR_PHASE + 90 * k)
        out.append(cyl_y(M4_CLR / 2, BACK_Y0 - 1, BACK_Y1 + 1, mx + dx, mz + dz))
        out.append(cyl_y(4.1, BACK_Y0 - 1, BACK_Y0 + STATOR_RECESS, mx + dx, mz + dz))
        out.append(Pos(mx + dx, BACK_Y0 + STATOR_RECESS - 0.01, mz + dz) * Rot(-90, 0, 0) * Cone(4.1, 2.05, 2.25, align=MIN))
    return out


def wire_bay(which, length=70.0):
    """Cut on motor `which`'s connector side: everything past WIRE_BAY_R, between the outboard standoffs."""
    mx, mz = MA if which == "A" else MB
    a = (180.0 if which == "A" else 0.0) + MOTOR_TURN[which]
    cx, cz = polar(WIRE_BAY_R, a)
    return Pos(mx + cx, BACK_Y0 - 1, mz + cz) * Rot(0, -a, 0) * Box(
        length, BACK_Y1 - BACK_Y0 + 2, 2 * WIRE_BAY_HALF_W, align=(Align.MIN, Align.MIN, Align.CENTER))


def back_plate():
    y0, y1 = BACK_Y0, BACK_Y1
    p = plate_blank(y0, y1, rib_side=-1, skeleton=True)
    for i, (mx, mz) in enumerate((MA, MB)):
        # magnet holder clearance
        p -= cyl_y(MAGNET_HOLDER_CLEAR_D / 2, y0 - 1, y1 + 1, mx, mz)
        # stator screws: M4 countersunk from the rear, PCD 40
        for c_ in stator_cuts(mx, mz):
            p -= c_
        # open wire bay: the motor overhangs the plate where its phase wires leave
        p -= wire_bay("AB"[i])
        # moteus: printed bosses for the two far holes, flush inserts for the two standoffs
        for hx, hz in moteus_holes("AB"[i], "boss"):
            p += cyl_y(3.8, PCB_Y, y0 + 0.5, hx, hz)
            p -= cyl_y(M2_5_INSERT / 2, PCB_Y - 1, PCB_Y + 5.0, hx, hz)
        for hx, hz in moteus_holes("AB"[i], "standoff"):
            p -= cyl_y(M2_5_FLUSH / 2, y0 - 0.01, y0 + 4.5, hx, hz)
    # drum A bearing housing: A1 from the rear, A2 from the front, a 21.5 mm gap between them
    ox, oz = OUT
    p += cyl_y(A_HOUSE_R, y0, A_R2_Y + B30[2], ox, oz)
    # conical buttress round the housing's root (fea_check.py: without the gantry, drum A's front 6806
    # levers the bare housing off the plate at SF 0.9); inside the output boss, clear of the motors
    p += Pos(ox, y1, oz) * Rot(-90, 0, 0) * Cone(A_BUTTRESS[0], A_HOUSE_R, A_BUTTRESS[1] - y1, align=MIN)
    p -= cyl_y((B30[1] + PRESS) / 2, y0 - 1, A_R1_Y + B30[2], ox, oz)
    p -= cyl_y(18.0, y0 - 1, A_R2_Y + 1, ox, oz)
    p -= cyl_y((B30[1] + PRESS) / 2, A_R2_Y, A_R2_Y + B30[2] + 1, ox, oz)
    return p


def front_plate():
    # v6.2: enlarge the 6808 load ring and its two lower struts to the post bosses. Leave the motor
    # struts slim so reinforcement follows the output bearing load path.
    p = plate_blank(FRONT_Y0, FRONT_Y1, rib_side=+1, skeleton=True, out_r=WALL_OUT_R,
                    low_strut_r=WALL_LOWER_STRUT_R,
                    extra=[(x, z, GANTRY_POST_R + 2.0) for x, z in GANTRY_POSTS])
    # raised bearing bosses on the front face (after the Aaed Musa-style frame, but solid): the
    # three bearings sit in 12 mm of plastic instead of 8, and the 6808's lip is 5 mm thick
    p += cyl_y(WALL_BOSS_R, FRONT_Y1, FRONT_Y1 + WALL_BOSS_H, *OUT)
    for (mx, mz) in (MA, MB):
        p += cyl_y(20.0, FRONT_Y1, FRONT_Y1 + WALL_BOSS_H, mx, mz)
    for (mx, mz) in (MA, MB):                                                  # small-drum tip bearings
        p -= cyl_y((B15[1] + PRESS) / 2, FRONT_Y0 - 1, FRONT_Y1 - 1.0, mx, mz)
        p -= cyl_y(12.0, FRONT_Y0 - 1, FRONT_Y1 + WALL_BOSS_H + 1, mx, mz)
    p -= cyl_y((B40[1] + PRESS) / 2, FRONT_Y0 - 1, WALL_BRG_Y[1], *OUT)      # 6808 for drum B's journal
    p -= cyl_y(24.0, FRONT_Y0 - 1, FRONT_Y1 + WALL_BOSS_H + 1, *OUT)          # front lip on its outer race
    for x, z in GANTRY_POSTS:                                                  # gantry posts: M4 inserts from the front
        p -= cyl_y(M4_INSERT / 2, FRONT_Y1 - 7.0, FRONT_Y1 + 0.01, x, z)          # (nothing pokes into drum B's sweep)
    return p


WALL_WINDOW_DEG = {"A": ((20.0, 80.0), (100.0, 160.0), (180.0, 240.0)),   # A's solid arc (-120..20 deg) keeps
                   "B": ((100.0, 160.0), (20.0, 80.0), (-60.0, 0.0))}        # the gantry inserts; B mirrors A
WALL_WINDOW_R = (23.0, 45.0)          # inner / outer radius round the motor tip: 22 mm of opening, 5 mm rim
WALL_WINDOW_FILLET = 4.0


def wall_windows():
    """Stanley-style windows through the front wall round each motor tip: three rounded annular sectors
    per motor (60 deg each, 20 deg spokes between), clear of the bosses, posts and gantry inserts."""
    out = []
    h = FRONT_Y1 + WALL_BOSS_H + 2.0 - FRONT_Y0
    for w, (mx, mz) in (("A", MA), ("B", MB)):
        for a0, a1 in WALL_WINDOW_DEG[w]:
            win = pie(WALL_WINDOW_R[1], a0, a1, h) - Cylinder(WALL_WINDOW_R[0], h, align=MIN)
            try:
                win = win.fillet(WALL_WINDOW_FILLET, win.edges().filter_by(Axis.Z))
            except Exception:
                pass
            out.append(Pos(0, FRONT_Y1 + WALL_BOSS_H + 1.0, 0) * Rot(90, 0, 0) * Pos(mx, mz, 0) * win)
    return out


def end_bracket(front=False):
    """U-shaped end bracket: two flanges clamped under a pair of corner posts' bolt heads (front) and
    nuts (back) -- one ear above and one below the moteus -- joined by a web perpendicular to X.
    Rear (motor A's end): the web takes the ab/ad hub (6x M4 through-bolts). Front (motor B's end, v6):
    the web takes the trunnion (4x M4 into heat-set inserts) that runs in the cradle's front bearing."""
    x0, x1 = FB_X if front else END_X
    px, xe = (80.0, x1 - 5.0) if front else (-80.0, x0 + 5.0)
    ears = []
    for y0, y1 in ((BACK_Y0 - 8.0, BACK_Y0), (FRONT_Y1, FRONT_Y1 + 8.0)):
        for z in (45.0, -45.0):
            face = make_hull([e for c in (Pos(px, z) * Circle(POST_R + 3.0), Pos(xe, z - 8.0) * Circle(4.0),
                                          Pos(xe, z + 8.0) * Circle(4.0)) for e in c.edges()])
            ears.append(Pos(0, y1, 0) * Rot(90, 0, 0) * extrude(face, y1 - y0))
    b = ears[0]
    for e_ in ears[1:]:
        b += e_
    ys = (BACK_Y0 - 8.0, FRONT_Y1 + 8.0)
    zb = min(-53.0, AB_Z - 34.0)                                              # web reaches below the ab/ad hub
    face = make_hull([e for c in (Pos(ys[0] + 5.0, 53.0) * Circle(4.0), Pos(ys[1] - 5.0, 53.0) * Circle(4.0),
                                  Pos(ys[0] + 5.0, zb) * Circle(4.0), Pos(ys[1] - 5.0, zb) * Circle(4.0)) for e in c.edges()])
    web = Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(face, x1 - x0)
    web -= Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(
        make_hull([e for c in (Pos(15.0, 32.0) * Circle(9.0), Pos(75.0, 32.0) * Circle(9.0)) for e in c.edges()]), x1 - x0)
    # v9 lightening: a rounded slot across the middle of the web and a round window each side of the hub,
    # all >= 26 mm from the hub / trunnion centre (its bolt circle and heads) and >= 9 mm from the edges
    on = Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))
    zm = (AB_Z + 26.0 + 32.0 - 9.0) / 2                                       # between the hub's keep-out and the top window
    hm = (32.0 - 9.0 - (AB_Z + 26.0)) / 2 - 5.0
    if hm > 4.0:
        web -= on * extrude(make_hull([e for c in (Pos(18.0, zm) * Circle(hm), Pos(72.0, zm) * Circle(hm)) for e in c.edges()]), x1 - x0)
    for yy in (AB_Y - 36.0, AB_Y + 36.0):
        web -= on * extrude(Pos(yy, AB_Z - 2.0) * Circle(7.0), x1 - x0)
    if AB_Z > -5.0:                                                           # lower window only clear of a high hub
        web -= Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(
            make_hull([e for c in (Pos(15.0, -32.0) * Circle(9.0), Pos(75.0, -32.0) * Circle(9.0)) for e in c.edges()]), x1 - x0)
    b += web
    if front:
        for yy, zz in TRUNNION_SCREWS:                                         # M4 heat-set inserts from the outer face
            b -= _bore_x(x1 - 6.0, x1 + 0.01, yy, zz, M4_INSERT / 2)
    else:
        b -= _bore_x(AB_TUBE_END - 1, x1 + 1, AB_Y, AB_Z, 11.0)              # wire pass-through
        for yy, zz in AB_HUB_SCREWS:                                           # v9.1: 8x M4 x 16 into the hub's inserts
            b -= _bore_x(AB_TUBE_END - 1, x1 + 1, yy, zz, M4_CLR / 2)
            b -= _bore_x(x1 - 4.5, x1 + 0.01, yy, zz, 3.8)
    for x, z in (POSTS[1::2] if front else POSTS[0::2]):                       # that end's corner posts
        b -= cyl_y(M4_CLR / 2, BACK_Y0 - 20, FRONT_Y1 + 20, x, z)
    return b


def gantry():
    """Front gantry plate: 6902 on output A's stub, two M4 posts back to the front wall."""
    pts = [(OUT[0], OUT[1], 22.0)] + [(x, z, GANTRY_POST_R + 2.0) for x, z in GANTRY_POSTS]
    face = make_hull([e for (x, z, r) in pts for e in (Pos(x, z) * Circle(r)).edges()])
    g0, g1 = GANTRY_Y
    g = Pos(0, g1, 0) * Rot(90, 0, 0) * extrude(face, g1 - g0)
    g -= cyl_y((GANTRY_BRG[1] + PRESS) / 2, g0 - 1, g0 + GANTRY_BRG[2], *OUT)  # v9: 6903, from the leg side
    g -= cyl_y(9.0, g0 - 1, g1 + 1, *OUT)
    for x, z in GANTRY_POSTS:
        g -= cyl_y(M4_CLR / 2, g0 - 1, g1 + 1, x, z)
    # v6: a rounded window in the middle of the three bosses (5 mm wall to each)
    cx = (OUT[0] + sum(x for x, _ in GANTRY_POSTS)) / 3
    cz = (OUT[1] + sum(z for _, z in GANTRY_POSTS)) / 3
    rw = min(math.hypot(cx - OUT[0], cz - OUT[1]) - 14.0 - 5.0,
             min(math.hypot(cx - x, cz - z) - GANTRY_POST_R - 5.0 for x, z in GANTRY_POSTS))
    if rw > 3.0:
        g -= cyl_y(rw, g0 - 1, g1 + 1, cx, cz)
    return g


POST_INSERT_DEPTH = 10.0              # v6: M4 heat-set insert (8 long) in each end of every post and top rail


def post_insert_holes(length):
    return (Cylinder(M4_INSERT / 2, POST_INSERT_DEPTH, align=MIN) +
            Pos(0, 0, length - POST_INSERT_DEPTH) * Cylinder(M4_INSERT / 2, POST_INSERT_DEPTH, align=MIN))


def post(length):
    """Ø16 post with an M4 heat-set insert in each end: the motor plate and the front wall each screw to it."""
    return Cylinder(POST_R, length, align=MIN) - Cylinder(M4_CLR / 2, length, align=MIN) - post_insert_holes(length)


TOP_POSTS = POSTS[:2]                 # (+-80, 45): the upper posts are rails with a flat top for the mounts
RAIL_TOP = 62.0                       # model z of the rails' top face
RAIL_INSERT_Y = (12.0, 81.0)          # far apart: the mount moment loads the inserts less


def top_rail():
    """Upper corner post with a rib up to a flat top (model +Z) carrying 2x M4 inserts for the desk base
    or the rail carriage. Local frame: post axis along local z (= model +Y), local -y = model +Z."""
    length = FRONT_Y0 - BACK_Y1
    r = Cylinder(POST_R, length, align=MIN)
    r += Pos(0, -(RAIL_TOP - POSTS[0][1]) / 2, 0) * Box(24.0, RAIL_TOP - POSTS[0][1], length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    r -= Cylinder(M4_CLR / 2, length, align=MIN)
    r -= post_insert_holes(length)
    for y in RAIL_INSERT_Y:
        r -= Pos(0, -(RAIL_TOP - POSTS[0][1]), y - BACK_Y1) * Rot(-90, 0, 0) * Cylinder(M4_INSERT / 2, 8.0, align=MIN)
    return r


def rail_insert_points():
    return [(x, y) for x, _ in TOP_POSTS for y in RAIL_INSERT_Y]


DESK_T = 8.0
DESK_SCREWS = [(sx * 125.0, y) for sx in (-1, 1) for y in (-15.0, 105.0)]   # 4 wood screws, outside the module


def desk_base():
    """Desk version (no ab/ad): plate on the two top rails (4x M4 x 16 from the desk side, heads sunk),
    4 countersunk holes for #10 (5 mm) wood screws into the desk. Model +Z is the desk side."""
    b = Pos(0, -30.0, RAIL_TOP) * Box(270.0, 150.0, DESK_T, align=(Align.CENTER, Align.MIN, Align.MIN))
    for x, y in rail_insert_points():
        b -= Pos(x, y, RAIL_TOP) * Cylinder(M4_CLR / 2, DESK_T, align=MIN)
        b -= Pos(x, y, RAIL_TOP + DESK_T - 4.2) * Cylinder(4.0, 4.2, align=MIN)            # heads sunk on the desk side
    for x, y in DESK_SCREWS:
        b -= Pos(x, y, RAIL_TOP) * Cylinder(2.75, DESK_T, align=MIN)
        b -= Pos(x, y, RAIL_TOP) * Cone(5.2, 2.75, 2.6, align=MIN)                       # countersink, module side
    for x in (-40.0, 0.0, 40.0):                                                        # lightening
        b -= Pos(x, 45.0, RAIL_TOP) * Cylinder(14.0, DESK_T, align=MIN)
    return b


def test_arms(gantry=False):
    """Flange versions: the v3 test arms (M5 load holes at 50-150 mm). With the gantry, arm A carries
    the Ø15 front stub. name -> (printed solid, placed solid, colour)."""
    ox, oz = OUT
    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    arm_a = test_arm(31.0, A_ARM_SPIGOT_R + 0.1, 24.0, 30.0)
    if gantry:
        arm_a = front_stub(arm_a)
    arm_b = test_arm(47.0, 27.1, 35.0, 0.0)
    return {"Test_Arm_A": (arm_a, to_y(Rot(0, 0, phi_a + 300.0) * arm_a, 128.0, ox, oz), "red"),
            "Test_Arm_B": (arm_b, to_y(Rot(0, 0, phi_b + 180.0) * arm_b, 111.0, ox, oz), "black")}


def wood_screw(length=30.0):
    """5 mm countersunk wood screw: head face at local z 0 (Ø10), cone to the shank, shank to +length."""
    return Cone(5.0, 2.5, 2.6, align=MIN) + Pos(0, 0, 2.6) * Cylinder(2.4, length - 2.6, align=MIN)


# ----------------------------------------------------------------------------
# Small (motor) drums - modelled about local Z, z=0 at the rotor face
# ----------------------------------------------------------------------------
CORE_R = SMALL_PITCH_R - 0.2          # groove bottom sits 0.8 mm under rope centre


def small_drum(rope_y0, rope_y1, core_end_y, journal_y1):
    z = lambda y: y - ROTOR_FACE_Y
    # flange: four lobes round the rotor screws on a Ø24 centre (v6; a Ø40 disc before)
    d = Cylinder(12.0, 8.0, align=MIN)
    for k in range(4):
        x, y = polar(ROTOR_BC / 2, 45 + 90 * k)
        d += extrude(make_hull([e for c in (Circle(12.0), Pos(x, y) * Circle(4.5)) for e in c.edges()]), 8.0)
    d += Pos(0, 0, 8.0) * Cylinder(CORE_R, z(core_end_y) - 8.0, align=MIN)  # core
    d += Pos(0, 0, z(core_end_y)) * Cylinder(B15[0] / 2 - JOURNAL_CLR / 2, z(journal_y1) - z(core_end_y), align=MIN)
    d -= Cylinder((ROTOR_BOSS_D + 0.7) / 2, ROTOR_BOSS_H + 0.6, align=MIN)  # rotor boss pocket
    for k in range(4):
        x, y = polar(ROTOR_BC / 2, 45 + 90 * k)
        d -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
        d -= Pos(x, y, 8.0 - 3.2) * Cylinder(3.1, 3.3, align=MIN)            # M3 SHCS counterbore
    # helical rope groove
    h = z(rope_y1) - z(rope_y0)
    helix = Pos(0, 0, z(rope_y0)) * Helix(GROOVE_PITCH, h, SMALL_PITCH_R)
    sec = Plane(origin=helix @ 0, z_dir=helix % 0) * Circle(ROPE_D / 2 + 0.1)
    d -= sweep(sec, path=helix, is_frenet=True)
    # rope anchor cross-hole at mid length
    zm = (z(rope_y0) + z(rope_y1)) / 2
    d -= Pos(0, 0, zm) * Rot(0, 90, 0) * Cylinder(1.3, 2 * CORE_R + 2)
    return d


def small_drum_a():
    return small_drum(A_Y0 + LIP_T, A_Y1 - LIP_T, 89.5, 97.5)


def small_drum_b():
    return small_drum(B_Y0 + LIP_T, B_Y1 - LIP_T, 89.5, 97.5)


# ----------------------------------------------------------------------------
# Big (output) drums - modelled about local Z; z = 0 at the slab's rear face
# ----------------------------------------------------------------------------
RIM_R = BIG_PITCH_R - ROPE_D / 2      # 69
LIP_R = RIM_R + 1.5                   # 70.5 (keeps the other motor's drum clear)
# Push-screw tensioner: the slider drops into the channel from the drum face, an M4 nut drops
# into its own slot from the same face, and an M4x16 set screw is driven in radially from the
# rim with a 2 mm hex key. Screwing in pushes the slider (and the rope knotted to it) inward.
# The screw sits beside the rope hole, toward the sector end, and never stands proud of the rim.
# In-line rope tensioner at each sector end. The rope leaves the rim at the rope hole and runs in a
# straight line through the drum to a printed slider in an open pocket; an M3 x 12 socket screw goes
# in from the sector's end face, along the same line, into an M3 nut in the slider. Turning the
# screw pulls the slider (and the rope knotted in it) toward the end of the drum.
# Slider-local frame (ts_loc): x along the line from the end face toward the rim, y inward,
# z depth from the pocket's open face.
TS_D = 50.0                           # distance of the line from the drum axis
TS_THC = ROPE_HOLE_DEG + math.degrees(math.acos(TS_D / RIM_R))     # angle of the line's foot point
TS_END = TS_D * math.tan(math.radians(SECTOR_SPAN / 2 - TS_THC))  # line parameter at the end face
TS_RIM = -math.sqrt(RIM_R ** 2 - TS_D ** 2)                       # ... and at the rim
_SEC = {}                             # the sector being built: the legs' (default) or the ab/ad capstan's


def _sec_set(abad=False):
    span, hole = (AB_SECTOR_SPAN, AB_ROPE_HOLE_DEG) if abad else (SECTOR_SPAN, ROPE_HOLE_DEG)
    thc = hole + math.degrees(math.acos(TS_D / RIM_R))
    _SEC.update(half=span / 2, thc=thc, end=TS_D * math.tan(math.radians(span / 2 - thc)))


_sec_set()
TS_SPOT, TS_WALL = 1.2, 5.0           # spot face for the screw head; end wall to the pocket
TS_L, TS_W, TS_PD = 10.0, 7.6, 11.0   # slider length, width; pocket depth
TS_TRAVEL, TS_GAP = 4.0, 0.3          # tension stroke; slider rests TS_GAP off the wall when tight
TS_ZR, TS_ZS = 2.75, 6.5              # rope and screw depths from the open face
TS_NUT_X = 2.2                        # nut slot position in the slider (2.2 mm wall takes the rope pull)
# v6 winding-peg tensioner (replaces the slider): an M4 x 18 socket screw put in radially from the rim
# at the sector's end block (past the rope's wrap), head sunk in a counterbore, crossing the rope's
# tunnel PEG_X from the end face. The rope comes in through the rim hole, crosses a well round the
# shank, passes through a Ø2 cross-hole drilled in the shank and its tail leaves through the end
# face; turning the screw winds the rope onto the shank. An M4 nylon-insert nut in a slot below the
# well (dropped in from the drum face, flats against the slot walls) takes the thread and holds it.
# The well is open to the drum face, so the rope is threaded and its turns checked from there.
# The drums stay as thick as before and nothing stands proud of a face; the screw is reached from
# outside the rim, with the drum turned to bring the end to an opening.
PEG_X = 11.0                          # peg axis from the end face, along the rope line
PEG_Z = 4.5                           # peg axis depth from the open face (the tunnel slopes from TS_ZR at the rim)
PEG_WELL_R, PEG_WELL_H = 4.2, 6.0     # well round the shank: Ø8.4 (a turn of Ø2 rope each side), 6 long (3 turns)
PEG_L, PEG_HEAD = 18.0, (7.0, 4.0)    # M4 x 18 socket head (Ø7 x 4)
PEG_SINK = 0.6                        # head below the rim
PEG_NUT_AF, PEG_NUT_H = 7.0, 5.0      # M4 nylon-insert nut (DIN 985)
PEG_NUT_GAP = 1.5                     # plastic between the well and the nut


def ts_loc(s, w):
    """Location of the tensioner frame at sector end s (+1 front-open, -1 rear-open) in drum-local coords."""
    th = math.radians(s * _SEC["thc"])
    n = (math.cos(th), math.sin(th))
    u = (-s * math.sin(th), s * math.cos(th))                              # along the line, toward the end
    o = (TS_D * n[0] + _SEC["end"] * u[0], TS_D * n[1] + _SEC["end"] * u[1], w if s > 0 else 0.0)
    return Location(Plane(origin=o, x_dir=(-u[0], -u[1], 0), z_dir=(0, 0, -1 if s > 0 else 1)))


def peg_k_rim():
    """Distance along the peg's axis from the rope crossing to the rim surface."""
    t = _SEC["end"] - PEG_X
    return math.sqrt(RIM_R ** 2 - t ** 2) - TS_D


def peg_frame(s, w):
    """Location of the peg in drum-local coordinates: origin where it crosses the rope, z outward
    along its axis (toward the rim), x along the rope line toward the rim."""
    return ts_loc(s, w) * Location(Plane(origin=(PEG_X, 0, PEG_Z), x_dir=(1, 0, 0), z_dir=(0, _peg_sy(s, w), 0)))


def _seg(p0, p1, r):
    """Cylinder of radius r from p0 to p1 (tuples)."""
    d = [b - a for a, b in zip(p0, p1)]
    n = math.sqrt(sum(c * c for c in d))
    return Location(Plane(origin=p0, z_dir=tuple(c / n for c in d))) * Cylinder(r, n, align=MIN)


def peg_cuts(s, w):
    """Tunnel, well, screw hole, head counterbore and nut slot for the peg at sector end s."""
    k_rim, hd, hh = peg_k_rim(), PEG_HEAD[0], PEG_HEAD[1]
    k_head = k_rim - PEG_SINK - hh                                         # head's underside
    k_nut = -PEG_WELL_H / 2 - PEG_NUT_GAP                                  # nut's top face
    pf = peg_frame(s, w)
    c = Pos(0, 0, -PEG_WELL_H / 2) * Cylinder(PEG_WELL_R, PEG_WELL_H, align=MIN)                  # well
    c += Pos(0, 0, k_nut - PEG_NUT_H - (PEG_L - 4.0)) * Cylinder(M4_CLR / 2, 40.0, align=MIN)    # screw hole
    c += Pos(0, 0, k_head) * Cylinder(hd / 2 + 0.4, k_rim - k_head + 5.0, align=MIN)            # head counterbore
    cut = pf * c
    # nut slot: flats across the rope line, open to the drum's open face (toward frame -z of ts_loc)
    loc = ts_loc(s, w)
    cut += loc * Pos(PEG_X, 0, -0.01) * Box(2 * PEG_WELL_R, PEG_WELL_H, PEG_Z + 0.01,              # well open to the face:
                                            align=(Align.CENTER, Align.CENTER, Align.MIN))         # rope threaded from there
    cut += loc * Pos(PEG_X, _peg_sy(s, w) * (k_nut - PEG_NUT_H / 2), -0.01) * Box(
        PEG_NUT_AF + 0.3, PEG_NUT_H + 0.3, PEG_Z + PEG_NUT_AF / math.sqrt(3) + 0.25, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # rope tunnel: rim hole (TS_ZR deep) down to the peg (PEG_Z), then on to the end face
    rim_x = _SEC["end"] - TS_RIM
    cut += loc * _seg((rim_x + 3.0, 0, TS_ZR), (PEG_X, 0, PEG_Z), 1.3)
    cut += loc * _seg((PEG_X, 0, PEG_Z), (-1.0, 0, PEG_Z), 1.3)
    return cut


def _peg_sy(s, w):
    loc = ts_loc(s, w)
    o = (loc * Pos(PEG_X, 0, PEG_Z)).position
    q = (loc * Pos(PEG_X, 10.0, PEG_Z)).position
    return 1.0 if math.hypot(q.X, q.Y) > math.hypot(o.X, o.Y) else -1.0


def sector_body(w, hub_r, abad=False):
    _sec_set(abad)
    try:
        return _sector_body(w, hub_r)
    finally:
        _sec_set()


def _sector_body(w, hub_r):
    a = _SEC["half"]
    body = pie(RIM_R, -a, a, w) - Cylinder(RIM_R - 6.0, w, align=MIN)        # rim
    for z0 in (0.0, w - LIP_T):                                               # lips
        body += Pos(0, 0, z0) * (pie(LIP_R, -a, a, LIP_T) - Cylinder(RIM_R - 1, LIP_T, align=MIN))
    web = pie(RIM_R - 5.0, -a, a, 6.0) - Cylinder(hub_r - 1, 6.0, align=MIN)
    body += Pos(0, 0, w / 2 - 3.0) * web
    for s in (-1, 1):                                                         # end blocks
        lo, hi = sorted((s * (a - ANCHOR_DEG - 4), s * a))
        body += pie(RIM_R - 1, lo, hi, w) - Cylinder(hub_r + 6.5, w, align=MIN)
    body += Cylinder(hub_r, w, align=MIN)
    # lightening windows: three tapered, fully rounded slots through the web with 12 deg spokes
    # between them, and a small rounded slot in each end block; rim and hub keep generous walls
    w_in, w_out = hub_r + 7.0, RIM_R - 12.0
    edge = a - ANCHOR_DEG - 4 - 5.0                                          # 5 deg short of the end blocks
    nw = 3 if a < 100 else 5                                                # v8: five windows in the long sector
    span = (2 * edge - (nw - 1) * 12.0) / nw
    for k in range(nw):
        th = math.radians(-edge + span / 2 + k * (span + 12.0))
        r1 = 0.40 * w_in * math.radians(span)                                # round ends fit the window's width
        r2 = 0.40 * w_out * math.radians(span)
        p1, p2 = w_in + r1, w_out - r2
        body -= chain([(p1 * math.cos(th), p1 * math.sin(th), r1), (p2 * math.cos(th), p2 * math.sin(th), r2)], w)
    # (v6: the end blocks stay solid; the tensioner's nut slot sits where their small windows were)
    # winding-peg tensioners, radial from the rim at each end block
    for s in (-1, 1):
        body -= peg_cuts(s, w)
    return body


def big_drum_a():
    w = A_Y1 - A_Y0
    d = sector_body(w, 22.0)
    # rear shaft through A1 and A2 in the plate housing, shoulder on A2's inner race
    z_shaft = A_SHAFT_Y0 - A_Y0
    d += Pos(0, 0, z_shaft) * Cylinder(B30[0] / 2 - JOURNAL_CLR / 2, -z_shaft, align=MIN)
    d += Pos(0, 0, A_R2_Y + B30[2] - A_Y0) * Cylinder(17.0, A_Y0 - A_R2_Y - B30[2], align=MIN)
    d -= Pos(0, 0, z_shaft) * Cylinder(3.2, 10.0, align=MIN)                     # M5 insert for the rear cap
    # inner output tube
    z = lambda y: y - A_Y0
    t = Pos(0, 0, w) * Cylinder(17.0, B_Y0 - A_Y1, align=MIN)
    t += Pos(0, 0, z(B_Y0)) * Cylinder(B30[0] / 2 - JOURNAL_CLR / 2, B2_Y + B30[2] - B_Y0, align=MIN)   # B1, B2 seats
    t += Pos(0, 0, z(B2_Y + B30[2])) * Cylinder(A_NOSE_R, 120.0 - NOSE_HEX_L - B2_Y - B30[2], align=MIN)  # nose
    t += Pos(0, 0, z(120.0 - NOSE_HEX_L)) * double_d(A_NOSE_R, NOSE_HEX_AF, NOSE_HEX_L)  # double-D drive
    d += t
    d -= Pos(0, 0, w) * Cylinder(9.5, z(B2_Y + B30[2] - 2.0) - w, align=MIN)               # hollow under the 6806 seats
    d -= Pos(0, 0, A_SHAFT_Y0 + 8.5 - A_Y0) * Cylinder(9.0, A_Y1 - A_SHAFT_Y0 - 8.5, align=MIN)   # open to the insert pocket  # hollow shaft and hub, past the M5 insert
    for k in range(4):                                                        # M3 heat-set inserts for flange A
        x, y = polar(A_NOSE_SCREW_R, 45 + 90 * k)
        d -= Pos(x, y, 120.0 - A_Y0 - 6.5) * Cylinder(M3_INSERT / 2, 6.5, align=MIN)
    phi = -dir_deg(OUT, MA)
    return Rot(0, 0, phi) * d


def big_drum_b():
    w = B_Y1 - B_Y0
    d = sector_body(w, 26.0)
    z = lambda y: y - B_Y0
    j0, j1 = WALL_BRG_Y
    d += Pos(0, 0, w) * Cylinder(22.0, z(j0) - w, align=MIN)                  # shoulder on the 6808's inner race
    d += Pos(0, 0, z(j0)) * Cylinder(B40[0] / 2 - JOURNAL_CLR / 2, j1 - j0, align=MIN)  # Ø40 journal in the front wall
    d -= Cylinder((B30[1] + PRESS) / 2, 7.0, align=MIN)                        # B1 from the rear
    d -= Pos(0, 0, 7.0) * Cylinder(B30[1] / 2 + 0.4, z(B2_Y) - 7.0, align=MIN)  # relief: B2 passes B1's seat
    d -= Pos(0, 0, z(B2_Y)) * Cylinder((B30[1] + PRESS) / 2, B30[2], align=MIN)  # B2 from the rear, 4 mm behind B1
    d -= Pos(0, 0, z(B2_Y + B30[2])) * Cylinder(18.5, 1.0, align=MIN)          # step on B2's outer race only
    d -= Cylinder(A_NOSE_R + 1.0, z(j1), align=MIN)                           # clears A's nose
    for k in range(6):                                                        # M3 inserts in the journal end for flange B
        x, y = polar(B_HUB_SCREW_R, 30 + 60 * k)
        d -= Pos(x, y, z(j1) - 6.5) * Cylinder(M3_INSERT / 2, 6.5, align=MIN)
    phi = -dir_deg(OUT, MB)
    return Rot(0, 0, phi) * d


def output_flange_b():
    """Separate outer output flange; fit only after the front wall is installed. Its hub bolts to
    the end of drum B's journal and clamps the 6808's inner race against B's shoulder."""
    hub = WALL_BRG_Y[1] - 105.0                                           # hub reaches back to the bearing
    f = Cylinder(44.0, 6.0, align=MIN) - Cylinder(A_NOSE_R + 1.5, 6.0, align=MIN)
    f += Pos(0, 0, hub) * (Cylinder(22.0, -hub, align=MIN) - Cylinder(A_NOSE_R + 1.5, -hub, align=MIN))
    # no arm spigot: the front face stays flat so the flange prints disc-down; the six bolts locate arm B
    for k in range(6):
        x, y = polar(B_HUB_SCREW_R, 30 + 60 * k)                         # fasten flange to drum B
        f -= Pos(x, y, hub) * Cylinder(M3_CLR / 2, 6.0 - hub, align=MIN)
        f -= Pos(x, y, 3.0) * Cylinder(3.1, 3.0, align=MIN)
        x, y = polar(35.0, 60 * k)                                        # v9.1: the thigh's 6x M3 into heat-set inserts
        f -= Pos(x, y, 0) * Cylinder(M3_INSERT / 2, 6.0, align=MIN)        # pressed in from the front face (v9: captured nuts)
    for k in range(6):                                                    # v9.1: six plain radial pill slots between the inserts
        th = 30 + 60 * k
        f -= chain([(*polar(24.5, th), 2.75), (*polar(31.0, th), 2.75)], 6.0)
    return Rot(0, 0, -dir_deg(OUT, MB)) * f


def a_shaft_cap():
    """Clamps A1's inner race to the shaft from behind: M5 x 12 into the shaft's insert."""
    c = Cylinder(17.0, 3.0, align=MIN) - Cylinder(2.75, 3.0, align=MIN)
    return c


def output_flange_a():
    # local z = 0 at Y 120 (front face of A's Ø22 nose; 1 mm beyond B crank)
    f = Cylinder(29.0, 8.0, align=MIN)                                    # solid: the nose behind it is solid
    f += Pos(0, 0, -NOSE_HEX_L) * Cylinder(17.0, NOSE_HEX_L, align=MIN)          # boss round A's double-D
    f -= Pos(0, 0, -NOSE_HEX_L - 0.01) * double_d(A_NOSE_R + 0.2, NOSE_HEX_AF + 0.4, NOSE_HEX_L + 0.01)
    f += Pos(0, 0, 8.0) * (Cylinder(A_ARM_SPIGOT_R, 2.0, align=MIN) - Cylinder(A_ARM_SPIGOT_R - 1.5, 2.0, align=MIN))
    for k in range(4):
        x, y = polar(A_NOSE_SCREW_R, 45 + 90 * k)                         # 4x M3 into the nose inserts, PCD 14
        f -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
        f -= Pos(x, y, 8.0 - 3.5) * Cylinder(3.1, 3.5, align=MIN)          # head counterbore
    for k in range(6):
        x, y = polar(24.0, 30 + 60 * k)
        f -= Pos(x, y, 8.0 - 6.0) * Cylinder(M3_INSERT / 2, 6.0, align=MIN)
    return f                                                              # v9.1: no blind pockets (only the crank's 6 inserts and the nose's 4 screws)


def test_arm(hub_ro, hub_ri, bolt_r, bolt_phase, length=150.0, t=8.0):
    """Detachable tapered crank with a known 100 mm load point and fixture holes."""
    disks = [Circle(hub_ro), Pos(0, -length) * Circle(14.0)]
    a = extrude(make_hull([e for disk in disks for e in disk.edges()]), t)
    a -= Cylinder(hub_ri, t, align=MIN)
    for k in range(6):
        x, y = polar(bolt_r, bolt_phase + 60 * k)
        a -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, t, align=MIN)
        a -= Pos(x, y, t - 3.2) * Cylinder(3.1, 3.2, align=MIN)
    for radius in (50.0, 75.0, 100.0, 125.0, 150.0):
        a -= Pos(0, -radius, 0) * Cylinder(2.75, t, align=MIN)  # M5 load pin / fastener
    for y0, y1 in ((56.0, 69.0), (81.0, 94.0), (106.0, 119.0)):  # v6: rounded windows between the load holes
        if y1 > length - 20.0:
            continue
        hw = lambda y: hub_ro + (14.0 - hub_ro) * y / length          # the arm's half-width
        r0, r1 = min(5.0, hw(y0) - 5.0), min(5.0, hw(y1) - 5.0)
        if min(r0, r1) > 2.0:
            a -= window([(0, -y0 - r0 + 3.0, r0), (0, -y1 + r1 - 3.0, r1)], t)
    for x in (-7.0, 7.0):
        a -= Pos(x, -135.0, 0) * Cylinder(2.25, t, align=MIN)  # M4 fixture pair
    return a


def base_mount():
    """Flat base the module stands on (upright). Clamp its rear flat (y < -25) to the desk."""
    m = Pos(0, MOUNT_Y0, Z_TOP) * Box(2 * TOP_HALF_W, MOUNT_Y1 - MOUNT_Y0, MOUNT_T, align=(Align.CENTER, Align.MIN, Align.MIN))
    for y in (BACK_Y0 + (BACK_Y1 - BACK_Y0) / 2 - RIB_T / 2, FRONT_Y1 - (FRONT_Y1 - FRONT_Y0) / 2 + RIB_T / 2):
        for x in (-60, 60):
            m -= Pos(x, y, Z_TOP) * Cylinder(M4_CLR / 2, MOUNT_T, align=MIN)
            m -= Pos(x, y, Z_TOP + MOUNT_T - 4.2) * Cylinder(4.0, 4.2, align=MIN)   # head counterbore
    return m


# ----------------------------------------------------------------------------
# Reference parts (not printed)
# ----------------------------------------------------------------------------
def bearing(d, D, w):
    return Cylinder(D / 2, w, align=MIN) - Cylinder(d / 2, w, align=MIN)


def ref_motor(which):
    m = import_step(os.path.join(REF, "ref_cad", "eaglepower_8308.step"))
    x, z = MA if which == "A" else MB
    rz = 0 if which == "A" else 180
    # STEP stator seating face is at local Y 0; XZ-plane turn d == Rot(0, -d, 0)
    return Pos(x, BACK_Y1, z) * Rot(0, -MOTOR_TURN[which], 0) * Rot(0, rz, 0) * m


def ref_moteus(which):
    step = os.path.join(REF, "ref_cad", "moteousr4.step")
    if not os.path.exists(step):                    # the board as exported before (already placed), if the STEP is absent
        from build123d import import_stl
        return import_stl(os.path.join(REF, "ref_cad", f"moteus_r4_placed_{which}.stl"))
    b = import_step(step)
    x, z = MA if which == "A" else MB
    # STEP: PCB in XZ (y 0..1.65), motor side toward -Y, encoder at (0, 8); flip to face +Y,
    # then flip 180 deg and turn 90 deg about the encoder so the connector end points outboard
    return Pos(x, 0, z) * Rot(0, -MOTEUS_TURN[which], 0) * Rot(0, 180, 0) * Pos(0, PCB_Y, -8.0) * Rot(0, 0, 180) * b


# ----------------------------------------------------------------------------
# Build, place, export
# ----------------------------------------------------------------------------
def build():
    ox, oz = OUT
    parts = {}   # name -> (printed-orientation solid, [placed copies], qty, colour)

    def add(name, solid, placed, color="orange"):
        parts[name] = (solid, placed if isinstance(placed, list) else [placed], color)

    bp, fp = back_plate(), front_plate()
    add("Motor_Plate", Rot(90, 0, 0) * bp, bp, "black")
    add("Front_Wall", Rot(90, 0, 0) * fp, fp, "black")

    ps = post(FRONT_Y0 - BACK_Y1)
    add("Post", ps, [to_y(ps, BACK_Y1, x, z) for (x, z) in POSTS[2:]], "dimgray")
    tr = top_rail()
    add("Top_Rail", Rot(90, 0, 0) * tr, [to_y(tr, BACK_Y1, x, z) for (x, z) in TOP_POSTS], "dimgray")  # prints on its side

    sa, sb = small_drum_a(), small_drum_b()
    add("Small_Drum_A", sa, to_y(sa, ROTOR_FACE_Y, *MA), "silver")
    add("Small_Drum_B", sb, to_y(sb, ROTOR_FACE_Y, *MB), "silver")

    da, db = big_drum_a(), big_drum_b()
    add("Big_Drum_A", da, to_y(da, A_Y0, ox, oz), "teal")                # v9.1: one colour per drum
    add("Big_Drum_B", db, to_y(db, B_Y0, ox, oz), "amber")

    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    fa, fb = output_flange_a(), output_flange_b()
    add("Output_Flange_A", fa, to_y(Rot(0, 0, phi_a) * fa, 120.0, ox, oz), "red")
    add("Output_Flange_B", Rot(180, 0, 0) * fb, to_y(fb, 105.0, ox, oz), "red")   # printed disc-down

    eb = end_bracket()
    add("End_Bracket", Rot(0, -90, 0) * eb, eb, "black")                  # prints web-down, ears up
    gt = gantry()
    add("Gantry", Rot(-90, 0, 0) * gt, gt, "black")
    gp = Cylinder(GANTRY_POST_R, GANTRY_Y[0] - FRONT_Y1, align=MIN) - Cylinder(M4_CLR / 2, GANTRY_Y[0] - FRONT_Y1, align=MIN)
    add("Gantry_Post", gp, [to_y(gp, FRONT_Y1, x, z) for x, z in GANTRY_POSTS], "dimgray")

    cap = a_shaft_cap()
    add("A_Shaft_Cap", cap, to_y(Rot(180, 0, 0) * cap, A_R1_Y, ox, oz), "red")


    return parts


# ----------------------------------------------------------------------------
# Five-bar robot leg (optional): thighs bolt to the output flanges in place of the
# test arms, two shins meet at the foot. Modelled hanging (motors up, leg down).
# ----------------------------------------------------------------------------
LEG_L1, LEG_L2 = 100.0, 200.0         # thigh (= the test arm's 100 mm hole), shin
# Leg joints: an M6 socket head bolt clamps the fixed link; the link that turns runs on a bearing
# in a pocket with a lip. The loaded 4-bar knee uses a 636ZZ; the other pivots retain 626ZZ.
PIN_BRG = (6.0, 22.0, 7.0)            # v9.1: 636ZZ on M6 pins, C0 1.37 kN (v9: 626ZZ, 0.95 kN, s0 1.07 at the crouched 891 N)
KNEE_BRG = (10.0, 26.0, 8.0)         # v9: 6000ZZ at the loaded 4-bar knee, C0 1.96 kN (v8: 636ZZ, 6 x 22 x 7, 1.37 kN)
KNEE_PIN_D, KNEE_PIN_CLR = 10.0, 10.3  # v9: Ø10 x M8 shoulder screw (ISO 7379) at the knee
KNEE_BOSS = 2.0                        # v9: the shin's knee boss in front of the core (v8: 1), keeps a 2 mm lip
PIN_D, PIN_CLR = 6.0, 6.4
PIN_BRG_POCKET = (PIN_BRG[1] + PRESS) / 2
KNEE_BRG_POCKET = (KNEE_BRG[1] + PRESS) / 2


PIN_LIP, PIN_LIP_R, PIN_SPACER_R = 8.0 - PIN_BRG[2], 7.3, 4.5   # lip on the outer ring; Ø9 spacer on the inner ring


def brg_pocket(lip_front):
    """636ZZ pocket in an 8 mm link, open toward the neighbouring link (the bearing sits flush with that
    face) with a 1 mm lip on the outer race on the other face (axial location only: the neighbouring link
    holds it the other way); the lip's Ø14.6 hole clears the Ø9 inner-race spacer that the pin clamps."""
    w = PIN_BRG[2]
    z0 = 0.0 if lip_front else PIN_LIP
    return Pos(0, 0, z0) * Cylinder(PIN_BRG_POCKET, w, align=MIN) + Cylinder(PIN_LIP_R, 8.0, align=MIN)


def knee_brg_pocket():
    """v9: 6000ZZ flush at the back, with a 2 mm outer-race lip in the knee's 10 mm local boss."""
    return (Cylinder(KNEE_BRG_POCKET, KNEE_BRG[2], align=MIN)
            + Cylinder(PIN_LIP_R, 8.0 + KNEE_BOSS, align=MIN))
LEG_Y = {"Leg_Thigh_B": 111.0, "Leg_Shin_B": 119.5, "Leg_Foot": 128.0, "Leg_Thigh_A": 128.0, "Leg_Shin_A": 136.5}


def leg_shin():
    """Knee hole at the origin, foot hole LEG_L2 along -y; M5 pins."""
    disks = [Circle(10.0), Pos(0, -LEG_L2) * Circle(10.0)]
    s = extrude(make_hull([e for d in disks for e in d.edges()]), 8.0)
    slot = [Pos(0, -30.0) * Circle(4.0), Pos(0, -LEG_L2 + 30.0) * Circle(4.0)]
    s -= extrude(make_hull([e for d in slot for e in d.edges()]), 8.0)
    for y in (0.0, -LEG_L2):                                          # 2x MR105ZZ in each end
        s -= Pos(0, y, 0) * Cylinder(PIN_BRG_POCKET, 8.0, align=MIN)
    return s


def leg_foot():
    return Cylinder(12.0, 8.0, align=MIN) - Cylinder(2.75, 8.0, align=MIN)


def leg_rest():
    """Rest pose (outputs centred): thigh angles (deg, global XZ) and knee/foot points."""
    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    q = {"A": (90.0 - phi_a + 180) % 360 - 180, "B": (90.0 - (phi_b + 180.0) + 180) % 360 - 180}
    knee = {w: (OUT[0] + LEG_L1 * math.cos(math.radians(q[w])), OUT[1] + LEG_L1 * math.sin(math.radians(q[w])))
            for w in "AB"}
    mx, mz = (knee["A"][0] + knee["B"][0]) / 2, (knee["A"][1] + knee["B"][1]) / 2
    dx, dz = knee["B"][0] - knee["A"][0], knee["B"][1] - knee["A"][1]
    half = math.hypot(dx, dz) / 2
    h = math.sqrt(LEG_L2 ** 2 - half ** 2)
    nx, nz = -dz / (2 * half), dx / (2 * half)
    if (mx + nx - OUT[0]) ** 2 + (mz + nz - OUT[1]) ** 2 < (mx - nx - OUT[0]) ** 2 + (mz - nz - OUT[1]) ** 2:
        nx, nz = -nx, -nz                                                  # foot away from the hip
    return q, knee, (mx + h * nx, mz + h * nz)


def leg_ik(r, phi):
    """Thigh angles (deg) putting the foot r mm from the hip at global XZ angle phi (deg).
    A is the thigh at phi - spread, B at phi + spread."""
    lo, hi = 0.0, 179.0
    for _ in range(60):                                                    # r falls as the spread grows
        mid = (lo + hi) / 2
        sp = math.radians(mid)
        rr = LEG_L1 * math.cos(sp) + math.sqrt(LEG_L2 ** 2 - (LEG_L1 * math.sin(sp)) ** 2)
        lo, hi = (mid, hi) if rr > r else (lo, mid)
    return phi - lo, phi + lo


def leg_parts():
    """name -> (printed solid, placed solid, colour), at the rest pose."""
    ox, oz = OUT
    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    q, knee, foot = leg_rest()
    thigh_a = test_arm(31.0, A_ARM_SPIGOT_R + 0.1, 24.0, 30.0, length=LEG_L1)
    thigh_b = test_arm(47.0, 27.1, 35.0, 0.0, length=LEG_L1)
    out = {"Leg_Thigh_A": (thigh_a, to_y(Rot(0, 0, phi_a) * thigh_a, LEG_Y["Leg_Thigh_A"], ox, oz), "red"),
           "Leg_Thigh_B": (thigh_b, to_y(Rot(0, 0, phi_b + 180.0) * thigh_b, LEG_Y["Leg_Thigh_B"], ox, oz), "black")}
    shin = leg_shin()
    for w in "AB":
        kx, kz = knee[w]
        alpha = math.degrees(math.atan2(foot[1] - kz, foot[0] - kx))
        out[f"Leg_Shin_{w}"] = (shin, to_y(Rot(0, 0, 90.0 - alpha) * shin, LEG_Y[f"Leg_Shin_{w}"], kx, kz),
                                "red" if w == "A" else "black")
    ft = leg_foot()
    out["Leg_Foot"] = (ft, to_y(ft, LEG_Y["Leg_Foot"], *foot), "dimgray")
    return out


# ----------------------------------------------------------------------------
# 4-bar leg (optional): output B swings the thigh; output A turns a crank that drives the
# knee through a parallel coupler rod (a parallelogram on one side of the leg), so the shin
# always points opposite the crank. Thigh and shin 150 mm, crank and shin lever 45 mm.
# ----------------------------------------------------------------------------
# CAPSTAN_LEG_L (mm) lengthens thigh and shin together (the quadruped's longer legs); default 150 = this leg
L4_LEN = float(os.environ.get("CAPSTAN_LEG_L", "200"))
L4_K = L4_LEN / 150.0                 # shin features along its length scale with it
L4_L1, L4_L2, L4_C = L4_LEN, L4_LEN, 55.0
L4_DELTA = -20.0                      # shin lever bent 20 deg off the shin line (a boomerang shin)
L4_FOOT_R = 23.0                      # bulb foot radius (soft TPU / cast silicone, as on Stanley)
L4_CLAW_R = (13.0, 19.0)              # the shin's two-prong claw, inside the bulb
L4_Q0 = {"thigh": -60.0, "shin": -120.0}                       # rest angles (deg, global XZ), outputs centred
L4_Y = {"Leg4_Thigh": 111.0, "Leg4_Shin": 119.5, "Leg4_Crank": 128.0, "Leg4_Coupler": 136.5}


def chain(points, t=8.0):
    """Smooth printed link: union of hulls of consecutive (x, y, r) circles, extruded t."""
    body = None
    for (x0, y0, r0), (x1, y1, r1) in zip(points, points[1:]):
        h = extrude(make_hull([e for c in (Pos(x0, y0) * Circle(r0), Pos(x1, y1) * Circle(r1)) for e in c.edges()]), t)
        body = h if body is None else body + h
    return body


def window(points, t=8.0):
    return chain(points, t)


def soften(solid, r=1.2):
    """Round the outline edges of a flat link (both faces); fall back to sharp if OCCT refuses."""
    try:
        edges = solid.edges().filter_by(Axis.Z, reverse=True)
        return solid.fillet(r, edges.group_by(Axis.Z)[0] + edges.group_by(Axis.Z)[-1])
    except Exception:
        return solid


def leg4_thigh(bolt_phase):
    """Tapered thigh on flange B: hub with the arm-B bolt circle, M5 knee 150 mm out, long window."""
    th = chain([(0, 0, 47.0), (0, -L4_L1, 15.0)])
    th -= window([(0, -58.0, 11.0), (0, -L4_L1 + 30.0, 6.0)])
    th -= Cylinder(27.1, 8.0, align=MIN)
    for k in range(6):
        x, y = polar(35.0, bolt_phase + 60 * k)
        th -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
        th -= Pos(x, y, 8.0 - 3.2) * Cylinder(3.1, 3.2, align=MIN)
    th -= Pos(0, -L4_L1, 0) * Cylinder(KNEE_PIN_CLR / 2, 8.0, align=MIN)    # v9: Ø10 knee pin
    return soften(th)


CRANK_SHOULDER_R = 10.5               # v9: Ø21 shoulder to the 6903's inner race (v8: Ø19)
CRANK_PIN_R = 15.0                     # v9: the crank's pin end (v8: 13.5, v6.2: 11); SF 1.9 at 9:1 with 13.5
CRANK_STUB = (8.0, GANTRY_Y[0] + GANTRY_BRG[2] + 2.0 - 128.0)       # crank-local z range of the Ø15 front stub


def front_stub(part):
    """Fill the hub bore past flange A's locating recess and add the Ø15 stub that runs in the gantry's
    6902 (output A held on both sides of the leg). For parts on flange A: local z 0 = Y 128, 8 thick."""
    part += Pos(0, 0, 2.5) * Cylinder(A_ARM_SPIGOT_R + 0.2, 8.0 - 2.5, align=MIN)
    part += Pos(0, 0, CRANK_STUB[0]) * Cylinder(GANTRY_BRG[0] / 2 - JOURNAL_CLR / 2, CRANK_STUB[1] - CRANK_STUB[0], align=MIN)
    part += Pos(0, 0, CRANK_STUB[0]) * Cylinder(CRANK_SHOULDER_R, GANTRY_Y[0] - 128.0 - CRANK_STUB[0] - 0.5, align=MIN)  # shoulder
    return part


def leg4_crank(bolt_phase, stub=True):
    """Crank on output A: arm A's hub and bolt circle, M5 pin L4_C out; with the gantry, a Ø15 stub."""
    c = chain([(0, 0, 31.0), (0, -L4_C, CRANK_PIN_R)])
    c -= Cylinder(A_ARM_SPIGOT_R + 0.1, 8.0, align=MIN)                      # bore over flange A's locating ring
    if stub:
        c = front_stub(c)
    for k in range(6):
        x, y = polar(24.0, bolt_phase + 60 * k)
        c -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
        c -= Pos(x, y, 8.0 - 3.2) * Cylinder(3.1, 3.2, align=MIN)
    c -= Pos(0, -L4_C, 0) * Cylinder(PIN_CLR / 2, 8.0, align=MIN)
    return soften(c)


def leg4_claw():
    """Two-prong claw at the shin's end: a ring open at the bottom (local -y), centred on the foot point.
    The bulb foot slips over it; an M3 screw through each prong tip holds it."""
    r0, r1 = L4_CLAW_R
    c = Cylinder(r1, 8.0, align=MIN) - Cylinder(r0, 8.0, align=MIN)
    c -= pie(r1 + 1.0, -90.0 - 50.0, -90.0 + 50.0, 8.0)                       # the gap between the prongs
    for a in (-90.0 - 62.0, -90.0 + 62.0):
        x, y = polar((r0 + r1) / 2, a)
        c -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
    return c


def leg4_foot():
    """Bulb foot (local z = shin plane mid-depth, centred on the foot point): a slightly flattened ball
    with an annular slot for the claw and two M3 cross holes. Print in TPU or cast in silicone."""
    r = L4_FOOT_R
    b = Sphere(r) & Box(2 * r, 2 * r, 2 * r * 0.8)                          # flattened across the shin
    r0, r1 = L4_CLAW_R
    b -= (Cylinder(r1 + 0.4, 8.6) - Cylinder(r0 - 0.4, 8.6)) - pie(r1 + 2.0, -140.0, -40.0, 8.6).moved(Pos(0, 0, -4.3))
    b -= Pos(4.0, r * 0.6, 0) * Box(30.0, r, 8.6)                            # entry for the shin above the claw
    for a in (-90.0 - 62.0, -90.0 + 62.0):
        x, y = polar((r0 + r1) / 2, a)
        b -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 2 * r)
        for sgn in (-1, 1):
            b -= Pos(x, y, sgn * (r * 0.8 - 2.5)) * Cylinder(3.0, 5.0)          # counterbore / nut pocket
    return b


# v6 sandwich shin (Stanley's tibia / fibula): the 8 mm core (knee, lever, claw) plus a 4 mm plate on
# each face below the knee, 16 mm thick in all, clamped by 3x M3 x 16 countersunk screws and nuts.
# The back plate starts 51 mm below the knee, 3 mm past where it would meet the thigh in the deepest
# crouch; the front plate (nothing in its plane there) reaches up to 34 mm, clear of the knee pin's head.
SHIN_PLATE_T = 4.0
SHIN_PLATE_TOP = {False: (7.3, -51.0, 15.5), True: (4.9, -34.0, 16.2)}           # back, front
SHIN_PLATE_PTS = [(10.0, -70.0 * L4_K, 16.5), (5.4, -112.0 * L4_K, 12.5)]                       # flush with the reinforced core plus 0.5
SHIN_BOLTS = [(6.6, -46.0 * L4_K), (8.6, -84.0 * L4_K), (5.6, -113.0 * L4_K)]
SHIN_WINDOWS = [[(7.9, -55.0 * L4_K, 4.0), (9.8, -73.0 * L4_K, 4.0)], [(7.8, -92.0 * L4_K, 3.5), (6.5, -104.0 * L4_K, 3.5)]]
SHIN_LEVER_R = 18.0                       # 7 mm wall around the lever 636
SHIN_CORE_R = (19.0, 16.0, 11.0)          # wider core at knee and bend; plates follow below the knee


def leg4_shin_plate(front):
    """Side plate of the sandwich shin (shin-local x, y; z 0..4). Front: countersunk for the M3 heads;
    back: hex pockets for the nuts on its outer face. Two windows show the red core."""
    p = chain([SHIN_PLATE_TOP[front]] + SHIN_PLATE_PTS, SHIN_PLATE_T)
    for w in SHIN_WINDOWS:
        p -= window(w, SHIN_PLATE_T)
    for x, y in SHIN_BOLTS:
        p -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, SHIN_PLATE_T, align=MIN)
        if front:
            p -= Pos(x, y, SHIN_PLATE_T - 1.8) * Cone(M3_CLR / 2, 3.15, 1.8, align=MIN)
        else:                                                             # v9.1: short M3 heat-set insert from the outer face
            p -= Pos(x, y, 0) * Cylinder(M3_INSERT / 2, SHIN_PLATE_T, align=MIN)   # (v9: a captured nut in a hex pocket)
    return soften(p, 0.8)


SHIN_NECK_WEB = [(4.0, 23.0, 11.5), (-5.0, 11.0, 4.0), (13.0, 11.0, 4.0)]   # (x, y above the foot point, r)


def shin_neck_web():
    """Web blending the shin's neck into the claw ring (fea_check.py: the bare neck/ring corner was a
    notch at about twice the shin's bending stress). It stays inside the bulb's entry slot (x -11..19,
    above the foot point + 2.3) and outside the bulb's core (r < the claw's inner radius)."""
    fy = -L4_L2
    web = make_hull([e for x, y, r in SHIN_NECK_WEB for e in (Pos(x, fy + y) * Circle(r)).edges()])
    return extrude(web, 8.0) - Pos(0, fy, 0) * Cylinder(L4_CLAW_R[0], 8.0, align=MIN)


def leg4_shin():
    """Boomerang shin core: knee at the origin, foot ring centred LEG length along -y, lever bent off the
    shin line to the coupler pin. Local angle phi maps to global -(phi + psi) when placed."""
    lx, ly = polar(L4_C, 90.0 - L4_DELTA)
    r0, r1, r2 = SHIN_CORE_R
    s = chain([(lx, ly, SHIN_LEVER_R), (0, 0, r0), (10.0, -70.0 * L4_K, r1), (4.0, -L4_L2 + L4_CLAW_R[1] + 2.0, r2)])
    for x, y in SHIN_BOLTS:
        s -= Pos(x, y, 0) * Cylinder(M3_CLR / 2, 8.0, align=MIN)
    s += Pos(0, -L4_L2, 0) * leg4_claw()
    s += shin_neck_web()
    s += Pos(0, 0, 8.0) * Cylinder(17.0, KNEE_BOSS, align=MIN)         # v9: local 10 mm boss keeps a 2 mm lip on the 8 mm 6000
    s -= knee_brg_pocket()                                              # 636ZZ at the knee (thigh behind)
    s -= Pos(lx, ly, 0) * brg_pocket(lip_front=False)                 # 636ZZ at the lever pin (coupler in front)
    return soften(s, 1.0)


COUPLER_PROFILE = [(0.0, 29.0), (30.0, 26.0), (65.0, 26.0), (105.0, 48.0), (170.0, 44.0), (L4_L1, 34.0)]   # v9.1: (mm from C, width)
COUPLER_RIB = (34.0, 4.0, 90.0, 182.0)  # back rib: width, depth (1 mm short of the shin's front plate), from / to (mm from C)
A_PUSH_TAU = 25.0                     # N*m: output A's software limit in the direction that pushes the coupler
COUPLER_WAIST_DX = float(os.environ.get("CAPSTAN_WAIST_DX", "-10.5"))  # waist offset sideways, away from the crank's gantry
                                      # stub at the knee limit (straight: 461 mm3 into it; -9 clears; v9's bow: 16)


def leg4_coupler():
    """v9.1: a solid bar, 8 thick: 29 mm at the crank pin, a 26 mm waist offset 10.5 mm sideways where it passes
    the hip (30-65 mm out: folded to the knee limit, the crank's Ø21 gantry stub passes there; ~15 MPa of
    bending at 891 N), 48 in the middle, 34 at the lever;
    a 34 x 4 mm rib on its back from 90 mm out (clear of the crank's r31 hub). It pulls when standing
    (655-891 N) and is pushed only when output A drives the shin: out-of-plane buckling ~1020 N pin-ended, so
    output A is limited to 25 N*m in that direction (455 N, SF 2.2; check_strength.py). v9: a 16 mm bow,
    ~63 MPa of bending in its 13 mm neck."""
    c = chain([(COUPLER_WAIST_DX if 0.0 < s < 100.0 else 0.0, -s, w / 2) for s, w in COUPLER_PROFILE])
    rw, rd, r0, r1 = COUPLER_RIB
    c += Pos(0, 0, -rd) * chain([(0, -r0 - rw / 2, rw / 2), (0, -r1 + rw / 2, rw / 2)], rd)
    c -= brg_pocket(lip_front=True)                                    # 636ZZ on the crank pin (crank behind)
    c -= Pos(0, -L4_L1, 0) * Cylinder(PIN_CLR / 2, 8.0, align=MIN)     # clamped on the shin lever pin
    return soften(c, 1.0)


def leg4_points(qt, qs):
    """Model (x, z) of the knee, foot (ring centre), crank pin C and shin lever pin D."""
    u = lambda q: (math.cos(math.radians(q)), math.sin(math.radians(q)))
    ox, oz = OUT
    ql = qs + 180.0 + L4_DELTA                                         # lever (= crank) direction
    k = (ox + L4_L1 * u(qt)[0], oz + L4_L1 * u(qt)[1])
    f = (k[0] + L4_L2 * u(qs)[0], k[1] + L4_L2 * u(qs)[1])
    c = (ox + L4_C * u(ql)[0], oz + L4_C * u(ql)[1])
    d = (k[0] + L4_C * u(ql)[0], k[1] + L4_C * u(ql)[1])
    return k, f, c, d


def leg4_ik(r, phi):
    """Thigh and shin angles (deg) for the foot r mm from the hip at angle phi; knee bends forward."""
    cos_k = (r ** 2 - L4_L1 ** 2 - L4_L2 ** 2) / (2 * L4_L1 * L4_L2)
    kb = math.degrees(math.acos(max(-1.0, min(1.0, cos_k))))          # 0 = straight
    a = math.degrees(math.atan2(L4_L2 * math.sin(math.radians(kb)), L4_L1 + L4_L2 * math.cos(math.radians(kb))))
    qt = phi + a
    return qt, qt - kb


def leg4_parts(gantry=True):
    """name -> (printed solid, placed solid, colour), at the rest pose."""
    ox, oz = OUT
    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    qt, qs = L4_Q0["thigh"], L4_Q0["shin"]
    k, f, c, d = leg4_points(qt, qs)
    psi_t, psi_c = 90.0 - qt, 90.0 - (qs + 180.0 + L4_DELTA)          # local -y -> global angle 90 - psi
    thigh = leg4_thigh((phi_b + 180.0 - psi_t) % 60.0)
    crank = leg4_crank((phi_a + 330.0 - psi_c) % 60.0, stub=gantry)
    shin, coupler = leg4_shin(), leg4_coupler()
    return {"Leg4_Thigh": (thigh, to_y(Rot(0, 0, psi_t) * thigh, L4_Y["Leg4_Thigh"], ox, oz), "black"),
            "Leg4_Crank": (crank, to_y(Rot(0, 0, psi_c) * crank, L4_Y["Leg4_Crank"], ox, oz), "red"),
            "Leg4_Shin": (shin, to_y(Rot(0, 0, 90.0 - qs) * shin, L4_Y["Leg4_Shin"], *k), "red"),
            "Leg4_Coupler": (coupler, to_y(Rot(0, 0, psi_t) * coupler, L4_Y["Leg4_Coupler"], *c), "red"),
            "Leg4_Shin_Plate_Back": (leg4_shin_plate(False), to_y(Rot(0, 0, 90.0 - qs) * leg4_shin_plate(False),
                                                                   L4_Y["Leg4_Shin"] - SHIN_PLATE_T, *k), "black"),
            "Leg4_Shin_Plate_Front": (leg4_shin_plate(True), to_y(Rot(0, 0, 90.0 - qs) * leg4_shin_plate(True),
                                                                  L4_Y["Leg4_Shin"] + 8.0, *k), "black"),
            "Leg4_Foot": (leg4_foot(), to_y(Rot(0, 0, 90.0 - qs) * Pos(0, -L4_L2, 4.0) * leg4_foot(), L4_Y["Leg4_Shin"], *k), "dimgray")}


M6_LENGTHS = (16, 20, 25, 30, 35, 40, 45, 50)


def pin_joint(x, z, layers, knee=False):
    """One leg pivot at model (x, z), axis along Y. layers: [(y0, y1, turns_on_bearings)] in +Y order.
    Bolt head at the front, washer each side, nylon-insert nut at the back, a 0.5 mm shim in each
    gap between links (a spacer where the gap is wider). Returns (hardware, bearings, spec)."""
    hw, brg = [], []
    y_back, y_front = layers[0][0], layers[-1][1]
    grip = (y_front - y_back) + 1.2                                       # front washer only: the nut sits on the
    if knee:                                                              # v9: Ø10 x M8 shoulder screw, M8 nylon nut
        length = next(L for L in (16, 20, 25, 30) if L >= grip)
        hw.append(to_y(Pos(0, 0, -length) * Cylinder(5.0, length, align=MIN) + Pos(0, 0, -length - 12.0) * Cylinder(4.0, 12.0, align=MIN)
                       + Cylinder(8.0, 10.0, align=MIN) - Pos(0, 0, 5.5) * hex_prism_z(5.0, 4.6), y_front + 1.6, x, z))
        ring = lambda od, t: Cylinder(od / 2, t, align=MIN) - Cylinder(5.3, t, align=MIN)
        hw.append(to_y(ring(20.0, 1.6), y_front, x, z))                       # Ø10.5 x 20 x 1.6 washer
        hw.append(to_y(hex_prism_z(13.0, 8.0) - Cylinder(4.0, 8.0, align=MIN), y_back - 12.0 + 4.0, x, z))   # M8 nylon-insert nut
        spacer_r = 6.5
    else:
        length = next(L for L in M6_LENGTHS if L >= grip + 6.0 + 1.0)     # fixed link or the inner-race spacer
        hw.append(to_y(shcs(6, length, 10.0, 6.0), y_front + 1.2, x, z))
        ring = lambda od, t: Cylinder(od / 2, t, align=MIN) - Cylinder(3.2, t, align=MIN)
        hw.append(to_y(ring(12.0, 1.2), y_front, x, z))                       # M6 washers, 6.4 x 12 x 1.2
        hw.append(to_y(hex_prism_z(10.0, 6.0) - Cylinder(3.0, 6.0, align=MIN), y_back - 6.0, x, z))   # M6 nylon-insert nut
        spacer_r = PIN_SPACER_R
    gaps = []
    for (a0, a1, _), (b0, b1, _) in zip(layers, layers[1:]):              # Ø8: touches only the inner race
        if b0 - a1 > 0.05:
            hw.append(to_y(ring(2 * spacer_r, b0 - a1), a1, x, z))
            gaps.append(round(b0 - a1, 1))
    bearing_spec = KNEE_BRG if knee else PIN_BRG
    lip = 2.0 if knee else PIN_LIP
    for i, (y0, y1, on_brg) in enumerate(layers):
        if on_brg:                                                          # bearing flush with the open face
            lip_front = i == len(layers) - 1
            yb = y0 if lip_front else y1 - bearing_spec[2]
            brg.append(to_y(bearing(*bearing_spec), yb, x, z))
            ys = y1 - lip if lip_front else y0                              # spacer touches the inner race through the lip
            hw.append(to_y(ring(2 * spacer_r, lip), ys, x, z))
    spec = {"bolt": f"Ø10 x {length} shoulder screw (M8), M8 nylon nut" if knee else f"M6 x {length}", "bearings": len(brg), "gaps": gaps}
    return Compound(children=hw), Compound(children=brg), spec


def leg_hardware(variant):
    """label -> (hardware or bearings compound, colour, joint, spec) at the rest pose, model coordinates."""
    out = {}
    if variant == "fivebar":
        q, knee, foot = leg_rest()
        y = LEG_Y
        joints = {"knee_A": (knee["A"], [(y["Leg_Thigh_A"], y["Leg_Thigh_A"] + 8, False), (y["Leg_Shin_A"], y["Leg_Shin_A"] + 8, True)]),
                  "knee_B": (knee["B"], [(y["Leg_Thigh_B"], y["Leg_Thigh_B"] + 8, False), (y["Leg_Shin_B"], y["Leg_Shin_B"] + 8, True)]),
                  "foot": (foot, [(y["Leg_Shin_B"], y["Leg_Shin_B"] + 8, True), (y["Leg_Foot"], y["Leg_Foot"] + 8.0, False),
                                  (y["Leg_Shin_A"], y["Leg_Shin_A"] + 8, True)])}
        prefix = "Leg_HW_"
    else:
        k, f, c, d = leg4_points(L4_Q0["thigh"], L4_Q0["shin"])
        y = L4_Y
        joints = {"knee": (k, [(y["Leg4_Thigh"], y["Leg4_Thigh"] + 8, False), (y["Leg4_Shin"], y["Leg4_Shin"] + 8 + KNEE_BOSS, True)]),
                  "crank": (c, [(y["Leg4_Crank"], y["Leg4_Crank"] + 8, False), (y["Leg4_Coupler"], y["Leg4_Coupler"] + 8, True)]),
                  "lever": (d, [(y["Leg4_Shin"], y["Leg4_Shin"] + 8, True), (y["Leg4_Coupler"], y["Leg4_Coupler"] + 8, False)])}
        prefix = "Leg4_HW_"
        qs = L4_Q0["shin"]
        r0, r1 = L4_CLAW_R
        y0 = L4_Y["Leg4_Shin"] + 4.0 - L4_FOOT_R * 0.8                    # bulb's far face
        screws = []
        for a in (-90.0 - 62.0, -90.0 + 62.0):                              # 2x M3 x 40 across the bulb and claw
            px, py = polar((r0 + r1) / 2, a)
            local = Rot(0, 0, 90.0 - qs) * Pos(px, py - L4_L2, 0)
            pt = to_y(local * Pos(0, 0, 0) * Cylinder(0.1, 0.1), 0.0, *k)
            cx, cz = pt.center().X, pt.center().Z
            screws.append(to_y(Rot(180, 0, 0) * shcs(3, 40, 5.5, 3.0), y0, cx, cz))
            screws.append(to_y(hex_prism_z(5.5, 2.4) - Cylinder(1.5, 2.4, align=MIN), y0 + 2 * L4_FOOT_R * 0.8 - 2.4, cx, cz))
        out[f"{prefix}foot_screws"] = (Compound(children=screws), "screw", "shin", {"bolt": "M3 x 40", "bearings": 0, "gaps": []})
        cs = Cone(3.0, 1.5, 1.7, align=MIN) + Pos(0, 0, 1.7) * Cylinder(1.5, 16.0 - 1.7, align=MIN)   # M3 x 16 DIN 7991
        loc = lambda q: to_y(Rot(0, 0, 90.0 - qs) * q, L4_Y["Leg4_Shin"], *k)
        out[f"{prefix}shin_screws"] = (Compound(children=[loc(Pos(x, y, 8.0 + SHIN_PLATE_T) * Rot(180, 0, 0) * cs)
                                                          for x, y in SHIN_BOLTS]),
                                       "screw", "shin", {"bolt": "M3 x 16 countersunk", "bearings": 0, "gaps": []})
        out[f"{prefix}shin_inserts"] = (Compound(children=[loc(Pos(x, y, -SHIN_PLATE_T) * insert(4.6, 2.6, 3.8)) for x, y in SHIN_BOLTS]),   # flush with the outer face
                                        "brass", "shin", {"bolt": "M3 x 3.8 heat-set insert", "bearings": 0, "gaps": []})
    for j, ((x, z), layers) in joints.items():
        hw, brg, spec = pin_joint(x, z, layers, knee=(variant != "fivebar" and j == "knee"))
        out[f"{prefix}{j}_pin"] = (hw, "screw", j, spec)
        if brg.solids():
            out[f"{prefix}{j}_brg"] = (brg, "steel", j, spec)
    return out


def stand_hardware():
    """4x M3 x 12 through the carriage upright into the MGN12H block (move with the rig, model coordinates)."""
    zc = Z_TOP + MOUNT_T + CARRIAGE_UP / 2
    yf = CARRIAGE_Y0 + 8.0                                             # carriage upright's front face
    screws = [Pos(x, yf, z) * Rot(-90, 0, 0) * shcs(3, 12, 5.5, 3.0)              # 4 mm into the block
              for x in (-10.0, 10.0) for z in (zc - 10.0, zc + 10.0)]
    return {"Carriage_HW_block_screws": (Compound(children=screws), "screw")}


# ----------------------------------------------------------------------------
# Test stand (optional): the rig hangs from a printed carriage on an MGN12 rail, on a vertical
# 2040 V-slot extrusion with a 2040 T-foot. Stand parts are in world coordinates (floor at z 0);
# the carriage and its block move with the rig (model coordinates).
# ----------------------------------------------------------------------------
STAND_Y = {"extrusion": (-175.0, -155.0), "rail": (-155.0, -147.0), "block": (-147.0, -137.0)}
CARRIAGE_Y0, CARRIAGE_UP = -137.0, 80.0


def vslot(size_x, size_y, length, axis="z"):
    """V-slot-like extrusion: a box with a 6 mm groove centred in every 20 mm of each face."""
    b = Box(size_x, size_y, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for i in range(int(size_x // 20)):
        x = -size_x / 2 + 10 + 20 * i
        for sy in (-1, 1):
            b -= Pos(x, sy * size_y / 2, 0) * Box(6.0, 6.0, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for i in range(int(size_y // 20)):
        y = -size_y / 2 + 10 + 20 * i
        for sx in (-1, 1):
            b -= Pos(sx * size_x / 2, y, 0) * Box(6.0, 6.0, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Cylinder(2.1, length, align=MIN)
    return b


def stand_parts(x0=0.0, dy=0.0):
    """Static stand in world coordinates (rail at x = x0, moved dy in y): name -> solid."""
    y0, y1 = STAND_Y["extrusion"]
    yc = (y0 + y1) / 2
    out = {"Stand_Upright_2040x900": Pos(0, yc, 0) * vslot(40, 20, 900.0)}
    out["Stand_Foot_X_2040x500"] = Pos(0, -175.0, 10.0) * Rot(0, 90, 0) * Pos(0, 0, -250.0) * vslot(20, 40, 500.0)
    for sx in (-1, 1):
        out[f"Stand_Foot_Y_{'L' if sx < 0 else 'R'}_2040x300"] = (
            Pos(sx * 230.0, -195.0, 10.0) * Rot(90, 0, 0) * vslot(40, 20, 300.0))
    r0, r1 = STAND_Y["rail"]
    out["Stand_MGN12_rail_700"] = Pos(0, (r0 + r1) / 2, 120.0) * Box(12.0, r1 - r0, 700.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return {n: Pos(x0, dy, 0) * s for n, s in out.items()}


def rail_carriage():
    """Replaces the base mount for the leg: same footprint and M4 holes into the rib inserts,
    plus an upright plate at the back for an MGN12H block (4x M3, 20 x 20)."""
    ys = (BACK_Y0 + (BACK_Y1 - BACK_Y0) / 2 - RIB_T / 2, FRONT_Y1 - (FRONT_Y1 - FRONT_Y0) / 2 + RIB_T / 2)
    c = Pos(0, CARRIAGE_Y0, Z_TOP) * Box(2 * TOP_HALF_W, MOUNT_Y1 - CARRIAGE_Y0, MOUNT_T, align=(Align.CENTER, Align.MIN, Align.MIN))
    c += Pos(0, CARRIAGE_Y0, Z_TOP) * Box(70.0, 8.0, MOUNT_T + CARRIAGE_UP, align=(Align.CENTER, Align.MIN, Align.MIN))
    for x in (-31.0, 31.0):                                                # gussets
        g = Polygon((0, 0), (60, 0), (0, CARRIAGE_UP - 5), align=None)
        c += Location(Plane(origin=(x - 4.0, CARRIAGE_Y0 + 8.0, Z_TOP + MOUNT_T), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(g, 8.0)
    for y in ys:
        for x in (-60, 60):
            c -= Pos(x, y, Z_TOP) * Cylinder(M4_CLR / 2, MOUNT_T, align=MIN)
            c -= Pos(x, y, Z_TOP + MOUNT_T - 4.2) * Cylinder(4.0, 4.2, align=MIN)
    zc = Z_TOP + MOUNT_T + CARRIAGE_UP / 2
    for x in (-10.0, 10.0):
        for z in (zc - 10.0, zc + 10.0):
            c -= Pos(x, CARRIAGE_Y0 - 1, z) * Rot(-90, 0, 0) * Cylinder(M3_CLR / 2, 10.0, align=MIN)
    return c


def carriage_parts():
    """Moving with the rig (model coordinates): the printed carriage and the MGN12H block."""
    b0, b1 = STAND_Y["block"]
    zc = Z_TOP + MOUNT_T + CARRIAGE_UP / 2
    block = Pos(0, (b0 + b1) / 2, zc) * Box(27.0, b1 - b0, 45.4)
    return {"Rail_Carriage": (rail_carriage(), "gray"), "REF_MGN12H_block": (block, "steel")}


# ----------------------------------------------------------------------------
# 3-DOF version (compact, Stanley-style): a third 8308 adds hip ab/ad about an axis along the
# body (X) at the motors' height. The ab/ad unit sits at the module's back end (-X): a yoke on the
# module (it replaces the base mount and bolts to the same rib inserts) grips the hollow hub of the
# ab/ad drum; the hub runs in two 6808s, one in motor C's plate and one in the outer end plate; motor
# C sits above the axis between the plates and drives the drum through a third 8:1 capstan, which is
# drum B's stage reused: ab_loc maps that stage (main frame) onto the ab/ad axis.
# Ab/ad angle phi turns the module about +X (y toward z).
# ----------------------------------------------------------------------------
AB_Y, AB_Z = 45.0, -60.0              # ab/ad axis (model y, z), along X; v5: almost level with the hip output
AB_X0 = -142.0                        # x of the sector's outer face (= main y B_Y0); main +Y -> global +X
AB_SHIFT = 30.0                       # motor C sits 30 mm closer to its drum than motor B does
AB_RANGE = 20.0                       # ab/ad used in the films (deg); the sector allows +-34
# along X from the module outward: yoke | inner plate (6808 + motor C's tip 6902) | ab/ad drum |
# motor C | outer plate (motor C, 6808, moteus C behind the motor's magnet)
AB_X = {"yoke": END_X, "inner": (-122.0, -114.0), "outer": (-186.5, -179.0)}
AB_TUBE_R, AB_TUBE_END = B40[0] / 2 - JOURNAL_CLR / 2, END_X[0]  # hub journal in the 6808s; the hub end bears on the bracket web
# (the inner 6808 floats on the hub; the outer one, on the drum's shoulder, locates it)
AB_POSTS = [(128.0, AB_Z - 20.0), (128.0, AB_Z + 60.0)]  # (y, z) of the posts: front side, past the drum's sweep
# v6 cradle: the module turns on the hub at its rear end AND on a trunnion at its front end. A front
# bracket (mirror of the end bracket) carries a bolted-on Ø30 trunnion in a 6806 in the cradle's front
# plate; a spine behind the module ties the outer, inner and front plates together and carries the rail
# blocks (leg builds) or the two desk angles (desk builds). The spine sits 7 mm clear of the module's
# swept volume at the sector's full +-34 deg; the rear 6808s locate the module axially, the front 6806 floats.
AB_RAIL_Y0 = -85.0                    # back face of the spine (v5 rail plate: -60, behind the ab/ad unit only)
SPINE_T = 10.0
AB_RAIL_X = -40.0                     # the rail sits behind the middle of the cradle, near the rig's centre of mass
AB_BLOCKS_Z = (AB_Z - 10.0, AB_Z + 90.0)                  # two MGN12H blocks, 100 mm apart, take the hanging leg's moment
FB_X = (98.0, 106.0)                  # front bracket web
TRUNNION_X = (106.0, 110.0, 119.0)    # flange face on the web, flange end / journal start, journal end
TRUNNION_D = 29.9                     # floating journal in the 6806 (0.1 mm clearance: slides as the module grows)
HF_X = (111.0, 119.0)                 # cradle front plate: 6806 pressed in from outside against a 1 mm lip (prints lip-down)
TRUNNION_SCREWS = [(AB_Y + 20.0 * math.cos(math.radians(45 + 90 * k)), AB_Z + 20.0 * math.sin(math.radians(45 + 90 * k)))
                   for k in range(4)]  # 4x M4 x 10 countersunk, PCD 40
CARRIAGE5_Y0 = -60.0                  # leg without ab/ad: the rail carriage's upright (back face)


def _ab_cs():
    al = math.radians(dir_deg(OUT, MB) + 90.0)                    # motor C ends up straight above the axis
    return math.cos(al), math.sin(al)


def ab_loc(shift=0.0):
    """Location mapping drum B's stage (main frame) onto the ab/ad axis: main +Y -> global +X (toward
    the module); shift moves motor-side parts toward the drum (main +Y)."""
    c, s_ = _ab_cs()
    return (Location(Plane(origin=(AB_X0, AB_Y, AB_Z), x_dir=(0.0, c, s_), z_dir=(0.0, s_, -c))) *
            Pos(-OUT[0], -AB_STAGE_Y[0] + shift, -OUT[1]))


def ab_point(px, py, pz=0.0):
    """Global (x, y, z) of a point given in drum-B-local coordinates (before Rot(phi_b) and to_y)."""
    phi = math.radians(-dir_deg(OUT, MB))
    x, y = px * math.cos(phi) - py * math.sin(phi), px * math.sin(phi) + py * math.cos(phi)
    dx, d, dz = x, pz, -y                                               # to_y, relative to OUT and B_Y0
    c, s_ = _ab_cs()
    return AB_X0 + d, AB_Y + dx * c + dz * s_, AB_Z + dx * s_ - dz * c


def _plate_x(x0, x1, pts):
    """Plate perpendicular to X between x0 and x1: hull of (y, z, r) circles."""
    face = make_hull([e for (y, z, r) in pts for e in (Pos(y, z) * Circle(r)).edges()])
    return Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(face, x1 - x0)


def _skeleton_x(x0, x1, body, struts):
    """Cradle plate (v6 skeleton): hull of the `body` (y, z, r) circles plus a tapered strut for each
    ((y, z, r), (y, z, r)) pair, inside corners filleted."""
    face = fillet2d([hull2d(body)] + [hull2d([a, b]) for a, b in struts], PLATE_FILLET)
    return Location(Plane(origin=(x0, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(face, x1 - x0)


def _bore_x(x0, x1, y, z, r):
    return Location(Plane(origin=(x0, y, z), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * Cylinder(r, x1 - x0, align=MIN)


AB_HUB_R = 15.0                       # v9.1: hub screw circle (M4 heat-set inserts in the hub's end; v9: M4 nuts on r16)
AB_HUB_N = 8                          # v9.1: 8x M4 (the inserts' pull-out sets the count: SF 2.5 at the hub's moment)
AB_HUB_SCREWS = [(AB_Y + AB_HUB_R * math.cos(math.radians(22.5 + 45 * k)), AB_Z + AB_HUB_R * math.sin(math.radians(22.5 + 45 * k)))
                 for k in range(AB_HUB_N)]                       # end bracket -> hub
AB_HUB_NUT_X = -137.5                 # captured M4 nuts inside the drum hub (x of their outer face)


def ab_drum():
    """Ab/ad drum (global): drum B's sector moved onto the axis, a Ø40 hub through the inner plate's
    6808 to the yoke, and a Ø40 stub across to the outer plate's 6808 (two bearings ~65 mm apart)."""
    w = AB_STAGE_Y[1] - AB_STAGE_Y[0]
    d = ab_loc() * to_y(Rot(0, 0, -dir_deg(OUT, MB)) * sector_body(w, 26.0, abad=True), AB_STAGE_Y[0], *OUT)
    d += _bore_x(AB_X0 + w, AB_TUBE_END, AB_Y, AB_Z, AB_TUBE_R)          # hub toward the yoke
    ox0, ox1 = AB_X["outer"]
    d += _bore_x(ox1 + 2.0, AB_X0, AB_Y, AB_Z, AB_TUBE_R)                 # stub toward the outer plate
    d += _bore_x(ox1, ox1 + 2.0, AB_Y, AB_Z, 23.0)                        # shoulder on the outer 6808's inner race
    d += _bore_x(ox1 - 7.0, ox1, AB_Y, AB_Z, AB_TUBE_R)                   # journal in the outer 6808
    d -= _bore_x(ox1 - 7.01, AB_TUBE_END + 0.01, AB_Y, AB_Z, 9.5)         # hollow (wires can pass; v9.1 Ø19)
    for y, z in AB_HUB_SCREWS:                                                # v9.1: 6x M3 heat-set inserts in the hub's end face,
        d -= _bore_x(AB_TUBE_END - 8.5, AB_TUBE_END + 0.01, y, z, M4_INSERT / 2)   # pressed on the bench (v9: nuts in slots)
    return d


def ab_yoke():
    """Rides on the module: the base mount's plate (4x M4 into the rib inserts) plus a flange at the
    back end that bolts to the ab/ad hub; a boss on the flange clamps the inner 6808's inner race."""
    ys = (BACK_Y0 + (BACK_Y1 - BACK_Y0) / 2 - RIB_T / 2, FRONT_Y1 - (FRONT_Y1 - FRONT_Y0) / 2 + RIB_T / 2)
    x0, x1 = AB_X["yoke"]
    y = Pos((x0 + 72.0) / 2, -16.0, Z_TOP) * Box(72.0 - x0, 126.0, MOUNT_T, align=(Align.CENTER, Align.MIN, Align.MIN))
    y += _plate_x(x0, x1, [(-12.0, Z_TOP + 4.0, 4.0), (106.0, Z_TOP + 4.0, 4.0), (AB_Y, AB_Z, 30.0)])
    y -= _plate_x(x0, x1, [(AB_Y - 32.0, 35.0, 8.0), (AB_Y + 32.0, 35.0, 8.0), (AB_Y, 22.0, 6.0)])   # window
    y += _bore_x(AB_TUBE_END, x0, AB_Y, AB_Z, 23.0)                       # boss on the inner race
    y -= _bore_x(AB_TUBE_END - 1, x1 + 1, AB_Y, AB_Z, 11.0)
    for yy, zz in AB_HUB_SCREWS:
        y -= _bore_x(AB_TUBE_END - 1, x1 + 1, yy, zz, M3_CLR / 2)
        y -= _bore_x(x1 - 3.5, x1 + 0.01, yy, zz, 3.1)
    for yy in ys:
        for x in (-60, 60):
            y -= Pos(x, yy, Z_TOP) * Cylinder(M4_CLR / 2, MOUNT_T, align=MIN)
            y -= Pos(x, yy, Z_TOP + MOUNT_T - 4.2) * Cylinder(4.0, 4.2, align=MIN)
    return y


def ab_motor_plate_features():
    """Motor C's mount (main frame, as on the motor plate around motor B): (adds, cuts)."""
    mx, mz = MB
    cuts = [cyl_y(MAGNET_HOLDER_CLEAR_D / 2, BACK_Y0 - 1, BACK_Y1 + 1, mx, mz)] + stator_cuts(mx, mz)
    cuts.append(wire_bay("B"))
    adds = []
    for hx, hz in moteus_holes("B", "boss"):
        adds.append(cyl_y(3.8, PCB_Y, BACK_Y0 + 0.5, hx, hz))
        cuts.append(cyl_y(M2_5_INSERT / 2, PCB_Y - 1, PCB_Y + 5.0, hx, hz))
    for hx, hz in moteus_holes("B", "standoff"):
        cuts.append(cyl_y(M2_5_FLUSH / 2, BACK_Y0 - 0.01, BACK_Y0 + 4.5, hx, hz))
    return adds, cuts


def _rear_tab(x0, x1, inward, zs=None, zr=None):
    """14 mm thick rear tab of a cradle plate, 2x M4 inserts in its back face for the spine."""
    zs, (za, zb) = zs or TAB_Z, zr or SPINE_Z
    xa, xb = (x0, x0 + 14.0) if inward > 0 else (x1 - 14.0, x1)
    y0 = AB_RAIL_Y0 + SPINE_T
    t = Pos((xa + xb) / 2, y0, za) * Box(14.0, 22.0, zb - za, align=(Align.CENTER, Align.MIN, Align.MIN))
    for z in zs:
        t -= Location(Plane(origin=((xa + xb) / 2, y0 - 0.01, z), x_dir=(1, 0, 0), z_dir=(0, 1, 0))) * Cylinder(M4_INSERT / 2, 8.0, align=MIN)
    return t, (xa + xb) / 2


TAB_Z = (AB_Z - 15.0, AB_Z + 100.0)
SPINE_Z = (AB_Z - 40.0, AB_Z + 125.0)
SPINE_X = (AB_X["outer"][0], HF_X[0] + 14.0)
FRONT_TAB_Z = (AB_Z - 35.0, AB_Z + 35.0)       # the front plate's tab is short: the spine steps down to it
FRONT_TAB_ZR = (AB_Z - 40.0, AB_Z + 40.0)
SPINE_SLOPE = ((-20.0, SPINE_Z[1]), (SPINE_X[1], FRONT_TAB_ZR[1] + 5.0))   # top edge falls from here to the front end


def spine_top(x):
    (xa, za), (xb, zb) = SPINE_SLOPE
    return za if x <= xa else za + (zb - za) * (x - xa) / (xb - xa)


def plate_back_pts():
    """(y, z, r) hull points that carry a cradle plate back to its tab."""
    yb = AB_RAIL_Y0 + SPINE_T + 10.5                                   # 0.5 inside the tab's back face (clean mesh)
    return [(yb, AB_Z - 30.0, 10.0), (yb, AB_Z + 115.0, 10.0)]


def hip_trunnion():
    """Front trunnion (global): a flange on the front bracket's web, 4x M4 countersunk into inserts, and
    a Ø29.9 journal that floats in the cradle's 6806. Hollow journal, solid flange."""
    xf, xj, xe = TRUNNION_X
    t = _bore_x(xf, xj, AB_Y, AB_Z, 26.0) + _bore_x(xj, xe, AB_Y, AB_Z, TRUNNION_D / 2)
    t -= _bore_x(xj, xe + 0.01, AB_Y, AB_Z, 8.0)
    for y, z in TRUNNION_SCREWS:
        t -= _bore_x(xf - 0.01, xj + 0.01, y, z, M4_CLR / 2)
        t -= Location(Plane(origin=(xj - 2.3, y, z), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * Cone(M4_CLR / 2, 4.15, 2.31, align=MIN)
    return t


def hip_front_plate():
    """Cradle front plate (fixed): the 6806 for the trunnion, a window, and its tab on the spine."""
    x0, x1 = HF_X
    yb = AB_RAIL_Y0 + SPINE_T + 10.0
    p = _plate_x(x0, x1, [(AB_Y, AB_Z, 30.0), (yb, FRONT_TAB_ZR[0] + 10.0, 10.0), (yb, FRONT_TAB_ZR[1] - 10.0, 10.0)])
    tab, _ = _rear_tab(x0, x1, inward=+1, zs=FRONT_TAB_Z, zr=FRONT_TAB_ZR)
    p += tab
    p -= _bore_x(x1 - B30[2], x1 + 0.01, AB_Y, AB_Z, (B30[1] + PRESS) / 2)    # 6806 from outside
    p -= _bore_x(x0 - 1, x1 + 1, AB_Y, AB_Z, 18.0)                         # lip on the outer race, module side
    p -= _plate_x(x0 - 1, x1 + 1, [(-38.0, AB_Z, 13.0), (-2.0, AB_Z, 13.0)])          # window
    return p


def hip_spine():
    """The cradle's backbone (fixed): a 10 mm plate behind the module, 311 mm long, from the outer plate to
    the front plate, with a 12 mm rib along each long edge (stopped at the tabs). M4 holes at the six tab
    points, M3 holes for two MGN12H blocks, and windows."""
    x0, x1 = SPINE_X
    z0, z1 = SPINE_Z
    yb, yf = AB_RAIL_Y0, AB_RAIL_Y0 + SPINE_T
    xz = lambda pts: Location(Plane(origin=(0, yb, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0)))     # local (x, -z)
    outline = [(x0, z1), SPINE_SLOPE[0], (x1, spine_top(x1)), (x1, z0), (x0, z0)]   # counter-clockwise in local (x, -z)
    sp = xz(None) * extrude(Polygon(*[(x, -z) for x, z in outline], align=None), SPINE_T)
    tabs = [(AB_X["outer"][0], AB_X["outer"][0] + 14.0), (AB_X["inner"][1] - 14.0, AB_X["inner"][1]), (HF_X[0], HF_X[0] + 14.0)]
    gaps = [(tabs[0][1] + 1.0, tabs[1][0] - 1.0), (tabs[1][1] + 1.0, tabs[2][0] - 1.0)]
    for ga, gb in gaps:                                                 # bottom rib; top rib up to the slope
        sp += Pos((ga + gb) / 2, yf, z0) * Box(gb - ga, 12.0, 8.0, align=(Align.CENTER, Align.MIN, Align.MIN))
        gt = min(gb, SPINE_SLOPE[0][0])
        if gt > ga:
            sp += Pos((ga + gt) / 2, yf, z1 - 8.0) * Box(gt - ga, 12.0, 8.0, align=(Align.CENTER, Align.MIN, Align.MIN))
    win = lambda x, za, zb, r: Location(Plane(origin=(0, yb - 1, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))) * \
        extrude(make_hull([e for c in (Pos(x, -za) * Circle(r), Pos(x, -zb) * Circle(r)) for e in c.edges()]), SPINE_T + 2)
    for x, r in ((-150.0, 13.0), (-83.0, 15.0), (-2.0, 13.0), (34.0, 13.0), (70.0, 12.0)):
        sp -= win(x, z0 + 22.0 + r, spine_top(x + r) - 16.0 - r, r)
    for x, z in ab_tab_points():
        sp -= Pos(x, yb - 1, z) * Rot(-90, 0, 0) * Cylinder(M4_CLR / 2, SPINE_T + 2, align=MIN)
    for x, z in ab_block_points():
        sp -= Pos(x, yb - 1, z) * Rot(-90, 0, 0) * Cylinder(M3_CLR / 2, SPINE_T + 2, align=MIN)
    return sp


def ab_block_points():
    return [(x, zc + dz) for zc in AB_BLOCKS_Z for x in (AB_RAIL_X - 10.0, AB_RAIL_X + 10.0) for dz in (-10.0, 10.0)]


def ab_parts(desk=False):
    """3-DOF additions. Returns (moving, fixed): name -> (solid, colour).
    `moving` turns with the module about the ab/ad axis; `fixed` is the ab/ad frame and motor C."""
    loc, loc_m = ab_loc(), ab_loc(AB_SHIFT)
    mv, fx = {}, {}
    mv["Abad_Drum"] = (ab_drum(), "violet")
    phi_b = -dir_deg(OUT, MB)
    ref = reference()
    for tag in ("minus", "plus"):                                           # winding pegs, as drum B's
        for part in ("M4x18_screw", "nut"):
            hw_ = peg_hardware(-1 if tag == "minus" else 1, AB_STAGE_Y[1] - AB_STAGE_Y[0], abad=True)[part]
            mv[f"Abad_REF_Tensioner_{tag}_{part}"] = (loc * to_y(Rot(0, 0, phi_b) * hw_, AB_STAGE_Y[0], *OUT),
                                                      "screw" if "screw" in part else "steel")
    yx0, yx1 = AB_X["yoke"]
    mv["Abad_HW_hub_bolts"] = (Compound(children=[                            # v9.1: 8x M4 x 16 from the bracket's counterbores
        Location(Plane(origin=(yx1 - 4.5, y, z), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * shcs(4, 16, 7.0, 4.0)
        for y, z in AB_HUB_SCREWS]), "screw")
    mv["Abad_HW_hub_inserts"] = (Compound(children=[                          # v9.1: pressed into the hub's end on the bench
        _bore_x(AB_TUBE_END - 8.0, AB_TUBE_END, y, z, 3.0) - _bore_x(AB_TUBE_END - 8.1, AB_TUBE_END + 0.1, y, z, 1.8)
        for y, z in AB_HUB_SCREWS]), "brass")

    # ---- fixed: outer plate (motor C, moteus C), inner plate, posts, rail plate, bearings ----
    ox0, ox1 = AB_X["outer"]
    # v6: hub and motor C joined, struts to the two posts and to the spine tab; motor C overhangs
    hub, (bl, bh) = (AB_Y, AB_Z, 34.0), plate_back_pts()
    struts = [((AB_Y, AB_Z, PLATE_STRUT_R), (y, z, 10.0)) for y, z in AB_POSTS] + \
             [((AB_Y, AB_Z, PLATE_STRUT_R), bl), ((AB_Y, AB_Z, PLATE_STRUT_R), bh), (bl, bh)]
    op = _skeleton_x(ox0, ox1, [hub, (AB_Y, AB_Z + 84.0, MOTOR_BOSS_R)], struts)
    tab, tab_op = _rear_tab(ox0, ox1, inward=+1)
    op += tab
    adds, cuts = ab_motor_plate_features()
    for a_ in adds:
        op += loc_m * a_
    for c_ in cuts:
        op -= loc_m * c_
    op -= _bore_x(ox1 - 7.0, ox1 + 0.01, AB_Y, AB_Z, (B40[1] + PRESS) / 2)       # 6808, from the drum side
    op -= _bore_x(ox0 - 1, ox1 + 1, AB_Y, AB_Z, 21.0)
    for y, z in AB_POSTS:
        op -= _bore_x(ox0 - 1, ox1 + 1, y, z, M4_CLR / 2)
    fx["Hip_Outer_Plate"] = (op, "black")
    ix0, ix1 = AB_X["inner"]
    ip = _skeleton_x(ix0, ix1, [hub, (AB_Y, AB_Z + 84.0, 22.0)], struts)
    tab, tab_ip = _rear_tab(ix0, ix1, inward=-1)
    ip += tab
    ip -= _bore_x(ix1 - 7.0, ix1 + 0.01, AB_Y, AB_Z, (B40[1] + PRESS) / 2)       # 6808, from the yoke side
    ip -= _bore_x(ix0 - 1, ix1 + 1, AB_Y, AB_Z, AB_TUBE_R + 0.5)
    ip -= loc * cyl_y((B15[1] + PRESS) / 2, FRONT_Y0 - 1, FRONT_Y1 - 1.0, *MB)      # motor C's tip 6902
    ip -= loc * cyl_y(12.0, FRONT_Y0 - 1, FRONT_Y1 + 1, *MB)
    for y, z in AB_POSTS:                                                   # M4 inserts, pressed in from the post side
        ip -= _bore_x(ix0 - 0.01, ix0 + 6.0, y, z, M4_INSERT / 2)
    fx["Hip_Inner_Plate"] = (ip, "black")
    fx["Hip_Posts"] = (Compound(children=[_bore_x(ox1, ix0, y, z, 8.0) - _bore_x(ox1 - 1, ix0 + 1, y, z, M4_CLR / 2)
                                          for y, z in AB_POSTS]), "dimgray")
    # ---- v6 cradle: front plate, trunnion, front bracket, spine ----
    fx["Hip_Front_Plate"] = (hip_front_plate(), "black")
    fx["Hip_Bearing_front"] = (_bore_x(HF_X[1] - B30[2], HF_X[1], AB_Y, AB_Z, B30[1] / 2) -
                               _bore_x(HF_X[1] - B30[2] - 1, HF_X[1] + 1, AB_Y, AB_Z, B30[0] / 2), "steel")
    mv["End_Bracket_Front"] = (end_bracket(front=True), "black")
    mv["Hip_Trunnion"] = (hip_trunnion(), "red")
    cs4 = Cone(4.0, 2.0, 2.3, align=MIN) + Pos(0, 0, 2.3) * Cylinder(2.0, 10.0 - 2.3, align=MIN)          # M4 x 10 DIN 7991
    mv["Hip_HW_trunnion_screws"] = (Compound(children=[
        Location(Plane(origin=(TRUNNION_X[1], y, z), x_dir=(0, 1, 0), z_dir=(-1, 0, 0))) * cs4 for y, z in TRUNNION_SCREWS]), "screw")
    mv["Hip_HW_trunnion_inserts"] = (Compound(children=[
        _bore_x(FB_X[1] - 6.0, FB_X[1], y, z, 2.9) - _bore_x(FB_X[1] - 7, FB_X[1] + 1, y, z, 2.0) for y, z in TRUNNION_SCREWS]), "brass")
    fx["Hip_Spine"] = (hip_spine(), "black")
    t_head = AB_RAIL_Y0 - (DESK_T if desk else 0.0)                        # desk builds: through the desk angles too
    fx["Hip_HW_rail_screws"] = (Compound(children=[
        Pos(x, t_head, z) * Rot(90, 0, 0) * shcs(4, 25 if desk else 16, 7.0, 4.0) for x, z in ab_tab_points()]), "screw")
    fx["Hip_HW_block_screws"] = (Compound(children=[                    # from the spine's front, 4 mm into each block
        Pos(x, AB_RAIL_Y0 + SPINE_T, z) * Rot(-90, 0, 0) * shcs(3, 14, 5.5, 3.0) for x, z in ab_block_points()]), "screw")
    dy = AB_RAIL_Y0 - CARRIAGE_Y0
    b0, b1 = STAND_Y["block"]
    fx["Hip_REF_MGN12H_blocks"] = (Compound(children=[Pos(AB_RAIL_X, (b0 + b1) / 2 + dy, zc) * Box(27.0, b1 - b0, 45.4)
                                                      for zc in AB_BLOCKS_Z]), "steel")
    fx["Hip_HW_post_bolts"] = (Compound(children=[                      # M4 x 70 from the outer plate into the inserts:
        Location(Plane(origin=(ox0, y, z), x_dir=(0, 1, 0), z_dir=(-1, 0, 0))) * shcs(4, 70, 7.0, 4.0)   # nothing on the
        for y, z in AB_POSTS]), "screw")                                                                    # inner face
    fx["Hip_HW_post_inserts"] = (Compound(children=[_bore_x(ix0, ix0 + 6.0, y, z, 2.9) - _bore_x(ix0 - 1, ix0 + 7, y, z, 2.0)
                                                    for y, z in AB_POSTS]), "brass")
    # motor C, moteus C, small drum, tip bearing and fasteners: drum B's stage mapped onto the unit
    fx["Abad_Motor"] = (loc_m * ref_motor("B"), "motor")
    fx["Abad_moteus"] = (loc_m * ref_moteus("B"), "pcb")
    sd = small_drum(AB_STAGE_Y[0] + LIP_T - AB_SHIFT, AB_STAGE_Y[1] - LIP_T - AB_SHIFT, 89.5 - AB_SHIFT, 97.5 - AB_SHIFT)
    fx["Abad_Small_Drum"] = (loc_m * to_y(sd, ROTOR_FACE_Y, *MB), "silver")
    fst = fasteners()
    for n, tag in (("FST_Stator_B", "stator_screws"), ("FST_Insert_moteus_B", "moteus_inserts"),
                   ("FST_Standoff_moteus_B", "moteus_standoffs"), ("FST_Screw_moteus_B", "moteus_screws")):
        fx[f"Abad_HW_motor_{tag}"] = (loc_m * fst[n], "brass" if "Insert" in n or "Standoff" in n else "screw")
    fx["Abad_HW_small_drum_screws"] = (loc_m * fst["FST_Rotor_B"], "screw")
    fx["Abad_Bearing_tip"] = (loc * to_y(bearing(*B15), FRONT_Y0, *MB), "steel")
    fx["Abad_Bearings_6808"] = (Compound(children=[
        _bore_x(ix1 - 7.0, ix1, AB_Y, AB_Z, B40[1] / 2) - _bore_x(ix1 - 8, ix1 + 1, AB_Y, AB_Z, B40[0] / 2),
        _bore_x(ox1 - 7.0, ox1, AB_Y, AB_Z, B40[1] / 2) - _bore_x(ox1 - 8, ox1 + 1, AB_Y, AB_Z, B40[0] / 2)]), "steel")
    return mv, fx


def ab_tab_points(which=("outer", "inner", "front")):
    """(x, z) of the M4 inserts in the cradle plates' rear tabs (the spine, and the desk angles, bolt here)."""
    xs = {"outer": AB_X["outer"][0] + 7.0, "inner": AB_X["inner"][1] - 7.0, "front": HF_X[0] + 7.0}
    return [(xs[w], z) for w in which for z in (FRONT_TAB_Z if w == "front" else TAB_Z)]


DESK_Z_AB = 77.0                      # ab/ad desk versions: underside (model) of the desk angle's foot


DESK_ANGLE_X = {"rear": (AB_X["outer"][0] - 75.0, AB_X["inner"][0]), "front": (FB_X[0] + 2.0, HF_X[0] + 14.0 + 66.0)}
DESK_ANGLE_UP = {"rear": (AB_X["outer"][0], AB_X["inner"][1]), "front": (FB_X[0] + 2.0, HF_X[0] + 14.0 + 66.0)}
DESK_ANGLE_TABS = {"rear": ("outer", "inner"), "front": ("front",)}
DESK_FOOT_Y = (-140.0, 170.0)         # 310 deep: fits the H2C


def desk_angle(end):
    """Ab/ad desk versions (v6: one at each end of the cradle). An upright behind the spine, bolted through
    the spine into the cradle plates' tab inserts (M4 x 25), and a foot (model +Z = desk side) held down by
    4 wood screws, with two gussets. The feet stay clear of the module's swing; the front foot passes over
    the front bracket's web 5 mm clear at the sector's full +-34 deg."""
    x0, x1 = DESK_ANGLE_X[end]
    u0, u1 = DESK_ANGLE_UP[end]
    yb = AB_RAIL_Y0 - DESK_T                                           # upright's back face
    z_top = DESK_Z_AB + DESK_T
    a = Pos((u0 + u1) / 2, yb, AB_Z - 40.0) * Box(u1 - u0, DESK_T, z_top - (AB_Z - 40.0), align=(Align.CENTER, Align.MIN, Align.MIN))
    a += Pos((x0 + x1) / 2, DESK_FOOT_Y[0], DESK_Z_AB) * Box(x1 - x0, DESK_FOOT_Y[1] - DESK_FOOT_Y[0], DESK_T,
                                                            align=(Align.CENTER, Align.MIN, Align.MIN))
    for x in (u0 + 30.0, u1 - 30.0):                                    # gussets, behind the upright
        g = Polygon((0, 0), (-45, 0), (0, -110), align=None)
        a += Location(Plane(origin=(x - 4.0, yb, DESK_Z_AB), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(g, 8.0)
    for x, z in ab_tab_points(DESK_ANGLE_TABS[end]):
        a -= Pos(x, yb - 1, z) * Rot(-90, 0, 0) * Cylinder(M4_CLR / 2, DESK_T + 2, align=MIN)
    for x, y in desk_angle_screws(end):
        a -= Pos(x, y, DESK_Z_AB) * Cylinder(2.75, DESK_T, align=MIN)
        a -= Pos(x, y, DESK_Z_AB) * Cone(5.2, 2.75, 2.6, align=MIN)
    return a


def desk_angle_screws(end=None):
    if end is None:
        return desk_angle_screws("rear") + desk_angle_screws("front")
    x0, x1 = DESK_ANGLE_X[end]
    xs = (x0 + 11.0, x1 - 8.0) if end == "rear" else (x0 + 8.0, x1 - 8.0)
    return [(x, y) for x in xs for y in (DESK_FOOT_Y[0] + 14.0, DESK_FOOT_Y[1] - 15.0)]


RAIL5_BLOCKS_Z = (100.0, 190.0)       # leg without ab/ad: two MGN12H blocks on the carriage upright


def rail_carriage5():
    """Leg version without ab/ad: plate on the two top rails (4x M4 x 16, heads on top) with an upright
    at the back for two MGN12H blocks (8x M3 x 12), and two gussets."""
    c = Pos(0, CARRIAGE5_Y0, RAIL_TOP) * Box(212.0, 106.0 - CARRIAGE5_Y0, DESK_T, align=(Align.CENTER, Align.MIN, Align.MIN))
    c += Pos(0, CARRIAGE5_Y0, RAIL_TOP) * Box(70.0, 8.0, 230.0 - RAIL_TOP, align=(Align.CENTER, Align.MIN, Align.MIN))
    for x in (-31.0, 31.0):
        g = Polygon((0, 0), (70, 0), (0, 90), align=None)
        c += Location(Plane(origin=(x - 4.0, CARRIAGE5_Y0 + 8.0, RAIL_TOP + DESK_T), x_dir=(0, 1, 0), z_dir=(1, 0, 0))) * extrude(g, 8.0)
    for x, y in rail_insert_points():
        c -= Pos(x, y, RAIL_TOP) * Cylinder(M4_CLR / 2, DESK_T, align=MIN)
    for zc in RAIL5_BLOCKS_Z:
        for x in (-10.0, 10.0):
            for z in (zc - 10.0, zc + 10.0):
                c -= Pos(x, CARRIAGE5_Y0 - 1, z) * Rot(-90, 0, 0) * Cylinder(M3_CLR / 2, 10.0, align=MIN)
    return c


def reference():
    ox, oz = OUT
    ref = {
        "REF_Motor_A": ref_motor("A"), "REF_Motor_B": ref_motor("B"),
        "REF_moteus_A": ref_moteus("A"), "REF_moteus_B": ref_moteus("B"),
        "REF_Bearing_A_30x42_rear": to_y(bearing(*B30), A_R1_Y, ox, oz),
        "REF_Bearing_A_30x42_front": to_y(bearing(*B30), A_R2_Y, ox, oz),
        "REF_Bearing_30x42_B1": to_y(bearing(*B30), B_Y0, ox, oz),
        "REF_Bearing_30x42_B2": to_y(bearing(*B30), B2_Y, ox, oz),
        "REF_Bearing_40x52_wall": to_y(bearing(*B40), WALL_BRG_Y[0], ox, oz),
        "REF_Bearing_17x30_gantry": to_y(bearing(*GANTRY_BRG), GANTRY_Y[0], ox, oz),
        "REF_Bearing_15x28_tipA": to_y(bearing(*B15), FRONT_Y0, *MA),
        "REF_Bearing_15x28_tipB": to_y(bearing(*B15), FRONT_Y0, *MB),
    }
    # Each winding peg: M4 x 20 (cross-drilled for the rope) on a DIN 9021 washer, nylon-insert nut in
    # the far face's pocket. Threads are omitted.
    for stage, y0, w, phi_c in (("A", A_Y0, A_Y1 - A_Y0, -dir_deg(OUT, MA)),
                                 ("B", B_Y0, B_Y1 - B_Y0, -dir_deg(OUT, MB))):
        for side in (-1, 1):
            for part, shape in peg_hardware(side, w).items():
                ref[f"REF_Tensioner_{stage}_{'minus' if side < 0 else 'plus'}_{part}"] = to_y(Rot(0, 0, phi_c) * shape, y0, ox, oz)
    return ref


def peg_hardware(s, w, abad=False):
    """Winding peg at sector end s, drum-local: the screw (cross-drilled for the rope) and its nut."""
    _sec_set(abad)
    try:
        return _peg_hardware(s, w)
    finally:
        _sec_set()


def _peg_hardware(s, w):
    k_rim, hd, hh = peg_k_rim(), PEG_HEAD[0], PEG_HEAD[1]
    k_head = k_rim - PEG_SINK - hh
    k_nut = -PEG_WELL_H / 2 - PEG_NUT_GAP
    pf = peg_frame(s, w)
    screw = Pos(0, 0, k_head) * shcs(4, PEG_L, hd, hh)
    screw -= Rot(0, 90, 0) * Cylinder(1.0, 8.0)                                                  # rope cross-hole
    nut_ = Pos(0, 0, k_nut - PEG_NUT_H) * Rot(0, 0, 30) * nut(PEG_NUT_AF, PEG_NUT_H, 4.0)
    return {"M4x18_screw": pf * screw, "nut": pf * nut_}


def ref_motor_split(which):
    """The 8308 STEP as separately placed rotor, stator and magnet holder (for animation)."""
    m = import_step(os.path.join(REF, "ref_cad", "eaglepower_8308.step"))
    x, z = MA if which == "A" else MB
    rz = 0 if which == "A" else 180
    return {c.label: Pos(x, BACK_Y1, z) * Rot(0, -MOTOR_TURN[which], 0) * Rot(0, rz, 0) * c for c in m.children}


# ----------------------------------------------------------------------------
# Fasteners (animation only; threads omitted). Each group sits in the local
# frame of the part it is fitted to, so it shares that part's placement.
# ----------------------------------------------------------------------------
def shcs(d, length, head_d=None, head_h=None):
    """Socket head cap screw: head on z 0..head_h, shank down to -length."""
    head_d, head_h = head_d or 1.83 * d, head_h or d
    s = Pos(0, 0, -length) * Cylinder(d / 2, length, align=MIN)
    s += Cylinder(head_d / 2, head_h, align=MIN) - Pos(0, 0, head_h * 0.45) * hex_prism_z(d * 0.85, head_h * 0.56)
    return s


def insert(od, id_, h):
    return Cylinder(od / 2, h, align=MIN) - Cylinder(id_ / 2, h, align=MIN)


def nut(af, h, d):
    return hex_prism_z(af, h) - Cylinder(d / 2, h, align=MIN)


def ring_of(shape, r, phase, n, z=0.0):
    out = []
    for k in range(n):
        x, y = polar(r, phase + 360 / n * k)
        out.append(Pos(x, y, z) * shape)
    return Compound(children=out)


def fasteners(front_bracket=False):
    ox, oz = OUT
    phi_a, phi_b = -dir_deg(OUT, MA), -dir_deg(OUT, MB)
    f = {}
    for w, (mx, mz) in (("A", MA), ("B", MB)):
        # stator: 4x M4 countersunk from the rear of the motor plate
        cs = Cone(4.0, 2.0, 2.2, align=MIN) + Pos(0, 0, 2.2) * Cylinder(2.0, 10.0, align=MIN)
        f[f"FST_Stator_{w}"] = to_y(ring_of(cs, STATOR_BC / 2, -STATOR_PHASE, 4), BACK_Y0 + STATOR_RECESS, mx, mz)
        # small drum: 4x M3x10 into the rotor, heads in the flange counterbores
        f[f"FST_Rotor_{w}"] = to_y(ring_of(shcs(3, 10.8, 5.5, 3.0), ROTOR_BC / 2, 45, 4, 4.8), ROTOR_FACE_Y, mx, mz)
    for w, (mx, mz) in (("A", MA), ("B", MB)):
        # moteus: M2.5 inserts in the two printed bosses and flush in the plate for the two
        # M-F standoffs (8 mm body, fitted after the stator screws), 4x M2.5x6 through the board
        holes = moteus_holes(w)
        f[f"FST_Insert_moteus_{w}"] = Compound(
            children=[to_y(insert(3.9, 2.1, 4.0), PCB_Y, hx, hz) for hx, hz in moteus_holes(w, "boss")] +
                     [to_y(insert(3.5, 2.1, 4.0), BACK_Y0, hx, hz) for hx, hz in moteus_holes(w, "standoff")])
        standoff = (hex_prism_z(5.0, MOTEUS_STANDOFF_L) - Cylinder(1.05, MOTEUS_STANDOFF_L, align=MIN)) + \
            Pos(0, 0, MOTEUS_STANDOFF_L) * Cylinder(1.2, 4.0, align=MIN)                   # male thread into the insert
        f[f"FST_Standoff_moteus_{w}"] = Compound(children=[to_y(Pos(0, 0, -MOTEUS_STANDOFF_L) * standoff, BACK_Y0, hx, hz)
                                                           for hx, hz in moteus_holes(w, "standoff")])
        pcb_back = PCB_Y - 1.65
        f[f"FST_Screw_moteus_{w}"] = Compound(children=[to_y(Rot(180, 0, 0) * shcs(2.5, 6, 4.5, 2.5), pcb_back, hx, hz)
                                                        for hx, hz in holes])
    # drum B: 6x M3 inserts in the end of the journal (PCD 32)
    f["FST_Insert_B_hub"] = to_y(Rot(0, 0, phi_b) * ring_of(insert(4.6, 2.6, 5.7), B_HUB_SCREW_R, 30, 6, WALL_BRG_Y[1] - 5.7 - B_Y0), B_Y0, ox, oz)
    # flange B: 6x M3 nuts for the arm (PCD 70), 6x M3x16 into the journal inserts
    f["FST_Insert_B_flange"] = to_y(Rot(0, 0, phi_b) * ring_of(insert(4.6, 2.6, 5.7), 35.0, 0, 6, 0.3), 105.0, ox, oz)   # v9.1
    f["FST_Screw_B_flange"] = to_y(Rot(0, 0, phi_b) * ring_of(shcs(3, 16, 5.5, 3.0), B_HUB_SCREW_R, 30, 6, 3.0), 105.0, ox, oz)
    # arm B: 6x M3x10 into the flange's inserts (a 12 would reach B's hub shoulder)
    f["FST_Screw_B_arm"] = to_y(Rot(0, 0, phi_b) * ring_of(shcs(3, 10, 5.5, 3.0), 35.0, 0, 6, 4.8), 111.0, ox, oz)
    # flange A: 4x M3x10 into heat-set inserts in A's Ø22 nose (PCD 14), 6x M3 inserts for the arm (PCD 48)
    f["FST_Insert_A_nose"] = to_y(Rot(0, 0, phi_a) * ring_of(insert(4.6, 2.6, 5.7), A_NOSE_SCREW_R, 45, 4, -5.7), 120.0, ox, oz)
    f["FST_Screw_A_flange"] = to_y(Rot(0, 0, phi_a) * ring_of(shcs(3, 10, 5.5, 3.0), A_NOSE_SCREW_R, 45, 4, 4.5), 120.0, ox, oz)
    f["FST_Insert_A_flange"] = to_y(Rot(0, 0, phi_a) * ring_of(insert(4.6, 2.6, 5.7), 24.0, 30, 6, 2.3), 120.0, ox, oz)
    # arm A: 6x M3x10 into the flange inserts
    f["FST_Screw_A_arm"] = to_y(Rot(0, 0, phi_a) * ring_of(shcs(3, 10, 5.5, 3.0), 24.0, 30, 6, 4.8), 128.0, ox, oz)
    # frame: 5x M4 through wall, posts and motor plate, nuts at the rear
    # frame (v6): an M4 heat-set insert in each end of every post; one screw through the front wall and one
    # through the motor plate into each post (M4 x 25 where an end bracket's ear sits under the head)
    eared = POSTS[0::2] + (POSTS[1::2] if front_bracket else [])
    ear = lambda xz: 8.0 if xz in eared else 0.0
    f["FST_Insert_posts"] = Compound(children=[to_y(insert(5.8, 4.0, 8.0), y0, x, z) for (x, z) in POSTS
                                               for y0 in (BACK_Y1, FRONT_Y0 - 8.0)])
    f["FST_Bolt_posts"] = Compound(children=[to_y(shcs(4, 25.0 if ear((x, z)) else 16.0, 7.0, 4.0), FRONT_Y1 + ear((x, z)), x, z)
                                             for (x, z) in POSTS])
    f["FST_Screw_posts_back"] = Compound(children=[to_y(Rot(180, 0, 0) * shcs(4, 25.0 if ear((x, z)) else 16.0, 7.0, 4.0),
                                                        BACK_Y0 - ear((x, z)), x, z) for (x, z) in POSTS])
    gb = shcs(4, 65.0, 7.0, 4.0)                                             # v8 gantry: M4 x 65 through gantry (10) and post (48), 7 into the insert
    f["FST_Bolt_gantry"] = Compound(children=[to_y(gb, GANTRY_Y[1], x, z) for x, z in GANTRY_POSTS])
    f["FST_Insert_gantry"] = Compound(children=[to_y(insert(5.8, 4.0, 7.0), FRONT_Y1 - 7.0, x, z) for x, z in GANTRY_POSTS])
    # base mount: 4x M4 rib inserts, 4x M4x12 from its underside
    ys = (BACK_Y0 + (BACK_Y1 - BACK_Y0) / 2 - RIB_T / 2, FRONT_Y1 - (FRONT_Y1 - FRONT_Y0) / 2 + RIB_T / 2)

    # drum A shaft: M5 heat-set insert in the rear end, M5 x 12 through the cap
    f["FST_Insert_A_shaft"] = to_y(insert(7.0, 4.2, 8.0), A_SHAFT_Y0, ox, oz)
    f["FST_Screw_A_cap"] = to_y(Rot(180, 0, 0) * shcs(5, 12, 8.5, 5.0), A_R1_Y - 3.0, ox, oz)
    return f


# ----------------------------------------------------------------------------
# The eight builds: test arms (desk) or 4-bar leg (rail stand), each with or without the ab/ad unit
# and with or without the front gantry.
# ----------------------------------------------------------------------------
VARIANTS = {}
for _leg in (False, True):
    for _gantry in (False, True):
        for _abad in (False, True):
            _n = ("leg" if _leg else "flange") + ("_abad" if _abad else "") + ("_gantry" if _gantry else "")
            VARIANTS[_n] = dict(leg=_leg, abad=_abad, gantry=_gantry)
A_PARTS = ("Big_Drum_A", "Output_Flange_A", "A_Shaft_Cap", "Test_Arm_A", "Leg4_Crank")
B_PARTS = ("Big_Drum_B", "Output_Flange_B", "Test_Arm_B", "Leg4_Thigh")
A_FST = ("FST_Insert_A_nose", "FST_Screw_A_flange", "FST_Insert_A_flange", "FST_Screw_A_arm", "FST_Insert_A_shaft", "FST_Screw_A_cap")
B_FST = ("FST_Insert_B_hub", "FST_Insert_B_flange", "FST_Screw_B_flange", "FST_Screw_B_arm")
DESK_SLAB = 40.0


def leg_floor_drop():
    """Model z of the floor below the hip when the 4-bar foot stands at rest."""
    _, foot4, _, _ = leg4_points(L4_Q0["thigh"], L4_Q0["shin"])
    return foot4[1] - L4_FOOT_R


def mount_parts(v):
    """Mount for a build: label -> (solid, group, colour, kind). group 'module' or 'world'."""
    out = {}
    hw = lambda c: Compound(children=c)
    if not v["abad"]:
        out["FST_Insert_rails"] = (hw([Pos(x, y, RAIL_TOP - 8.0) * insert(5.8, 4.0, 8.0) for x, y in rail_insert_points()]),
                                   "module", "brass", "hw")
    if not v["abad"] and not v["leg"]:                                  # desk base on the top rails
        out["Desk_Base"] = (desk_base(), "module", "black", "part")
        out["Mount_HW_base_screws"] = (hw([Pos(x, y, RAIL_TOP + DESK_T - 4.2) * shcs(4, 10, 7.0, 4.0)
                                           for x, y in rail_insert_points()]), "module", "screw", "hw")
        out["Mount_HW_wood_screws"] = (hw([Pos(x, y, RAIL_TOP) * wood_screw(30.0) for x, y in DESK_SCREWS]), "module", "screw", "hw")
        out["REF_Desk"] = (Pos(0, 45.0, RAIL_TOP + DESK_T) * Box(700.0, 500.0, DESK_SLAB, align=(Align.CENTER, Align.CENTER, Align.MIN)),
                           "world", "desk", "ref")
    elif not v["abad"]:                                                 # rail carriage on the top rails
        out["Rail_Carriage"] = (rail_carriage5(), "module", "black", "part")
        out["Mount_HW_carriage_screws"] = (hw([Pos(x, y, RAIL_TOP + DESK_T) * shcs(4, 14, 7.0, 4.0)
                                               for x, y in rail_insert_points()]), "module", "screw", "hw")
        out["Mount_HW_block_screws"] = (hw([Pos(x, CARRIAGE5_Y0 + 8.0, z) * Rot(-90, 0, 0) * shcs(3, 12, 5.5, 3.0)
                                            for zc in RAIL5_BLOCKS_Z for x in (-10.0, 10.0) for z in (zc - 10.0, zc + 10.0)]),
                                        "module", "screw", "hw")
        b0, b1 = STAND_Y["block"]
        dy = CARRIAGE5_Y0 - CARRIAGE_Y0
        out["REF_MGN12H_blocks"] = (hw([Pos(0, (b0 + b1) / 2 + dy, zc) * Box(27.0, b1 - b0, 45.4) for zc in RAIL5_BLOCKS_Z]),
                                    "module", "steel", "ref")
    elif not v["leg"]:                                                  # desk angles at both ends of the cradle
        out["Desk_Angle_Rear"] = (desk_angle("rear"), "world", "black", "part")
        out["Desk_Angle_Front"] = (desk_angle("front"), "world", "black", "part")
        out["Mount_HW_wood_screws"] = (hw([Pos(x, y, DESK_Z_AB) * wood_screw(30.0) for x, y in desk_angle_screws()]),
                                       "world", "screw", "hw")
        out["REF_Desk"] = (Pos(-40.0, 10.0, DESK_Z_AB + DESK_T) * Box(620.0, 520.0, DESK_SLAB, align=(Align.CENTER, Align.CENTER, Align.MIN)),
                           "world", "desk", "ref")
    if v["leg"]:
        x0 = AB_RAIL_X if v["abad"] else 0.0
        drop = leg_floor_drop()
        for n, so in stand_parts(x0, (AB_RAIL_Y0 if v["abad"] else CARRIAGE5_Y0) - CARRIAGE_Y0).items():
            out[n] = (Pos(0, 0, drop) * so, "world", "silver", "ref")
    return out


def assembly(name):
    """Every placed solid of one build: label -> (solid, group, colour, kind).
    group: 'module' (motor plate side), 'A' / 'B' (turn with output A / B), 'leg_shin', 'leg_coupler',
    'leg_knee', 'leg_crank', 'leg_lever' (4-bar links and their pin stacks), 'world' (does not move:
    the ab/ad frame, desk, stand). With ab/ad everything except 'world' turns about the ab/ad axis.
    kind: 'part' (printed), 'hw' (screw, nut, insert, bearing, pin), 'ref' (motor, board, block, stand)."""
    v = VARIANTS[name]
    out = {}
    for n, (_, placed, color) in build().items():
        if n in ("Gantry", "Gantry_Post") and not v["gantry"]:
            continue
        for i, p in enumerate(placed):
            lab = f"{n}_{i + 1}" if len(placed) > 1 else n
            grp = "A" if n in A_PARTS else "B" if n in B_PARTS else "module"
            out[lab] = (p, grp, color, "part")
    for n, p in reference().items():
        if n == "REF_Bearing_17x30_gantry" and not v["gantry"]:
            continue
        grp = ("A" if "_A_" in n else "B") if n.startswith("REF_Tensioner") else \
              "B" if n.startswith("REF_Bearing_30x42_B") else "module"
        out[n] = (p, grp, "steel", "hw" if n.startswith(("REF_Bearing", "REF_Tensioner")) else "ref")
    for n, c in fasteners(front_bracket=v["abad"]).items():
        if n in ("FST_Bolt_gantry", "FST_Insert_gantry") and not v["gantry"]:
            continue
        out[n] = (c, "A" if n in A_FST else "B" if n in B_FST else "module", "brass" if n.startswith(("FST_Insert", "FST_Standoff")) else "screw", "hw")
    if v["leg"]:
        groups = {"Leg4_Thigh": "B", "Leg4_Crank": "A", "Leg4_Shin": "leg_shin", "Leg4_Foot": "leg_shin", "Leg4_Coupler": "leg_coupler",
                  "Leg4_Shin_Plate_Back": "leg_shin", "Leg4_Shin_Plate_Front": "leg_shin"}
        for n, (_, p, color) in leg4_parts(gantry=v["gantry"]).items():
            out[n] = (p, groups[n], color, "part")
        for n, (c, color, j, _) in leg_hardware("4bar").items():
            out[n] = (c, "leg_shin" if j == "shin" else f"leg_{j}", color, "hw")
    else:
        for n, (_, p, color) in test_arms(gantry=v["gantry"]).items():
            out[n] = (p, "A" if n.endswith("A") else "B", color, "part")
    if v["abad"]:
        mv, fx = ab_parts(desk=not v["leg"])
        for n, (so, color) in mv.items():
            out[n] = (so, "module", color, "hw" if "HW" in n or "REF" in n else "part")
        for n, (so, color) in fx.items():
            if not v["leg"] and n in ("Hip_HW_block_screws", "Hip_REF_MGN12H_blocks"):
                continue                                                # desk builds: no rail blocks on the spine
            kind = "ref" if n in ("Abad_Motor", "Abad_moteus", "Hip_REF_MGN12H_blocks") else \
                   "hw" if "HW" in n or "Bearing" in n else "part"
            out[n] = (so, "world", color, kind)
    out.update(mount_parts(v))
    return out


def pose_assembly(asm, ta=0.0, tb=0.0, legq=None, phi=0.0):
    """Place a build: outputs A, B turned ta, tb (deg); legq = (thigh, shin) 4-bar angles (then ta, tb
    follow from the leg); ab/ad phi (deg). Returns label -> (solid, group)."""
    ox, oz = OUT
    rot = lambda p, a, cx, cz: Pos(cx, 0, cz) * Rot(0, -a, 0) * Pos(-cx, 0, -cz) * p
    if legq is not None:
        qt, qs = legq
        tb, ta = qt - L4_Q0["thigh"], qs - L4_Q0["shin"]
        k0, _, c0, d0 = leg4_points(L4_Q0["thigh"], L4_Q0["shin"])
        k, _, c, d = leg4_points(qt, qs)
        move = {"leg_shin": lambda p: Pos(k[0] - k0[0], 0, k[1] - k0[1]) * rot(p, ta, *k0),
                "leg_coupler": lambda p: Pos(c[0] - c0[0], 0, c[1] - c0[1]) * rot(p, tb, *c0),
                "leg_knee": lambda p: Pos(k[0] - k0[0], 0, k[1] - k0[1]) * p,
                "leg_crank": lambda p: Pos(c[0] - c0[0], 0, c[1] - c0[1]) * p,
                "leg_lever": lambda p: Pos(d[0] - d0[0], 0, d[1] - d0[1]) * p}
    else:
        move = {}
    move["A"] = lambda p: rot(p, ta, ox, oz)
    move["B"] = lambda p: rot(p, tb, ox, oz)
    ab = lambda p: Pos(0, AB_Y, AB_Z) * Rot(phi, 0, 0) * Pos(0, -AB_Y, -AB_Z) * p
    out = {}
    for n, (so, grp, _, _) in asm.items():
        p = move[grp](so) if grp in move else so
        if phi and grp != "world":
            p = ab(p)
        out[n] = (p, grp)
    return out


if __name__ == "__main__":
    print(f"v9.1 buildable capstan  ratio {RATIO:.2f}:1  output range +-{OUT_RANGE / 2:.1f} deg", flush=True)
    for sub in ("", "leg_4bar", "abad", "mounts"):
        os.makedirs(os.path.join(ROOT, "print", sub), exist_ok=True)
        os.makedirs(os.path.join(ROOT, "step", sub), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "assembly"), exist_ok=True)

    def export_part(sub, name, solid):
        export_step(solid, os.path.join(ROOT, "step", sub, f"{name}.step"))
        export_stl(solid, os.path.join(ROOT, "print", sub, f"{name}.stl"), tolerance=0.02, angular_tolerance=0.15)

    def flat(solid, rot):
        """Print orientation: rotated, centred on the bed, resting on z 0."""
        pr = rot * solid
        b = pr.bounding_box()
        return Pos(-b.center().X, -b.center().Y, -b.min.Z) * pr

    for name, (solid, placed, color) in build().items():
        export_part("", name, solid)
        print(f"  {name:24s} qty {len(placed)}")
    for gt, suffix in ((False, ""), (True, "_Gantry")):
        arm = test_arms(gantry=gt)["Test_Arm_A"][0]
        export_part("", f"Test_Arm_A{suffix}", flat(arm, Rot(0, 0, 0)))
    export_part("", "Test_Arm_B", flat(test_arms()["Test_Arm_B"][0], Rot(0, 0, 0)))
    for gt, suffix in ((True, ""), (False, "_NoGantry")):
        for name, (solid, placed, color) in leg4_parts(gantry=gt).items():
            if suffix and name != "Leg4_Crank":
                continue
            export_part("leg_4bar", name + suffix, solid)
    mv, fx = ab_parts()
    printed = {"Abad_Drum": Rot(0, -90, 0), "Abad_Small_Drum": Rot(0, -90, 0), "Hip_Outer_Plate": Rot(0, -90, 0),
               "Hip_Inner_Plate": Rot(0, 90, 0), "Hip_Spine": Rot(90, 0, 0), "Hip_Front_Plate": Rot(0, -90, 0),
               "Hip_Trunnion": Rot(0, -90, 0), "End_Bracket_Front": Rot(0, 90, 0)}   # flattest face on the bed
    for name, (solid, color) in list(mv.items()) + list(fx.items()):
        if name in printed:
            export_part("abad", name, flat(solid, printed[name]))
    hp = Cylinder(8.0, AB_X["inner"][0] - AB_X["outer"][1], align=MIN) - Cylinder(M4_CLR / 2, 57.0, align=MIN)
    export_part("abad", "Hip_Post", hp)
    export_part("mounts", "Desk_Base", flat(desk_base(), Rot(0, 0, 0)))
    for end in ("rear", "front"):
        export_part("mounts", f"Desk_Angle_{end.title()}", flat(desk_angle(end), Rot(180, 0, 0)))   # foot on the bed, upright up
    export_part("mounts", "Rail_Carriage", flat(rail_carriage5(), Rot(0, 0, 0)))     # plate on the bed

    tint = {"screw": "dimgray", "steel": "lightsteelblue", "motor": "goldenrod", "pcb": "green", "silver": "silver",
            "yellow": "gold", "gray": "lightgray", "black": "dimgray", "red": "firebrick", "white": "gold", "brass": "goldenrod",
            "desk": "burlywood"}
    for vname in VARIANTS:
        children = []
        for name, (solid, grp, color, kind) in assembly(vname).items():
            solid.label, solid.color = name, Color(tint.get(color, color))
            children.append(solid)
        export_step(Compound(children=children, label=f"Capstan_v6_2_{vname}"), os.path.join(ROOT, "assembly", f"{vname}.step"))
        print(f"  assembly/{vname}.step  {len(children)} items", flush=True)
    print("done")
