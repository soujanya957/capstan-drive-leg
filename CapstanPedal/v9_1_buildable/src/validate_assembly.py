"""Sample the actual insertion directions, not only the final assembly poses."""
import generate as g
from build123d import Pos

parts = {name: placed[0] for name, (_solid, placed, _color) in g.build().items() if len(placed) == 1}
ox, oz = g.OUT
b1 = g.to_y(g.bearing(*g.B30), g.B_Y0, ox, oz)
b2 = g.to_y(g.bearing(*g.B30), g.B2_Y, ox, oz)
parts["B1_bearing"] = b1
parts["B2_bearing"] = b2
parts["A1_bearing"] = g.to_y(g.bearing(*g.B30), g.A_R1_Y, ox, oz)
parts["A2_bearing"] = g.to_y(g.bearing(*g.B30), g.A_R2_Y, ox, oz)
parts["Wall_bearing"] = g.to_y(g.bearing(*g.B40), g.WALL_BRG_Y[0], ox, oz)


def overlapping_boxes(a, b):
    aa, bb = a.bounding_box(), b.bounding_box()
    return all(getattr(aa.min, c) < getattr(bb.max, c) - 1e-5 and
               getattr(bb.min, c) < getattr(aa.max, c) - 1e-5 for c in "XYZ")


def check_insertion(name, fixed, sign=1, carry=()):
    """Back `name` (and anything fitted with it) out along sign*Y from its seat."""
    shape = parts[name]
    for c in carry:
        shape = shape + parts[c]
    for distance in list(range(0, 81, 2)):
        moving = Pos(0, sign * distance, 0) * shape
        for fixed_name in fixed:
            other = parts[fixed_name]
            if not overlapping_boxes(moving, other):
                continue
            intersection = moving & other
            volume = intersection.volume if intersection is not None else 0.0
            if volume > 0.5:
                raise AssertionError(f"{name} hits {fixed_name} at Y offset {distance} mm: {volume:.2f} mm^3")
    print(f"{name}: clear along {'+' if sign > 0 else '-'}Y insertion path (2 mm samples)")


# Sequence is deliberately ordered: the removable B flange and both arms go on
# only after the front wall has passed the B hub. Rope is routed before the wall.
check_insertion("Big_Drum_A", ["Motor_Plate", "Small_Drum_A", "Small_Drum_B"])
check_insertion("Big_Drum_B", ["Motor_Plate", "Small_Drum_A", "Small_Drum_B", "Big_Drum_A"])
# drum A's bearings go into the plate housing: A1 from the rear, A2 from the front
check_insertion("A1_bearing", ["Motor_Plate"], sign=-1)
check_insertion("A2_bearing", ["Motor_Plate"])
# drum B's bearings: both from the rear, B2 first (4 mm gap between them)
check_insertion("B2_bearing", ["Big_Drum_B"], sign=-1)
check_insertion("B1_bearing", ["Big_Drum_B"], sign=-1)
# the 6808 goes into the front wall from the rear, against the front lip
check_insertion("Wall_bearing", ["Front_Wall"], sign=-1)
check_insertion("A_Shaft_Cap", ["Motor_Plate"], sign=-1)
check_insertion("Front_Wall", ["Motor_Plate", "Small_Drum_A", "Small_Drum_B", "Big_Drum_A", "Big_Drum_B"],
                carry=("Wall_bearing",))                         # the 6808 slides onto B's journal with the wall
check_insertion("Output_Flange_B", ["Front_Wall", "Big_Drum_A", "Big_Drum_B"])
# v4: the 4-bar leg goes on from the front, then the gantry over the crank's stub
legs4 = g.leg4_parts()
for n_, (_, pl_, _) in legs4.items():
    parts[n_] = pl_
gp = full_build = g.build()
parts["Gantry"] = gp["Gantry"][1][0]
check_insertion("Leg4_Thigh", ["Front_Wall", "Big_Drum_A", "Big_Drum_B", "Output_Flange_B"])
check_insertion("Output_Flange_A", ["Front_Wall", "Big_Drum_A", "Big_Drum_B", "Output_Flange_B", "Leg4_Thigh"])
check_insertion("Leg4_Crank", ["Front_Wall", "Big_Drum_A", "Big_Drum_B", "Output_Flange_B", "Leg4_Thigh", "Output_Flange_A"])
check_insertion("Gantry", ["Front_Wall", "Big_Drum_A", "Big_Drum_B", "Output_Flange_B", "Leg4_Thigh", "Output_Flange_A",
                           "Leg4_Crank", "Leg4_Shin", "Leg4_Coupler", "Leg4_Foot", "Leg4_Shin_Plate_Back", "Leg4_Shin_Plate_Front"])


def check_insertion_x(name, fixed, sign=-1, carry=()):
    """Back `name` (and anything fitted with it) out along sign*X from its seat: the end brackets slide in
    from the module's ends with their ears on the plates' outer faces; the cradle's front plate slides
    over the trunnion."""
    shape = parts[name]
    for c in carry:
        shape = shape + parts[c]
    for distance in range(0, 81, 2):
        moving = Pos(sign * distance, 0, 0) * shape
        for fixed_name in fixed:
            other = parts[fixed_name]
            if not overlapping_boxes(moving, other):
                continue
            x = moving & other
            if x is not None and x.volume > 0.5:
                raise AssertionError(f"{name} hits {fixed_name} at X offset {sign * distance} mm: {x.volume:.2f} mm^3")
    print(f"{name}: clear along {'+' if sign > 0 else '-'}X insertion path (2 mm samples)")


check_insertion_x("End_Bracket", ["Motor_Plate", "Front_Wall", "Big_Drum_A", "Big_Drum_B", "Small_Drum_A"])
# v6 cradle: the front bracket (trunnion already bolted on) slides in from motor B's end; the cradle's
# front plate, with its 6806 pressed in, slides over the trunnion
mv, fx = g.ab_parts()
for n_ in ("End_Bracket_Front", "Hip_Trunnion", "Hip_HW_trunnion_screws"):
    parts[n_] = mv[n_][0]
for n_ in ("Hip_Front_Plate", "Hip_Bearing_front"):
    parts[n_] = fx[n_][0]
check_insertion_x("End_Bracket_Front", ["Motor_Plate", "Front_Wall", "Big_Drum_A", "Big_Drum_B", "Small_Drum_B"], sign=+1,
                  carry=("Hip_Trunnion", "Hip_HW_trunnion_screws"))
check_insertion_x("Hip_Front_Plate", ["End_Bracket_Front", "Hip_Trunnion", "Hip_HW_trunnion_screws"], sign=+1,
                  carry=("Hip_Bearing_front",))

# Tensioners (winding pegs): each nylon nut drops into its slot from the open drum face, the M4 x 18
# backs out radially past the rim, and a 3 mm hex key (60 mm shown) reaches its head from outside.
from build123d import Cylinder, Rot
import math
ref = g.reference()
for stage, drum, y0, phi_c in (("A", "Big_Drum_A", g.A_Y0, -g.dir_deg(g.OUT, g.MA)),
                               ("B", "Big_Drum_B", g.B_Y0, -g.dir_deg(g.OUT, g.MB))):
    body = parts[drum]
    w = (g.A_Y1 - g.A_Y0) if stage == "A" else (g.B_Y1 - g.B_Y0)
    for side in (-1, 1):
        tag = f"REF_Tensioner_{stage}_{'minus' if side < 0 else 'plus'}"
        out_y = 1 if side > 0 else -1                                   # open face: front for +, rear for -
        for d in range(0, 21, 1):
            x = Pos(0, out_y * d, 0) * ref[f"{tag}_nut"] & body
            v = x.volume if x is not None else 0.0
            assert v < 0.5, f"{tag} nut jams {d} mm from its slot: {v:.2f} mm^3"
        frame = Rot(0, 0, phi_c) * g.peg_frame(side, w)
        for d in range(0, 31, 1):                                      # back the screw out along its axis
            sm = g.to_y(frame * Pos(0, 0, d + g.peg_k_rim() - g.PEG_SINK - g.PEG_HEAD[1]) *
                        g.shcs(4, g.PEG_L, *g.PEG_HEAD), y0, *g.OUT)
            x = sm & body
            v = x.volume if x is not None else 0.0
            assert v < 0.5, f"{tag} screw jams {d} mm out: {v:.2f} mm^3"
        key = g.to_y(frame * Pos(0, 0, g.peg_k_rim() - g.PEG_SINK) * Cylinder(1.5, 60.0, align=g.MIN), y0, *g.OUT)
        x = key & body
        assert (x.volume if x is not None else 0.0) < 0.1, f"{tag}: hex key line hits the drum"
    print(f"{drum}: peg nuts drop in from the face, pegs back out past the rim, key access clear")

assert 24.0 > 22.0, "flange B hub passes the wall's front lip"
assert g.A_NOSE_R + 1.0 < g.B_HUB_SCREW_R - g.M3_INSERT / 2 - 1.5, "journal wall inside the inserts"
assert g.B_HUB_SCREW_R + g.M3_INSERT / 2 + 1.5 < g.B40[0] / 2, "journal wall outside the inserts"
assert g.A_HOUSE_R < 46 - g.POST_R, "A housing clears the lower posts"
assert g.A_NOSE_SCREW_R + 3.1 < g.A_ARM_SPIGOT_R - 1.5, "flange A counterbores clear the arm spigot"
assert g.A_NOSE_SCREW_R * 2 * math.sin(math.radians(45)) > 2 * 3.1, "flange A counterbores stay separate"
assert g.A_NOSE_SCREW_R + g.M3_INSERT / 2 + 1.5 < g.A_NOSE_R, "nose wall outside its inserts"
assert g.A_ARM_SPIGOT_R + 0.1 < 24.0 - 1.7, "arm A bore clears its screws"

phi_a, phi_b = -g.dir_deg(g.OUT, g.MA), -g.dir_deg(g.OUT, g.MB)
def aligned_sixfold(a, b):
    return abs(((a-b+30) % 60)-30) < 1e-6
assert aligned_sixfold(phi_a, phi_a), "A flange to A tube pattern"
assert aligned_sixfold(phi_a + 30, phi_a + 300 + 30), "A crank pattern"
assert aligned_sixfold(phi_b + 30, phi_b + 30), "B flange to B hub pattern"
assert aligned_sixfold(phi_b, phi_b + 180), "B crank pattern"
print("pass: insertion paths, radial clearances, and bolt circle phases")
