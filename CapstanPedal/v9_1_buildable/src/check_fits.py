"""Print fits: every designed clearance or interference between a printed part and what fits in it,
against the usual FDM ranges (0.4 mm nozzle, holes print ~0.1-0.2 mm under size), plus the thinnest
walls left round inserts and nuts. Diametral values in mm; + = clearance, - = interference.

Journals (a printed shaft in a bearing's bore) are drawn nominal: printed outside diameters come out ~0.05-0.15
over, a light press that keeps the inner race from creeping (v9 drew them 0.05-0.10 under and this table
mis-signed them as interference). Tune first: print the fit coupon (one bearing pocket, one Ø30 journal,
one insert hole, one tensioner peg nut slot; BOM.md) and adjust PRESS / JOURNAL_CLR.
"""
import math

import generate as g

fits = [  # (fit, part, mating item, designed (diametral), allowed range, kind)
    ("Bearing pockets (press)", "plates, drums, gantry, links", "6806/6808/6902/6903/636/6000 OD", g.PRESS, (0.05, 0.25)),
    ("Drum A shaft in 6806s", "Big_Drum_A", "6806 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("A's tube in drum B's 6806s", "Big_Drum_A", "6806 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("Drum B journal in 6808", "Big_Drum_B", "6808 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("Small-drum tips in 6902", "Small_Drum_A/B", "6902 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("Crank / arm stub in gantry 6903", "Leg4_Crank, Test_Arm_A_Gantry", "6903 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("Ab/ad hub in 6808s", "Abad_Drum", "6808 bore", g.JOURNAL_CLR, (-0.05, 0.05)),
    ("Front trunnion in the 6806 (floating)", "Hip_Trunnion", "6806 bore", 30.0 - g.TRUNNION_D, (0.05, 0.25)),
    ("M3 clearance holes", "all", "M3 screw", g.M3_CLR - 3.0, (0.3, 0.6)),
    ("M4 clearance holes", "all", "M4 screw", g.M4_CLR - 4.0, (0.3, 0.7)),
    ("M6 pin holes (fixed links)", "crank, coupler", "M6 pin", g.PIN_CLR - g.PIN_D, (0.2, 0.6)),
    ("Knee pin hole (thigh)", "Leg4_Thigh", "Ø10 shoulder screw", g.KNEE_PIN_CLR - g.KNEE_PIN_D, (0.2, 0.6)),
    ("M3 heat-set inserts", "all", "insert (Ø4.6 knurl)", g.M3_INSERT - 4.6, (-0.8, -0.3)),
    ("M4 heat-set inserts", "all", "insert (Ø6.0 knurl)", g.M4_INSERT - 6.0, (-0.8, -0.3)),
    ("M2.5 heat-set inserts", "motor plate", "insert (Ø4.0 knurl)", g.M2_5_INSERT - 4.0, (-0.8, -0.3)),

    ("Double-D drive (flange A on A's nose)", "Output_Flange_A", "nose flats 18", 0.4, (0.2, 0.5)),
    ("Arm / crank bore on flange A's ring", "Leg4_Crank, Test_Arm_A", "Ø28 locating ring", 0.2, (0.1, 0.4)),
    ("Tensioner peg nut slot (across flats)", "big drums", "M4 nylon nut 7 AF", 0.3, (0.2, 0.5)),
    ("Tensioner peg head counterbore", "big drums", "M4 head Ø7", 0.8, (0.4, 1.2)),
    ("Lip hole vs inner-race spacer", "shin, coupler", "Ø9 spacer", 2 * g.PIN_LIP_R - 2 * g.PIN_SPACER_R, (1.0, 8.0)),
]
walls = [  # (where, wall thickness, minimum)
    ("Ab/ad hub: M4 insert to the Ø19 bore (v9.1)", g.AB_HUB_R - g.M4_INSERT / 2 - 9.5, 1.6),
    ("Ab/ad hub: M4 insert to the Ø40 surface (v9.1)", g.B40[0] / 2 - g.AB_HUB_R - g.M4_INSERT / 2, 1.6),
    ("Flange B: M3 insert to the rim (v9.1)", 44.0 - 35.0 - g.M3_INSERT / 2, 1.6),
    ("Flange B: pill slot to the hub screw counterbore (v9.1)", 24.5 - 2.75 - (g.B_HUB_SCREW_R + 3.1), 1.6),
    ("A's nose: M3 insert to the double-D flat", g.NOSE_HEX_AF / 2 - (g.A_NOSE_SCREW_R * math.cos(math.radians(45)) + g.M3_INSERT / 2), 1.6),
    ("A's nose: M3 insert to the Ø22 surface", g.A_NOSE_R - g.A_NOSE_SCREW_R - g.M3_INSERT / 2, 1.6),
    ("Drum B journal: M3 insert to the bore", g.B_HUB_SCREW_R - g.M3_INSERT / 2 - (g.A_NOSE_R + 1.0), 1.6),
    ("Drum B journal: M3 insert to the Ø40 surface", g.B40[0] / 2 - g.B_HUB_SCREW_R - g.M3_INSERT / 2, 1.6),
    ("Tensioner peg: well to the nut slot", g.PEG_NUT_GAP, 1.2),
    ("Tensioner peg: well to the end face", g.PEG_X - g.PEG_WELL_R, 4.0),
    ("Tensioner peg: nut slot to the drum's far face", 18.0 - (g.PEG_Z + g.PEG_NUT_AF / math.sqrt(3) + 0.25), 4.0),
    ("Ab/ad hub: M4 hole to the Ø40 surface", 20.0 - 16.0 - g.M4_CLR / 2, 1.6),
    ("Front wall 6808 radial wall at output ring", g.WALL_OUT_R - (g.B40[1] + g.PRESS) / 2, 8.0),
    ("636 lip (outer ring, axial location only; 5 layers)", g.PIN_LIP, 1.0),
    ("6000 knee lip (outer ring support)", 8.0 + g.KNEE_BOSS - g.KNEE_BRG[2], 1.6),
    ("Shin knee boss round the 6000", 17.0 - g.KNEE_BRG_POCKET, 3.0),
    ("Shin lever boss round the 636", g.SHIN_LEVER_R - g.PIN_BRG_POCKET, 2.0),
    ("Coupler crank end round the 636", g.COUPLER_PROFILE[0][1] / 2 - g.PIN_BRG_POCKET, 2.0),
    ("Top rail: M4 insert to the post bore", (g.RAIL_TOP - 8.0) - g.POSTS[0][1] - g.M4_CLR / 2, 2.0),
    ("Front bracket web behind the trunnion inserts", (g.FB_X[1] - g.FB_X[0]) - 6.0, 1.6),
    ("Trunnion flange outside the countersinks", 26.0 - (20.0 + 4.15), 1.6),
    ("Trunnion journal wall (hollow)", g.TRUNNION_D / 2 - 8.0, 3.0),
    ("Cradle front plate lip on the 6806", (g.HF_X[1] - g.HF_X[0]) - g.B30[2], 1.0),
    ("Shin back plate: thickness for its 3.8 mm M3 inserts", g.SHIN_PLATE_T, 3.8),
    ("Spine: tab M4 insert depth vs 8 mm insert (M4 x 16 engagement)", 16.0 - g.SPINE_T, 5.0),
]
bad = []
print(f"{'fit':40s} {'designed':>9s}  range")
for name, part, mate, v, (lo, hi) in fits:
    ok = lo - 1e-9 <= v <= hi + 1e-9
    print(f"{'ok  ' if ok else 'FAIL'}{name:36s} {v:+9.2f}  {lo:+.2f} .. {hi:+.2f}   {part} / {mate}")
    if not ok:
        bad.append(name)
print(f"\n{'wall':48s} {'mm':>5s}  min")
for name, t, mn in walls:
    ok = t >= mn - 1e-9
    print(f"{'ok  ' if ok else 'FAIL'}{name:44s} {t:5.2f}  {mn}")
    if not ok:
        bad.append(name)
assert not bad, f"fits out of range: {bad}"
print("\npass: every fit in its FDM range and every listed wall at least its minimum")
