"""Exports the body stress test (stress_test.py) as a Fusion 360 static-stress package in ../stress_test_fusion360:
the three front bulkheads as STEP (body coordinates, mm), every load and constraint (loads.json, LOADS.md), a
Fusion script that imports the parts and marks the load points, and the steps to set the studies up.

Fusion's API cannot create or solve simulation studies, so the studies are set up by hand from LOADS.md.

Run: python export_fusion_study.py
"""
import json
import os
import shutil

import numpy as np
from build123d import export_step

import generate_quad as q
import stress_test as st

g = q.g
DIR = os.path.join(os.path.dirname(st.HERE), "stress_test_fusion360")
KINDS = (("outer", "Centre_Bulkhead_F"), ("inner", "Inner_Bulkhead_F"), ("front", "End_Frame_F"))


def load_table(G):
    xs = [G["x"]["outer"], G["x"]["inner"], G["x"]["front"]]
    T = st.TAU_AB / (g.BIG_PITCH_R / 1000.0)
    cases = [("Push-landing 520N", (st.F_PUSH, st.F_PUSH), 2.0), ("Impact x2 1040N", (2 * st.F_PUSH, 2 * st.F_PUSH), 1.0),
             ("One leg x2 1040N", (2 * st.F_PUSH, 0.0), 1.0)]
    out = {}
    for kind, name in KINDS:
        xa, xb = q.bulkhead_x(kind, True)
        seat_r = {"outer": (g.B40[1] + g.PRESS) / 2, "inner": (g.B40[1] + g.PRESS) / 2, "front": (g.B30[1] + g.PRESS) / 2}[kind]
        brg = {"outer": "6808 (hub, outer)", "inner": "6808 (hub, inner)", "front": "6806 (trunnion)"}[kind]
        studies = []
        for tag, (FL, FR), target in cases:
            loads = []
            for side, sy, F in (("left (+Y)", 1, FL), ("right (-Y)", -1, FR)):
                if F == 0:
                    continue
                R = st.beam_reactions(xs, G["hip_x"], F)[G["x"][kind]]
                loads.append(dict(type="Bearing load", on=f"{brg} seat, {side} pod", centre=[round((xa + xb) / 2, 2), round(sy * G["y_ax"], 2), G["z_ax"]],
                                  axis=[1, 0, 0], seat_radius=round(seat_r, 3), force=[0.0, 0.0, round(R, 1)]))
                if kind in ("outer", "inner"):
                    my, mz = G["motorC"]
                    d = np.array([0.0, sy * G["y_ax"] - sy * my, G["z_ax"] - mz])
                    d = d / np.linalg.norm(d)
                    Fm = (0.5 * T * d).round(1).tolist()
                    if kind == "outer":
                        loads.append(dict(type="Force (distributed)", on=f"motor C stator seat (face at X {xb}, within 34 mm of the motor axis), {side}",
                                          centre=[xb, round(sy * my, 2), mz], force=Fm))
                    else:
                        loads.append(dict(type="Bearing load", on=f"motor C tip bearing seat (15x28), {side}", centre=[round((xa + xb) / 2, 2), round(sy * my, 2), mz],
                                          axis=[1, 0, 0], seat_radius=round((g.B15[1] + g.PRESS) / 2, 3), force=Fm))
            studies.append(dict(name=f"{name} - {tag}", target_sf=target, loads=loads))
        out[name] = dict(step=f"step/{name}.step", x_range=[xa, xb],
                         constraints=[dict(type="Fixed", on=f"top edge faces and tabs (Z >= {q.Z_TOP - 0.3})"),
                                      dict(type="Fixed", on=f"bottom edge faces and tabs (Z <= {q.Z_BOT + 0.3})")],
                         studies=studies)
    return out


FUSION_SCRIPT = r'''"""Fusion 360 script: imports the stress-test bulkheads (STEP, body coordinates, mm) into a new design and
adds a construction point at every load centre in loads.json, named after the load, to pick faces by.
Install: Utilities > Add-Ins > Scripts and Add-Ins > + (My Scripts) > choose this folder; then Run."""
import json
import os
import traceback

import adsk.core
import adsk.fusion

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        data = json.load(open(os.path.join(PKG, "loads.json")))
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(app.activeProduct)
        design.fusionUnitsManager.distanceDisplayUnits = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
        root = design.rootComponent
        im = app.importManager
        for name, part in data.items():
            opts = im.createSTEPImportOptions(os.path.join(PKG, part["step"]))
            im.importToTarget(opts, root)
            occ = root.occurrences.item(root.occurrences.count - 1)
            occ.component.name = name
            pts = occ.component.constructionPoints
            seen = set()
            for study in part["studies"]:
                for ld in study["loads"]:
                    c = tuple(ld["centre"])
                    if c in seen:
                        continue
                    seen.add(c)
                    inp = pts.createInput()
                    inp.setByPoint(adsk.core.Point3D.create(c[0] / 10.0, c[1] / 10.0, c[2] / 10.0))      # cm
                    pt = pts.add(inp)
                    pt.name = ld["on"][:60]
        ui.messageBox("Imported %d bulkheads. Simulation: new Static Stress study per row of LOADS.md." % len(data))
    except Exception:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
'''

MANIFEST = {"autodeskProduct": "Fusion360", "type": "script", "author": "", "description": {"": "Import the quadruped body stress-test parts"},
            "supportedOS": "windows|mac", "editEnabled": True}


def main():
    os.makedirs(os.path.join(DIR, "step"), exist_ok=True)
    pl = q.cradle_plates()
    for kind, name in KINDS:
        export_step(q.bulkhead(kind, True, pl), os.path.join(DIR, "step", f"{name}.step"))
        print("  step", name, flush=True)
    G = st.geometry()
    table = load_table(G)
    with open(os.path.join(DIR, "loads.json"), "w") as f:
        json.dump(table, f, indent=1)
    sd = os.path.join(DIR, "ImportStressParts")
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "ImportStressParts.py"), "w") as f:
        f.write(FUSION_SCRIPT)
    with open(os.path.join(sd, "ImportStressParts.manifest"), "w") as f:
        json.dump(MANIFEST, f, indent=1)
    lines = ["# Loads and constraints", "",
             "All coordinates in the body frame, mm (X along the body, Y to the left, Z up); forces in N. Each bulkhead STEP is",
             "the printed part (8 mm; the 7.5 mm centre one). For the laser build use the same solid with 6061-T6.", ""]
    for name, part in table.items():
        lines += [f"## {name}", "", f"`{part['step']}`, X {part['x_range'][0]}–{part['x_range'][1]}.", "",
                  "Constraints (every study): " + "; ".join(f"**{c['type']}** on {c['on']}" for c in part["constraints"]) + ".", ""]
        for s in part["studies"]:
            lines += [f"### {s['name']} (target SF {s['target_sf']})", "", "| Load | On | Centre (X, Y, Z) | Axis | Seat radius | Force (X, Y, Z) |", "|---|---|---|---|---|---|"]
            for ld in s["loads"]:
                lines.append(f"| {ld['type']} | {ld['on']} | {ld['centre']} | {ld.get('axis', '–')} | {ld.get('seat_radius', '–')} | {ld['force']} |")
            lines.append("")
    with open(os.path.join(DIR, "LOADS.md"), "w") as f:
        f.write("\n".join(lines))
    res = os.path.join(os.path.dirname(st.HERE), "stress_test", "results.md")
    if os.path.exists(res):
        shutil.copy(res, os.path.join(DIR, "results_python_fea.md"))
    print("written", DIR)


if __name__ == "__main__":
    main()
