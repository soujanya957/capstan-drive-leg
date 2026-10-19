import sys, itertools
sys.path.insert(0, ".")
import generate as g
from build123d import *
parts = g.build(); ref = g.reference()
ox, oz = g.OUT
A_GROUP = {"Big_Drum_A", "Output_Flange_A", "Test_Arm_A", "A_Shaft_Cap"}
B_GROUP = {"Big_Drum_B", "Output_Flange_B", "Test_Arm_B"}
def items_at(aa, ab):
    out = []
    for n,(s,placed,c) in parts.items():
        for i,p in enumerate(placed):
            nm=f"{n}#{i+1}"
            if n in A_GROUP or (n=="Rope_Tensioner_Slider" and i<2):
                p = Pos(ox,0,oz)*Rot(0,aa,0)*Pos(-ox,0,-oz)*p
            if n in B_GROUP or (n=="Rope_Tensioner_Slider" and i>=2):
                p = Pos(ox,0,oz)*Rot(0,ab,0)*Pos(-ox,0,-oz)*p
            out.append((nm,p))
    for n,s in ref.items():
        if n.startswith("REF_Tensioner_A"):
            s = Pos(ox,0,oz)*Rot(0,aa,0)*Pos(-ox,0,-oz)*s
        elif n.startswith("REF_Tensioner_B"):
            s = Pos(ox,0,oz)*Rot(0,ab,0)*Pos(-ox,0,-oz)*s
        for j,so in enumerate(s.solids()): out.append((f"{n}.{j}", so))
    return out
def bb(s): b=s.bounding_box(); return (b.min.X,b.min.Y,b.min.Z,b.max.X,b.max.Y,b.max.Z)
def ov(a,b): return all(a[i]<b[i+3]-1e-6 and b[i]<a[i+3]-1e-6 for i in range(3))
def grp(n): return n.split("#")[0].split(".")[0]
R=round(g.OUT_RANGE/2,1)
bad = []
for aa, ab in [(0,0),(R,-R),(-R,R),(R,R),(-R,-R),
               (20,0),(-20,0),(0,20),(0,-20)]:
    items = items_at(aa, ab); bbs={n:bb(s) for n,s in items}; hits=[]
    for (n1,s1),(n2,s2) in itertools.combinations(items,2):
        # Screw and nut thread volumes are deliberately simplified and overlap.
        if n1.startswith("REF_Tensioner") and n2.startswith("REF_Tensioner"): continue
        if grp(n1)==grp(n2) and grp(n1).startswith("REF"): continue
        if not ov(bbs[n1],bbs[n2]): continue
        x=(s1 & s2); v=x.volume if x is not None else 0.0
        if v>0.5: hits.append((round(v,1),n1,n2))
    print(f"angles A={aa:+} B={ab:+}: {len(items)} bodies, clashes:", sorted(hits,reverse=True) or "none")
    bad += hits
assert not bad, "module clashes"
print("pass: module clear at all nine output poses")
