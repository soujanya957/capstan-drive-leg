"""The MuJoCo model of the quadruped, built from the URDF (../urdf/capstan_quad.urdf, made from the CAD by
build_urdf.py) and completed with what URDF cannot say:

- the drives: motor C on ab/ad, motor B on the hip (output B), motor A on the crank (output A, which sets
  the shin's absolute angle through the parallelogram). 31.5 N*m each; the controller applies the
  torque-speed curve and output A's 25 N*m push limit (the same motor model as the RL work).
- the 4-bar's loop: the coupler's far end pinned to the shin's lever pin (a connect equality); knee and
  coupler are passive joints.
- the motor rotors (with their small drums) turn at -9x their output: joint equalities (URDF <mimic>).
- reflected rotor inertia (armature) on the three driven joints.
- a floor, lights, and the visual meshes kept out of collisions (contacts: body box, pod boxes, link
  capsules, 23 mm ball feet).
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

URDF = Path(__file__).resolve().parent.parent / "urdf" / "capstan_quad.urdf"
PODS = ("LF", "LR", "RF", "RR")
TAU = 31.5                                     # N*m per output (9:1)
W0 = 2000.0 / 60.0 * 2 * math.pi / 9.0         # no-load output speed, rad/s (KV 90 on 6S)
A_PUSH = 25.0                                  # output A in the direction that pushes the coupler (buckling)
ARMATURE = 1.9e-4 * 81.0                       # 8308 rotor through 9:1


def build(urdf=URDF, floor_friction=1.0):
    urdf = Path(urdf)
    root = ET.parse(urdf).getroot()
    mimics = []                                                       # (joint, of, multiplier)
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            mimics.append((j.get("name"), m.get("joint"), float(m.get("multiplier", 1.0))))
            j.remove(m)
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", meshdir=str(urdf.parent / "meshes"), discardvisual="false",
                  balanceinertia="true", fusestatic="false", strippath="true")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))

    # free-floating torso (URDF roots are fixed)
    torso = spec.body("torso")
    torso.add_freejoint(name="root")
    torso.pos = [0.0, 0.0, 0.48]
    torso.add_site(name="imu", pos=[0, 0, 0])

    # looks: sky, floor, lights
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.32, 0.36, 0.42], rgb2=[0.05, 0.06, 0.08], width=64, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.20, 0.22, 0.25], rgb2=[0.13, 0.14, 0.16], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.3, 0.33, 0.36], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid",
                friction=[floor_friction, 0.005, 0.0001])
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 3], dir=[-0.35, -0.45, -0.82], diffuse=[0.6, 0.6, 0.58],
                 castshadow=1)
    wb.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 3], dir=[0.6, 0.3, -0.4], diffuse=[0.25, 0.26, 0.3],
                 castshadow=0)
    spec.visual.headlight.ambient = [0.3, 0.3, 0.32]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.quality.shadowsize = 4096
    spec.visual.map.znear = 0.005
    spec.visual.map.haze = 0.0                                      # no bright band at the horizon
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.stat.extent = 1.5
    spec.stat.center = [0, 0, 0.3]
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # visual meshes: drawn, never collide; collision primitives: hidden (group 3)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.contype, g.conaffinity, g.group = 0, 0, 1
            g.density = 0
        elif g.name != "floor":
            g.group = 3
            g.contype, g.conaffinity = 1, 1
            g.rgba = [0.8, 0.3, 0.2, 0.4]
    for p in PODS:
        foot = spec.geom(f"foot_{p}")
        foot.friction = [floor_friction, 0.01, 0.0001]
        foot.condim = 3
        spec.body(f"shin_{p}").add_site(name=f"foot_{p}", pos=foot.pos)

    driven = ("abad", "hip", "crank")
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            rotor = "_rotor" in j.name
            j.armature = ARMATURE if j.name.endswith(driven) else 0.0
            j.damping = [0.0 if rotor else 0.05, 0.0, 0.0]
            j.limited = mujoco.mjtLimited.mjLIMITED_FALSE if rotor else mujoco.mjtLimited.mjLIMITED_TRUE
    for jn, of, mult in mimics:                                        # rotors: -9x their output
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=jn, name2=of, data=[0, mult, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    import json
    frames = json.load(open(urdf.parent / "frames.json"))
    for p in PODS:                                                     # close the parallelogram
        spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, name1=f"coupler_{p}", name2=f"shin_{p}",
                          objtype=mujoco.mjtObj.mjOBJ_BODY,
                          data=list(frames[p]["lever_on_coupler"]) + list(frames[p]["lever_on_shin"]) + [0] * 5,
                          solref=[0.004, 1.0])
        for name, target in ((f"{p}_mC", f"{p}_abad"), (f"{p}_mB", f"{p}_hip"), (f"{p}_mA", f"{p}_crank")):
            ttype = mujoco.mjtTrn.mjTRN_JOINT
            a = spec.add_actuator(name=name, target=target, trntype=ttype)
            a.ctrllimited, a.ctrlrange, a.gear[0] = mujoco.mjtLimited.mjLIMITED_TRUE, [-TAU, TAU], 1.0
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    return spec.compile()


if __name__ == "__main__":
    import time
    m = build()
    d = mujoco.MjData(m)
    print(f"nbody {m.nbody} njnt {m.njnt} ngeom {m.ngeom} nmesh {m.nmesh} neq {m.neq} nu {m.nu} "
          f"mass {m.body_subtreemass[1]:.2f} kg")
    t0 = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    print(f"2000 steps (4 s, motors off): {time.perf_counter() - t0:.2f} s wall, torso z {d.qpos[2]:.3f}")
