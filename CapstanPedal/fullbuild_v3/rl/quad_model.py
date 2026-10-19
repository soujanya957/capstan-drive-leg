"""MuJoCo physics model of the fullbuild_v3 quadruped (v9.1 legs), for reinforcement learning.

Geometry and masses come from the CAD (generate_quad / the v9.1 module): torso box over the body, four ab/ad
pods (the module's turning part, motors A and B in it), thighs and shins as capsules, 23 mm ball feet.
SI units (m, kg, s). The torso frame is the CAD body frame; the robot walks toward -X (knees trailing).

Drive, as built:
- ab/ad: motor C through its 9:1 capstan, torque on the pod's hinge about X.
- each leg's 4-bar: output B turns the thigh; output A's crank sets the shin's ABSOLUTE angle through the
  parallel coupler. With q_hip (thigh) and q_knee (shin relative to thigh), output A turns q_hip + q_knee:
  motor B acts on the hip joint, motor A on a fixed tendon hip + knee (generalised forces tau_A + tau_B on
  the hip, tau_A on the knee: the parallelogram's statics exactly).
- limits: each output +-90 deg (the capstan's travel; output A's as a tendon range), knee +-60 deg from rest
  (the thigh meets the shin plate past +60, the 4-bar's dead centre at -75), ab/ad +-20 deg (the body).
- torque: 3.5 N*m motor x 9 = 31.5 N*m per output, falling to zero at the no-load speed (KV 90 on 6S,
  23.3 rad/s at the output); reflected rotor inertia as joint armature. Rope compliance is not modelled.
"""
import math

import numpy as np

TAU = 31.5                      # N*m per output (v9.1 9:1)
W0 = 2000.0 / 60.0 * 2 * math.pi / 9.0       # no-load output speed, rad/s
ARMATURE = 1.9e-4 * 81.0        # 8308 rotor (~1.9e-4 kg m^2) reflected through 9:1
LEG_L = 0.200
FOOT_R = 0.023
HIP = (0.2215, 0.055, -0.0698)  # hip (output axis) of the LF pod, torso frame
LEG_Y = 0.1785                  # leg plane
AB = (0.100, -0.060)            # ab/ad axis y, z
Q0 = (-60.0, -120.0)            # thigh, shin rest angles (deg, from +X toward +Z) with the outputs centred
PODS = ("LF", "LR", "RF", "RR")
MASS = {"torso": 6.2, "pod": 2.15, "thigh": 0.14, "shin": 0.16}   # kg (estimated from the CAD; see masses())
STAND_Z = FOOT_R + 2 * LEG_L * math.sin(math.radians(60.0)) - HIP[2]  # torso origin height standing at rest


def pod_frame(p):
    sx = 1.0 if p[1] == "F" else -1.0
    sy = 1.0 if p[0] == "L" else -1.0
    return sx, sy


def mjcf(masses=None, floor_friction=1.0, terrain=""):
    m = dict(MASS, **(masses or {}))
    u = lambda d: (math.cos(math.radians(d)), math.sin(math.radians(d)))
    tx, tz = u(Q0[0])
    sxv, szv = u(Q0[1])
    knee = (LEG_L * tx, LEG_L * tz)
    foot = (LEG_L * sxv, LEG_L * szv)
    legs, tendons, acts = [], [], []
    for p in PODS:
        sx, sy = pod_frame(p)
        hx, hz = sx * HIP[0], HIP[2]
        ay, az = sy * AB[0], AB[1]
        ly = sy * LEG_Y
        pod_c = (sx * 0.20, sy * 0.10, -0.035)                       # the pod's mass centre (motors over the hip)
        legs.append(f'''
    <body name="{p}_pod" pos="0 {ay} {az}">
      <joint name="{p}_abad" type="hinge" axis="1 0 0" range="-20 20" armature="{ARMATURE}" damping="0.05"/>
      <inertial pos="{pod_c[0]} {pod_c[1] - ay} {pod_c[2] - az}" mass="{m['pod']}" diaginertia="0.012 0.010 0.008"/>
      <geom name="{p}_podbox" type="box" pos="{sx * 0.19} {sy * 0.10 - ay} {-0.03 - az}" size="0.15 0.06 0.10" rgba="0.2 0.2 0.2 0.25" contype="1" conaffinity="1"/>
      <body name="{p}_thigh" pos="{hx} {ly - ay} {hz - az}">
        <joint name="{p}_hip" type="hinge" axis="0 1 0" range="-90 90" armature="{ARMATURE}" damping="0.05"/>
        <inertial pos="{knee[0] * 0.45} 0 {knee[1] * 0.45}" mass="{m['thigh']}" diaginertia="0.0006 0.0006 0.00005"/>
        <geom name="{p}_thighg" type="capsule" fromto="0 0 0 {knee[0]} 0 {knee[1]}" size="0.014" rgba="0.1 0.1 0.1 1" contype="2" conaffinity="1"/>
        <body name="{p}_shin" pos="{knee[0]} 0 {knee[1]}">
          <joint name="{p}_knee" type="hinge" axis="0 1 0" range="-60 60" armature="{ARMATURE}" damping="0.05"/>
          <inertial pos="{foot[0] * 0.55} 0 {foot[1] * 0.55}" mass="{m['shin']}" diaginertia="0.0008 0.0008 0.00005"/>
          <geom name="{p}_shing" type="capsule" fromto="0 0 0 {foot[0] * 0.88} 0 {foot[1] * 0.88}" size="0.012" rgba="0.8 0.2 0.15 1" contype="2" conaffinity="1"/>
          <geom name="{p}_foot" type="sphere" pos="{foot[0]} 0 {foot[1]}" size="{FOOT_R}" rgba="0.3 0.3 0.3 1" friction="{floor_friction} 0.01 0.0001" contype="2" conaffinity="1" condim="3"/>
          <site name="{p}_footsite" pos="{foot[0]} 0 {foot[1]}"/>
        </body>
      </body>
    </body>''')
        tendons.append(f'''    <fixed name="{p}_outA" limited="true" range="{-math.pi / 2} {math.pi / 2}"><joint joint="{p}_hip" coef="1"/><joint joint="{p}_knee" coef="1"/></fixed>''')
        acts.append(f'''    <motor name="{p}_mC" joint="{p}_abad" gear="1" ctrllimited="true" ctrlrange="{-TAU} {TAU}"/>
    <motor name="{p}_mB" joint="{p}_hip" gear="1" ctrllimited="true" ctrlrange="{-TAU} {TAU}"/>
    <motor name="{p}_mA" tendon="{p}_outA" gear="1" ctrllimited="true" ctrlrange="{-TAU} {TAU}"/>''')
    return f'''<mujoco model="capstan_quadruped_v3">
  <compiler angle="degree" inertiafromgeom="false"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default><geom friction="{floor_friction} 0.005 0.0001"/></default>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="40 40 0.1" rgba="0.3 0.32 0.35 1" contype="1" conaffinity="3"/>
    {terrain}
    <body name="torso" pos="0 0 {STAND_Z + 0.01}">
      <freejoint name="root"/>
      <inertial pos="0 0 -0.03" mass="{m['torso']}" diaginertia="0.10 0.38 0.42"/>
      <geom name="torso_box" type="box" pos="0 0 -0.0325" size="0.345 0.13 0.11" rgba="0.15 0.15 0.15 0.4" contype="1" conaffinity="1"/>
      <site name="imu" pos="0 0 0"/>
{''.join(legs)}
    </body>
  </worldbody>
  <tendon>
{chr(10).join(tendons)}
  </tendon>
  <actuator>
{chr(10).join(acts)}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <velocimeter name="vel" site="imu"/>
    <framequat name="quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>'''


def masses(meta, tris):
    """Estimate the four masses from the quadruped's CAD meshes (as jump_estimate's density table)."""
    vol = lambda t: abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6) / 1000.0
    dens = lambda n, c: (1.27 * 0.6 if c in ("black", "red", "teal", "orange", "violet") else 1.6 if c == "cf" else
                         7.8 if c in ("steel", "silver", "screw") else 1.2 if c == "dimgray" else 3.0)
    out = {"torso": 0.0, "pod": 0.0, "thigh": 0.0, "shin": 0.0}
    for k, t in tris.items():
        v = meta[k]
        if k == "Q_Build_Stand":
            continue
        n = v.get("base", k)
        c = v["color"]
        if c in ("motor", "motor_dark", "magnet"):
            gm = 150.0                                               # two meshes per 8308 (~0.3 kg)
        elif c == "blue":
            gm = 290.0 if "Pack" in n else 0.0                       # 6S 1800 mAh
        else:
            gm = vol(t) * dens(n, c)
        if "pod" not in v:
            out["torso"] += gm
        elif v.get("body_fixed"):
            out["torso"] += gm                                       # ab/ad drive's fixed parts ride the body
        elif n.startswith(("Leg4_Thigh", "Leg4_Crank", "Leg4_Coupler", "Leg4_HW_knee", "Leg4_HW_crank")):
            out["thigh"] += gm / 4
        elif n.startswith(("Leg4_Shin", "Leg4_Foot", "Leg4_HW_shin", "Leg4_HW_foot", "Leg4_HW_lever")):
            out["shin"] += gm / 4
        else:
            out["pod"] += gm / 4
    return {k: v / 1000.0 for k, v in out.items()}
