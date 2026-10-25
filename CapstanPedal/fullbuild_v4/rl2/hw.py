"""The v4 robot's sensors and timing as they will be on the hardware: what the policy can read, how often, how
late and how wrong. Training (gpu_env.py), evaluation (play.py) and teleop.py all read these numbers, so a policy
is trained on the same data format the robot will give it.

The signal chain on the robot (../BOM.md):
  Raspberry Pi 5 runs the policy at 50 Hz.
  mjbots pi3hat r4.5: CAN-FD to the 12 moteus r4.11 (one bus per leg), and its IMU with an attitude filter.
  moteus r4.11: one per motor, position/velocity from its on-board magnetic encoder on the rotor, PD loop at
    the drive's own rate (tens of kHz; the simulation runs it at the 500 Hz physics rate).
  NEW, for body velocity (no other sensor measures it): a downward-facing optical-flow sensor (PMW3901) and a
    time-of-flight range sensor (VL53L1X) under the bottom plate, on the Pi's SPI and I2C. Flow (pixels per frame)
    times height gives ground speed; this is how small drones hold position.

Numbers marked "assumed" are typical datasheet-class values, to be replaced by measurements on the build.
"""
import math

# --- timing -------------------------------------------------------------------------------------------------
POLICY_HZ = 50                       # policy on the Pi 5
PHYS_DT = 0.002                      # simulation step; the moteus PD runs much faster on the robot
DELAY_MS = (4.0, 20.0)               # assumed: sensor sample -> new targets at the motors (inference ~1 ms,
                                     # pi3hat CAN-FD round trip ~1-2 ms, Linux scheduling jitter), per episode

# --- IMU: pi3hat r4.5 on-board IMU and attitude reference ------------------------------------------------------
IMU_HZ = 1000                        # the policy reads the latest sample
GYRO_NOISE = 0.02                    # rad/s rms, assumed
GYRO_BIAS = 0.02                     # rad/s, per-axis bias drawn per episode (+-), assumed
TILT_BIAS = math.radians(1.5)        # attitude-filter tilt error per episode (+-), assumed
TILT_NOISE = math.radians(0.4)       # rad rms

# --- joint encoders: moteus r4.11 on-board magnetic encoder on the rotor (14 bit), 9:1 to the output ----------
ENC_BITS = 14
ENC_RATIO = 9.0
ENC_QUANT = 2 * math.pi / 2 ** ENC_BITS / ENC_RATIO      # rad at the output
ENC_OFFSET = 0.01                    # rad, zero-calibration error per joint per episode (+-), assumed (cable seating)
ENC_NOISE = 0.001                    # rad rms
VEL_NOISE = 0.05                     # rad/s rms on the drive's velocity estimate

# --- NEW: optical flow (PMW3901) + range (VL53L1X), facing down from the bottom plate ---------------------------
FLOW_HZ = 100                        # assumed frame rate used
FLOW_MIN_H = 0.08                    # m: the PMW3901 does not focus closer than ~80 mm
FLOW_MAX_H = 2.0
FLOW_NOISE = 0.05                    # m/s rms on the computed ground speed, assumed
FLOW_SCALE = 0.08                    # +- per episode: height and lens calibration error
FLOW_DROPOUT = (0.0, 0.3)            # chance a reading is invalid (low surface quality), drawn per episode
FLOW_DEAD = 0.15                     # share of episodes with no flow at all (dark, shiny or featureless floor, a
                                     # loose cable): a first walk policy without these fell 3 times in 4 when it failed
FLOW_LATENCY_STEPS = 1               # readings are one policy step (20 ms) old by the time the policy uses them
TOF_HZ = 50
TOF_NOISE = 0.005                    # m rms, plus 1 % of range
TOF_REL = 0.01
TOF_RANGE = (0.04, 1.3)              # short-distance mode

# --- power ----------------------------------------------------------------------------------------------------
BATTERY_SAG = (0.83, 1.0)            # 6S: 25.2 V full, ~21 V loaded and low; the motors' top speed scales with it

# --- BOM lines for the new parts (../BOM.md "Sensing") ------------------------------------------------------------
NEW_PARTS = [
    ("PMW3901 optical-flow breakout (e.g. Pimoroni PIM453)", "SPI to the Pi 5; under the bottom plate, lens down"),
    ("VL53L1X time-of-flight breakout (e.g. Pimoroni PIM373)", "I2C to the Pi 5; beside the flow sensor"),
]
