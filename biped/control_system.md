# Wheeled Biped Robot - Control System Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Balance Control](#balance-control)
3. [Locomotion Control](#locomotion-control)
4. [State Machine](#state-machine)
5. [Sensor Fusion](#sensor-fusion)
6. [Motor Control](#motor-control)
7. [Implementation Details](#implementation-details)

---

## 1. System Architecture

### Control Hierarchy

```
┌────────────────────────────────────────────────────────┐
│              HIGH-LEVEL PLANNER                        │
│  (User commands, behavior selection, safety)           │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│           BEHAVIOR COORDINATOR                         │
│  (Walking, hopping, balancing state machine)           │
└─────────────────┬──────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
┌──────────────┐    ┌─────────────────┐
│  BALANCE     │    │   GAIT          │
│  CONTROLLER  │    │   GENERATOR     │
│  (200 Hz)    │    │   (50 Hz)       │
└──────┬───────┘    └────────┬────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
┌────────────────────────────────────────────────────────┐
│           MOTOR COMMAND ARBITRATOR                     │
│  (Combines balance + gait, applies limits)             │
└─────────────────┬──────────────────────────────────────┘
                  │
        ┌─────────┼──────────┬────────────┐
        ▼         ▼          ▼            ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ WHEEL  │ │ WHEEL  │ │  HIP   │ │  HIP   │
   │ LEFT   │ │ RIGHT  │ │ LEFT   │ │ RIGHT  │
   │ PID    │ │  PID   │ │  PID   │ │  PID   │
   └────────┘ └────────┘ └────────┘ └────────┘
        │         │          │            │
        ▼         ▼          ▼            ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │MOTOR 1 │ │MOTOR 2 │ │MOTOR 3 │ │MOTOR 4 │
   └────────┘ └────────┘ └────────┘ └────────┘
```

### Dual-Core Task Distribution (ESP32)

**Core 0 - Real-Time Control (200 Hz):**
```cpp
Task 1: IMU Reading & Filtering        (200 Hz)
Task 2: Balance Control Loop           (200 Hz)
Task 3: Wheel Speed Control             (200 Hz)
Task 4: Encoder Processing              (Interrupt-driven)
```

**Core 1 - High-Level Control (50 Hz):**
```cpp
Task 5: State Machine Update            (50 Hz)
Task 6: Gait Generation                 (50 Hz)
Task 7: Hip/Knee Position Control       (50 Hz)
Task 8: Bluetooth Communication         (As needed)
Task 9: Safety Monitoring               (10 Hz)
```

---

## 2. Balance Control

### Inverted Pendulum Model

The robot is modeled as an inverted pendulum on wheels:

```
         θ (tilt angle)
          ↑
          │
    ┌─────┴─────┐
    │   BODY    │  m = mass, h = height
    │           │  I = moment of inertia
    └─────┬─────┘
          │
          │ L (length to CoM)
          │
       ───┴───      x (wheel position)
         ◯ ◯  ←────
```

**Dynamics:**
```
θ̈ = (g/L) * sin(θ) - (a/L) * cos(θ)

where:
  θ̈  = angular acceleration
  g  = gravity (9.81 m/s²)
  L  = height to center of mass
  a  = wheel acceleration
```

**Linearized (small angles):**
```
θ̈ ≈ (g/L) * θ - (a/L)
```

### PID Balance Controller

#### Control Law
```
wheel_velocity = Kp * θ + Kd * θ̇ + Ki * ∫θdt + Kv * ẋ

where:
  θ   = tilt angle (degrees)
  θ̇   = tilt rate (degrees/sec)
  ∫θ  = accumulated tilt error
  ẋ   = wheel velocity (m/s)
```

#### Gain Tuning Guidelines

| Gain | Typical Value | Effect | Too High | Too Low |
|------|---------------|--------|----------|---------|
| Kp | 40-80 | Proportional response | Oscillation | Slow correction |
| Kd | 0.5-2.0 | Damping | Sluggish | Overshoot |
| Ki | 0.1-1.0 | Steady-state error | Instability | Drift |
| Kv | 0.05-0.2 | Velocity damping | Can't move | Drift when moving |

#### Tuning Procedure

**Step 1: Start Conservative**
```cpp
Kp = 40.0;
Kd = 0.5;
Ki = 0.0;  // Start with zero
Kv = 0.0;  // Start with zero
```

**Step 2: Increase Kp**
- Slowly increase until robot oscillates
- Back off 20-30%
- Robot should quickly return to vertical

**Step 3: Add Kd**
- Increase to dampen oscillations
- Should see smooth return to vertical
- Too much → sluggish response

**Step 4: Add Ki** (optional)
- Only if robot drifts over time
- Start very small (0.1)
- Increase until drift eliminated

**Step 5: Add Kv**
- Helps when robot is moving
- Prevents runaway acceleration
- Start at 0.05, adjust as needed

### Implementation (Pseudocode)

```cpp
// Balance control loop (200 Hz)
void balanceControl() {
    // Read IMU
    float tilt_angle = imu.getAngle();
    float tilt_rate = imu.getGyroZ();
    
    // Read wheel encoders
    float wheel_velocity = (left_encoder.velocity + right_encoder.velocity) / 2.0;
    
    // Integral term (with anti-windup)
    integral_error += tilt_angle * dt;
    integral_error = constrain(integral_error, -max_integral, max_integral);
    
    // PID calculation
    float correction = Kp * tilt_angle 
                     + Kd * tilt_rate 
                     + Ki * integral_error
                     + Kv * wheel_velocity;
    
    // Convert to motor commands
    left_wheel_speed = correction + turn_command;
    right_wheel_speed = correction - turn_command;
    
    // Apply limits
    left_wheel_speed = constrain(left_wheel_speed, -max_speed, max_speed);
    right_wheel_speed = constrain(right_wheel_speed, -max_speed, max_speed);
    
    // Send to motors
    setWheelSpeed(LEFT, left_wheel_speed);
    setWheelSpeed(RIGHT, right_wheel_speed);
}
```

### Anti-Windup Strategies

**Integral Clamping:**
```cpp
if (abs(integral_error) > MAX_INTEGRAL) {
    integral_error = MAX_INTEGRAL * sign(integral_error);
}
```

**Conditional Integration:**
```cpp
// Only integrate when not saturated
if (abs(motor_command) < motor_limit) {
    integral_error += error * dt;
}
```

---

## 3. Locomotion Control

### Gait Generation

#### Standing (Balanced)
```
State: STANDING
├─ Both legs straight (knee angle = 0°)
├─ Hips neutral (hip angle = 0°)
├─ Balance control ACTIVE
└─ Wheel speed determined by balance only
```

#### Walking Gait

**Cycle Duration:** 2.0 seconds per full cycle (1.0s per step)

**Phase Sequence:**
```
Phase 1: LEFT SWING (0.0 - 0.5s)
├─ Left hip: Swing forward (0° → 20°)
├─ Left knee: Lift foot (0° → 15°)
├─ Right leg: Support (straight, 0°)
└─ Balance: Compensate for weight shift

Phase 2: LEFT PLANT (0.5 - 1.0s)
├─ Left hip: Complete swing (20° → 25°)
├─ Left knee: Lower foot (15° → 0°)
├─ Weight shift: Left → Center
└─ Balance: Active throughout

Phase 3: RIGHT SWING (1.0 - 1.5s)
├─ Right hip: Swing forward (0° → 20°)
├─ Right knee: Lift foot (0° → 15°)
├─ Left leg: Support (straight, 0°)
└─ Balance: Compensate for weight shift

Phase 4: RIGHT PLANT (1.5 - 2.0s)
├─ Right hip: Complete swing (20° → 25°)
├─ Right knee: Lower foot (15° → 0°)
├─ Weight shift: Right → Center
└─ Balance: Active throughout
```

**Trajectory Equations:**

```cpp
// Smooth sinusoidal motion
float hip_angle(float t, float cycle_time) {
    float phase = (t / cycle_time) * 2 * PI;
    return HIP_AMPLITUDE * sin(phase);  // ±20°
}

float knee_angle(float t, float cycle_time, bool swing_leg) {
    if (!swing_leg) return 0.0;  // Support leg stays straight
    
    float phase = (t / cycle_time) * 2 * PI;
    // Only bend during swing phase
    if (sin(phase) > 0) {
        return KNEE_LIFT_ANGLE * sin(phase);  // 0-15°
    }
    return 0.0;
}
```

#### Hopping Sequence

**Total Duration:** 1.5 seconds

```
Phase 1: CROUCH (0.0 - 0.4s)
├─ Both hips: Bring legs slightly under body (0° → 10°)
├─ Both knees: Deep bend (0° → 90°)
├─ Wheel speed: Slow forward drift
└─ Balance: Extra active (lower CoM)

Phase 2: LAUNCH (0.4 - 0.5s) [100ms - EXPLOSIVE]
├─ Both hips: Push down (10° → -15°)
├─ Both knees: Extend rapidly (90° → 0°)
├─ Wheel speed: Maximum acceleration forward
└─ All 6 motors: Coordinated maximum output!

Phase 3: AIRBORNE (0.5 - 0.8s) [~300ms]
├─ IMU detects: Acceleration > 1.2g (liftoff)
├─ Wheel motors: Reduce power (no ground contact)
├─ Hip adjustment: Pitch control in air
├─ Knees: Pre-bend for landing (0° → 30°)
└─ Balance: Predictive (preparing for impact)

Phase 4: LANDING (0.8 - 0.9s) [100ms - CRITICAL]
├─ IMU detects: Rapid deceleration (impact)
├─ Knees: Flex to absorb shock (30° → 60°)
├─ Hips: Adjust for balance recovery
├─ Wheel speed: Aggressive balance correction
└─ Balance gains: Temporarily increased (Kp × 1.5)

Phase 5: RECOVERY (0.9 - 1.5s)
├─ Knees: Gradually extend (60° → 0°)
├─ Hips: Return to neutral (any angle → 0°)
├─ Balance: Return to normal gains
└─ Stabilize: Resume standing state
```

**Critical Timing:**
- **Launch must be fast (<100ms)** - slow extension won't generate enough force
- **Landing detection must be immediate** - delay causes fall
- **Recovery must be gradual** - sudden movements destabilize

---

## 4. State Machine

### State Diagram

```
                    ┌─────────────┐
                    │   STARTUP   │
                    │ (Initialize)│
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
            ┌──────►│  STANDING   │◄──────┐
            │       │  (Balance)  │       │
            │       └──────┬──────┘       │
            │              │              │
            │        User Commands        │
            │              │              │
            │       ┌──────┼──────┐       │
            │       │      │      │       │
            │       ▼      ▼      ▼       │
            │   ┌────┐ ┌────┐ ┌────┐     │
            │   │WALK│ │HOP │ │TURN│     │
            │   └──┬─┘ └──┬─┘ └──┬─┘     │
            │      │      │      │        │
            │      └──────┼──────┘        │
            │             │               │
            │      Complete/Cancel        │
            │             │               │
            └─────────────┴───────────────┘
                          │
                    Fall Detected
                          │
                          ▼
                    ┌─────────────┐
                    │   FALLEN    │
                    │  (Safe mode)│
                    └─────────────┘
```

### State Definitions

#### STARTUP
```cpp
Entry Actions:
- Initialize sensors (IMU calibration)
- Zero all encoders
- Set motors to coast mode
- Load default parameters

Exit Conditions:
- IMU calibrated AND
- Robot vertical (±10°) AND
- User confirms ready

Duration: 5-10 seconds
```

#### STANDING
```cpp
Active Controllers:
- Balance control (200 Hz)
- Wheel speed control
- Small disturbance rejection

Allowed Transitions:
- → WALKING (user command)
- → HOPPING (user command)
- → TURNING (user command)
- → FALLEN (tilt > 45°)

Notes:
- Default/idle state
- Legs remain straight
- Minimal energy consumption
```

#### WALKING
```cpp
Active Controllers:
- Balance control (200 Hz)
- Gait generator (50 Hz)
- Hip position control
- Knee position control

Gait Parameters:
- Step length: 5-10cm
- Step height: 2-3cm
- Cycle time: 2.0s

Exit Conditions:
- User stop command OR
- Target distance reached OR
- Obstacle detected OR
- Balance failure

Fallback: → STANDING
```

#### HOPPING
```cpp
Sub-States:
1. CROUCH (0.4s
