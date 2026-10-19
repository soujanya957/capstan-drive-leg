# Wheeled Biped Robot - Complete Wiring Diagram

## System Overview

```
┌────────────────────────────────────────────────────────────┐
│                       POWER FLOW                           │
│                                                            │
│  Battery (7.4V 2S) → Switch → Distribution                │
│                                ├→ L293D Shield (Motors)    │
│                                └→ Buck Converter (ESP32)   │
└────────────────────────────────────────────────────────────┘
```

---

## 1. Power Distribution System

### Main Power Wiring

```
[7.4V 2S LiPo Battery]
        │
        │ (XT60 connector)
        │
    [Power Switch] ────── OFF/ON
        │
        ├─────────────────────┬─────────────────────┐
        │                     │                     │
        │                     │                     │
    VIN (+7.4V)          VIN (+7.4V)           GND (0V)
        │                     │                     │
  [L293D Shield]      [Buck Converter]     [Common Ground]
        │                     │                     │
    Motors           → 5V Output              All components
                            │                  connected here
                      [ESP32 VIN]
                            │
                      ESP32 Powers:
                      - MPU6050 (3.3V)
                      - Logic circuits
```

### Detailed Power Connections

#### Battery to Switch:
```
Battery XT60 (+) ──────► Switch IN
Battery XT60 (-) ──────► Common Ground Bus
```

#### Switch to Devices:
```
Switch OUT ──┬──► L293D Shield VIN (screw terminal or power jack)
             │
             └──► Buck Converter VIN+ (screw terminal)

Common GND ──┬──► L293D Shield GND
             ├──► Buck Converter VIN-
             ├──► ESP32 GND
             └──► Capacitor (-)
```

#### Buck Converter to ESP32:
```
Buck OUT+ (5V) ──┬──► ESP32 VIN pin
                 │
                 └──► Capacitor (+) [1000µF, 16V]

Buck OUT- ──────────► Common Ground Bus
```

### Power Specifications

| Rail | Voltage | Max Current | Consumers |
|------|---------|-------------|-----------|
| Battery | 7.4V (6.0-8.4V) | 25C × 2.5Ah = 62.5A | Everything |
| Motor Rail | 7.4V | ~2.0A typical | 4× DC motors, 2× servos |
| Logic Rail | 5.0V | ~0.5A | ESP32, MPU6050 |

---

## 2. ESP32 to L293D Shield Connections

### Pin Mapping Table

| Function | ESP32 Pin | L293D Shield Pin | Notes |
|----------|-----------|------------------|-------|
| **Wheel Motor 1 (Left)** | | | |
| Direction | GPIO 27 | M1 DIR / IN1 | HIGH = forward |
| PWM Speed | GPIO 14 | M1 PWM / EN1 | 0-255 duty cycle |
| Encoder A | GPIO 34 | - | Connect to motor encoder |
| Encoder B | GPIO 35 | - | Connect to motor encoder |
| **Wheel Motor 2 (Right)** | | | |
| Direction | GPIO 26 | M2 DIR / IN2 | HIGH = forward |
| PWM Speed | GPIO 12 | M2 PWM / EN2 | 0-255 duty cycle |
| Encoder A | GPIO 32 | - | Connect to motor encoder |
| Encoder B | GPIO 33 | - | Connect to motor encoder |
| **Hip Motor 3 (Left)** | | | |
| Direction | GPIO 25 | M3 DIR / IN3 | HIGH = leg forward |
| PWM Speed | GPIO 13 | M3 PWM / EN3 | 0-255 duty cycle |
| Encoder A | GPIO 36 | - | Connect to motor encoder |
| Encoder B | GPIO 39 | - | Connect to motor encoder |
| **Hip Motor 4 (Right)** | | | |
| Direction | GPIO 33 | M4 DIR / IN4 | HIGH = leg forward |
| PWM Speed | GPIO 15 | M4 PWM / EN4 | 0-255 duty cycle |
| Encoder A | GPIO 4 | - | Connect to motor encoder |
| Encoder B | GPIO 16 | - | Connect to motor encoder |
| **Servo Connections** | | | |
| Knee Servo 1 (Left) | GPIO 18 | SERVO1 pin | PWM signal |
| Knee Servo 2 (Right) | GPIO 19 | SERVO2 pin | PWM signal |
| **Power** | | | |
| Logic Power | 5V pin | 5V (if needed) | Power shield logic |
| Common Ground | GND | GND | Critical! |

### Important Notes:
1. **L293D Shield Compatibility:** If shield is designed for Arduino, pin labels may differ
2. **Encoder Pins:** Must use interrupt-capable GPIO pins (34-39 are input-only)
3. **PWM Frequency:** Use 1kHz-20kHz for motor PWM
4. **Servo PWM:** 50Hz with 1000-2000µs pulse width

---

## 3. Motor Encoder Connections

### DC Motor with Encoder Pinout (Typical)

```
Motor Connector (6-pin):
┌─────────────────┐
│ 1. Motor +      │───► L293D Shield Motor Terminal A
│ 2. Motor -      │───► L293D Shield Motor Terminal B
│ 3. Encoder VCC  │───► ESP32 3.3V output
│ 4. Encoder GND  │───► Common Ground
│ 5. Encoder A    │───► ESP32 GPIO (interrupt pin)
│ 6. Encoder B    │───► ESP32 GPIO (interrupt pin)
└─────────────────┘
```

### Encoder Signal Connections

#### Left Wheel Motor (M1):
```
M1 Encoder VCC ──► ESP32 3.3V
M1 Encoder GND ──► Common Ground
M1 Encoder A   ──► GPIO 34 (input only, ADC1_CH6)
M1 Encoder B   ──► GPIO 35 (input only, ADC1_CH7)
```

#### Right Wheel Motor (M2):
```
M2 Encoder VCC ──► ESP32 3.3V
M2 Encoder GND ──► Common Ground
M2 Encoder A   ──► GPIO 32 (touch pin, ADC1_CH4)
M2 Encoder B   ──► GPIO 33 (touch pin, ADC1_CH5)
```

#### Left Hip Motor (M3):
```
M3 Encoder VCC ──► ESP32 3.3V
M3 Encoder GND ──► Common Ground
M3 Encoder A   ──► GPIO 36 (input only, ADC1_CH0, SENSOR_VP)
M3 Encoder B   ──► GPIO 39 (input only, ADC1_CH3, SENSOR_VN)
```

#### Right Hip Motor (M4):
```
M4 Encoder VCC ──► ESP32 3.3V
M4 Encoder GND ──► Common Ground
M4 Encoder A   ──► GPIO 4  (ADC2_CH0)
M4 Encoder B   ──► GPIO 16 (UART2 RX - be careful)
```

### Encoder Pull-up Resistors

Most encoder outputs are open-collector. Add 10kΩ pull-ups if needed:

```
     3.3V
      │
     [10kΩ]
      │
      ├────► ESP32 GPIO pin
      │
Encoder Output
```

**Note:** ESP32 has internal pull-ups (can enable in code), but external resistors provide more reliable signals.

---

## 4. IMU (MPU6050) Connections

### I2C Bus Wiring

```
[MPU6050 Module]          [ESP32]
┌──────────────┐         ┌──────────┐
│ VCC          │────────►│ 3.3V     │
│ GND          │────────►│ GND      │
│ SCL          │────────►│ GPIO 22  │ (I2C Clock)
│ SDA          │────────►│ GPIO 21  │ (I2C Data)
│ XDA          │         │          │ (not used)
│ XCL          │         │          │ (not used)
│ AD0          │────────►│ GND      │ (I2C address: 0x68)
│ INT          │────────►│ GPIO 23  │ (optional interrupt)
└──────────────┘         └──────────┘
```

### I2C Pull-up Resistors

Many MPU6050 breakouts have built-in pull-ups. If not, add:

```
    3.3V              3.3V
     │                 │
   [4.7kΩ]          [4.7kΩ]
     │                 │
    SCL────────────►GPIO 22
    
    SDA────────────►GPIO 21
```

### MPU6050 Specifications
- **I2C Address:** 0x68 (AD0=LOW) or 0x69 (AD0=HIGH)
- **I2C Speed:** 100kHz (standard) or 400kHz (fast mode)
- **Interrupt Pin:** Optional, for DMP motion detection
- **Power:** 3.3V only (NOT 5V tolerant)

---

## 5. Servo Motor Connections

### Servo Pinout (Standard)

```
[Servo Connector]
┌──────────────┐
│ Brown/Black  │───► GND
│ Red          │───► VCC (4.8-7.2V, from L293D servo rail)
│ Yellow/White │───► Signal (PWM from ESP32)
└──────────────┘
```

### Wiring Details

#### Left Knee Servo (Servo 1):
```
Servo 1 Signal ──► ESP32 GPIO 18 (PWM capable)
Servo 1 VCC    ──► L293D Shield SERVO VCC (typically 5-6V from shield)
Servo 1 GND    ──► Common Ground
```

#### Right Knee Servo (Servo 2):
```
Servo 2 Signal ──► ESP32 GPIO 19 (PWM capable)
Servo 2 VCC    ──► L293D Shield SERVO VCC
Servo 2 GND    ──► Common Ground
```

### Servo Power Notes
- Most L293D shields provide regulated 5-6V servo power
- If shield doesn't regulate servo power, add 5V BEC (Battery Eliminator Circuit)
- Peak current: ~1-2A per servo under load
- Add 470µF capacitor near servo power connection to absorb current spikes

```
Servo Rail (+) ───┬─── Servo 1 VCC
                  ├─── Servo 2 VCC
                  │
                 [470µF, 16V]
                  │
Common GND ────────┴─── All servo grounds
```

---

## 6. Complete Wiring Schematic (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BATTERY PACK (7.4V 2S)                      │
│                              [XT60 Male]                            │
└───────────────────────┬──────────────────┬──────────────────────────┘
                        │(+)               │(-)
                        │                  │
                   [ON/OFF Switch]         │
                        │                  │
                        │                  └────────┐
                        │                           │
      ┌─────────────────┴────────────────┐          │
      │                                  │          │
      ▼ VIN                              ▼ VIN      ▼ GND
┌─────────────────┐              ┌──────────────┐  │
│  L293D SHIELD   │              │BUCK CONVERTER│  │
│                 │              │  LM2596      │  │
│  Motor Outputs: │              │              │  │
│  M1: Wheel L    │              │ OUT: 5V      │  │
│  M2: Wheel R    │              └──────┬───────┘  │
│  M3: Hip L      │                     │          │
│  M4: Hip R      │                     ▼ 5V       │
│                 │              ┌──────────────┐  │
│  Servo Outputs: │              │   ESP32      │  │
│  S1: Knee L     │              │   DevKit     │  │
│  S2: Knee R     │              │              │  │
│                 │              │  GPIO Pins:  │  │
│  Control Pins:  │◄─────────────┤  27,26,25,33 │  │ (Motor DIR)
│  M1_DIR  ◄GPIO27│              │  14,12,13,15 │  │ (Motor PWM)
│  M1_PWM  ◄GPIO14│              │  18,19       │  │ (Servo PWM)
│  M2_DIR  ◄GPIO26│              │              │  │
│  M2_PWM  ◄GPIO12│              │  I2C Pins:   │  │
│  M3_DIR  ◄GPIO25│              │  21 (SDA)────┼──┼─► MPU6050 SDA
│  M3_PWM  ◄GPIO13│              │  22 (SCL)────┼──┼─► MPU6050 SCL
│  M4_DIR  ◄GPIO33│              │              │  │
│  M4_PWM  ◄GPIO15│              │  Encoder In: │  │
│  S1_SIG  ◄GPIO18│              │  34,35,32,33 │  │ (M1,M2 enc A,B)
│  S2_SIG  ◄GPIO19│              │  36,39,4,16  │  │ (M3,M4 enc A,B)
│                 │              │              │  │
│  GND ───────────┼──────────────┤ GND          │  │
└─────────────────┘              └──────┬───────┘  │
      │                                 │          │
      │                          ┌──────┴──────┐   │
      │                          │  MPU6050    │   │
      │                          │  IMU Module │   │
      │                          │             │   │
      │                          │ VCC ◄ 3.3V  │   │
      │                          │ GND ────────┼───┘
      │                          │ SCL ◄ GPIO22│
      │                          │ SDA ◄ GPIO21│
      │                          └─────────────┘
      │
      └─────────► [Common Ground Bus - All Components]

[1000µF Capacitor across ESP32 5V and GND]
[Heat Sinks on L293D Shield ICs]
```

---

## 7. Physical Layout Recommendations

### Component Placement (Top View)

```
┌──────────────────────────────────────┐
│           BODY CHASSIS               │
│                                      │
│  [Battery Pack]     [ESP32]          │
│   (bottom layer)    (mounted)        │
│                                      │
│  [Buck Converter]   [L293D Shield]   │
│   (near battery)    (stacked or side)│
│                                      │
│       [MPU6050]     [Switch]         │
│       (centered)    (accessible)     │
│                                      │
└───────┬──────────────────┬───────────┘
        │                  │
   [Left Leg]         [Right Leg]
```

### Wire Management

**Recommended Wire Colors:**
- **Red:** Power (+7.4V, +5V, +3.3V)
- **Black:** Ground (all ground connections)
- **Yellow:** Motor PWM signals
- **Green:** Motor direction signals
- **Blue:** I2C (SDA/SCL)
- **White:** Servo signals
- **Orange:** Encoder signals

**Cable Routing:**
1. **Keep power wires short and thick** (16-18 AWG for battery)
2. **Separate signal wires from power wires** (reduce EMI)
3. **Twist encoder wires together** (reduce noise pickup)
4. **Use cable ties** to secure wire bundles
5. **Label all connections** for troubleshooting

---

## 8. Grounding Strategy

### Critical Grounding Rules

```
               [Battery Ground]
                      │
                      │ (Heavy gauge wire)
                      │
              ┌───────┴───────┐
              │               │
         [Power GND]    [Signal GND]
              │               │
      ┌───────┴───────┐   ┌───┴────────┐
      │               │   │            │
  L293D GND      Buck GND ESP32 GND  Sensor GND
```

**Star Grounding Approach:**
- All grounds eventually connect to battery negative terminal
- Avoid ground loops
- Keep signal ground separate from motor ground paths
- Use thick wire for motor ground returns

### Ground Loop Prevention

**Wrong (Creates Loops):**
```
Battery(-) → Shield GND → ESP32 GND → MPU6050 GND → Battery(-)
                              ↓
                        Buck Converter GND
```

**Correct (Star Pattern):**
```
              Battery(-)
                  │
       ┌──────────┼──────────┐
       │          │          │
  Shield GND  Buck GND  ESP32 GND
                            │
                       MPU6050 GND
```

---

## 9. Testing Procedures

### Power-Up Sequence (Safe Testing)

#### Step 1: Power Supply Testing (Before Battery)
```
[Lab Power Supply] → Set to 7.4V, Current Limit 2A
         ↓
Connect to VIN points (without battery)
         ↓
Verify all voltages:
- L293D VIN: 7.4V ✓
- Buck output: 5.0V ✓
- ESP32 VIN: 5.0V ✓
- ESP32 3.3V pin: 3.3V ✓
```

#### Step 2: Component Testing (Individual)
1. **ESP32 alone:** Upload blink sketch, verify operation
2. **MPU6050:** Run I2C scanner, verify 0x68 address detected
3. **Motors (one at a time):** Test with LOW speed, verify direction
4. **Encoders:** Monitor pulses with serial output
5. **Servos:** Sweep test, verify smooth motion

#### Step 3: Integration Testing
```
Power on with battery → Measure all voltages
                     → Upload test sketch
                     → Test each subsystem
                     → Monitor current draw
```

### Multimeter Checkpoints

| Test Point | Expected Voltage | Notes |
|------------|------------------|-------|
| Battery terminals | 7.4V (6.0-8.4V) | Fully charged: 8.4V |
| After switch | 7.4V | Same as battery |
| L293D VIN | 7.4V | Motor power rail |
| Buck converter output | 5.0V ±0.1V | Adjust pot if needed |
| ESP32 VIN pin | 5.0V | From buck converter |
| ESP32 3.3V pin | 3.3V | Internal regulator |
| MPU6050 VCC | 3.3V | From ESP32 |

### Current Draw Testing

| Condition | Expected Current | Max Acceptable |
|-----------|------------------|----------------|
| Idle (no motors) | 150-250mA | 500mA |
| One motor running | 600-800mA | 1.2A |
| All motors running | 2.0-2.5A | 4.0A |
| Peak (hopping) | 3.5-4.5A | 6.0A |

**If current exceeds max acceptable:**
- Check for short circuits
- Verify motor voltage rating (should be 6V)
- Check for binding in mechanical parts

---

## 10. Safety & Protection

### Overcurrent Protection

**L293D Built-in Protection:**
- Thermal shutdown at ~150°C
- Current limiting per channel
- **Add heat sinks to prevent shutdown!**

**Battery Protection:**
- **NEVER discharge below 6.0V total (3.0V per cell)**
- Monitor voltage during operation
- Set ESP32 to alert/shutdown at 6.2V

```cpp
// Battery monitoring code
float batteryVoltage = analogRead(BATTERY_PIN) * (7.4 / 4095.0) * 2.0;
if (batteryVoltage < 6.2) {
    // Stop all motors
    // Flash LED warning
    // Shutdown after 30 seconds
}
```

### Reverse Polarity Protection

**XT60 Connectors:** Physically prevent reverse connection (keyed)

**Additional Protection (Optional):**
```
Battery (+) ──►|──┐    (Schottky diode)
              │  │
              │  └─► System VIN
              │
          [10kΩ]     (bleeder resistor)
              │
Battery (-) ──┴────► System GND
```

### Inductive Spike Protection

**L293D Shield:** Includes flyback diodes (internal)

**Additional Protection (if needed):**
- TVS diodes across motor terminals
- Snubber capacitors (0.1µF ceramic) across motor leads

```
Motor Terminal A ──┬──[Motor]──┬── Motor Terminal B
                   │           │
                 [0.1µF]     [0.1µF]
                   │           │
                   └─── GND ───┘
```

---

## 11. Troubleshooting Guide

### Power Issues

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| ESP32 won't power on | Buck not outputting 5V | Check buck input, adjust pot |
| Motors won't run | No power to L293D | Check battery, switch, connections |
| ESP32 resets randomly | Voltage sag from motors | Add/upgrade capacitor (1000µF+) |
| Battery drains quickly | Short circuit | Check all connections with multimeter |

### Motor Issues

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Motor doesn't spin | Wrong wiring | Swap motor +/- wires |
| Motor spins wrong direction | Direction pin inverted | Flip DIR signal in code |
| Motor runs intermittently | Poor connection | Check screw terminals, resolder |
| Motor overheats | Stalled/blocked | Check mechanical assembly |

### Encoder Issues

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| No encoder pulses | No power to encoder | Check encoder VCC/GND |
| Noisy encoder readings | EMI interference | Twist encoder wires, add capacitor |
| Encoder counts drift | Pullup resistors missing | Enable internal pullups or add 10kΩ |
| Wrong count direction | Channels swapped | Swap A/B in code or wiring |

### IMU Issues

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| I2C device not found | Wrong wiring | Check SDA/SCL connections |
| Incorrect readings | Not calibrated | Run calibration routine |
| Noisy data | Power supply noise | Add 0.1µF capacitor near IMU VCC |
| I2C timeout | Wrong pullups | Check 4.7kΩ pullups on SDA/SCL |

### Servo Issues

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Servo jitters | Insufficient power | Check servo power rail voltage |
| Servo doesn't move | Wrong signal | Verify PWM frequency (50Hz) |
| Servo overheats | Stalled against stop | Check mechanical range |
| Erratic movement | Voltage drops | Add 470µF capacitor near servo power |

---

## 12. Advanced: EMI/Noise Reduction

### Electromagnetic Interference (EMI) Sources
1. **DC motor brushes** (primary noise source)
2. **PWM switching** (high-frequency noise)
3. **Encoder pulses** (fast edges)
4. **Power supply switching** (buck converter)

### Noise Mitigation Techniques

#### Motor Noise Suppression
```
      [Motor]
        │ │
    [0.1µF] [0.1µF]  ← Ceramic capacitors
        │ │
    ────┴─┴────      ← Motor case/GND
```

**Placement:** Solder directly across motor terminals

#### I2C Bus Protection
```
ESP32 ──[100Ω]──┬── MPU6050 SDA
                │
              [4.7kΩ]
                │
              3.3V

ESP32 ──[100Ω]──┬── MPU6050 SCL
                │
              [4.7kΩ]
                │
              3.3V
```

**100Ω series resistors:** Dampen high-frequency ringing

#### Power Supply Filtering
```
Buck Output (+5V) ──┬──[10µF tantalum]──┬── ESP32 VIN
                    │                    │
                    └──[0.1µF ceramic]───┘
                                         │
                                        GND
```

**Capacitor selection:**
- Bulk: 1000µF electrolytic (low frequency)
- Decoupling: 10µF tantalum (medium frequency)
- Bypass: 0.1µF ceramic (high frequency)

---

## 13. PCB Layout Recommendations (If Making Custom Board)

### Layer Stack (2-layer board)

**Top Layer:**
- Signal traces (encoder, servo, I2C)
- Component pads
- Minimal power routing

**Bottom Layer:**
- Solid ground plane (90% coverage)
- Power distribution (thick traces)
- Return paths for all signals

### Trace Width Guidelines

| Current | Trace Width (1oz copper) |
|---------|--------------------------|
| 0.1A | 0.25mm (10 mil) |
| 0.5A | 0.50mm (20 mil) |
| 1.0A | 1.00mm (40 mil) |
| 2.0A | 2.00mm (80 mil) |
| 5.0A+ | 4.00mm+ (160 mil+) or multiple layers |

### Critical Design Rules

1. **Keep motor power traces away from signal traces** (>5mm separation)
2. **Use ground plane as shield** between noisy and sensitive circuits
3. **Star ground at battery connection**
4. **Place decoupling caps close to IC power pins** (<5mm)
5. **Keep I2C traces short and equal length**

---

## 14. Maintenance Schedule

### Daily (During Development)
- [ ] Check all screw terminals for tightness
- [ ] Verify battery voltage before use
- [ ] Inspect wires for wear/damage

### Weekly
- [ ] Clean encoder optical sensors (if optical)
- [ ] Check heat sink thermal contact
- [ ] Re-balance charge battery

### Monthly
- [ ] Check bearing smoothness
- [ ] Inspect solder joints
- [ ] Test battery capacity (charge/discharge cycle)
- [ ] Re-calibrate IMU if needed

### Signs of Problems

**Immediate attention needed:**
- Burning smell → POWER OFF IMMEDIATELY
- Smoke → POWER OFF, disconnect battery
- Hot components → Allow cooling, check current draw
- Loose connections → Re-secure before next use

---

## 15. Wiring Checklist (Before First Power-On)

### Visual Inspection
- [ ] All polarized connectors correct orientation (XT60, capacitors)
- [ ] No exposed wire strands (potential shorts)
- [ ] Heat sinks properly mounted with thermal compound
- [ ] All screw terminals tight
- [ ] No wires touching moving parts

### Continuity Testing (Multimeter - Beep Mode)
- [ ] Battery (+) to Switch IN
- [ ] Switch OUT to L293D VIN
- [ ] Switch OUT to Buck VIN
- [ ] Buck OUT to ESP32 VIN
- [ ] All grounds connected to common ground
- [ ] NO continuity between (+) and (-) rails (no shorts!)

### Resistance Testing
- [ ] Battery (+) to Battery (-): Should be >10kΩ (no short)
- [ ] Motor terminals: Should be 5-20Ω (motor coil resistance)
- [ ] Servo signal to GND: Should be >100kΩ (not shorted)

### Voltage Testing (Battery Connected, Switch OFF)
- [ ] Battery voltage: 7.4V ±0.5V
- [ ] No voltage on any other rails (switch working)

### Voltage Testing (Switch ON, No Load)
- [ ] L293D VIN: 7.4V
- [ ] Buck output: 5.0V ±0.1V
- [ ] ESP32 VIN: 5.0V
- [ ] ESP32 3.3V pin: 3.3V
- [ ] MPU6050 VCC: 3.3V

---

## 16. Bill of Materials - Wire & Connectors

| Item | Specification | Quantity | Purpose |
|------|---------------|----------|---------|
| Silicone wire 16AWG | Red/Black pair | 1m each | Battery to switch/distribution |
| Silicone wire 22AWG | Various colors | 5m | Motor/servo connections |
| Dupont connectors | 2.54mm pitch | 50 pcs | Encoder/sensor connections |
| Heat shrink tubing | 3mm, 5mm, 10mm | Assorted | Insulate connections |
| XT60 connectors | Male/female | 1 pair | Battery connection |
| JST connectors | 2-pin | 4 sets | Motor power (optional) |
| Servo connectors | 3-pin female | 2 pcs | Servo signal/power |
| Cable ties | 100mm, 200mm | 20 pcs | Wire management |

---

## Diagram Legend

### Symbols Used
```
───►  : Wire connection, signal flow
  │   : Vertical connection
  ├   : Branch point
  └   : Terminal connection
 [X]  : Component
◄───  : Input
───►  : Output
 ✓    : Verified/OK
 ✗    : Error/Problem
```

---

## Additional Resources

### Recommended Tools for Assembly
- Soldering iron (temperature controlled, 350°C)
- Solder (60/40 or lead-free)
- Wire strippers (20-16 AWG)
- Crimping tool (for Dupont connectors)
- Heat gun (for heat shrink)
- Multimeter (voltage, continuity, current)
- Logic analyzer (optional, for encoder debugging)

### Software for Testing
- Arduino IDE or PlatformIO
- I2C scanner sketch (MPU6050 detection)
- Motor test sketches (individual motor control)
- Encoder test code (pulse counting verification)

---

**End of Circuit Diagram Documentation**

For control system details, see `control_system.md`
For complete parts list, see `part_list.md`
