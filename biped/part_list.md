# Wheeled Biped Robot - Complete Parts List

## Budget Summary
**Total Cost: $115**  
**Timeline: 1-2 weeks for delivery**

---

## 1. Motors & Actuators

### Wheel Motors (Qty: 2)
- **Part:** DC Gear Motor with Encoder
- **Specifications:**
  - Voltage: 6V DC
  - Speed: 200 RPM (no load)
  - Current: <500mA (under load)
  - Encoder: Quadrature, 11 PPR motor shaft
  - Gear Ratio: 1:34 or similar
  - Output Torque: 0.8-1.2 kg·cm
- **Purpose:** Drive wheels for balance and locomotion
- **Cost:** $20 ($10 each)
- **Example:** JGA25-370 6V 200RPM encoder motor
- **Notes:** Must be <500mA to stay within L293D current limits

### Hip Motors (Qty: 2)
- **Part:** DC Gear Motor with Encoder
- **Specifications:**
  - Voltage: 6V DC
  - Speed: 100 RPM (no load)
  - Current: <500mA (under load)
  - Encoder: Quadrature, 11 PPR motor shaft
  - Gear Ratio: 1:68 or similar
  - Output Torque: 1.5-2.0 kg·cm
- **Purpose:** Swing legs forward/backward for walking and hopping
- **Cost:** $18 ($9 each)
- **Example:** JGA25-370 6V 100RPM encoder motor
- **Notes:** Slower speed provides more torque for leg control

### Knee Servos (Qty: 2)
- **Part:** MG996R Digital Servo
- **Specifications:**
  - Operating Voltage: 4.8-7.2V
  - Stall Torque: 11 kg·cm @ 6V
  - Speed: 0.17 sec/60° @ 6V
  - Metal Gears
  - Weight: 55g
  - Pulse Width: 1000-2000µs
- **Purpose:** Bend/extend knees for gait and hopping
- **Cost:** $16 ($8 each)
- **Notes:** Metal gears essential for landing impact resistance

---

## 2. Electronics & Control

### ESP32 Development Board (Already Owned)
- **Part:** ESP32-WROOM-32
- **Specifications:**
  - Dual-core processor @ 240MHz
  - 520KB SRAM
  - WiFi + Bluetooth
  - 34 GPIO pins
  - 12-bit ADC
- **Purpose:** Main microcontroller for balance and control
- **Cost:** $0 (already owned)

### Motor Driver Shield (Already Owned)
- **Part:** L293D Motor Driver Shield
- **Specifications:**
  - 4× DC motor channels (0.6A continuous, 1.2A peak each)
  - 2× Servo motor connections
  - Voltage Range: 4.5V - 36V
  - Built-in flyback diodes
  - Logic Level: 5V compatible
- **Purpose:** Drive all motors from ESP32 commands
- **Cost:** $0 (already owned)
- **Critical Note:** 0.6A per channel limit requires <500mA motors

### IMU Sensor (Qty: 1)
- **Part:** MPU6050 (GY-521 breakout board)
- **Specifications:**
  - 3-axis Gyroscope: ±250 to ±2000°/s
  - 3-axis Accelerometer: ±2g to ±16g
  - 16-bit ADC
  - I2C Interface (400kHz)
  - Operating Voltage: 3.3V-5V
  - DMP (Digital Motion Processor) capable
- **Purpose:** Measure tilt angle for balance control
- **Cost:** $4
- **Notes:** MOST CRITICAL SENSOR - robot cannot balance without this

### Buck Converter (Qty: 1)
- **Part:** LM2596 DC-DC Step-Down Converter
- **Specifications:**
  - Input: 4.5-40V DC
  - Output: 1.25-37V DC (adjustable)
  - Output Current: 3A max
  - Efficiency: ~85%
  - Fixed 5V versions available
- **Purpose:** Convert 7.4V battery to 5V for ESP32
- **Cost:** $2
- **Notes:** Prevents ESP32 damage from overvoltage

---

## 3. Power System

### LiPo Battery (Qty: 1)
- **Part:** 2S LiPo Battery Pack
- **Specifications:**
  - Voltage: 7.4V nominal (8.4V fully charged)
  - Capacity: 2500mAh
  - Discharge Rate: 25C minimum (higher is better)
  - Connector: XT60 or Deans
  - Configuration: 2S (2 cells in series)
- **Purpose:** Main power source for entire robot
- **Cost:** $18
- **Runtime:** 25-30 minutes typical use
- **SAFETY:** Never discharge below 3.0V per cell (6.0V total)

### LiPo Battery Charger (Qty: 1)
- **Part:** 2S-3S Balance Charger
- **Specifications:**
  - Supports: 2S-3S LiPo batteries
  - Balance charging capability
  - Auto-cutoff at 4.2V per cell
  - LED status indicators
  - Input: AC 100-240V or DC 12V
- **Purpose:** Safely charge LiPo battery
- **Cost:** $12
- **SAFETY:** REQUIRED - improper charging can cause fires

### XT60 Connector (Qty: 1 pair)
- **Part:** XT60 Male/Female Connector Pair
- **Specifications:**
  - Current Rating: 60A continuous
  - Contact Resistance: <0.5mΩ
  - Gold-plated contacts
  - Polarized (prevents reverse connection)
- **Purpose:** Safe, secure battery connection
- **Cost:** $2
- **Notes:** Prevents short circuits and reverse polarity

### Power Switch (Qty: 1)
- **Part:** Rocker Switch (SPST)
- **Specifications:**
  - Current Rating: 10A @ 250VAC (15A @ 125VAC)
  - Mounting: Panel mount
  - Terminals: Screw or solder
  - Size: 20mm × 15mm typical
- **Purpose:** Main power on/off control
- **Cost:** $2

### Capacitor (Qty: 1)
- **Part:** Electrolytic Capacitor
- **Specifications:**
  - Capacitance: 1000µF
  - Voltage Rating: 16V minimum
  - Type: Radial electrolytic
  - Low ESR preferred
- **Purpose:** Power rail smoothing, prevent brownouts
- **Cost:** $1
- **Placement:** Across 5V rail near ESP32

---

## 4. Mechanical Components

### Bearings (Qty: 4)
- **Part:** 608ZZ Ball Bearings
- **Specifications:**
  - Size: 8mm ID × 22mm OD × 7mm Width
  - Type: Deep groove ball bearing
  - Shields: Double shielded (ZZ)
  - Material: Chrome steel
- **Purpose:** Smooth rotation at hip and knee joints
- **Cost:** $6 ($1.50 each)
- **Notes:** Standard skateboard bearing size

### Hardware Kit (Qty: 1)
- **Part:** M3 Metric Hardware Assortment
- **Contents:**
  - M3 × 10mm screws (20pcs)
  - M3 × 20mm screws (20pcs)
  - M3 × 30mm screws (10pcs)
  - M3 nuts (30pcs)
  - M3 lock nuts (20pcs)
  - M3 washers (30pcs)
  - M3 × 10mm standoffs (10pcs)
  - M3 × 20mm standoffs (10pcs)
- **Purpose:** Assemble all mechanical components
- **Cost:** $8
- **Material:** Stainless steel preferred (rust resistant)

### Wheels (Already Owned)
- **Purpose:** Ground contact for balance and locomotion
- **Cost:** $0 (already owned)
- **Recommended Specs:** 60-80mm diameter, rubber tire

---

## 5. Cooling & Protection

### Heat Sinks (Qty: 2)
- **Part:** Small Aluminum Heat Sinks
- **Specifications:**
  - Size: 20mm × 20mm × 10mm (approximate)
  - Material: Aluminum
  - Fins: Multiple fins for surface area
  - Mounting: Thermal adhesive or paste
- **Purpose:** Cool L293D motor driver chips
- **Cost:** $3 ($1.50 each)
- **Notes:** Prevents thermal shutdown during continuous use

---

## 6. Wiring & Connections

### Jumper Wires (Qty: 20)
- **Part:** Dupont Jumper Wires
- **Specifications:**
  - Length: 20cm (8 inches)
  - Type: Male-to-Female
  - Wire Gauge: 22-24 AWG
  - Connector: 2.54mm pitch
- **Purpose:** Connect ESP32 to sensors and motor shield
- **Cost:** $3
- **Color Coding Recommended:**
  - Red: Power (+)
  - Black: Ground (-)
  - Yellow/Green: Signal
  - Blue: I2C (SDA/SCL)

---

## 7. Materials (Already Owned)

### 3D Printing Filament
- **Material:** PLA or PETG
- **Amount Needed:** ~300-400g
- **Purpose:** Print chassis, leg components, motor mounts
- **Cost:** $0 (already owned)

---

## Shopping Links Categories

### Amazon Shopping List:
1. **Motors & Servos** (~$54)
   - 2× 6V 200RPM wheel motors with encoders
   - 2× 6V 100RPM hip motors with encoders
   - 2× MG996R servos

2. **Electronics** (~$6)
   - 1× MPU6050 IMU
   - 1× LM2596 buck converter

3. **Power** (~$34)
   - 1× 7.4V 2S 2500mAh LiPo battery
   - 1× 2S-3S LiPo charger
   - 1× XT60 connector pair
   - 1× Power switch
   - 1× 1000µF capacitor

4. **Hardware** (~$17)
   - 4× 608ZZ bearings
   - 1× M3 hardware kit
   - 2× Heat sinks

5. **Wiring** (~$3)
   - 20× Jumper wires (male-female)

---

## Tools Required (Not Included in Budget)

### Essential Tools:
- Soldering iron and solder
- Wire cutters/strippers
- Screwdriver set (Phillips, flathead)
- Allen key set (hex keys)
- Multimeter (voltage/continuity testing)
- 3D printer (for chassis fabrication)
- Hot glue gun (optional but helpful)

### Optional but Recommended:
- Helping hands/PCB holder
- Heat shrink tubing
- Cable ties/zip ties
- Label maker (for wire organization)

---

## Part Selection Notes

### Why 6V Motors (Not 12V)?
- L293D limit: 0.6A per channel
- 6V motors: ~400-500mA (SAFE ✓)
- 12V motors: ~800-1500mA (EXCEEDS LIMIT ✗)

### Why Encoders on All DC Motors?
- Closed-loop control for precision
- Know exact position/velocity
- Detect slipping
- Coordinate left/right synchronization
- Essential for smooth gait

### Why MG996R Servos?
- Metal gears (won't strip on impact)
- Sufficient torque (11 kg·cm)
- Fast enough (0.17s/60°)
- Budget-friendly (~$8 each)
- Well-documented, reliable

### Why 2S LiPo (Not 3S)?
- 2S = 7.4V nominal (matches 6V motors)
- 3S = 11.1V (too high, wastes energy on regulation)
- Lower voltage = safer, less heat

### Why MPU6050 (Not Other IMUs)?
- Excellent price/performance ($4)
- Well-supported Arduino/ESP32 libraries
- DMP (Digital Motion Processor) for sensor fusion
- Widely used, extensively documented
- I2C interface (simple 2-wire connection)

---

## Verification Checklist

Before ordering, verify:
- [ ] Motors are 6V (not 12V) with encoders
- [ ] Servos have metal gears
- [ ] Battery is 2S (7.4V) with appropriate capacity
- [ ] Charger supports 2S balance charging
- [ ] Buck converter outputs 5V (or is adjustable)
- [ ] Bearings are 608ZZ size
- [ ] All hardware is M3 metric
- [ ] Heat sinks fit L293D package size
- [ ] Wire length is sufficient (20cm recommended)

---

## Estimated Delivery Time
- **Electronics/Motors:** 1-2 weeks (if ordering from Asia)
- **Battery/Charger:** 1 week (domestic)
- **Hardware:** 1 week (domestic)
- **Total:** Plan for 2 weeks to receive everything

---

## Storage & Safety

### Battery Storage:
- Store at 3.8V per cell (7.6V total) for long-term
- Keep in LiPo-safe bag
- Never store fully charged for extended periods
- Check voltage monthly if not in use

### Safety Equipment:
- LiPo charging bag (highly recommended)
- Fire extinguisher nearby during charging
- Charge on non-flammable surface

---

## Next Steps After Parts Arrive

1. **Inventory Check:** Verify all parts received and undamaged
2. **Motor Testing:** Test each motor individually with lab power supply
3. **Encoder Testing:** Verify encoder signals with oscilloscope/logic analyzer
4. **Power System:** Assemble and test power distribution (battery → switch → converters)
5. **IMU Calibration:** Mount and calibrate MPU6050
6. **Shield Testing:** Verify L293D shield with test motors
7. **Integration:** Assemble complete electrical system
8. **Mechanical Assembly:** Print and assemble chassis
9. **Software Upload:** Load control code to ESP32
10. **Tuning:** Calibrate PID gains and gait parameters

**Estimated Assembly Time:** 2-3 full days (assuming parts printed)
