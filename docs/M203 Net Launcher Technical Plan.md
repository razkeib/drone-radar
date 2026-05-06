# M203 40mm Net Launcher — Technical Plan

## Context

This plan is the neutralization arm of the drone-radar project (see `Plan.md`, `Technical Plan.md`). The audio-detection system solves the "see the drone" problem; this system solves the "kill it cheaply and fast" problem. The goal is a 40mm round for the M203 grenade launcher that any infantry soldier already carries, which fires a net that entangles drone propellers at close-to-medium range. No special equipment, no training beyond standard grenade launcher qualification, no expensive electronics on the round itself.

---

## 1. M203 Platform — Confirmed Specifications

**Rifling:** The M203 barrel IS rifled (right-hand twist, 1:47" — one complete turn per 47 inches of barrel travel). This is critical and works in your favor: the projectile exits with significant spin.

**Spin rate calculation at v₀ = 76 m/s:**
```
v₀ = 76 m/s = 2,992 in/s
twist = 47 in/rev
n = 2992 / 47 = 63.7 rev/s = 3,823 RPM
ω = 2π × 63.7 = 400 rad/s
```

This spin is more than sufficient to centrifugally deploy a lightweight net near-instantaneously (see §4).

**Muzzle velocity:** 76 m/s is confirmed for the standard M433 HEDP round (228g). A lighter net payload with the same propellant charge will exit faster — see §5 for the correction required.

---

## 2. Ballistics Formula — Error Correction

**The original formula `y = -0.000849x² + x + 1.5` is mathematically inconsistent.** It mixes coefficients from two different trajectories.

### Derivation
For a projectile launched at elevation angle θ, initial velocity v₀, from height h₀ = 1.5 m:

```
y = h₀ + tan(θ)·x - [g / (2v₀²cos²θ)] · x²
```

### Flat trajectory (θ = 0°, aimed horizontally):
```
y = 1.5 - [9.81 / (2 × 76²)] · x²
y = 1.5 - 0.000849·x²          ← coefficient IS correct here
```
The +x term is absent because tan(0°) = 0. Ground impact at x = √(1.5/0.000849) ≈ **42 m**.

### 45° trajectory (maximum range angle):
```
cos²(45°) = 0.5
y = 1.5 + x - [9.81 / (2 × 76² × 0.5)] · x²
y = 1.5 + x - 0.001699·x²      ← coefficient is 2× larger
```
Ground impact at x ≈ **590 m** (ignoring air resistance).

### The error explained
The formula `y = -0.000849x² + x + 1.5` takes the flat-shot coefficient (0.000849 for θ=0°) but pairs it with the +x slope term (tan(45°) = 1). This produces a fictitious ~1,180 m range that is physically impossible.

**Correct formulas by elevation angle:**
| Angle | Formula | Vacuum range |
|---|---|---|
| 0° (flat, soldier shoulder height) | y = 1.5 − 0.000849x² | 42 m |
| 20° | y = 1.5 + 0.364x − 0.000960x² | ~320 m |
| 45° (max range) | y = 1.5 + x − 0.001699x² | ~590 m |

**Practical note:** Maximum ballistic range does not equal effective net range. See §6.

---

## 3. The Critical Design Problem

**This is the most important section of this plan.**

The intuition was: "the cartridge should be made of metal which should have a mechanism that opens up about 2-3 meters from launch so it won't tangle with anything on ground." This is correct for SAFETY — but if the net itself opens at 2-3 m, it cannot reach the drone.

### Physics of an open net in flight

An open 2 m diameter net experiences enormous aerodynamic drag:
```
F_drag = ½ × ρ × Cd × A × v²
       = ½ × 1.225 × 0.5 × π(1)² × 76²
       ≈ 5,560 N

For a 100 g net payload:
a = F/m = 5560 / 0.1 = 55,600 m/s²  ≈ 5,670 g's

Velocity decay: v(t) = v₀ / (1 + k·v₀·t)   where k = ρ·Cd·A / (2m) ≈ 9.6 m⁻¹

At t = 0.1 s: v = 76 / (1 + 9.6 × 76 × 0.1) = 76 / 73.9 ≈ 1.0 m/s

Total distance traveled before stopping: x = (1/k)·ln(1 + k·v₀·t) ≈ 0.47 m
```

**Conclusion:** An open net stops within ~0.5 m of deployment. It cannot travel to a drone.

### The correct two-stage design

```
Stage 1 — Safety (2–3 m from muzzle):
  Outer metal casing splits open and falls away.
  This is the "safety arming" event — prevents ground entanglement.
  The NET remains packed inside a compact inner sub-projectile.

Stage 2 — Deployment (near drone, set by soldier):
  Inner sub-projectile (net slug) reaches target range.
  A timer/fuze triggers release of the net.
  Centrifugal spin (still ~3,800 RPM) opens net in ~5 ms.
  Net wraps around drone propellers.
```

The metal casing opening and the net deployment are TWO SEPARATE events at TWO DIFFERENT distances.

---

## 4. Net Design

### Material
**UHMWPE / Dyneema** — optimal choice:
- Tensile strength: ~2.5 GPa (stronger than steel by weight)
- Density: 0.97 g/cm³ (floats on water — lightest structural fiber available)
- 0.1 mm diameter thread: ~10-15 kg breaking load
- Virtually no stretch (high modulus) — does not bounce off propellers
- Cut-resistant

### Dimensions
| Parameter | Value | Rationale |
|---|---|---|
| Open diameter | 2.0–2.5 m | Catches most tactical drones with ±1 m aim error |
| Mesh size | 15 mm × 15 mm | ~3× smaller than any propeller blade; thread catches regardless of impact angle |
| Thread diameter | 0.1 mm | Thin enough to fold to <1 g/m², strong enough to resist propeller tension |
| Edge weights | 6–8 × 2 g steel balls | Distributed evenly around perimeter; maximize centrifugal opening force |

### Centrifugal deployment speed
```
At ω = 400 rad/s, edge weight at r = 0 → 1 m:
Centripetal acceleration at r_avg = 0.5 m: a = ω² × r = 400² × 0.5 = 80,000 m/s²

Time to travel 1 m outward under this acceleration:
t = √(2r/a) = √(2/80,000) ≈ 0.005 s (5 ms)
```
Net deployment is essentially instantaneous — no timing precision needed after trigger.

### Packed dimensions
A 2 m Dyneema net (15 mm mesh, 0.1 mm thread) weighs approximately 20–40 g.
With edge weights: total ~50–70 g.
Folded accordion-style into 40 mm × 45 mm cylinder: easily achievable. Commercial fishing nets of similar spec compress to fist-size.

---

## 5. Propellant and Muzzle Velocity for Net Payload

**Standard M433 round:** 228 g projectile, 76 m/s.

A lighter net payload with the same propellant produces a higher muzzle velocity (same energy, less mass):
```
½·m₁·v₁² ≈ ½·m₂·v₂²
v₂ = v₁ × √(m₁/m₂)

If net payload = 100 g:
v₂ = 76 × √(228/100) = 76 × 1.51 = 114.8 m/s
```

At 115 m/s, spin rate becomes ~5,750 RPM — higher than needed, and higher chamber pressure/recoil. **The propellant charge must be reduced** to bring muzzle velocity back to ~76 m/s. This is standard ballistic engineering: tune propellant charge mass to achieve target v₀ for the given payload mass.

Consequence for spin: a reduced charge targeting 76 m/s keeps spin at ~3,820 RPM regardless of payload weight, since spin is determined solely by muzzle velocity and rifling twist rate.

**Design specification:** Propellant charge tuned to achieve 75–80 m/s muzzle velocity with the net payload, verified by chronograph during live-fire testing.

---

## 6. Effective Range Analysis

Drones in tactical scenarios typically operate at 10–80 m altitude and 20–150 m horizontal distance. Relevant engagement geometries:

| Drone altitude | Horizontal dist | Elevation angle | Slant range | Time of flight (approx) | Required lead at 10 m/s drone |
|---|---|---|---|---|---|
| 20 m | 30 m | 34° | 36 m | 0.47 s | 4.7 m |
| 30 m | 50 m | 31° | 58 m | 0.76 s | 7.6 m |
| 50 m | 50 m | 45° | 71 m | 0.93 s | 9.3 m |
| 20 m | 80 m | 14° | 82 m | 1.08 s | 10.8 m |

**With a 2 m net (1 m radius), aim error tolerance is ±1 m.** Lead prediction to within 1 m at moving targets requires training. This is the primary accuracy challenge for the soldier, not the weapon itself.

**Practical effective range:** 20–80 m slant range. Beyond 80 m, time of flight exceeds ~1 second, lead error for a moving drone exceeds the net radius, and success probability drops significantly without assisted aiming.

Maximum ballistic range (~590 m at 45°) is irrelevant. The net is relevant only where the drone can be seen and lead-aimed by a human: **20–80 m slant range**.

---

## 7. Fuze / Deployment Timing Mechanism

This is the most critical engineering component.

### Option A — Mechanical time fuse (recommended for POC)
- Soldier estimates range to drone (e.g., 50 m)
- Sets dial on round (like a grenade fuze ring) to corresponding time setting
- At t_fuse, pyrotechnic or spring mechanism releases net from inner slug
- Simple, no electronics, low cost
- Downside: soldier must estimate range correctly; wrong estimate = deployment too early/late

**Time settings needed:**
```
20 m: t ≈ 0.26 s
30 m: t ≈ 0.39 s
50 m: t ≈ 0.66 s
80 m: t ≈ 1.05 s
```
These assume straight-line travel at v₀ ≈ 76 m/s. In practice, air resistance on the compact slug (Cd ≈ 0.3) causes ~5–10% velocity loss over 50–80 m. Fuze settings should be calibrated empirically during Phase 4 live-fire testing.

### Option B — Fixed distance (spring-loaded, non-adjustable)
- Net deploys at a fixed slant range, e.g., always at 40 m
- Simpler to manufacture; eliminates soldier range-estimation requirement
- Limits flexibility — only effective at one optimal range

### Option C — Electronic proximity fuze (post-POC)
- Small IR or RF sensor detects drone within 3–5 m
- Triggers net deployment on detection
- Highest accuracy, most complex, adds cost and failure points
- Reserve for production version

**Recommendation for POC:** Option A with 4 dial positions (20 m / 40 m / 60 m / 80 m).

---

## 8. Complete Cartridge Architecture

```
[M203 cartridge — cross section]

┌─────────────────────────────────────────────┐
│ Cartridge case (brass/steel, 40×46mm NATO)  │
│ Propellant charge (reduced, ~76 m/s target) │
├─────────────────────────────────────────────┤
│ Rotating bands (engage M203 rifling)         │
├────────────────────────────────────────────┤
│ OUTER CASING (aluminum, splits at 2–3 m)    │
│   - Held by shear pins, breaks on setback   │
│   - Falls away 2–3 m downrange (safety)     │
├────────────────────────────────────────────┤
│ INNER SUB-PROJECTILE (net slug, continues)  │
│   - Compact cylinder, ~38 mm OD             │
│   - Contains folded Dyneema net             │
│   - Contains 6–8 edge weights (2 g steel)   │
│   - Timed fuze (dial-set before firing)     │
│   - Spring-loaded release collar            │
├────────────────────────────────────────────┤
│ FUZE MECHANISM                              │
│   - Clockwork or electronic timer           │
│   - Arms after 2–3 m (setback mechanism)   │
│   - Fires at t_set, releases net collar     │
└────────────────────────────────────────────┘
```

**Materials:**
- Outer casing: aluminum (light, splits cleanly)
- Inner slug body: ABS plastic or aluminum
- Net: Dyneema 0.1 mm monofilament
- Edge weights: steel balls, standard fishing-net sinkers
- Fuze: off-the-shelf pyrotechnic timer (used in illumination rounds) or small electronics

---

## 9. Wind and Environmental Effects

The inner slug (~100–130 g, compact) has manageable drag in flight. Over 50–80 m:
- Crosswind drift at 5 m/s wind, 1-second flight: ~5 m lateral displacement
- A 2 m net catches to within 1 m of center → 5 m crosswind at 80 m range → misses

**Implication:** At longer ranges (>60 m) in significant wind, the soldier must compensate. At short range (<40 m), 5 m/s wind causes ~2 m drift — marginal with a 2 m net.

Recommend a **2.5 m net** for operational units to add wind tolerance margin.

The net itself, once deployed at t_fuse, experiences only ~5 ms of additional flight before reaching the drone. Wind effect during those 5 ms is negligible (<5 mm).

---

## 10. Entanglement Mechanics

For the net to disable the drone, it must entangle propellers:
- Consumer quadrotor propellers: 2–6 inch diameter, spin at 8,000–20,000 RPM
- Net mesh (15 mm) vs propeller chord (~20–30 mm): thread catches on propeller edge
- 0.1 mm Dyneema under tension resists propeller cutting force for ~0.5–1 second
- Within that time: motor stalls mechanically → ESC overcurrent → motor shutdown
- A 2+ m net catches multiple rotors simultaneously, which is required to bring the drone down

**Thread strength verification:**
```
Propeller tip force at stall: ~2–5 N per propeller
Dyneema 0.1 mm breaking load: ~10–15 kg = 100–150 N
Safety factor: 20–30× — thread will not break before motor stalls
```

**Key requirement:** Mesh size ≤ 20 mm. Larger mesh risks propellers passing through without catching.

---

## 11. Safety and Regulatory Considerations

1. **Safe-and-arm distance:** 2–3 m minimum, consistent with NATO STANAG 4187 requirements for 40mm rounds. The outer casing shedding is the safety event.
2. **Dud rate:** The net deployment fuze must have a self-destruct/passivation mode if the round misses. Unexploded rounds on the ground are a hazard.
3. **Friendly fire hazard:** A missed net that deploys late could entangle nearby soldiers. Safety training required; max "check-fire" range must be documented.
4. **Export control:** The cartridge is a weapon — ITAR (US) or equivalent controls apply for production/sale. POC development is generally permissible.
5. **Test range requirements:** Live-fire testing requires a licensed military test range.

---

## 12. POC Step-by-Step Plan

### Phase 0 — Specification Lock (1–2 weeks, no hardware)
- [ ] Confirm target engagement scenario: drone type, typical range, typical altitude, drone speed
- [ ] Lock net diameter (recommend 2.0 m) and mesh size (15 mm)
- [ ] Lock propellant spec: target muzzle velocity and payload mass
- [ ] Define "success" criteria for each phase

### Phase 1 — Net Fabrication & Static Testing (2–3 weeks, ~$200)
- [ ] Order 10 m × 1 m Dyneema netting, 15 mm mesh, 0.1 mm thread (marine/fishing suppliers)
- [ ] Cut circular nets, 2 m diameter; attach 6–8 steel ball weights at edge
- [ ] Manual tangle test: hold net above consumer drone mid-flight; confirm propeller entanglement and motor stall within 1 second
- [ ] Rotary deployment test: mount folded net on drill at 3,800 RPM; confirm opens to full 2 m in <50 ms
- [ ] **Do not proceed if propellers cut thread** — increase thread diameter

### Phase 2 — Sub-Projectile Mechanical Design (3–4 weeks, ~$500)
- [ ] Design and 3D print inner slug body (38 mm OD, 50 mm length)
- [ ] Integrate folded net + edge weights into slug; confirm fits inside 40 mm OD outer casing
- [ ] Design spring-loaded net release collar (spring holds net in; fuze trigger releases)
- [ ] Design outer casing with shear pins (calibrated to break at ~3 m setback + centrifugal)
- [ ] Bench test: spin assembly on lathe at 3,800 RPM, trigger collar release; verify net opens without tangling on slug

### Phase 3 — Compressed Air Ballistic Test (2–3 weeks, ~$1,000)
- [ ] Build or procure compressed air launcher producing ~76 m/s at 40 mm bore
- [ ] Fire inert slug (no net, equivalent mass) to verify structural integrity under setback forces
- [ ] Fire full net assembly with inert fuze → confirm outer casing sheds at 2–3 m, inner slug continues
- [ ] Fire with active timed fuze → verify net deploys at target distance
- [ ] Use high-speed camera (1,000–5,000 fps) to document every phase
- [ ] **Checkpoint:** net deploys within ±5 m of dial-set target range in 10/10 shots

### Phase 4 — Live Fire on Test Range (4–8 weeks, ~$5,000+, requires military cooperation)
- [ ] Submit design to military/defense authority for safety review
- [ ] Coordinate with military test range (Guy — Military Communicator — leads this)
- [ ] Verify propellant charge via chronograph; confirm muzzle velocity with net payload
- [ ] 20 shots at suspended targets at 20 m, 40 m, 60 m to confirm deployment timing
- [ ] 10 shots at a tethered hovering drone (remote-pilot safety protocol)
- [ ] **Checkpoint:** ≥7/10 shots entangle drone at 40 m with fuze set to 40 m

### Phase 5 — Iteration and Documentation (2–4 weeks)
- [ ] Adjust mesh size, net diameter, or edge weight count based on Phase 4 failure modes
- [ ] Document soldier training requirements (range estimation, lead angle, wind compensation)
- [ ] Estimate unit cost at scale (material, machining, propellant): target <$50/round
- [ ] Prepare demonstration package for military acquisition

---

## 13. Open Questions (Resolve Before Phase 1)

1. **What is the target drone?** FPV kamikaze (~20 cm span) vs DJI Mavic-type reconnaissance (~30 cm) vs larger logistics drone (~60 cm). Net mesh and diameter design changes significantly.
2. **What is the maximum engagement range the military requires?** If 50 m is acceptable, fuze is simple. If 100 m is required, a proximity fuze (Option C above) may be needed.
3. **Is capture (intact drone) required, or just downing it?** If capture (for intelligence exploitation), the net needs a descent parachute. If just downing, any entanglement is a success.
4. **Matan (team mathematician) should verify** the drag equations and fuze timing calculations in §7 — specifically the velocity-decay formula and fuze time vs range table, which are first-order approximations assuming constant Cd and ignoring spin-down effects.

---

## 14. Bill of Materials (Phase 1–3 Estimate)

| Item | Quantity | Est. cost |
|---|---|---|
| Dyneema fishing net, 15 mm mesh, 0.1 mm thread | 5 m² | ~$30 |
| Steel fishing sinkers, 2 g | 50 | ~$10 |
| 3D printer filament (PLA/PETG for slug prototypes) | 1 kg | ~$25 |
| Drill press with RPM control (Phase 2 spin test) | 1 | ~$150 |
| Compressed air launcher components | 1 set | ~$400 |
| High-speed camera rental (1,000 fps) | 1 day | ~$500 |
| Fuze components (springs, shear pins, pyrotechnic delay) | bulk | ~$200 |
| **Phase 1–3 subtotal** | | **~$1,315** |

---

## 15. Alternative Payload: High-Density Pellet Round ("Wall" Approach)

This section covers a second, complementary design for the same M203 platform — a shotgun-style pellet round that creates a dense cloud of projectiles. It is mechanically simpler than the net round but has different tradeoffs. Both designs should be prototyped and compared.

---

### 15.1 Why the M576 Exits Slightly Faster than the M433

The M576 buckshot round exits at ~82 m/s vs the M433 HEDP at ~76 m/s. The reason is simply lighter projectile mass with a similar propellant charge:

```
½·m₁·v₁² ≈ ½·m₂·v₂²
v₂ = v₁ × √(m₁/m₂)

M576 projectile (~185 g) vs M433 (~228 g):
v₂ = 76 × √(228/185) = 76 × 1.11 = 84 m/s  ← matches observed ~82 m/s
```

This is the same principle as §5: lighter payload exits faster with the same propellant charge. It is not a special property of buckshot — a heavier pellet round would exit at the same ~76 m/s as the M433 if masses matched.

---

### 15.2 Shotguns Against Drones — Real-World Effectiveness

From documented military testing, field reports, and FPV pilot communities:

| Weapon | Range | Effectiveness |
|---|---|---|
| 12-gauge birdshot (#7.5) | <20 m | Reliable. Propellers shattered, immediate crash. |
| 12-gauge birdshot (#7.5) | 20–40 m | Inconsistent. Works on small drones, not reliable on larger ones. |
| 12-gauge buckshot (#4) | <30 m | Reliable. Multiple propeller hits. |
| 12-gauge buckshot (#4) | 30–60 m | Inconsistent. Spread too wide, energy marginal. |
| 12-gauge 00 buckshot | <50 m | Reliable on small drones. Higher energy per pellet compensates for lower count. |
| 12-gauge 00 buckshot | 50–80 m | Marginal. Occasionally works. Not tactically reliable. |
| 12-gauge slug | <100 m | Works if aimed precisely. No tolerance for aim error. |

**Why propeller hits work:** Consumer and FPV propellers are plastic (ABS, nylon, carbon-filled nylon). A steel pellet at impact velocity above ~30 m/s carries enough kinetic energy (~1–3 J) to crack or shatter a propeller blade. Breaking even one of four propellers is usually enough — flight controllers cannot compensate for a broken prop and the drone crashes. The goal is not to penetrate the drone body — just to touch the propellers.

**Why standard shotguns fail at range:** Two simultaneous problems as range increases: (1) pellets lose kinetic energy to drag and arrive below the damage threshold, and (2) the cloud spreads so wide that per-pellet hit probability drops to near zero. The M576's 20 pellets makes this worse — the cloud is already too sparse at 40 m to be tactically reliable.

---

### 15.3 Pellet Size Tradeoff: Energy vs Count

Smaller pellets → more fit per cartridge → denser cloud → higher hit probability. But smaller pellets also decelerate faster because they have a higher drag-to-mass ratio (surface area scales with r², mass scales with r³).

**Drag equation for a sphere:**
```
dv/dx = −k·v   →   v(x) = v₀·e^(−k·x)
                    KE(x) = KE₀·e^(−2k·x)

where k = (ρ · Cd · π · r²) / (2m)
      ρ = 1.225 kg/m³,  Cd = 0.47 (sphere)
```

**KE at range for standard pellet sizes (v₀ = 82 m/s):**

| Pellet type | Diameter | Mass | KE at 0 m | KE at 40 m | KE at 60 m | KE at 80 m |
|---|---|---|---|---|---|---|
| #7.5 birdshot | 2.4 mm | 0.06 g | 0.20 J | 0.08 J | 0.05 J | 0.03 J |
| #2 birdshot | 3.8 mm | 0.26 g | 0.87 J | 0.47 J | 0.34 J | 0.24 J |
| **#3 buckshot** | **5.7 mm** | **0.96 g** | **3.23 J** | **2.17 J** | **1.78 J** | **1.46 J** |
| **#4 buckshot** | **6.1 mm** | **1.70 g** | **5.70 J** | **4.36 J** | **3.79 J** | **3.30 J** |
| 00 buckshot | 8.4 mm | 4.40 g | 14.8 J | 12.0 J | 10.8 J | 9.70 J |

**Minimum KE for reliable propeller damage:** ~1–2 J (plastic propeller fracture energy at >30 m/s impact velocity).

**Verdict by pellet type:**
- **#7.5 birdshot:** Useless beyond 20 m. Too little energy.
- **#2 birdshot:** Effective to ~40 m. Borderline at 60 m.
- **#3 buckshot:** Effective to ~70–80 m. Best balance of count and energy. ← primary recommendation
- **#4 buckshot:** Effective to ~100 m. Slightly fewer fit per cartridge than #3.
- **00 buckshot:** Maximum energy but very few pellets → sparse cloud → low hit probability despite high per-pellet energy.

**Optimal choice: #3 or #4 buckshot (5.7–6.1 mm steel spheres, 0.96–1.70 g each).** This is the range that balances sufficient energy at 60–80 m against enough pellet count to create a genuine wall.

---

### 15.4 How Many Pellets Fit in a 40mm Cartridge

The M576 packs only 20 pellets because its sabot cup design is not optimized for pellet density. A purpose-built anti-drone round can do significantly better.

**Available cavity in a 40mm projectile body:**
```
Inner diameter: ~36 mm (2 mm walls)
Usable length:  ~50 mm (leaving room for wad base and crimp)
Inner volume:   π × (18 mm)² × 50 mm ≈ 50,900 mm³
```

**Pellet volumes and theoretical packing:**
```
#3 buckshot (d = 5.7 mm): V_pellet = (4/3)π(2.85 mm)³ = 97 mm³
#4 buckshot (d = 6.1 mm): V_pellet = (4/3)π(3.05 mm)³ = 119 mm³

Random sphere packing efficiency: ~64%

#3: N = 50,900 × 0.64 / 97  ≈ 335 pellets (theoretical maximum)
#4: N = 50,900 × 0.64 / 119 ≈ 274 pellets (theoretical maximum)
```

**Practical pellet count** (accounting for wad structure, base cup, manufacturing tolerances): **100–180 pellets** — 5–9× more than the M576's 20.

---

### 15.5 "Wall" Density and Hit Probability

**Pellet spread model:**
The M576 achieves roughly 0.5 m diameter spread at 22 m range. This gives a half-angle spread rate of:
```
α = arctan(0.25 / 22) ≈ 0.65°
Cloud diameter at range x:  D(x) ≈ 0.023·x
```
The M203's rifling adds a centrifugal component as pellets exit the spinning wad cup, slightly increasing spread. Empirical calibration needed but this is conservative.

**Hit probability** (drone cross-section ≈ 0.09 m² for a 30 cm quadrotor):
```
P(≥1 hit) = 1 − (1 − A_drone / A_cloud)^N
```

| Range | Cloud diameter | P(hit), 20 pellets [M576] | P(hit), 150 pellets |
|---|---|---|---|
| 20 m | 0.46 m | 66% | >99.9% |
| 40 m | 0.92 m | 24% | >99.9% |
| 60 m | 1.38 m | 11% | 99.8% |
| 80 m | 1.84 m | 6% | 99.3% |
| 100 m | 2.30 m | 4% | 94.8% |

The M576 is genuinely inadequate (24% at 40 m). 150 pellets of #3 buckshot creates a near-certain wall to **80 m**, with useful probability to 100 m.

**Expected hits and energy on target (150 × #3 buckshot):**

| Range | Expected hits | KE per pellet | Total energy delivered |
|---|---|---|---|
| 20 m | ~80 | 3.0 J | ~240 J |
| 40 m | ~20 | 2.2 J | ~44 J |
| 60 m | ~9 | 1.8 J | ~16 J |
| 80 m | ~5 | 1.5 J | ~7.5 J |

At 60–80 m: 5–9 propeller hits at 1.5–1.8 J each is catastrophic to any plastic-propeller drone.

---

### 15.6 Wad/Sabot Design for the Rifled M203 Barrel

The M203 barrel is rifled, which creates a constraint absent from smoothbore shotguns. The wad cup engages the rifling and exits at ~3,820 RPM. When the cup petals open at the muzzle, pellets are flung outward with a slight centrifugal component on top of their forward velocity — this actually helps by producing a more uniform radial spread than a smoothbore would.

The wad cup itself weighs <5 g and experiences extreme aerodynamic deceleration immediately after muzzle exit. It falls to the ground within 1–3 m and is not a downrange hazard.

**Wad architecture:**
```
[Wad cross-section]

  ┌──────────────────────────────┐
  │  Rotating band               │
  │  (engages rifling, spins wad)│
  ├──────────────────────────────┤
  │  Base wad (gas seal)         │
  ├──────────────────────────────┤
  │  Pellet cup — 4 scored petals│
  │  open under air pressure /   │
  │  centrifugal force at muzzle │
  │                              │
  │  ○ ○ ○ ○ ○ ○ ○ ○ ○           │
  │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○          │
  │  ○ ○ ○ ○ ○ ○ ○ ○ ○  (pellets)│
  │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○          │
  │  ○ ○ ○ ○ ○ ○ ○ ○ ○           │
  ├──────────────────────────────┤
  │  Shear-pin crimp ring        │
  │  (holds petals closed for    │
  │   first 2–3 m — see §15.7)   │
  └──────────────────────────────┘
```

**Material:** Polyethylene or ABS cup. Petal walls scored to open under combined air pressure differential and centrifugal force at muzzle. Total wad weight target: <5 g.

---

### 15.7 Safety: The Shear-Pin Crimp Ring ("Auto-Fuse at 2–3 m")

For the net round, the outer casing sheds at 2–3 m while the inner net slug continues. For the pellet round, the analogous safety mechanism is a **shear-pin crimp ring on the wad cup** that keeps all pellets bundled together until the round has cleared the immediate area around the shooter.

**Without the crimp:** Pellets begin spreading immediately at the muzzle. Within 2–3 m, the cloud is still very dense and close — any friendly soldier within that radius to the side is at risk from stray pellets.

**With the crimp ring:**
- 0–2 m: all pellets travel as a single compact slug inside the closed cup (~130–280 g). Essentially a solid projectile.
- At ~2–3 m: shear pin breaks (driven by setback deceleration or centrifugal force from spin at 3,820 RPM)
- Cup petals open, pellets spread into cloud pattern
- Wad decelerates to zero within the next 1–3 m and falls to ground
- From ~5 m onward: clean spreading pellet cloud, no heavy wad fragment

**No pyrotechnics needed.** The shear pin is calibrated to break at a specific mechanical force — the same concept as the outer casing on the net round, just simpler. The crimp ring is a stamped metal band with notched break points.

---

### 15.8 Effective Range Summary

| Range | Cloud diameter | Expected hits (150 pellets) | KE/hit | Assessment |
|---|---|---|---|---|
| 10–20 m | 0.2–0.5 m | 20–80+ | 2.9–3.1 J | Excellent. Very dense wall. May destroy drone completely. |
| 20–50 m | 0.5–1.1 m | 12–20 | 2.1–2.9 J | Optimal zone. Near-certain multiple propeller hits. Reliable kill. |
| 50–80 m | 1.1–1.8 m | 5–12 | 1.5–2.1 J | Good. 5+ propeller hits expected. Reliable at 60 m. |
| 80–100 m | 1.8–2.3 m | 3–5 | 1.2–1.5 J | Marginal. 3–5 hits each near minimum damage threshold. |
| >100 m | >2.3 m | <3 | <1.2 J | Unreliable. Spread too wide, energy too low. |

**Practical effective range: 20–80 m** — identical envelope to the net round, but with zero fuze-timing requirement from the soldier.

---

### 15.9 Collateral Damage — The Key Tactical Limitation

Unlike the net (which falls immediately after deployment), pellets **continue flying beyond the drone**. Pellets fired at elevation to reach a drone at altitude will follow a parabolic arc and land 200–400 m downrange. They retain ~1–2 J at that distance — enough to injure exposed skin or eyes.

This makes the pellet round **unsuitable for urban combat or when friendly forces are beyond the target drone**. It is well-suited for open field engagement (the primary military scenario) and specifically for intercepting incoming FPV drones flying directly toward the shooter — in that case, nothing is behind the drone from the shooter's perspective.

---

### 15.10 Pellet Round vs Net Round — Direct Comparison

| Property | Net Round | Pellet Round |
|---|---|---|
| Effective range | 20–80 m | 20–80 m |
| Soldier skill required | High (fuze timing + aim) | Low (aim only) |
| Works against fast-moving drones | Harder (fuze timing critical) | Yes — fire in drone's path |
| Works against armored drones | Yes (mechanical entanglement) | No (pellets deflect off armor) |
| Drone fate after hit | Definite fall (propellers stalled) | Probable fall (damage-dependent) |
| Collateral risk beyond target | Very low (net falls immediately) | Real (pellets travel 200–400 m) |
| Mechanism complexity | High (two-stage fuze) | Low (wad + shear-pin crimp) |
| Phase 1–3 cost | ~$1,315 | ~$305 |
| Phase 1–3 timeline | 8–10 weeks | 4–6 weeks |

**Recommended development order:** Build and test the pellet round first — cheaper, faster to prototype, and operationally simpler for soldiers. Lessons learned from the M203 ballistics, wad behavior, and range calibration directly inform the net round fuze timing design. Run both in parallel once the compressed air launcher is built (shared infrastructure).

---

### 15.11 Pellet Round POC Steps

#### Phase 1 — Component Validation (1–2 weeks, ~$100)
- [ ] Purchase #3 and #4 buckshot (hunting supply store)
- [ ] Manual impact test: drop ~20 pellets from 1.5 m onto a hovering consumer drone's propeller arc — confirm plastic propellers crack or shatter at low impact velocity
- [ ] If inconclusive: increase pellet size to #1 buckshot and retest
- [ ] **Do not proceed if pellets bounce off without damage** — drone has hardened propellers, different approach needed

#### Phase 2 — Wad Design and Packing (2–3 weeks, ~$300)
- [ ] Design 40mm pellet cup in CAD (4-petal scored design, rotating band, base wad)
- [ ] 3D print prototype cups; fill with pellets; count achieved (target: 120–150 × #3 buckshot)
- [ ] Design shear-pin crimp ring; calibrate break force on tensile test rig (target: breaks at ~3–5 m setback force)
- [ ] Static spin test: mount loaded wad on lathe at 3,820 RPM with crimp removed; confirm petals open cleanly and pellets spread radially without tangling

#### Phase 3 — Compressed Air Launch Test (2–3 weeks, ~$800)
- [ ] Fire inert wad (no pellets, equivalent mass) — confirm structural integrity and wad shedding within 3 m (high-speed camera)
- [ ] Fire full pellet round with crimp ring at paper targets at 20 m, 40 m, 60 m, 80 m
- [ ] Count pellet holes per target; measure spread diameter; compare to model predictions in §15.5
- [ ] Verify crimp ring sheds at 2–3 m, not at muzzle and not at 10+ m
- [ ] **Checkpoint:** at 40 m, ≥40% of target area within 0.9 m diameter covered by pellet holes

#### Phase 4 — Live Fire (combined with net round Phase 4, shared range time)
- [ ] 20 shots at paper targets at 20 / 40 / 60 / 80 m — validate spread model with real M203 spin
- [ ] 5 shots at tethered hovering drone; confirm disabling hit each time
- [ ] **Checkpoint:** drone disabled (crash or loss of control) in ≥4/5 shots at 40 m

---

### 15.12 Pellet Round Bill of Materials (Phase 1–3)

| Item | Quantity | Est. cost |
|---|---|---|
| #3 buckshot, 25 lb bag (hunting supply) | 1 | ~$30 |
| #4 buckshot, 25 lb bag (for comparison) | 1 | ~$30 |
| ABS/PE filament for wad cup prototypes | 1 kg | ~$25 |
| Spring scale / tensile test rig for shear-pin calibration | 1 | ~$50 |
| Paper targets for spread pattern measurement | 50 | ~$20 |
| Consumer drone for Phase 1 impact test (sacrificial) | 1 | ~$150 (used/cheap) |
| High-speed camera rental | shared with net round Phase 3 | — |
| Compressed air launcher | shared with net round Phase 3 | — |
| **Phase 1–3 subtotal** | | **~$305** |

**Combined budget for both designs in parallel (Phase 1–3):** ~$1,620. The pellet round adds only $305 to the overall project and shares all major hardware.
