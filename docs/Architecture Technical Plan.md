# Technical Macro Plan: Audio-Based Drone Classification & Distance Estimation

## Table of Contents
1. [Core Decisions](#core-decisions)
2. [Research Findings & Prior Art](#research-findings--prior-art)
3. [Two-Track Strategy](#two-track-strategy)
4. [Track A — Quick POC](#track-a--quick-poc)
   - [Phase A1 — Hardware Setup & YAMNet Baseline](#phase-a1--hardware-setup--yamnet-baseline)
   - [Phase A2 — Intensity-Based Distance Proxy](#phase-a2--intensity-based-distance-proxy)
   - [Phase A3 — Basic Multi-Mic Triangulation Test](#phase-a3--basic-multi-mic-triangulation-test)
5. [Track B — Precise Model](#track-b--precise-model)
   - [Phase B1 — Hardware Setup & Recording Pipeline](#phase-b1--hardware-setup--recording-pipeline)
   - [Phase B2 — Dataset Collection](#phase-b2--dataset-collection)
   - [Phase B3 — Audio Preprocessing Pipeline](#phase-b3--audio-preprocessing-pipeline)
   - [Phase B4 — Model Training](#phase-b4--model-training)
   - [Phase B5 — Field Validation](#phase-b5--field-validation)
6. [Phase 5 — Multi-Microphone & Triangulation](#phase-5--multi-microphone--triangulation)
7. [Phase 6 — Directional Detection & Long-Range Architecture](#phase-6--directional-detection--long-range-architecture)
8. [End-to-End Latency Breakdown](#end-to-end-latency-breakdown)
9. [What to Check / Validate Before Acting](#what-to-check--validate-before-acting)
10. [Tools & Libraries Summary](#tools--libraries-summary)
11. [Hardware Shopping List](#hardware-shopping-list)
12. [Glossary](#glossary) — YAMNet, Model Backbone, Model Quantization, Transfer Learning, Fine-tuning, Domain Gap, SVM, MFCC, Mel Spectrogram, CNN, CNN-BiLSTM, BPF, TDOA, GCC-PHAT, MAE, Huber Loss, Inverse Square Law

---

## Core Decisions

| Question | Decision | Reason |
|---|---|---|
| Analog vs digital mic | Digital MEMS (I²S) for deployment; MEMS also for training recordings | Deployment constraints (low power, small, rugged, cheap) require MEMS. Training must use the same model to avoid [domain gap](#domain-gap) — model learns mic-specific response, not just drone acoustics |
| Which MEMS mic | INMP441 or ICS-43434 | Wide frequency response (50 Hz – 15 kHz); omnidirectional; weatherproofable |
| Why not a measurement mic for training | Use MEMS even for data collection | A measurement condenser has a flatter response and lower noise floor, but training on it and deploying on MEMS creates a [domain gap](#domain-gap). Side-by-side verification is useful to confirm your MEMS captures the key harmonics |
| One mic or many for training | One mic model, same unit used in deployment | Model learns mic-specific response; different mics add noise not signal |
| Distance measurement (Track A) | Audio intensity proxy via [Inverse Square Law](#inverse-square-law) | Fast, requires only one calibration recording at a known distance. Rough but functional for POC |
| Distance measurement (Track B) | GPS on both mic and drone | Gives precise ground-truth distance per audio sample. Required for training a real distance model |
| Distance output format (Track A) | Distance brackets (0–50 m, 50–150 m, 150–300 m, >300 m) | Intensity proxy is too noisy for continuous estimation — brackets are the most honest output at this precision level |
| Distance output format (Track B) | Continuous regression (meters) as primary output; bracket classification as secondary confidence check | GPS labels are already continuous values — converting them to brackets throws away information. Regression gives ±5–30 m accuracy depending on range, which is far more useful for triangulation than 100-m-wide brackets |
| Metadata in training data | Label everything (wind speed, environment, noise sources) but do NOT feed as model input | Model should be robust *to* conditions via diverse training data, not informed *of* conditions at inference time. Labels are for dataset analysis and debugging, not model inputs. Exception: a wind sensor could be added post-POC to dynamically adjust detection threshold |
| Mic hardware as model input | A 5-value feature vector `[mic_type, dish_diameter_cm, dish_depth_cm, dish_material, dish_wall_thickness_mm]` IS fed as a model input — specifically into the distance heads, not the classification head | Mic hardware is not an ambient condition; it is a known, deterministic property that systematically changes the acoustic signature. Each parameter captures a distinct physical effect: **`mic_type`** (int: 0=omni, 1=parabolic) is the primary flag. **`dish_diameter_cm`** (float, 0.0 for omni) encodes aperture size — gain scales roughly with D²×f², so a larger dish produces more total gain and a narrower beam, shifting the apparent loudness balance across harmonics. **`dish_depth_cm`** (float, 0.0 for omni) encodes bowl depth — determines focal length (f = D²/16d) and therefore the acceptance cone angle; a deeper dish focuses tightly (directional) while a shallower dish accepts sound from a wider cone. **`dish_material`** (int: 0=none, 1=metal, 2=mesh, 3=plastic/fiberglass) encodes reflectivity: metal reflects nearly perfectly; mesh has frequency-dependent transmission loss above the aperture frequency (passes sound where wavelength < aperture opening size); plastic/fiberglass introduces surface absorption and internal resonances at specific frequencies. **`dish_wall_thickness_mm`** (float, 0.0 for omni) encodes stiffness — a thin wall flexes under acoustic pressure and can resonate at its natural frequencies, absorbing and re-radiating energy at those frequencies, while a thick or rigid wall reflects cleanly without flexing. The combination of material + wall thickness determines the effective surface impedance and which frequencies are reflected vs absorbed. Without all five values the distance head silently averages over incompatible input distributions. Must be added from the start — retrofitting later requires full retraining on relabeled data. |
| Track B training approach | Fine-tuning (not training from scratch) | Start from YAMNet's pretrained weights rather than random initialization — the backbone already knows how to listen. Fine-tuning all layers on your dataset converges faster, needs less data, and reaches better accuracy |
| Multi-drone training | No — single drone per recording session for POC | Multi-source separation is a separate hard problem; triangulation handles multi-drone position |
| Audio feature | [Mel Spectrogram](#mel-spectrogram) + [MFCC](#mfcc) fusion | Combining both consistently outperforms either alone in published literature |
| Pretrained backbone | [YAMNet](#yamnet) (MobileNetV1, trained on AudioSet) | Audio-domain pretraining beats ImageNet pretraining for audio tasks; proven in drone detection literature; achieves detection up to 500 m |
| Training framework | PyTorch + torchaudio | Best ecosystem for audio ML; YAMNet PyTorch ports available |

---

## Research Findings & Prior Art

### Publicly Available Datasets — Study, Don't Train On

These datasets exist and are worth knowing about. However, **do not add them to your Track B training set** for two reasons: they were recorded on different microphones (creating a [domain gap](#domain-gap)), and none have GPS-derived ground-truth distance labels (making them useless for the distance estimation task regardless of the mic issue).

**For Track A only:** DroneAudioset can be used to [fine-tune](#fine-tuning) YAMNet for binary classification. The [domain gap](#domain-gap) degrades accuracy, but Track A is explicitly a rough POC — that tradeoff is acceptable there.

| Dataset | Size | Contents | Access |
|---|---|---|---|
| **DroneAudioset** | 23.5 hours | Annotated, multiple drone types, SNR from -57 to -2.5 dB, various environments | HuggingFace: `ahlab-drone-project/DroneAudioSet` (MIT license) |
| **Multiclass Acoustic Dataset (2025)** | 3,200 recordings / 16,000 seconds | 32 distinct UAV models, spectrograms and [MFCC](#mfcc) plots | arxiv.org/abs/2509.04715 |
| **DroneAudioDataset** | Medium | Propeller noise, indoor, noise-augmented | github.com/saraalemadi/DroneAudioDataset |
| **DREGON** | 8-channel array recordings | Annotated sounds from a quadrotor UAV; useful for [TDOA](#tdoa) research | dregon.inria.fr |
| **AUDROK** | Large | EU-classified drone sounds (C0–C3) | mobilithek.info |

In all cases: listen to recordings and inspect their spectrograms before your first recording session. This tells you what drone harmonics should look like and helps you validate that your MEMS is actually capturing them.

### What Models Have Achieved

| Approach | Feature | Result |
|---|---|---|
| [CNN](#cnn) + STFT (Seo et al.) | Spectrogram | >98% detection accuracy, 1.28% false alarm rate |
| RNN + [MFCC](#mfcc) (Jeon et al.) | 40 mel filters, 240 ms window | F1 = 0.698 |
| [YAMNet](#yamnet) [transfer learning](#transfer-learning) | AudioSet embeddings | Detection up to 500 m; outperforms CNN-only (200 m) |
| AUDRON framework | Fused [MFCC](#mfcc) + STFT + [CNN](#cnn) + LSTM + autoencoder | Best published multi-class type recognition |
| [CNN-BiLSTM](#cnn-bilstm) | [Mel Spectrogram](#mel-spectrogram) | Best published approach for distance prediction |
| [SVM](#svm) | [MFCC](#mfcc) | Strong with small datasets and low compute; good baseline |

### Drone Acoustic Signature — Known Physics

- **[BPF](#bpf)** = (RPM / 60) × number of blades
  - DJI Phantom II at 3840 RPM, 2 blades: BPF = 128 Hz, shaft rate 64 Hz
  - Typical consumer quadcopter at 4,700–6,200 RPM, 2 blades: BPF = 158–205 Hz
  - Energy concentrated below 1,000 Hz; harmonics remain high-amplitude up to the 4th (~500–820 Hz)
- **Broadband noise:** turbulent airflow and vortex shedding at blade tips creates the characteristic "warbling" drone sound above the tonal BPF
- **Filtering implication:** band-pass must include 50 Hz–1,000 Hz minimum to capture shaft rate and BPF harmonics

### Known Hard Negatives

| Sound | Why it's dangerous |
|---|---|
| **Helicopters** | Most dangerous confusor — similar harmonic structure to drones, just lower [BPF](#bpf). A model without helicopter negatives will fail in military contexts |
| **Birds (flapping)** | Wing-beat harmonics; hardest overall confusor in open-field environments |
| **Insects (mosquitoes, bees)** | Mosquito [BPF](#bpf) ~600 Hz overlaps drone harmonics |
| **Ground vehicles** | Engine harmonics in the same frequency band |
| **Industrial machinery / generators** | Sustained tonal noise at fixed frequencies |

### Detection Range from Literature

- CNN-only: reliable up to **200 m**
- Linear discriminant: reliable up to **300 m**
- [YAMNet](#yamnet) transfer learning: reliable up to **500 m**

### Embedded Deployment — Latency Reality Check

| Hardware | Model | Latency |
|---|---|---|
| Raspberry Pi 5 | Quantized INT8 | ~5 ms |
| Jetson Nano | YOLOv8n INT8 | ~23 ms |

The <100 ms target is easily achievable on Raspberry Pi 5 with a [quantized](#model-quantization) model. Use INT8 quantization before deployment — it also reduces energy consumption by ~45%.

### Similar Projects & Reference Systems

These are real implementations — open-source, commercial, and research — worth studying for architecture decisions, code patterns, and validated approaches before building your own.

#### Open-Source

| Project | Description | Link |
|---|---|---|
| **Acoustic-Drones-Detection** (orcohen9826) | End-to-end pipeline: MEMS microphone capture → MFCC/spectrogram feature extraction → classification model. Good reference for pipeline structure and preprocessing code. | github.com/orcohen9826/Acoustic-Drones-Detection |
| **DroneAudioDataset** (saraalemadi) | Not just a dataset — includes baseline classification code alongside the recordings. Useful to see how others structured the audio → label → model pipeline. | github.com/saraalemadi/DroneAudioDataset |
| **RTKLIB** | Full suite for RTK GPS processing including the `str2str` NTRIP client used in Phase B1 for base station corrections. Reference implementation for all RTK correction workflows. | rtklib.com |
| **pyroomacoustics** | Python library implementing GCC-PHAT, TDOA estimation, and room acoustic simulation. The reference implementation for the triangulation algorithms in Phase 5. | github.com/LCAV/pyroomacoustics |

#### Research Systems (Inspect for Architecture Ideas)

| System / Paper | Key Idea | Why It's Relevant |
|---|---|---|
| **Tetrahedral microphone array + DNN** (MDPI Sensors 2026) | 4-mic tetrahedral array fused with sensor geometry metadata in a neural network. >95% F1. | Shows how to fuse spatial geometry into the model, not just audio — relevant for Phase 5 TDOA work |
| **Circular array + HMM beamforming** (EURASIP J. Wireless 2019) | HMM on top of circular array beamforming for simultaneous classification, positioning, and tracking. | Full pipeline from sound to track in one system — closest to the end goal of this project |
| **Self-rotational bi-microphone array** (J. Intelligent & Robotic Systems 2018) | Two-mic rotating arm on a ground robot for real-time sound source localization including Virtual Rotating Array (VRA) Doppler compensation. | The only published rotating-array localization system — directly relevant to Phase 6 Option 1 |
| **CRNN + parabolic microphone** | Convolutional-Recurrent NN trained on parabolic mic input achieves >95% F1 on drone detection. | Validates that standard classification models work with parabolic input without special retraining |
| **DREGON dataset + system** (INRIA) | 8-channel microphone array with annotated quadrotor recordings. The system paper describes their full TDOA pipeline. | Reference implementation for multi-channel TDOA; dataset usable for GCC-PHAT validation |

#### Commercial Systems (Reference for Performance Targets)

| System | Detection Range | Technology | Notes |
|---|---|---|---|
| **Hall Lidar acoustic drone detection** | Up to 200 m | Standalone MEMS array, ML classification | Low-cost commercial product; proves 200 m is achievable with commodity hardware |
| **Zvook parabolic acoustic system** | 5 km (Shahed drone), 7 km (cruise missile) | Large parabolic concentrators | Sets the upper bound for what parabolic acoustic detection can achieve; military-grade |
| **Dedrone DroneTracker** | Multi-sensor fusion | RF + acoustic + optical | Dominant commercial C-UAS; worth studying for sensor fusion architecture even though RF is their primary layer |
| **Squarehead Technology "Discovair"** | ~500 m | Acoustic camera (large mic array + beamforming) | Shows what a serious fixed acoustic camera array achieves; uses the same beamforming principles as Phase 6 Option 3 |

---

## Two-Track Strategy

| | Track A — Quick POC | Track B — Precise Model |
|---|---|---|
| **Goal** | Prove the concept works end-to-end | Production-quality detection and distance estimation |
| **Timeline** | 2–4 weeks | 3–6 months |
| **Data needed** | Public datasets (DroneAudioset) + one reference calibration recording | Your own MEMS recordings with GPS-synchronized distance labels — no shortcut |
| **Classification** | Binary (drone / not-drone) via [fine-tuned](#fine-tuning) [YAMNet](#yamnet) | Multi-class (drone type) + binary, via [YAMNet](#yamnet) [fine-tuned](#fine-tuning) on your data |
| **Distance method** | Audio intensity proxy via [Inverse Square Law](#inverse-square-law) — rough, no labeled data needed | [CNN-BiLSTM](#cnn-bilstm) trained on GPS-labeled recordings — precise brackets |
| **Distance accuracy** | Very rough — affected by drone orientation and type variation | Bracket-level (±50 m), improving with more data |
| **Triangulation** | Coarse area estimate (intersecting rough distance circles) | Precise area with [TDOA](#tdoa) / [GCC-PHAT](#gcc-phat) available |
| **[Domain gap](#domain-gap)** | Yes — public dataset mics ≠ your MEMS. Accepted tradeoff for POC speed | No — trained on your exact hardware |
| **Two models running in parallel** | Optional: run both Track A and Track B models simultaneously; flag detection if either fires (OR logic = maximum recall) | — |

**Recommended sequence:** Complete Track A first to validate the full pipeline (hardware → detection → triangulation → alert). Then build Track B in parallel — swap in the precise model once it's validated.

---

## Track A — Quick POC

**Goal:** A working end-to-end demo — microphone captures audio, model detects drone presence, rough distance estimated, triangulation produces an area — in 2–4 weeks, without collecting custom training data.

### Phase A1 — Hardware Setup & YAMNet Baseline

1. Purchase MEMS microphone (INMP441 or ICS-43434), connect to a Raspberry Pi via I²S.
2. Purchase a foam windjammer ("deadcat"). Wind hitting a bare membrane creates intense low-frequency noise that drowns out drone harmonics — required for every outdoor session.
3. Verify recording pipeline: capture .wav files at 16 kHz, 16-bit mono. Listen to playback outdoors — should be clean.
4. Download DroneAudioset from HuggingFace (`ahlab-drone-project/DroneAudioSet`).
5. Set up [YAMNet](#yamnet) with a binary classification [head](#model-backbone): drone / not-drone.
6. [Fine-tune](#fine-tuning) for 5–10 epochs on DroneAudioset. This will have a [domain gap](#domain-gap) relative to your MEMS, but should be sufficient to detect drones at close range.
7. Test against a live drone outdoors. If it detects at 30 m, the baseline works.

**Checkpoint:** Model detects a hovering drone at 30 m with fewer than 1 false positive per minute of background audio.

### Phase A2 — Intensity-Based Distance Proxy

Distance in Track A is estimated from audio amplitude using the [Inverse Square Law](#inverse-square-law). Sound intensity drops with the square of distance — if you know how loud a drone sounds at a reference distance, you can estimate its distance from any other loudness measurement.

**Calibration procedure (do once per drone type):**
1. Place the microphone on its tripod. Hover the drone at exactly 10 m.
2. Record 30 seconds. Compute the mean RMS amplitude of that recording → this is your reference amplitude $A_{ref}$ at $d_{ref} = 10$ m.
3. Store `{drone_type: A_ref}` in a small config file.

**Runtime distance estimation:**
$$d_{estimated} = d_{ref} \times \sqrt{\frac{A_{ref}}{A_{measured}}}$$

Then map to a bracket:

| Estimated distance | Bracket |
|---|---|
| < 50 m | Close |
| 50–150 m | Medium |
| 150–300 m | Far |
| > 300 m | Likely not detected |

**Known limitations of this approach:**
- Drone orientation affects amplitude (a drone flying away from you sounds quieter at the same distance)
- Different drone types have different reference loudness — requires calibration per type
- Wind noise interferes with amplitude measurement even with a deadcat
- Accuracy degrades significantly beyond 150 m

These limitations are acceptable for Track A. Track B replaces this entirely with a trained model.

**Checkpoint:** Given a drone at a known distance (measured by tape or GPS), the intensity proxy estimates the correct bracket at least 60% of the time at distances up to 100 m.

### Phase A3 — Basic Multi-Mic Triangulation Test

1. Set up 3 MEMS microphones at known GPS positions (e.g., corners of a triangle with 50–100 m sides).
2. Each mic runs the Track A pipeline independently and reports: `{mic_id, gps_position, detected: true/false, distance_bracket, timestamp}` to a central laptop.
3. When ≥2 mics detect simultaneously, compute the intersection of their distance circles on a 2D map. This gives a rough "likely drone area."
4. Fly a drone within the triangle at various positions. Observe whether the estimated area contains the actual drone position.
5. Log all results. Identify the dominant failure modes.

**Checkpoint:** When a drone is within 150 m of all three mics, the triangulated area contains the actual drone position at least 50% of the time. Not precise — but it proves the pipeline works and gives a starting point for Track B.

---

## Track B — Precise Model

**Goal:** A model trained on your own MEMS recordings with GPS-derived distance labels. Calibrated to your hardware. Precise distance brackets. Deployable.

The [YAMNet](#yamnet) [backbone](#model-backbone) is reused from Track A — the difference is what data you [fine-tune](#fine-tuning) it on and the addition of a [CNN-BiLSTM](#cnn-bilstm) distance head.

### Phase B1 — Hardware Setup & Recording Pipeline

#### Why Standard GPS Is Not Enough for Sample Collection

Your regression model can only be as accurate as its training labels. Standard consumer GPS gives ±3–5 m horizontal accuracy and ±10–15 m vertical accuracy. At close ranges this is a serious problem: a ±5 m error on a 30 m recording is a 17% label error — the model learns a fundamentally wrong distance mapping that it will carry permanently. At 200 m the same ±5 m error is only 2.5%, which is acceptable. So label accuracy matters most at close range, where you need the most training data.

Vertical accuracy is even worse. If the drone hovers at 40 m altitude and 40 m horizontal distance, the real 3D distance is $\sqrt{40^2 + 38.5^2} \approx 55$ m — but standard GPS might report horizontal as 38 m and vertical as 30 m, giving a computed distance of 48 m: a 7 m error at only 55 m range. A barometer solves the vertical problem cheaply; RTK solves both.

#### RTK GPS Setup (Recommended for Sample Collection)

RTK (Real-Time Kinematic) GPS uses a stationary base station at a known surveyed location to continuously measure GPS signal errors and broadcast corrections to mobile rover units. At ranges under 300 m (your entire recording area), this achieves **±2–5 cm horizontal and ±2–3 cm vertical** — effectively eliminating label noise from your training data.

**Hardware needed:**
- **1× RTK rover (ZED-F9P)** — ArduSimple simpleRTK2B Budget board (~€160) + TAN1216Q39 quadrifilar helix antenna (~$15–25) + SMA male-to-male patch cable (~$2–3). The same unit serves both the mic position measurement and the drone during flight — see workflow below.
- **RTK2Go as base station (free, no hardware)** — connect to the RTK2Go free NTRIP caster (rtk2go.com, port 2101) from a laptop during recording sessions. RTK2Go aggregates corrections from volunteer base stations worldwide. Find the nearest active station at rtk2go.com before your first session. Requires internet — a phone hotspot works. NTRIP client: `str2str` from RTKLIB (free, cross-platform).
- **No radio correction link needed** — corrections travel via internet (NTRIP) to the laptop, which forwards them to the ZED-F9P over USB. The radio correction link in the original plan was for a local base station and is no longer needed.
- **No second rover needed** — the mic is stationary, so its position only needs to be recorded once per session. The same ZED-F9P then moves to the drone for all flight recordings.

**Setup per session:**
1. Connect laptop to phone hotspot. Open `str2str` NTRIP client, connect to rtk2go.com:2101, select your nearest active base station, and forward corrections to the ZED-F9P rover over USB.
2. Place ZED-F9P rover next to the mic tripod. Wait for RTK Fixed status (status string "Fix" — not "Float"). Average the position for 2–5 minutes and record `mic_lat`, `mic_lon`, `mic_alt` — this is the mic's ground-truth position to ±2–5 cm. Store it before each session.
3. Move ZED-F9P from mic tripod to drone. Verify RTK Fixed status on the drone before any flight. The drone now logs its position continuously during all flights.
4. Mount a **MEMS barometer** (BMP580/BMP581) on the mic unit for the deployed capsule use case. For sample collection distance labels, RTK altitude (±2–3 cm) is far more accurate than barometric height and is the primary height source — the barometer is a cross-check only.

**Why barometer alongside RTK:** RTK vertical accuracy is ±2–3 cm, which is already excellent. However the barometer gives an independent height reading useful for detecting GPS anomalies (e.g., multipath error near trees causing a sudden jump in RTK vertical). If RTK height and barometer height disagree by more than 2 m, flag that clip as potentially mislabeled and inspect manually.

#### Logger

Write a logger that captures per-second:
```
timestamp | rtk_mic_lat | rtk_mic_lon | rtk_mic_alt | baro_mic_h |
            rtk_drone_lat | rtk_drone_lon | rtk_drone_alt | baro_drone_h |
            rtk_fix_status | audio_file_path
```

Compute 3D distance offline using horizontal RTK positions + barometric height difference:
$$d_{3D} = \sqrt{d_{horizontal}^2 + \Delta h_{baro}^2}$$

Discard clips where `rtk_fix_status ≠ Fixed` — those have degraded accuracy and should not become training labels.

**Checkpoint:** Hover drone at 4–5 measured distances (10 m, 30 m, 60 m, 100 m). RTK-derived distance must match a tape measure or laser rangefinder to within ±10 cm. Do not proceed until this works reliably.

### Phase B2 — Dataset Collection

**Goal:** ~1,500+ labeled clips across all classes from your own recordings.

Structure your dataset in two explicit sets:
- **Clean set** — quiet open field, low wind, no traffic. Teaches the model the pure acoustic signature.
- **Noisy set** — same distances and drone types, with real-world background noise (wind, cars, people, insects). What the model will face in deployment.

Both are required. A model trained only on clean audio will fail in the field.

| Class | Description | Min clips (clean + noisy each) |
|---|---|---|
| Drone — DJI Mavic (or similar consumer quad) | Hovering, approaching, receding, passing overhead | 200+ |
| Drone — fixed-wing FPV | If accessible | 100+ |
| Drone — heavy-lift hex/octo | If accessible | 100+ |
| Not drone — helicopter | Critical — similar harmonic structure; a model without this will fail in military contexts | 150+ |
| Not drone — wind | Strong wind, light wind, gusts | 150+ |
| Not drone — vehicle | Cars, motorbikes, trucks passing | 150+ |
| Not drone — insects | Bees, flies, mosquitoes ([BPF](#bpf) ~600 Hz overlaps drone harmonics) | 100+ |
| Not drone — birds | Flapping, wing-beats — hardest confusor in open-field environments | 100+ |
| Not drone — silence / ambient | Open field with no notable sounds | 100+ |

**Recording protocol:**
- Same microphone with deadcat fitted, 1.5 m above ground on a tripod
- Record drone at distances up to 300 m — literature confirms this range is achievable
- Record at many distances spread across the range, not just a few fixed points (e.g., 10, 20, 30, 50, 75, 100, 150, 200, 250, 300 m). The regression model needs a continuous spread to learn the distance-to-audio mapping — gaps in coverage create gaps in accuracy.
- Label each clip with **both continuous and categorical metadata:**
  - `distance_m` — RTK-derived 3D distance in meters: $\sqrt{d_{horizontal\_RTK}^2 + \Delta h_{baro}^2}$ (primary training label for regression)
  - `mic_type` — `omni_mems` or `parabolic` **(model input — not just a label; see Core Decisions)**. Encode as integer: 0 = MEMS omnidirectional, 1 = parabolic.
  - `dish_diameter_cm` — **(model input)** aperture diameter of the parabolic dish in centimetres; `0.0` for MEMS. Re-measure if switching to a different dish.
  - `dish_depth_cm` — **(model input)** physical depth of the bowl from rim plane to vertex in centimetres; `0.0` for MEMS. Determines focal length and acceptance cone width.
  - `dish_material` — **(model input)** encode as integer: 0 = none (MEMS), 1 = metal, 2 = mesh, 3 = plastic/fiberglass. Controls reflectivity and transmission loss profile across frequencies.
  - `dish_wall_thickness_mm` — **(model input)** wall thickness of the dish material in millimetres; `0.0` for MEMS. Thin walls flex and resonate at specific frequencies; thick or rigid walls reflect cleanly. Measure with a calliper — estimate is fine for POC.
  - `drone_type`, `behavior` (hover/approach/pass)
  - `environment` (open_field / urban / forest)
  - `wind_speed` (none / light / moderate / strong — estimate or measure with a phone anemometer app)
  - `noise_sources` (none / traffic / insects / people / machinery)
  - `time_of_day` (morning / afternoon / evening)
- **`mic_type` is the only label that is also a model input.** All other metadata labels are for dataset analysis and debugging only — if the model fails in a specific condition (e.g., always wrong in high wind), these labels tell you exactly what's missing from your dataset so you can go record more of that condition.
- Clip length: 2–5 seconds each
- Record at multiple times of day (morning ambient noise ≠ afternoon)
- **Doppler effect:** When recording approaching/receding passes, pitch shifts higher on approach and lower on recession. This is real data — record many passes so the model learns that a pitch-shifted drone is still a drone.

**Data augmentation (applied during training, not recording):**
- Add synthetic background noise at varying SNR levels
- Time-stretch (simulates RPM variation)
- Random volume scaling (simulates distance variation)
- GAN-based synthesis if data is scarce — validated in published drone detection research
- Do NOT augment distance labels

**Checkpoint:** ~1,500 clips stored at `data/raw/{class_name}/{session_id}_{timestamp}.wav` with a matching `data/labels.csv`. All clips have `rtk_fix_status = Fixed`. Distance labels verified against tape measure at 5 reference distances.

### Phase B3 — Audio Preprocessing Pipeline

**Goal:** Convert raw .wav files into model-ready feature tensors.

1. **Load audio:** Read .wav, resample to 16 kHz if needed.
2. **Band-pass filter:** High-pass at 50 Hz (captures shaft rate at ~64 Hz), low-pass at 10 kHz. Use `scipy.signal.butter`.
3. **Normalize amplitude:** Scale to consistent RMS level. Removes mic gain variation between sessions.
4. **Encode mic hardware feature vector:** Read the five mic hardware labels from the CSV and build a single float tensor `[mic_type, dish_diameter_cm, dish_depth_cm, dish_material, dish_wall_thickness_mm]`. Normalize the continuous values (`dish_diameter_cm`, `dish_depth_cm`, `dish_wall_thickness_mm`) to z-scores using training-set mean and std — this puts them on the same scale as the audio embedding dimensions. Keep the integer categoricals (`mic_type`, `dish_material`) as-is without normalization. This 5-element vector travels alongside the audio tensor through the entire pipeline and is concatenated to the 1024-dim YAMNet embedding before the distance heads (see Phase B4 architecture). Do not augment any of these values — they are fixed hardware properties for the recording session, not signal characteristics.
5. **Segment:** 1–2 second windows, 50% overlap. Each segment inherits its parent recording's label including `mic_type`. This window size is the single largest contributor to end-to-end detection latency — see [End-to-End Latency Breakdown](#end-to-end-latency-breakdown).
6. **Compute features — use both:**
   - **[Mel Spectrogram](#mel-spectrogram):** `n_fft=1024`, `hop_length=512`, `n_mels=128`, converted to dB. Output: `(128 × time_frames)`.
   - **[MFCC](#mfcc):** 40 coefficients, 240 ms window. Output: `(40 × time_frames)`.
   - Concatenate along the feature axis → fused input tensor.
7. **Save** to `data/processed/` as `.pt` files with updated labels CSV. Each `.pt` file stores a tuple of `(audio_tensor, mic_type_int, distance_m, drone_class)`.
8. **Split by session:** 70% train / 15% validation / 15% test. Split by recording session, not by clip — splitting by clip leaks information since clips from the same session sound nearly identical. Verify the split contains both `mic_type` values in train, val, and test — if all parabolic recordings land in one split, the model cannot generalize.

**Checkpoint:** Run the pipeline on 10 clips. Inspect spectrograms: a hovering drone should show clear horizontal harmonic bands. A wind clip should show mostly low-frequency uniform energy.

### Phase B4 — Model Training

**Goal:** A model that classifies drone type and estimates continuous distance in meters, with a bracket confidence check.

#### Why Continuous Regression, Not Brackets

Your GPS labels are already precise values — 73.4 m, 142.8 m, 211.0 m. Bucketing these into "50–150 m" throws away real information that was expensive to collect. A regression model uses the full precision of those labels to learn a fine-grained mapping between audio features and distance.

The output change is architecturally simple: instead of a softmax over 4 buckets, the distance head outputs a single number (meters). The tradeoff is that regression models have no built-in way to express uncertainty — a softmax can be "not confident," but a regression output is always a specific number. The solution is to run **two distance outputs in parallel**: a regression head for the best estimate, and a bracket head as a sanity check. If they disagree strongly (e.g., regression says 40 m but bracket says ">300 m"), the prediction is flagged as low-confidence and weighted down in triangulation.

**Expected accuracy per range:**

| Range | Expected error |
|---|---|
| 0–50 m | ±5–10 m |
| 50–150 m | ±15–30 m |
| 150–300 m | ±30–60 m |
| >300 m | Unreliable — beyond detection limits |

These limits come from physical factors the model cannot overcome: unknown drone throttle level, wind, drone orientation, and atmospheric absorption — not from model architecture. Brackets cannot do better on these same inputs; they just hide the imprecision by rounding.

For triangulation: three mics each with ±20 m estimates produce an intersection area roughly 20–40 m in radius. Compare this to bracket-based triangulation which produces a zone hundreds of meters wide. The regression approach is significantly better for your use case.

#### Architecture

[YAMNet](#yamnet) [backbone](#model-backbone) ([fine-tuned](#fine-tuning) from AudioSet pretraining), with three output [heads](#model-backbone):

```
Input A: Fused Mel Spectrogram + MFCC tensor
  → YAMNet backbone (pretrained on AudioSet, fine-tune all layers)
  → 1024-dim audio embedding
  │
  ├── Head 1: Dense(256) → ReLU → Dense(N_classes) → Softmax    [classification: drone type]
  │     └── uses audio embedding only — mic hardware does not affect drone classification
  │
Input B: mic hardware vector [mic_type, dish_diameter_cm, dish_depth_cm, dish_material, dish_wall_thickness_mm]
  → normalized continuous values (diameter, depth, thickness) z-scored to training-set stats
  → concatenated to audio embedding → 1029-dim combined vector
  │
  ├── Head 2: CNN-BiLSTM → Dense(256) → ReLU → Linear(1)        [distance regression: meters]
  └── Head 3: Dense(256) → ReLU → Dense(4) → Softmax            [distance bracket: confidence check]
```

Head 2 and Head 3 both predict distance from the combined 1029-dim vector. Head 1 (classification) uses only the 1024-dim audio embedding — mic hardware is irrelevant to whether the sound is a drone.

**Why mic hardware goes into distance heads only:** The parabolic dish applies frequency-dependent gain that varies with diameter, depth, material, and wall thickness — the same drone at the same distance sounds measurably different through each hardware combination. The distance head must know the hardware to interpret loudness correctly. The classification head is unaffected because harmonic structure (the classification feature) is preserved by the dish; only amplitude and frequency balance change.

The [CNN-BiLSTM](#cnn-bilstm) in Head 2 adds temporal reasoning — it reads how the signal evolves across multiple frames (getting louder = drone approaching, fading = receding). This temporal pattern is a strong distance cue that a static single-frame [CNN](#cnn) misses entirely.

**If you have limited data (<500 clips):** train an [SVM](#svm) on [MFCC](#mfcc) features first as a baseline. This validates your pipeline in minutes before committing to deep learning.

#### Training order

1. **Binary classifier only (drone vs not-drone):** achieve >90% F1 before moving on. Use class-weighted loss if your not-drone class is larger. `mic_type` is not yet wired in at this stage — the classifier doesn't use it.
2. **Add drone type classification:** expand to multi-class. Evaluate per-class precision/recall. Still no `mic_type` input needed.
3. **Add distance heads with mic_type:** train both Head 2 (regression) and Head 3 (bracket) simultaneously, now with the 1025-dim combined vector (audio embedding + `mic_type` scalar). Loss: `total_loss = classification_loss + λ₁ × huber_regression + λ₂ × bracket_crossentropy`. Use [Huber Loss](#huber-loss) instead of plain [MAE](#mae) — it is less sensitive to outlier clips where GPS had a bad fix. Start with λ₁=0.5, λ₂=0.3. **Verify that `mic_type` is making a difference:** after training, compare distance MAE on MEMS-only clips vs parabolic-only clips — if they're similar, the head is correctly learning separate distance mappings for each mic type.
4. **[Quantize](#model-quantization):** apply INT8 quantization once validated. Reduces inference time ~30% and energy ~45%. The `mic_type` scalar input is unaffected by quantization — it is passed as a full integer at inference time.

#### Key metrics
- Classification: Precision, Recall, F1 (prioritize Recall — a missed drone is worse than a false alarm)
- Distance regression: [MAE](#mae) in meters per range bracket; improvement over Track A intensity proxy is the success signal
- Distance confidence: disagreement rate between Head 2 and Head 3 (high disagreement = low-quality training data in that range)
- Latency: <100 ms on target hardware (Raspberry Pi 5 with INT8 quantization achieves ~5 ms)

**Checkpoint:** Binary classifier >90% recall on held-out test set. Distance regression [MAE](#mae) <20 m in the 0–150 m range. Head 2 and Head 3 agree on the correct bracket >80% of the time.

### Phase B5 — Field Validation

**Goal:** Test the model against drone audio it has never seen.

1. New recording session (different location, time, and drone if possible) — true out-of-distribution test.
2. Pipe audio into the inference script via 2-second sliding window.
3. Log all predictions with GPS distance ground truth.
4. Measure: at what distance does detection reliability drop off?
5. Identify failure modes — add them to training data and retrain.

**Checkpoint:** Model detects a hovering drone within 3 seconds, at distances up to 200 m (300–500 m achievable per literature with YAMNet), with <20% false positive rate.

---

## Phase 5 — Multi-Microphone & Triangulation

Applies to both tracks, with different precision levels.

### Deployed Mic Positioning

For triangulation to work, each mic capsule must know its own position and height accurately. This is a different problem from sample collection — you can't use a base station and rover setup during live deployment because the mics are dropped autonomously into a field.

#### Horizontal Position — Standard GPS (Sufficient for POC)

Standard consumer GPS (±3–5 m horizontal) is acceptable for deployed mic positioning. The triangulation target accuracy is ±20–40 m, so a ±5 m error in each mic's position shifts the result by ±5 m — within the acceptable margin. Each capsule records its GPS position upon landing and reports it to the command center.

#### Height — MEMS Barometer (Required)

Standard GPS vertical accuracy (±10–15 m) is too coarse for 3D triangulation — a mic's height error would dominate the geometry. A MEMS barometer (BMP580/BMP581 or MS5611, <$10, <1g) on each capsule provides **±0.5–1 m height accuracy**. On landing, the capsule takes a barometric altitude reading and reports it alongside its GPS position.

Barometers measure absolute atmospheric pressure, which drifts with weather. The solution: the command center also runs a barometer and broadcasts its reading to all capsules. Each capsule computes height relative to command center, removing weather drift. This requires only a one-way data link from command center to capsules — something you need anyway for coordination.

#### RTK for Deployed Mics — Enhanced Positioning Option

For higher triangulation precision, deployed mics can receive RTK corrections from a base station at the command center. This gives **±2–5 cm horizontal and ±2–3 cm vertical**, which combined with accurate distance estimates produces triangulation precision in the low single-digit meters.

**Option 1 — Within ~20 km of command center (direct RTK):**
The command center runs an RTK base station (u-blox F9P). Each capsule carries an RTK rover module and a radio receiver. The base continuously broadcasts corrections via radio (LoRa or RFD900 — range up to 40 km line-of-sight). The capsule applies corrections in real time. This is the simplest setup and works well within the 20 km range where ionospheric errors are highly correlated between base and rover.

**Option 2 — Beyond 20 km (Network RTK / NTRIP):**
Beyond ~20 km, the ionospheric errors at the rover location start to differ meaningfully from those at the base — correction quality degrades toward ±10–20 cm. The solution is Network RTK (NTRIP protocol): instead of one base station, a network of stations interpolates a "virtual" correction for any rover position. Each capsule connects to the NTRIP network via cellular modem, receives corrections computed specifically for its location, and achieves ±2–5 cm regardless of how far it is from any individual station. Requires cellular coverage in the deployment area.

**Option 3 — RTK relay (daisy-chaining corrections):**
If cellular is unavailable and some capsules are beyond 20 km, an intermediate capsule within 20 km can relay the base station's correction stream to farther capsules via radio. The relay capsule receives corrections from the command center, forwards them over a second radio link to distant capsules. Accuracy at the distant capsule degrades slightly compared to being within 20 km (roughly ±5–20 cm vs ±2–5 cm), because the corrections computed for a 20 km baseline are slightly wrong for a 40 km baseline — but still far better than standard GPS. This is called an **RTK repeater** configuration.

| Scenario | Method | Horizontal accuracy | Requires |
|---|---|---|---|
| POC, short range | Standard GPS | ±3–5 m | Nothing extra |
| Within ~20 km of command center | Direct RTK via radio | ±2–5 cm | RTK module + radio on each capsule |
| Beyond 20 km, cellular available | Network RTK (NTRIP) | ±2–5 cm | Cellular modem on each capsule |
| Beyond 20 km, no cellular | RTK relay via intermediate capsule | ±5–20 cm | Radio link between capsules |

**For the POC:** use standard GPS + barometer. RTK for deployed mics is a post-POC enhancement — add it when triangulation precision needs to improve beyond what ±5 m positioning allows.

### Detection & Triangulation

- Each microphone runs its own inference pipeline independently and reports to a central coordinator:
  - Track A: `{mic_id, gps_lat, gps_lon, baro_height_m, detected: bool, distance_bracket, timestamp}`
  - Track B: `{mic_id, gps_lat, gps_lon, baro_height_m, detected: bool, distance_m, distance_confidence, distance_bracket, timestamp}`
- **Triangulation (Track A — bracket-based):** if ≥2 mics detect simultaneously, compute intersection of distance circles using bracket midpoints. Gives a rough area hundreds of meters wide — sufficient to prove the concept.
- **Triangulation (Track B — regression-based):** use the continuous `distance_m` value from each mic as the circle radius. With ±20 m model accuracy per mic, the intersection area shrinks to roughly 20–40 m radius — precise enough to meaningfully guide a response. Weight each mic's contribution by its `distance_confidence` (low-confidence readings expand their circle radius to reflect uncertainty).
- **[TDOA](#tdoa) / [GCC-PHAT](#gcc-phat) (Track B, post-POC):** more precise — uses the difference in arrival time of the same sound across microphones to locate the source on a hyperbola. Multiple mic pairs give a precise intersection.
  - Requires clock synchronization between mics (GPS PPS signal or NTP <1 ms accuracy)
  - **Inter-mic spacing:** 15 cm for unambiguous detection up to ~1.1 kHz, derived from $f_{max} = c / (2d)$ where $c = 343$ m/s. For 3D localization, span 2–3 m in x/y and 0.5 m in z.
  - Published results: ~2 cm position error in ideal conditions
- **Two-model parallel detection:** run Track A and Track B models simultaneously on the same audio stream. Flag if either fires (OR logic). Different architectures have different failure modes — OR combination maximizes recall at acceptable false positive cost.
- **Multi-drone:** two drones merge into one estimated region. Handle post-POC via source separation (NMF or deep learning-based) or training on multi-drone recordings.

---

## Phase 6 — Directional Detection & Long-Range Architecture

This section covers hardware approaches for extending detection range beyond what omnidirectional MEMS arrays achieve, and for obtaining a bearing when only one microphone detects a drone. Both problems have the same solution: a directional acoustic module that pairs with the Phase 5 array.

### Why This Matters

Phase 5 triangulation requires ≥2 mics to detect simultaneously. At the edge of detection range (150–300 m), one mic may pick up a faint signal while others miss it entirely — leaving you with "detected, but can't triangulate." A directional module solves both problems at once: it extends range and gives a bearing even from a single detection point.

---

### Option 1: Spinning Microphone Array

A rotating arm carrying microphones (and optionally a camera) that sweeps azimuth continuously until it detects a drone, then stops and locks onto the bearing.

**Feasibility: Medium — mechanically possible, documented in robotics literature, but complex.**

- Self-rotating bi-microphone arrays for acoustic source localization exist in published robotics research (used on ground robots for sound-following).
- The scan-then-lock logic is sound: spin slowly (~1 rev/sec), monitor classification output, stop on detection trigger and report the heading at that moment.
- **Doppler concern:** At 0.3 m arm radius and 1 rev/sec, tangential velocity is ~1.9 m/s → ~0.5% frequency shift. Small but present. Solved in published literature via the **Virtual Rotating Array (VRA)** technique, which mathematically compensates for rotation-induced frequency shift in post-processing. This is a known solved problem.
- **Key weakness while spinning:** direction-of-arrival estimates are unreliable during rotation because the array geometry relative to the source changes continuously. You only get a trustworthy bearing after the array stops. For a slowly arriving threat this is acceptable; for a fast-moving drone it may miss the window.
- **Camera integration:** practical. Mount a camera co-axially with the array on the same rotating arm — once it stops on a bearing, the camera points at the detection direction. This is done in commercial systems.
- **Mechanical overhead:** motor, slip rings for power and data across the rotating joint, dynamic balancing. Each is solvable but adds failure surface.
- **Verdict:** feasible as a secondary confirmation module. Not recommended as the primary detection layer due to the mechanical complexity and the bearing ambiguity during rotation.

---

### Option 2: Parabolic Microphone

A dish reflector that concentrates sound from a narrow cone onto a microphone at its focal point.

**Feasibility: High — well-proven, commercially available, documented at very long range.**

Key properties:
- Gain is frequency-dependent. A 22-inch dish provides ~0 dB gain at 200 Hz but up to **35 dB gain at 10 kHz+**. Drone propeller fundamentals (128–205 Hz) receive little dish gain, but harmonics at 500–2000 Hz benefit significantly — and harmonic structure is the primary classification feature, so this works in your favor.
- Usable beam width: roughly **5–15°** depending on dish diameter and frequency. The parabolic mic must already be pointed toward the general direction of the drone.
- Reported detection ranges: 300–500 m for consumer-grade parabolic setups; military-grade systems (Zvook parabolic concentrators) reported 5 km for Shahed drones and 7 km for cruise missiles.
- **Passive noise rejection near personnel:** the dish is inherently directional. Pointed skyward with a sound-blocking baffle on the rear face, ground-level noise (machinery, vehicles) arrives well outside the acceptance cone and is attenuated passively — no active noise cancellation required. This makes parabolic mics viable near noisy personnel areas where omnidirectional MEMS would saturate.
- **Primary limitation:** useless for wide-area scanning. Requires a coarse bearing first (from the Phase 5 array) before pointing it.

---

### Option 3: Array of Parabolic Microphones (Recommended)

Multiple parabolic elements arranged in a fixed array, using electronic beamforming rather than physical rotation to steer the listening direction.

**Feasibility: High — this is the architecture used by serious commercial and military acoustic camera systems.**

- Each element contributes its individual parabolic dish gain.
- Beamforming across the array adds spatial selectivity on top of the per-element gain — the combined directivity is multiplicative.
- **Electronic beam steering** is instantaneous and fully reliable: no motors, no Doppler artifact, no mechanical failure modes. The beam can be redirected in microseconds.
- Can form **multiple simultaneous beams** in different directions, enabling sector-wide monitoring without any physical movement.
- A circular or cross arrangement of 4–8 parabolic elements provides full 360° azimuth coverage with electronic steering.
- The YAMNet classification model is unchanged — you are improving input SNR, not the model architecture.

---

### Recommended Two-Layer Architecture

This is the most practical balance of range, reliability, and noise rejection for this project:

**Layer 1 — Primary Detection (Phase 5 system, unchanged):**
Fixed circular array of 6–8 omnidirectional ICS-43434 MEMS microphones. Wide-field detection, coarse direction-of-arrival via TDOA / beamforming. No moving parts. Detection range: 200–500 m depending on environment and drone type. Outputs: `{detected, bearing_coarse, distance_m, distance_confidence}`.

**Layer 2 — Directional Confirmation & Extended Range:**
1–2 parabolic microphones mounted on a pan-tilt unit. Pointed by Layer 1's coarse bearing on detection. A camera (optical or thermal) co-mounted on the same pan-tilt for visual confirmation. Adds:
- Extended range (300–500 m+ for the same drone, due to dish gain at harmonic frequencies)
- Narrow-beam spatial noise rejection when aimed skyward
- Visual / thermal confirmation for operator situational awareness

**Data flow:**
```
Layer 1 (omni MEMS array) detects drone
  → reports coarse bearing + distance estimate to coordinator
  → coordinator commands Layer 2 pan-tilt to point to that bearing
Layer 2 (parabolic + camera) points and runs YAMNet on focused signal
  → classification on high-SNR audio → confident detection or rejection
  → camera captures visual for operator confirmation
  → if confirmed: Layer 2 bearing + Layer 1 triangulation → refined position
```

---

### Noise Rejection Near Personnel

Ground-level machinery and personnel sound sources arrive from the horizontal plane. A skyward-pointing directional array rejects these through three stacked mechanisms:

1. **Beam spatial filter (electronic):** beamforming attenuates signals arriving from outside the steering direction. Ground-level noise arrives at a large off-axis angle relative to a sky-pointing beam — it is suppressed electronically.
2. **Physical baffle (passive):** dense sound-blocking material mounted on the rear face of the array prevents rear-hemisphere noise from diffracting around to the microphones. Standard practice in outdoor acoustic camera systems. See **Acoustic Backing Materials** subsection below for options.
3. **Frequency-domain notch filter:** machinery harmonics at known fixed frequencies (e.g., diesel generator at 50/100/150 Hz) can be notch-filtered in software without significantly affecting drone harmonic detection above 200 Hz.

**Practical ceiling:** broadband machinery noise still partially overlaps the drone harmonic band (100–1000 Hz). At very high noise levels, raise the detection confidence threshold — this trades some recall for false-positive reduction. The directional approach substantially reduces this problem; it does not fully eliminate it at extreme noise levels. If the system must operate at very close range to loud machinery, physical separation of even 20–30 m makes a significant difference.

#### Acoustic Backing Materials

For environments with intense background noise — loud machinery, construction equipment (CATs), armored vehicles, tanks — backing the microphone array or parabolic dish with sound-blocking material substantially improves SNR. The goal is to stop rear-hemisphere noise from diffracting around the array or dish and reaching the microphone capsule.

**Option A — Mass-Loaded Vinyl (MLV) + Rockwool (recommended for stationary installations)**

MLV and rockwool are complementary: MLV is a dense, limp barrier that blocks low-frequency sound transmission (the primary threat from engines, diesel machinery, and armored vehicles — 50–300 Hz); rockwool (mineral wool) absorbs mid-to-high frequency energy (200 Hz – several kHz). Used alone, MLV transmits rather than absorbs; rockwool alone lets low frequencies pass. Layered together — rockwool layer against the array back, MLV layer on the outside — they provide broadband attenuation across the full drone harmonic band (50 Hz – 2 kHz).

- **MLV spec:** 2 lb/ft² (3–4 mm thick) achieves STC 31 on its own. Use mass-loaded vinyl, not standard acoustic foam — foam absorbs, MLV blocks.
- **Rockwool spec:** 50–100 mm thickness, high-density (60–100 kg/m³). Thicker is better for low-frequency absorption but adds weight.
- **Application for flat arrays:** cut MLV and rockwool panels to cover the full rear face. Attach rockwool directly to the back plate with adhesive, then MLV over it, then weatherproof cover (neoprene sheet or outdoor-rated fabric). For parabolic dishes, wrap the rear (non-reflecting) face of the dish with the same sandwich — the reflective front face must remain clear.
- **Weight:** ~1 lb/ft² for MLV + ~0.5 lb/ft² for 50 mm rockwool = ~1.5 lb/ft² total. A 0.5 m × 0.5 m backing panel adds ~0.35 kg (~0.8 lb) — manageable for stationary command-post units, heavy for field-deployed capsules.

**Option B — Solid rigid plate + acoustic foam (lightweight alternative for mobile capsules)**

A solid plate (steel, aluminium, or hard plastic) provides broadband reflection loss; a thin layer of closed-cell acoustic foam on the interior face absorbs the small fraction that transmits. This is lighter and more ruggedized than Option A but provides less low-frequency attenuation (steel plate STC ~25 vs MLV+rockwool ~35+).

Use this for field-deployed capsules where weight and packaging size matter. The directional beam pattern already rejects most ground-level noise electronically — the rigid plate handles the remainder.

**Option C — Active noise cancellation (post-POC, extreme environments)**

For operations at very close range to continuous high-intensity noise sources (idling tanks, active generators), passive materials alone may be insufficient. Add a reference microphone on the rear face of the array (pointed away from the sky, toward the noise source), and run a digital ANC algorithm that subtracts the reference signal from the main mics. The U.S. Navy has funded ANC development specifically for boom microphone applications in high-noise environments. This adds compute and tuning complexity — save for post-POC if passive options prove insufficient.

| Option | Best for | Low-freq attenuation | Weight | Complexity |
|---|---|---|---|---|
| MLV + Rockwool | Fixed command-post arrays, parabolic dish rear faces | High (STC 35+) | ~1.5 lb/ft² | Low |
| Rigid plate + foam | Field-deployed capsules, mobile setups | Medium (STC ~25) | ~0.5–1 lb/ft² | Very low |
| Active noise cancellation | Extreme noise, very close to source | Very high (30–40 dB broadband) | Minimal hardware, high compute | High |

**Placement notes:**
- On **flat MEMS arrays**: cover the full rear face. Leave the forward-facing microphone apertures completely unobstructed.
- On **parabolic dishes**: wrap the non-reflecting rear hemisphere of the dish only — never cover the reflective bowl or focal-point area. For the parabolic mic in the Layer 2 pan-tilt unit, attach backing material to the dish back plate and the pan-tilt arm structure behind the focal point.
- On **omnidirectional MEMS capsules (Phase 5 ground deployment)**: a rigid backing disc underneath the capsule, or a short cylindrical shroud with backing material on the bottom and sides below the horizontal plane, prevents upward-reflected ground noise from entering the microphone from below.

---

### Fine-Tuning for Directional Input

The existing YAMNet classification model (Track A and Track B) does not need to be retrained for directional use. The model classifies the audio signal regardless of how it was captured. Using a parabolic dish simply increases the SNR of the input — the model sees a cleaner version of the same acoustic signature it was trained on. If anything, classification accuracy improves because the drone signal dominates more clearly over background noise.

The only addition worth considering post-POC: record a subset of Track B training data through the parabolic dish and include it in fine-tuning. This ensures the model has seen the slightly different frequency response of dish-captured audio (parabolic gain varies with frequency, which mildly shapes the spectrogram). This is optional — YAMNet's robustness to frequency response variation typically handles it without explicit training.

---

### Hardware Additions for Layer 2

All items below are **post-POC additions** — build and validate the Phase 5 triangulation system first.

| Item | Purpose | Approx. cost |
|---|---|---|
| Pan-tilt unit (servo-driven, 360° azimuth × 90° elevation, outdoor-rated) | Points the parabolic mic and camera to the detected bearing | $50–100 (Servo City kit or PTZ camera mount) |
| Parabolic dish (18–24 inch) + ICS-43434 at focal point | Dish gain for extended range; directional noise rejection | $30–80 DIY (mesh satellite dish + 3D-printed focal mount) or $200–400 commercial (Wildtronics, Telinga) |
| Camera — Raspberry Pi Camera Module 3 Wide (CSI) or USB equivalent | Visual confirmation for operator | $25–50 |
| LIS3MDL compass (already in section 10 of shopping list) | Converts pan-tilt heading to geographic bearing | $5–8 |
| Servo controller (PCA9685 I²C PWM driver) | Drives pan-tilt servos from Raspberry Pi | $5–10 |

**Focal point placement:** the ICS-43434 must be positioned at the parabolic focal point. For a standard offset-feed satellite dish, the focal length f = D²/(16c) where D is dish diameter and c is dish depth. Measure your specific dish — focal length varies. A 3D-printed arm positions the mic precisely; small misalignment significantly degrades gain.

---

## End-to-End Latency Breakdown

The ~5 ms figure cited in the [Embedded Deployment — Latency Reality Check](#embedded-deployment--latency-reality-check) is **model inference only**. End-to-end detection latency — from drone entering detectable range to alert — is dominated by the audio capture window, not the model.

### Full Pipeline

| Stage | Estimated time | Notes |
|---|---|---|
| **1. Audio capture window** | 1,000–2,000 ms | The single largest contributor — audio must be fully buffered before any processing begins |
| **2. Audio preprocessing** | 10–20 ms | Resample → band-pass filter → normalize → [Mel Spectrogram](#mel-spectrogram) + [MFCC](#mfcc) fusion |
| **3. Model inference** | ~5 ms (RPi5, INT8) | What the latency table in Research Findings refers to |
| **4. Post-processing** | <1 ms | Bracket comparison, confidence check |
| **5. Radio transmission** | 1–50 ms | WiFi or serial radio hop to coordinator (Architecture B only; eliminated in Architecture A) |

### Why the Audio Window Dominates

Drone [BPF](#bpf) harmonics at 128–205 Hz need enough cycles to form visible structure in a [Mel Spectrogram](#mel-spectrogram). This sets a hard physics floor on window size:

- **500 ms** — aggressive minimum; enough to see 2–3 BPF cycles, but increases misclassification rate.
- **1 second** — practical minimum for reliable classification.
- **2 seconds** — what published literature uses; gives the model the most signal.

With 50% overlap (the Phase B3 default), a new inference fires every 1 second. Worst-case detection latency after a drone enters detectable range is therefore ~1 window fill time + preprocessing + inference.

**End-to-end estimates:**

| Configuration | Worst-case detection latency |
|---|---|
| 500 ms window, 75% overlap, edge inference (Architecture A) | ~750 ms |
| Plan default: 1–2 s window, 50% overlap | ~1.5–3 seconds |
| Architecture B worst case: 2 s window + WiFi hop | ~2.5–4 seconds |

### Minimizing Latency

| Technique | Effect | Notes |
|---|---|---|
| Shorten window to 500 ms–1 s | Cuts detection latency 50–75% | Biggest available win; trades some classification accuracy |
| Increase overlap to 75% | New inference every 250 ms instead of 1 s | Reduces worst-case latency with no hardware change needed |
| INT8 quantization | ~30% faster inference, ~45% less energy | Covered in [Phase B4](#phase-b4--model-training) |
| Edge inference (Architecture A) | Eliminates radio transmission hop | No WiFi/serial delay on the detection path |
| VAD gating (optional) | Skips preprocessing on silent frames | ESP32-S3 runs a simple energy threshold before streaming to the Pi; reduces average latency and radio bandwidth without changing worst-case. Only useful in Architecture B where audio is streamed rather than processed locally |

### Training Window vs Inference Window

**The short answer: train on the same window size you plan to use at inference — especially for the distance heads.**

**Why it matters:**

The mel spectrogram fed to the CNN backbone has shape `(128 mel bins × T time frames)`. The number of time frames T scales directly with window length: a 2 s window at hop_length=512 gives ~62 frames; a 500 ms window gives ~15 frames. A CNN trained on `(128×62)` spectrograms cannot directly process `(128×15)` spectrograms — the spatial shape is different and the learned convolutional filters don't map across.

**YAMNet specifically:** YAMNet's internal fixed frame is 960 ms. If you pass a shorter clip (e.g. 500 ms), the TensorFlow Hub implementation zero-pads the remaining ~460 ms with silence. This works for inference without retraining — the embedding is still produced — but the model was never trained on padded inputs, so the embedding quality degrades and classification accuracy drops noticeably (~5–15% in practice). The classification head tolerates this degradation reasonably well because harmonic structure is still visible in 500 ms. The BiLSTM distance head does not tolerate it — the temporal patterns it learned (signal building over 1.5 s = approaching drone) are simply absent in 500 ms padded clips.

**Per-head summary:**

| Head | Sensitive to window size mismatch? | Why |
|---|---|---|
| Head 1 — classification (drone type) | Tolerates mismatch with small accuracy cost | Harmonic structure is visible even in 500 ms; zero-padding adds silence, not noise |
| Head 2 — distance regression (BiLSTM) | Sensitive — retrain required for a different window | BiLSTM learns temporal patterns (approach/fade over multiple seconds); these patterns do not exist in short or padded clips |
| Head 3 — distance bracket (sanity check) | Moderately sensitive | Same temporal dependency as Head 2 but less precise, so degradation is less catastrophic |

**Recommendation — decide before collecting training data:**

Choose your target inference window size first, then collect training data and train on that exact window. The options:

| Window | Detection latency | Classification accuracy | Distance accuracy | Notes |
|---|---|---|---|---|
| **500 ms** | ~750 ms worst-case | ~85–90% of 2 s baseline | Reduced — BiLSTM has little temporal context | Acceptable for classification-only use; distance estimates less reliable |
| **1 s** | ~1.5 s worst-case | ~95% of 2 s baseline | Good | **Recommended default** — good balance of latency and accuracy |
| **2 s** | ~3 s worst-case | Baseline | Best | Published literature standard; use if latency is not a constraint |

**If you need both low latency and good distance accuracy:** train on 500 ms windows but aggregate consecutive predictions. Run inference every 250 ms (75% overlap) and take the median of the last 4 distance estimates — this gives you a 1 s effective temporal window while keeping detection latency at ~500 ms.

**Physics floor on minimum window:** The drone BPF at 128 Hz has a cycle time of ~8 ms. To see a recognizable harmonic pattern in a mel spectrogram you need at least 10–20 cycles = 80–160 ms. In practice, the neural network needs more statistical averaging to be confident — 500 ms is the empirically validated minimum. Below 300 ms, false positive rates rise sharply and detection at range degrades severely.

### Is This a Problem?

For a drone threat scenario, 1–3 seconds of detection latency is operationally acceptable — drones move slowly enough that this doesn't change the response window meaningfully. The latency floor is set by physics (acoustic capture time), not by software or hardware choices. If you decide to use 500 ms windows to cut latency, plan for it from day one of data collection — do not retrain later.

---

## What to Check / Validate Before Acting

| Decision point | What to verify before proceeding |
|---|---|
| Choosing a microphone | Record 30 seconds outdoors and verify the drone is audible in the spectrogram at 100 m |
| Starting Track A training | YAMNet fine-tuned on DroneAudioset detects drone at 30 m in a live test |
| Starting Track B data collection | RTK shows "Fixed" status on both mic and drone rovers; 3D distance labels verified against tape measure to ±10 cm at 5 reference distances |
| Starting Track B model training | Preprocessing pipeline produces clean spectrograms with visible harmonic structure; helicopter negatives are in the dataset |
| Declaring binary classifier "done" | Tested on a session not used during training; tested against helicopter clips specifically |
| Declaring distance "done" | Adjacent-bracket confusion is acceptable; 0–50 m vs >300 m confusion is a failure |
| Moving to TDOA | Single-mic Track B model is stable and field-validated at ≥200 m |

---

## Tools & Libraries Summary

| Task | Tool |
|---|---|
| Audio recording | Python `sounddevice` or `pyaudio` |
| Audio processing | `torchaudio`, `librosa`, `scipy.signal` |
| Spectrogram visualization | `librosa.display`, `matplotlib` |
| Model backbone | [YAMNet](#yamnet) (TensorFlow Hub or PyTorch port) |
| Model training | PyTorch + `torchaudio`; TensorFlow if using YAMNet natively |
| Model [quantization](#model-quantization) | `torch.quantization` (PyTorch) or TensorFlow Lite |
| GPS parsing | `gpsd`, `pynmea2` |
| RTK GPS hardware | u-blox F9P module (rover × 2 + base × 1 for sample collection); SparkFun GPS-RTK2 is a ready-made breakout board |
| RTK correction radio link | RFD900 or LoRa module pair (base ↔ rovers, up to 40 km line-of-sight) |
| Network RTK (beyond 20 km) | NTRIP client library (`str2str` from RTKLIB); requires cellular modem on each capsule |
| Barometer | BMP580/BMP581 or MS5611 (<$10); use `smbus2` Python library |
| Dataset management | Pandas (labels CSV) + structured file system |
| Experiment tracking | Weights & Biases (wandb) — free tier is enough |
| TDOA / [GCC-PHAT](#gcc-phat) | `pyroomacoustics` |
| Public datasets | HuggingFace `datasets`, direct GitHub download |

---

## Hardware Shopping List

### Capsule Compute Architecture

Each capsule needs a processor to handle audio capture and either run inference or stream audio. Two approaches:

**Architecture A — Edge Inference**
- Each capsule: mic + **Raspberry Pi** + GPS + barometer
- The Pi runs the full CNN-BiLSTM inference pipeline locally
- Capsule is fully independent — continues working even if the radio link drops
- Cost: ~$80/capsule for compute; power draw ~800 mA (needs a larger solar panel)
- **Recommended for Track B** — field deployment requires independence

**Architecture B — Central Inference**
- Each capsule: mic + **ESP32-S3** + GPS + barometer
- The ESP32-S3 captures audio via I²S and streams it to a central Raspberry Pi at the command post via one of three link options:
  - **WiFi** — easiest to set up, range ~100–200 m outdoors
  - **Serial radio (SiK / RFD900x)** — km-range, works without infrastructure, but RF-jammable
  - **Optical fiber (jam-proof backup)** — physically tethered, immune to RF jamming; see [section 9](#9-optical-fiber-link--architecture-b-backup)
- The central Pi runs inference for all capsules
- Cost: ~$12/capsule for compute; power draw ~80–120 mA (much smaller solar panel)
- Tradeoff: if the link drops (radio) or is cut (fiber), that capsule goes silent; the central Pi is a single point of failure
- **Recommended for Track A POC** — cheaper, faster to wire up, audio lands on your laptop immediately

**Recommendation:** Build Track A with Architecture B (ESP32-S3 capsules → central Pi). Once the model is working and you move to Track B field deployment, switch to Architecture A (RPi per capsule) for independence.

---

### Quick Reference Checklist

**Microphones**
- [ ] [ICS-43434 I2S MEMS Microphone Breakout — Adafruit #6049](#1-microphone--ics-43434) (~$8 each) × 4

**Barometers**
- [ ] [BMP580 / BMP581 barometer breakout (3.3V–5V) × 4 — verify current Adafruit or SparkFun product number before ordering (~$5–10 each)](#2-barometer--bmp580--bmp581)

**RTK GPS — Sample Collection**
- [ ] [ArduSimple simpleRTK2B Budget board (ZED-F9P) × 1 — rover only; RTK2Go NTRIP network acts as base station via internet (~€160)](#3-rtk-gps--ardusimple-simplertk2b-budget)
- [ ] [TAN1216Q39 Quadrifilar Helix GNSS Antenna × 1 — multiband L1/L2/L5/G1/G2/B1/B2/B3, SMA female (~$15–25)](#4-rtk-antenna--tan1216q39-quadrifilar-helix)
- [ ] [SMA male-to-male patch cable × 1 — ~10–30 cm; connects ZED-F9P SMA female to antenna SMA female (~$2–3)](#4-rtk-antenna--tan1216q39-quadrifilar-helix)
- [ ] Phone hotspot or laptop with internet — relay NTRIP corrections from RTK2Go to rover (no extra purchase needed)

**Standard GPS — Deployment Capsules (POC)**
- [ ] [GEPRC GEP-M10 × 3 — pure GPS, integrated patch antenna, 4-pin JST-GH UART (~$10–15 each)](#6-standard-gps--deployment-capsules-poc) — buy this if purchasing BMP580/581 separately
- [ ] [GEPRC GEP-M10-DQ × 3 — GPS + DPS310 barometer built-in, 6-pin JST-GH UART+I2C (~$15–20 each)](#6-standard-gps--deployment-capsules-poc) — alternative: replaces separate BMP purchase for deployment capsules
- [ ] [JST-GH 1.25mm 4-pin to dupont female cable × 3 — connects GEP-M10 to ESP32-S3 header pins (~$1 each)](#6-standard-gps--deployment-capsules-poc)

**Compass / Magnetometer (optional — for directional mic array or parabolic mic)**
- [ ] [GY-271 QMC5883L breakout × N — I2C, 3.3V, ±1–2° accuracy after calibration (~$1–3 each on AliExpress)](#10-compass--magnetometer-optional)
- [ ] [LIS3MDL breakout × N — Adafruit or clone, better accuracy/stability, I2C, 3.3V (~$5–8 each)](#10-compass--magnetometer-optional)

**Wind Protection**
- [ ] [Furry lavalier deadcat windscreen × 4 — Rycote "Softie" or Bubblebee "Windbubble" miniature, sized for ~12–19 mm capsule (~$10–15 each)](#7-windscreen-deadcat)
- [ ] [Foam windscreen fallback — open-cell foam, lavalier/miniature mic size, pack of 10 (~$10) — calm-day POC only](#7-windscreen-deadcat)

**Capsule Compute — Architecture B (Track A POC)**
- [ ] [ESP32-S3-DevKitC-1 × 3 — official Espressif dev board (~$12 each)](#8-capsule-compute--esp32-s3-architecture-b)

**Capsule Compute — Architecture A (Track B deployment)**
- [ ] Raspberry Pi 5 × 3 (~$80 each) — not yet needed; buy when moving to Track B field deployment

**Optical Fiber Link — Architecture B Backup (optional, field deployment)**
- [ ] [POF tier ≤50 m: HFBR-1521Z / HFBR-2521Z transceiver pair × 3 + 1 mm plastic optical fiber cable (~$15–25/capsule + ~$0.20/m cable)](#9-optical-fiber-link--architecture-b-backup)
- [ ] [Glass fiber tier ≤2 km: multimode SFP media converter pair × 3 + OM3/OM4 fiber patch cable (~$60–100/capsule + ~$1/m cable)](#9-optical-fiber-link--architecture-b-backup)

**Long-Range Radio (post-POC deployment)**
- [ ] [RFD900x radio modem pair — for km-range RTK correction link in field deployment (~$200–250/pair)](#5-rtk-correction-radio-link--sik-915-mhz-recording--rfd900x-deployment)

---

**Priority order:** Buy microphones and barometers first — cheap, fast to arrive, lets you start software and audio testing immediately. Buy the RTK kit second — most expensive but essential before any serious data collection. Buy standard GPS modules last — not needed until past Track A.

**Estimated total:**
| Phase | Items | Approx. cost |
|---|---|---|
| Immediate (audio dev) | 4× ICS-43434 + 4× BMP580/BMP581 | ~$72 |
| Before data collection | 1× simpleRTK2B + 1× TAN1216Q39 + SMA cable | ~€160 + $30 |
| Before deployment POC | 3× GEP-M10 GPS + windscreens | ~$60 |
| Post-POC | RFD900x pair | ~$225 |

---

### Detailed Hardware Notes

#### 1. Microphone — ICS-43434

The INMP441 (previously listed in earlier versions of this plan) is discontinued and no longer manufactured. The ICS-43434 is its direct successor and is better in every measurable way:

| Spec | INMP441 | ICS-43434 |
|---|---|---|
| SNR | 61–62 dBA | 64–65 dBA |
| Sensitivity tolerance | ±3 dB | ±1 dB |
| Frequency response | 60 Hz – 15 kHz | 20 Hz – 20 kHz |
| Power draw | 1.2 mA | 0.8 mA |
| Status | **Discontinued** | Active, in production |

The higher SNR means quieter self-noise — critical for detecting distant drones. The tighter sensitivity tolerance (±1 dB vs ±3 dB) means less unit-to-unit variation between capsules, which matters for triangulation accuracy. The wider low-end response (down to 20 Hz vs 60 Hz) captures the drone shaft rate at ~64 Hz that the INMP441 misses.

Adafruit released a breakout board (product #6049) in August 2024 that is in stock internationally (also at Micro Center, The Pi Hut, Core Electronics). Available as bare module on AliExpress, but stick to Adafruit or verified distributors for measurement-critical hardware — clone modules may not meet the ±1 dB sensitivity tolerance.

**What to look out for when buying:**
- Needs I2S interface on the host (Raspberry Pi has this natively via the `i2s` overlay)
- Buy the Adafruit breakout, not bare chip — the bare ICS-43434 is a tiny LGA package requiring reflow soldering; the breakout has all caps pre-populated
- The windscreen (furry or foam) must fit snugly around the mic capsule housing without pressing on the membrane — see [section 7](#7-windscreen-deadcat) for windscreen choice guidance

**Sources:**
- [ICS-43434 datasheet — TDK InvenSense](https://invensense.tdk.com/products/ics-43434/)
- [Adafruit ICS-43434 breakout product page (#6049)](https://www.adafruit.com/product/6049)
- [INMP441 vs ICS-43434 comparison — AliExpress Wiki/community](https://www.aliexpress.com/s/wiki-ssr/article/inmp441-vs-ics-43434)

---

#### 2. Barometer — BMP580 / BMP581

The BMP580 (and its near-identical sibling the BMP581) is Bosch's current-generation barometric sensor, superseding the BMP388 we originally listed. The BMP388 is still active and would work, but BMP580/BMP581 is what's readily available now and is better on every spec that matters for this project:

| Spec | BMP388 | BMP580 / BMP581 |
|---|---|---|
| Relative accuracy | ±0.08 hPa (±0.68 m) | ±0.06 hPa (~±0.51 m) |
| Noise floor | 0.20 Pa RMS | ~0.09 Pa RMS (~2× lower) |
| Power draw | ~3.4 µA @ 1 Hz | ~2 µA @ 1 Hz |
| Status | Older generation | **Current generation** |

The lower noise floor is the meaningful improvement — it means cleaner altitude readings and less jitter in the `d_3D = sqrt(d_horizontal² + Δh_baro²)` formula. Both sensors are more than accurate enough for ±0.5–1 m relative height; BMP580/581 just has more headroom.

**BMP580 vs BMP581:** Nearly identical performance. BMP581 is a slightly different package. Either is fine — buy whichever breakout board is in stock.

**On the voltage variants you saw:** The bare chip runs at 1.71V–3.6V. Breakout boards add onboard regulators and I²C level shifters to accept higher voltages:
- **3.3V–5V breakout** — accepts 3.3V or 5V on VIN, I²C logic shifted to match. This is what you want for ESP32-S3 (3.3V) or Raspberry Pi (3.3V GPIO).
- **2.2V–3.6V breakout** — no regulator; raw chip voltage. Only useful on a custom PCB. Do not buy this for prototyping.

**What to look out for when buying:**
- Buy a breakout board, not bare chip — BMP580/581 is a tiny LGA package requiring reflow soldering
- Buy the **3.3V–5V variant** (onboard regulator + level shifter) — matches ESP32-S3 and Raspberry Pi directly
- Search Adafruit or SparkFun for "BMP581" to find the current breakout product — verify the product number on the site before ordering, as product numbers change between generations
- I2C address is configurable (0x46 or 0x47 on BMP581 via SDO pin) — slightly different from BMP388's 0x76/0x77; update your I2C scan code accordingly
- Must zero both mic barometer and drone barometer relative to each other at the start of each recording session — do not use absolute altitude values, only relative height difference
- At deployment, the command center broadcasts its barometric reading; each capsule computes height relative to the command center to remove weather-driven pressure drift

**Sources:**
- [Bosch BMP580 product page](https://www.bosch-sensortec.com/products/environmental-sensors/pressure-sensors/bmp580/)
- [Bosch BMP581 product page](https://www.bosch-sensortec.com/products/environmental-sensors/pressure-sensors/bmp581/)
- [Bosch BMP388 (original recommendation, still active) — drone altitude stabilization press release](https://www.bosch-sensortec.com/news/barometric-pressure-sensor-bmp388-ce-drones.html)

---

#### 3. RTK GPS — ArduSimple simpleRTK2B Budget

The u-blox ZED-F9P is the industry-standard chip for budget RTK, achieving **±1–2 cm horizontal, ±2–3 cm vertical**. The ArduSimple simpleRTK2B Budget is the most popular ZED-F9P breakout board: the best-documented, active since 2018, extensive community support, and roughly half the price of the SparkFun GPS-RTK2 with better RTK-specific documentation.

You need **3 boards**: 1 configured as base station (placed on a fixed tripod at the recording site), 2 configured as rovers (one on the mic tripod, one mounted on the drone).

**What to look out for when buying:**
- **The board does not include an antenna** — buy TAN1216Q39 separately (see section 4) plus a short SMA male-to-male patch cable (~$2–3); both the board and antenna have SMA female connectors
- Requires 5V or 3.3V power; communicates via USB or UART — USB is simplest for laptop-based sample collection
- Configure rover mode using u-blox's free u-center software (Windows) or RTKLIB (cross-platform) — you only need rover mode, not base station mode
- Verify RTK Fixed status (not just RTK Float) before recording mic position or starting any flight — Float mode gives ±0.1–0.5 m, not centimeters
- RTK2Go requires a nearby volunteer base station — check rtk2go.com for active stations in your area before purchasing

**Sources:**
- [ArduSimple simpleRTK2B product page](https://www.ardusimple.com/product/simplertk2b/)
- [u-blox ZED-F9P module datasheet](https://www.u-blox.com/en/product/zed-f9p-module)
- [RTK2Go free NTRIP caster](https://rtk2go.com)
- [RTKLIB str2str NTRIP client](https://rtklib.com)

---

#### 4. RTK Antenna — TAN1216Q39 Quadrifilar Helix

Every ZED-F9P **requires a multiband (L1+L2) active antenna**. A standard single-band GPS patch antenna will not enable RTK — the F9P needs both frequencies to compute centimeter-level corrections. This is the most common purchasing mistake for first-time RTK setups.

The **TAN1216Q39** is a quadrifilar helix antenna covering L1/L2/L5, GLONASS G1/G2, BeiDou B1/B2/B3, Galileo E1/E5, and L-band. It was chosen over flat survey patch antennas because this ZED-F9P will be mounted on a drone: the quadrifilar helix has a hemispherical radiation pattern and maintains consistent performance regardless of tilt angle, whereas a patch antenna requires pointing straight up and degrades significantly when the drone banks.

**Why TAN1216Q39 specifically (vs other TAN variants):**

| Variant | GPS L1 | GPS L2 | GLONASS G1 | Notes |
|---|---|---|---|---|
| Q23 | 2 dBi | **−3 dBi** | 0 dBi | L2 too low for reliable RTK |
| Q25 | 2 dBi | 2 dBi | −1.5 dBi | GLONASS bands negative |
| Q35 | 2.5 dBi | −1 dBi | 1.5 dBi | IPEX/U.FL connector, not SMA; poor L2 |
| **Q39** | **2 dBi** | **2 dBi** | **2 dBi** | **All key bands positive — best balance** |
| Q50A | 1.5 dBi | 2.5 dBi | 1 dBi | Lower L1/G1 than Q39 |

Q39 is the only variant with no negative values on the three critical RTK bands (L1, L2, G1). Q35 is also disqualified because it has an IPEX/U.FL connector rather than SMA, requiring an adapter with added signal loss.

**Connector note:** Both the ZED-F9P board (ArduSimple simpleRTK2B) and the TAN1216Q39 have **SMA female** connectors. They cannot connect directly — you need a short **SMA male-to-male patch cable** (~10–30 cm, ~$2–3). Buy this at the same time as the antenna.

**What to look out for when buying:**
- Buy the SMA male-to-male cable at the same time — easy to forget and not always sold with the antenna
- Keep the cable short (10–30 cm); every additional meter adds ~0.5 dB of signal loss
- The antenna is passive (no LNA) — the ZED-F9P provides antenna power via the coaxial cable; no separate power needed
- Mount with the top face toward the sky; the hemispherical pattern means ±30° tilt has minimal effect

**Sources:**
- TAN1216Q39 product listing — EC Buying (AliExpress)

---

#### 5. RTK Correction Radio Link — SiK 915 MHz (recording) / RFD900x (deployment)

RTK correction data (RTCM3 format) requires only ~500–1000 bps of bandwidth — even the cheapest serial radio can handle this. The choice depends on range:

**For recording sessions (base ↔ rovers within 300 m):**
A pair of SiK 915 MHz telemetry radios ("3DR Radio" clones, ~$20–30 on Amazon) is sufficient and pairs directly with ArduSimple boards. Plug the USB end into the base station laptop/Pi, UART end into the rover board.

**For field deployment (command center ↔ capsules, potentially km range):**
The RFD900x achieves **40+ km line-of-sight** range, has AES encryption (relevant for military context), and operates at 900 MHz which penetrates vegetation better than 2.4 GHz. A pair costs ~$200–250.

**For deployment beyond 20 km (Network RTK / NTRIP):**
Instead of a radio link, each capsule uses a cellular modem to connect to an NTRIP caster (network RTK server). The capsule sends its approximate position, receives corrections computed for that exact location from the nearest network stations. This removes the 20 km base-station range limitation entirely. The NTRIP client software is `str2str` from RTKLIB (free, open source). Requires cellular coverage in the deployment area and a SIM card per capsule.

**For deployment beyond 20 km without cellular (RTK relay):**
A capsule within 20 km of the command center receives corrections via radio and re-broadcasts them to farther capsules on a second radio channel — this is an RTK repeater configuration. The distant capsule gets corrections computed for the 20 km baseline, not its actual 40 km position, so accuracy degrades to roughly ±5–20 cm instead of ±2–5 cm. Still far better than standard GPS.

**What to look out for when buying:**
- Match frequency to your region: **915 MHz** for Americas, **868 MHz** for Europe/Israel. Verify legal frequency bands before purchasing.
- Both radios in a pair must run the same firmware and baud rate as the GPS UART output (typically 115200 baud)
- RFD900x requires an antenna (dipole included) — keep antenna vertical for omnidirectional coverage

**Sources:**
- [RFD900x radio modem — RFDesign product page](https://rfdesign.com.au/modems/)
- [RTK broadcast via LoRa/serial radio — SparkFun community discussion](https://community.sparkfun.com/t/rtk-broadcast-via-lora-serial-radio/62612)

---

#### 6. Standard GPS — Deployment Capsules (POC)

For POC deployment capsules, standard GPS is sufficient — the triangulation target accuracy is ±20–40 m, and mic position errors of ±3–5 m are acceptable.

**GEPRC GEP-M10** (~$10–15): uses the u-blox M10 chipset, single-band L1. The key advantage over buying a raw u-blox module is the **integrated ceramic patch antenna** — it requires no separate antenna purchase, no external cable, and no antenna mounting. The "THIS SIDE UP" label on the module faces skyward; that's the entire antenna setup. Outputs standard NMEA over UART at 3.3V. Power draw ~1–2 mA — very solar-friendly.

**GEPRC GEP-M10-DQ** (~$15–20): same u-blox M10 GPS, but adds a **DPS310 barometer** and QMC5883L compass (compass not needed — ignored). The DPS310 has the same ±0.06 hPa relative accuracy as the BMP580 but ~5× higher noise floor (0.5 Pa vs 0.09 Pa RMS). This is fine for deployment capsule altitude reporting; it is not suitable for the sample collection rig where BMP580/581 is preferred. If using GEP-M10-DQ, skip the separate BMP580 purchase for deployment capsules.

**GPS generation comparison (why M10, not M8N or M9N):**

| | M8N | M9N | M10 (GEP-M10) |
|---|---|---|---|
| Bands | L1 only | L1 + L5 | L1 only |
| Accuracy (open sky) | ±3–5 m | ±1.5–3 m | ±2–3 m |
| Power draw | ~25 mA | ~18 mA | **~1–2 mA** |
| Integrated antenna | No | No | **Yes** |
| Price | ~$5–10 | ~$50 (breakout) | **~$10–15** |

M9N's dual-band L1+L5 gives better accuracy near trees and buildings, but the triangulation target (±20–40 m) doesn't require it. M10's extreme low power draw (25× lower than M8N) is the decisive factor for solar-powered capsules.

**Connector:** GEP-M10 uses a JST-GH 1.25mm 4-pin connector (G/V/TX/RX). The ESP32-S3-DevKitC-1 has standard 2.54mm header pins — you need a JST-GH 1.25mm 4-pin to dupont female jumper cable (~$1) to connect them. Buy these at the same time.

**What to look out for when buying:**
- The integrated antenna faces up — design the capsule housing so this face has an unobstructed sky view after landing
- No active antenna power needed — the integrated ceramic patch is passive; the M10 drives it internally
- UART baud rate defaults to 9600 bps on most modules; configurable via UBX protocol if higher rate needed
- GEP-M10-DQ I2C pins (for DPS310 barometer): C = SCL, D = SDA — connect to ESP32-S3 I2C pins

**Sources:**
- [GEPRC GEP-M10 product page](https://geprc.com/product/geprc-gep-m10/)
- [u-blox M10 chipset overview](https://www.u-blox.com/en/product/m10-shield)

---

#### 7. Windscreen (Deadcat)

**Foam vs furry — comparison:**

| | Foam (open-cell) | Furry (deadcat / windjammer) |
|---|---|---|
| Wind attenuation | ~10–20 dB | ~20–50 dB |
| Effective wind speed | Up to ~15–20 km/h | Up to 50+ km/h |
| Frequency impact | None | Minor HF rolloff above ~10–12 kHz |
| Relevance to drone audio | No concern (drone signatures <1 kHz) | No concern (drone signatures <1 kHz) |
| Cost | ~$1–2 each (pack of 10, ~$10) | ~$10–15 each |
| Use case | Calm-day POC, indoor testing | Field deployment, any outdoor recording |

**Recommendation: furry for all outdoor use.** Wind noise in field conditions sits in the 100–800 Hz range — exactly where drone blade harmonics live. A foam windscreen fails above a moderate breeze, corrupting both your training data and live detection. Furry windscreens eliminate this problem. The slight high-frequency rolloff is irrelevant since nothing useful for drone detection is above 1–2 kHz.

**Foam** is still fine for calm-day POC sessions and indoor bench testing — keep a pack as a fallback.

**Fitting note:** The ICS-43434 sits on a PCB (~18 × 18 mm), not a cylindrical mic body. Standard deadcats are designed for cylindrical barrels. Mount the mic inside a small 3D-printed or off-the-shelf cylindrical enclosure (~12–19 mm outer diameter) first, then fit the furry windscreen over that. Rycote "Softie Lyre" and Bubblebee "Windbubble" both make miniature versions sized for lavalier mics in this range.

**Test before any recording session:** fit the windscreen, record 10 seconds outdoors in the prevailing wind. Wind noise should be nearly inaudible. If you still hear low-frequency rumble, the fur is too thin for conditions or the fit is too loose around the capsule housing.

**Sources:**
- [Gemini research plan (this project) — deadcat recommendation for microphone wind protection](gemini%20plan.md)
- General audio field recording best practices — no single authoritative product page; verified by standard outdoor recording guidance.

---

#### 8. Capsule Compute — ESP32-S3 (Architecture B)

The standard ESP32 has 520 KB of SRAM — enough for FFT and simple DSP, but not for loading a CNN model (even quantized models are tens of MB). The **ESP32-S3** is the correct variant: it supports up to 8 MB external PSRAM (on the N8R8 module), has a faster 240 MHz Xtensa LX7 dual-core CPU, and includes native I²S peripheral support for the MEMS microphone.

**Role in Architecture B:** The ESP32-S3 handles audio capture (I²S → ICS-43434), optionally runs a lightweight VAD (Voice Activity Detector) to avoid streaming silence, and transmits the audio stream over WiFi or serial radio to the central Raspberry Pi. The Pi does all inference.

**Recommended board for POC:** ESP32-S3-DevKitC-1 (official Espressif development board, ~$12). Compact, well-documented, USB-C, I²S pins exposed.

**For production capsule hardware:** ESP32-S3-WROOM-1 N8R8 bare module (~$3–5) — 8 MB flash + 8 MB PSRAM variant. Requires soldering to a custom PCB but is much smaller and lighter.

**What to look out for when buying:**
- **Buy the N8R8 variant** (N8 = 8 MB flash, R8 = 8 MB PSRAM) — the base N8 has no PSRAM and is insufficient. Check the exact part number before ordering.
- Verify I²S pin availability on the specific board — the DevKitC-1 exposes all I²S pins; some compact ESP32-S3 boards route them elsewhere
- Power draw ~80–120 mA active at 240 MHz, drops to <1 mA in deep sleep — very solar-friendly
- If using WiFi for audio streaming: range is limited to ~100–200 m outdoors. For longer range, use a serial radio (SiK or RFD900x) for audio stream transport instead
- This item is **not needed** if you build Architecture A (Raspberry Pi per capsule) from the start

**Sources:**
- [ESP32-S3 datasheet — Espressif](https://www.espressif.com/en/products/socs/esp32-s3)
- [ESP32-S3-DevKitC-1 product page — Espressif](https://www.espressif.com/en/development-tools/esp32-s3-devkitc-1)
- [ESP32-S3-WROOM-1 module datasheet — Espressif](https://www.espressif.com/sites/default/files/documentation/esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf)

---

#### 9. Optical Fiber Link — Architecture B Backup

**Why fiber?** The enemy fiber-optic drones described in this project's problem statement are a threat precisely because fiber cannot be RF-jammed. The same property protects your capsule link: a fiber-tethered capsule continues streaming audio even under full RF jamming, making it the most survivable option for Architecture B in a contested environment.

**Bandwidth requirement:** 16 kHz, 16-bit mono audio = 256 kbps raw. Even with 4 capsules streaming simultaneously, total bandwidth is ~1 Mbps — trivially small for any fiber link. Bandwidth is not a constraint here.

**How it connects to the ESP32-S3:** The ESP32-S3 has no native fiber interface; a converter is needed at each end of the cable. Two approaches:

- **UART-to-fiber (simplest for POC):** ESP32-S3 UART → fiber transceiver → fiber cable → fiber transceiver → Raspberry Pi UART. Run at 460800 baud — enough for 16 kHz audio with headroom. Requires only two small transceiver modules per link, no additional chips.
- **Ethernet-to-fiber (cleaner for production):** Add a W5500 SPI Ethernet module (~$3) to the ESP32-S3 → RJ45 → media converter → fiber → media converter → Pi Ethernet. Supports standard TCP/UDP audio streaming, handles multiple capsules over a single central switch, more robust framing than raw UART.

**Range and cost by tier:**

| Tier | Technology | Interface | Max range | Cost per capsule (converter/transceiver) | Cable cost |
|---|---|---|---|---|---|
| Short (POC) | Plastic optical fiber (POF) | UART | ~50 m | ~$15–25 (HFBR transceiver pair) | ~$0.20/m |
| Medium (deployment) | Multimode glass fiber | Ethernet | ~500 m–2 km | ~$60–100 (SFP media converter pair) | ~$0.50–1.50/m |
| Long (deployment) | Single-mode glass fiber | Ethernet | Virtually unlimited | ~$100–180 (single-mode SFP pair) | ~$0.50–2/m |

**Recommended components:**
- **POF tier:** Broadcom HFBR-1521Z (transmitter) + HFBR-2521Z (receiver) — ~$5–8 each; work with standard 1 mm PMMA plastic optical fiber; run on 5V; UART-compatible up to 5 Mbaud. Sold individually on DigiKey/Mouser.
- **Glass fiber tier:** Any pair of 100BASE-FX or 1000BASE-SX SFP media converters (e.g. TP-Link MC200CM, ~$25–40 each) + standard OM3 multimode fiber patch cable. Buy at least 10–20% extra cable length for routing slack.
- **W5500 SPI Ethernet module** (for Ethernet-to-fiber approach): ~$3–5, widely available on AliExpress or Amazon. Supported by the `Ethernet` library in ESP-IDF and Arduino framework.

**Physical deployment options:**
- **Spool-deploy from the drop drone:** The capsule spools out fiber from the deployment drone as it descends — the same mechanism enemy fiber-optic drones use. The capsule lands with a taut fiber run back to the drone, which relays to the command post. No ground team needed.
- **Pre-run ground cable:** Fiber already laid along terrain before capsule drop; capsule has a field-connectorized fiber pigtail (e.g. SC or LC connector with a dust cap) that a soldier plugs in on landing.
- **Armored cable for field use:** Standard patch cable jacket is fragile. For field deployment use armored duplex fiber (~$1–3/m) — stainless steel or Kevlar-reinforced jacket, rated for outdoor ground routing.

**What to look out for when buying:**
- **POF bending radius:** plastic fiber breaks if bent below ~25 mm radius — route it carefully inside the capsule housing; avoid sharp corners near the connector
- **Glass fiber connectors:** SC connectors are easiest to field-terminate; LC connectors are smaller but harder to handle with gloves. For POC, buy pre-made patch cables with connectors already installed.
- **Polarity:** UART fiber links are directional — HFBR-1521Z is transmit-only, HFBR-2521Z is receive-only. You need one of each at both ends: TX on capsule → RX on Pi, TX on Pi → RX on capsule (two fibers in total, or one duplex cable)
- This item is **only needed for Architecture B** and only when RF survivability is required — it is not needed for the Track A POC

**Sources:**
- [Broadcom HFBR-1521Z datasheet (plastic fiber transmitter)](https://www.broadcom.com/products/fiber-optic-modules-components/industrial-fiber-optics/plastic-optical-fiber/hfbr-x52x)
- [TP-Link MC200CM media converter product page](https://www.tp-link.com/en/business-networking/accessory/mc200cm/)
- [W5500 SPI Ethernet controller datasheet — WIZnet](https://www.wiznet.io/product-item/w5500/)


---

#### 10. Compass / Magnetometer (optional)

A compass is **not needed for Track A or Track B audio detection**. It only becomes relevant if you build a **directional microphone array** or **parabolic microphone** — in those cases, the array has a specific pointing direction, and knowing its magnetic heading lets you convert a detection into a geographic bearing toward the drone.

**How it works:** the magnetometer reads the Earth's magnetic field and outputs a heading in degrees (0–360°). Apply your local magnetic declination offset (the difference between magnetic north and geographic north, typically 1–5° depending on location) to convert to a true geographic bearing. The drone bearing is then the heading the array was pointing when it detected the drone.

**Two recommended options:**

**GY-271 (QMC5883L)** — ~$1–3 on AliExpress. The same chip used inside the GEPRC GEP-M10Q GPS module. I2C (address 0x0D), 3.3V, ±1–2° accuracy after calibration. Needs hard-iron and soft-iron calibration before use — rotate the sensor slowly through a full sphere in open air (away from metal objects) until the calibration routine converges. Widely supported: `QMC5883L` Arduino/ESP-IDF library, Python `qmc5883l` package. Best value for testing directional array concepts.

**LIS3MDL** (Adafruit #4479 or clone) — ~$5–8. STMicroelectronics chip. I2C or SPI, 3.3V. ±0.5° accuracy after calibration — noticeably better than QMC5883L. Better temperature stability, which matters if the capsule sits in direct sunlight. Well-documented: Adafruit CircuitPython library, Arduino library. Recommended if you move beyond testing to field validation.

**Buying guidance:**
- Buy separately rather than paying the GPS module premium — GY-271 at ~$1–3 is the same QMC5883L chip that costs ~$3–7 more inside the GEP-M10Q
- For testing only: GY-271 is sufficient
- For field validation: LIS3MDL is worth the extra $4–5 for better accuracy and stability
- Keep the compass away from motors, power cables, and steel hardware — magnetic interference from nearby metal is the primary source of heading error; calibrate it in the actual capsule housing in its final mounting position

**Sources:**
- [QMC5883L datasheet — QST Corporation](https://datasheet.lcsc.com/szlcsc/QST-QMC5883L-TR_C192585.pdf)
- [LIS3MDL product page — STMicroelectronics](https://www.st.com/en/mems-and-sensors/lis3mdl.html)
- [Adafruit LIS3MDL breakout (#4479)](https://www.adafruit.com/product/4479)

---

## Glossary

### YAMNet
A neural network pretrained by Google on **AudioSet** — 2 million 10-second YouTube clips labeled with 521 audio event categories (engines, rotors, birds, machinery, etc.). It uses [MobileNetV1](#model-backbone) architecture (designed for mobile/edge devices). For each 960 ms chunk of audio it produces a 1024-number vector describing the sound — this is the [backbone](#model-backbone) embedding.

For this project: we take YAMNet, discard its 521-class output layer, and add our own output [heads](#model-backbone) (drone vs not-drone, distance bracket). Because it was pretrained on audio data including rotary machinery and vehicle sounds, it starts from a much better place than an image-pretrained network. Published results show YAMNet-based drone detectors reach 500 m, vs 200 m for image-pretrained CNNs on the same task.

---

### Model Backbone
The "feature extractor" part of a neural network — the layers that convert raw input (a spectrogram) into a rich numerical description of what's in it. Think of it as the model's perception system.

On top of the backbone you attach **heads** — small networks that make specific decisions from the backbone's description (Is this a drone? How far away?). The backbone is the expensive, complex part that benefits from pretraining on large datasets. The heads are small and trained from scratch on your specific data.

Example: YAMNet backbone → 1024-dim audio embedding → Head 1 (classification) + Head 2 (distance).

---

### Model Quantization
Neural network weights are normally stored as 32-bit floating point numbers (FP32) — 4 bytes each. INT8 quantization converts them to 8-bit integers — 1 byte each, 4× smaller.

**This is a software-only step — it requires no special hardware purchase.** It runs on any CPU, though chips with dedicated INT8 acceleration (ARM NEON on Raspberry Pi, Google Coral TPU) give larger speedups.

Effect: ~30% faster inference, ~45% less energy, 4× smaller model, at the cost of a very small accuracy drop (typically under 1%). For a solar-powered capsule this matters significantly — smaller model means less battery, smaller solar panel, lighter capsule. Apply using `torch.quantization` (PyTorch) or TensorFlow Lite before deployment.

---

### Transfer Learning
Using a model that was trained on a large dataset as the starting point for a new, more specific task. Instead of training from random initial weights (which requires massive amounts of data), you start from a model that already "knows" something useful.

For this project: YAMNet was trained on 2 million audio clips. We [fine-tune](#fine-tuning) it on drone audio. The model already knows how to listen — we just teach it what specifically to listen for.

---

### Fine-tuning
The process of taking a pretrained model and continuing to train it on your own (smaller, more specific) dataset. The pretrained weights are a starting point — training adjusts them toward your specific task.

In this project, fine-tuning YAMNet means: keep the backbone weights as initialization, add your classification/distance heads, and run training on your dataset. All layers update, but the backbone starts from a strong position rather than random.

---

### Domain Gap
The performance degradation that happens when a model is trained on data from one source (microphone, environment, conditions) but deployed on data from a different source.

In this project: if you train on recordings from a measurement condenser microphone but deploy on a MEMS microphone, the model learned frequency response patterns that literally don't exist in deployment audio. Accuracy drops. The solution is to train on the same hardware you deploy.

Track A accepts a domain gap (public dataset mics ≠ your MEMS) as a known tradeoff for speed. Track B eliminates it by recording your own data.

---

### SVM
**Support Vector Machine** — a classical machine learning algorithm (not a neural network) from the 1990s. It finds a mathematical boundary in a high-dimensional space that separates two classes with the maximum possible margin.

For audio: you feed it a flat vector of [MFCC](#mfcc) values and it predicts "drone" or "not-drone." It trains in seconds, needs as few as 50 clips per class, and is a valuable sanity check before committing to deep learning. If your SVM gets 85%+ accuracy on a small set, your preprocessing pipeline and labels are correct. Limitation: lower accuracy ceiling than CNNs, doesn't generalize as well across environments.

---

### MFCC
**Mel-Frequency Cepstral Coefficients** — a compact numerical description of an audio clip. The process: compute a [mel spectrogram](#mel-spectrogram), take the logarithm, then apply a cosine transform. The result is 13–40 numbers per time window that describe the spectral shape of the sound.

Think of MFCCs as a compressed fingerprint of the audio. They discard irrelevant detail while preserving the frequency structure that distinguishes one sound from another. Standard input for audio classification models, especially [SVMs](#svm) and traditional ML. When combined with a [mel spectrogram](#mel-spectrogram) as a fused input to a [CNN](#cnn), performance improves over either alone.

---

### Mel Spectrogram
A 2D representation of sound: time on the x-axis, frequency on the y-axis (scaled to the Mel scale — a perceptual frequency scale that compresses high frequencies), brightness representing loudness. Computed by applying the STFT (Short-Time Fourier Transform) to overlapping windows of audio and then scaling the frequency axis.

For drone detection: a hovering drone produces clear horizontal bands at regular frequency intervals (the [BPF](#bpf) and its harmonics). This visual structure is exactly what a [CNN](#cnn) is designed to detect. Used as a 128×128 "image" fed into the CNN backbone.

---

### CNN
**Convolutional Neural Network** — a neural network architecture designed to find local patterns in 2D grids (originally images, but works on spectrograms). Convolutional filters slide across the input looking for specific patterns. For a drone spectrogram, filters learn to detect harmonic bands, BPF periodicity, and frequency rolloff patterns.

CNNs are the standard architecture for audio classification when using spectrogram inputs. They can be applied to a single frame (static pattern recognition) or combined with temporal models (see [CNN-BiLSTM](#cnn-bilstm)) for sequence-aware tasks like distance estimation.

---

### CNN-BiLSTM
A hybrid architecture combining a [CNN](#cnn) and a Bidirectional LSTM.

- **CNN part:** reads each spectrogram frame and extracts "what" is in it — frequency content, harmonic structure.
- **LSTM (Long Short-Term Memory):** a recurrent network that processes a sequence of frames and remembers patterns over time. Learns things like "this signal has been getting louder for 800 ms — drone is approaching."
- **Bi (Bidirectional):** the LSTM reads the sequence both forward and backward simultaneously, giving context from both directions.

Why it's best for distance: distance estimation requires temporal reasoning. A drone at 200 m sounds different across time (signal builds slowly, sustains, fades gradually) vs a drone hovering at 30 m (loud and stable). The LSTM captures this evolution in a way a static CNN cannot. Best published approach for audio-based drone distance prediction.

---

### BPF
**Blade Pass Frequency** — the fundamental acoustic frequency produced by a drone's rotating propellers.

Formula: $BPF = (RPM / 60) \times \text{number of blades}$

Example: DJI Phantom at 3840 RPM with 2-bladed propellers → BPF = 128 Hz. Harmonics (256, 384, 512 Hz...) carry most of the classification information. Energy is concentrated below 1,000 Hz, with the strongest harmonics up to the 4th overtone (~512 Hz). Above that, broadband turbulence noise dominates.

This frequency range defines why the band-pass filter in preprocessing starts at 50 Hz (shaft rate) rather than a higher frequency.

---

### TDOA
**Time-Difference-of-Arrival** — a localization technique that computes the position of a sound source by measuring how much earlier the sound arrives at one microphone versus another.

If a drone is closer to Mic A than Mic B, the sound reaches Mic A first. The time difference constrains the source to lie on a hyperbola (in 2D) or hyperboloid (in 3D). With 3+ microphone pairs, the intersection of multiple hyperbolas gives the source position.

Requires precise clock synchronization between microphones (GPS PPS signal or NTP <1 ms accuracy). The standard algorithm for computing the time difference is [GCC-PHAT](#gcc-phat).

---

### GCC-PHAT
**Generalized Cross-Correlation Phase Transform** — the standard algorithm for computing the [TDOA](#tdoa) between two microphones. It cross-correlates the two audio signals in the frequency domain and applies a whitening filter (PHAT = Phase Transform) that makes it robust to noise and reverberation.

The output is a time-delay estimate with a sharp peak at the true delay time. Implemented in the `pyroomacoustics` Python library. Requires inter-mic spacing of ~15 cm for unambiguous detection up to ~1.1 kHz, based on $f_{max} = c / (2d)$ where $c = 343$ m/s.

---

### MAE
**Mean Absolute Error** — the average of how wrong your model's predictions are, measured in the same units as the output (meters, for distance estimation).

$$MAE = \frac{1}{n} \sum |predicted - actual|$$

Example: predictions [80 m, 120 m, 45 m] vs actual [73 m, 140 m, 50 m] → errors [7, 20, 5] → MAE = 10.7 m. Directly interpretable: "my model is off by 10.7 m on average."

Compare to **MSE** (Mean Squared Error), which squares the errors before averaging — a 20 m error counts 4× more than a 10 m error. MSE punishes large mistakes heavily, which makes it sensitive to outliers (e.g., a clip where GPS had a bad fix and reported 300 m instead of 30 m). For noisy GPS labels, MSE can warp the whole model trying to fit a handful of bad readings. MAE treats all errors equally, which is more robust — but it also means the model doesn't try as hard to avoid catastrophic mistakes. See [Huber Loss](#huber-loss) for the best-of-both-worlds solution used in this plan.

---

### Huber Loss
A loss function for regression that combines the best properties of [MAE](#mae) and MSE. It behaves like MSE for small errors (where precision matters) and like MAE for large errors (where outliers would otherwise dominate).

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

In plain terms: errors smaller than δ (a tunable threshold, e.g., 10 m) are treated with MSE precision. Errors larger than δ are treated with MAE robustness — they don't explode quadratically. This is the right choice for GPS-labeled training data because GPS occasionally produces bad readings (wrong fix, multipath error) that would otherwise distort the model with MSE. δ is a hyperparameter — start with δ = 10–20 m for drone distance estimation.

---

### Inverse Square Law
The physical law governing how sound intensity decreases with distance: $I \propto 1/r^2$. Double the distance → one quarter the intensity.

Applied as a distance proxy in Track A:
$$d_{estimated} = d_{ref} \times \sqrt{\frac{A_{ref}}{A_{measured}}}$$

Where $d_{ref}$ and $A_{ref}$ are measured in a one-time calibration session (drone hovers at a known distance). Limitations: assumes a point source in free space; real environments have reflections, wind, and varying drone orientations that break the clean relationship. Sufficient for Track A; replaced by a trained model in Track B.
