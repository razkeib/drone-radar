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
7. [What to Check / Validate Before Acting](#what-to-check--validate-before-acting)
8. [Tools & Libraries Summary](#tools--libraries-summary)
9. [Hardware Shopping List](#hardware-shopping-list)
10. [Glossary](#glossary) — YAMNet, Model Backbone, Model Quantization, Transfer Learning, Fine-tuning, Domain Gap, SVM, MFCC, Mel Spectrogram, CNN, CNN-BiLSTM, BPF, TDOA, GCC-PHAT, MAE, Huber Loss, Inverse Square Law

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
- **1× RTK base station** — a u-blox F9P module on a fixed tripod at the recording site, or connect to a free CORS (Continuously Operating Reference Station) network if one exists in your region
- **2× RTK rover modules** — one mounted on the mic tripod, one on the drone (u-blox F9P boards are ~$50–80 each; pre-built units like the SparkFun GPS-RTK2 are ~$200)
- **Radio correction link** — a pair of small radio modems (e.g., RFD900, or LoRa modules) to transmit corrections from base to both rovers. Range of several km — more than enough for a recording session

**Setup per session:**
1. Place base station on a fixed tripod. Let it average its position for 5–10 minutes (improves surveyed position accuracy).
2. Attach RTK rover to mic tripod and drone. Verify both show RTK Fixed status (green light or status string "Fix") before recording.
3. Mount a **MEMS barometer** (BMP388 or MS5611, ~$5) on both the mic unit and the drone. Zero both relative to each other at the start of each session — this removes the effect of atmospheric pressure drift and gives **±0.5–1 m relative height accuracy**. Even with RTK, the barometer provides a useful redundant height measurement.

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
  - `drone_type`, `behavior` (hover/approach/pass)
  - `environment` (open_field / urban / forest)
  - `wind_speed` (none / light / moderate / strong — estimate or measure with a phone anemometer app)
  - `noise_sources` (none / traffic / insects / people / machinery)
  - `time_of_day` (morning / afternoon / evening)
- **These metadata labels are for dataset analysis and debugging, not model inputs.** After training, if the model fails in a specific condition (e.g., always wrong in high wind), these labels tell you exactly what's missing from your dataset. You then go record more of that condition.
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
4. **Segment:** 1–2 second windows, 50% overlap. Each segment inherits its parent recording's label.
5. **Compute features — use both:**
   - **[Mel Spectrogram](#mel-spectrogram):** `n_fft=1024`, `hop_length=512`, `n_mels=128`, converted to dB. Output: `(128 × time_frames)`.
   - **[MFCC](#mfcc):** 40 coefficients, 240 ms window. Output: `(40 × time_frames)`.
   - Concatenate along the feature axis → fused input tensor.
6. **Save** to `data/processed/` as `.pt` files with updated labels CSV.
7. **Split by session:** 70% train / 15% validation / 15% test. Split by recording session, not by clip — splitting by clip leaks information since clips from the same session sound nearly identical.

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
Input: Fused Mel Spectrogram + MFCC tensor
  → YAMNet backbone (pretrained on AudioSet, fine-tune all layers)
  → 1024-dim audio embedding
  ├── Head 1: Dense(256) → ReLU → Dense(N_classes) → Softmax    [classification: drone type]
  ├── Head 2: CNN-BiLSTM → Dense(256) → ReLU → Linear(1)        [distance regression: meters]
  └── Head 3: Dense(256) → ReLU → Dense(4) → Softmax            [distance bracket: confidence check]
```

Head 2 and Head 3 both predict distance from the same embedding. If they agree, the estimate is trustworthy. If they disagree strongly, flag the output as low-confidence.

The [CNN-BiLSTM](#cnn-bilstm) in Head 2 adds temporal reasoning — it reads how the signal evolves across multiple frames (getting louder = drone approaching, fading = receding). This temporal pattern is a strong distance cue that a static single-frame [CNN](#cnn) misses entirely.

**If you have limited data (<500 clips):** train an [SVM](#svm) on [MFCC](#mfcc) features first as a baseline. This validates your pipeline in minutes before committing to deep learning.

#### Training order

1. **Binary classifier only (drone vs not-drone):** achieve >90% F1 before moving on. Use class-weighted loss if your not-drone class is larger.
2. **Add drone type classification:** expand to multi-class. Evaluate per-class precision/recall.
3. **Add distance heads:** train both Head 2 (regression) and Head 3 (bracket) simultaneously. Loss: `total_loss = classification_loss + λ₁ × MAE_regression + λ₂ × bracket_crossentropy`. Use [Huber Loss](#huber-loss) instead of plain [MAE](#mae) — it is less sensitive to outlier clips where GPS had a bad fix. Start with λ₁=0.5, λ₂=0.3.
4. **[Quantize](#model-quantization):** apply INT8 quantization once validated. Reduces inference time ~30% and energy ~45%.

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

Standard GPS vertical accuracy (±10–15 m) is too coarse for 3D triangulation — a mic's height error would dominate the geometry. A MEMS barometer (BMP388 or MS5611, <$5, <1g) on each capsule provides **±0.5–1 m height accuracy**. On landing, the capsule takes a barometric altitude reading and reports it alongside its GPS position.

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
| Barometer | BMP388 or MS5611 (<$5); use `smbus2` or `pyms5611` Python library |
| Dataset management | Pandas (labels CSV) + structured file system |
| Experiment tracking | Weights & Biases (wandb) — free tier is enough |
| TDOA / [GCC-PHAT](#gcc-phat) | `pyroomacoustics` |
| Public datasets | HuggingFace `datasets`, direct GitHub download |

---

## Hardware Shopping List

### Quick Reference Checklist

**Microphones**
- [ ] [ICS-43434 I2S MEMS Microphone Breakout — Adafruit #6049](#1-microphone--ics-43434) (~$8 each) × 4

**Barometers**
- [ ] [BMP388 Precision Barometric Pressure & Altimeter — Adafruit #3966](#2-barometer--bmp388) (~$10 each) × 4

**RTK GPS — Sample Collection (buy together as a kit)**
- [ ] [ArduSimple simpleRTK2B Budget board (ZED-F9P) × 3 — 1 base + 2 rovers (~€160 each)](#3-rtk-gps--ardusimple-simplertk2b-budget)
- [ ] [ArduSimple Budget Survey Multiband GNSS Antenna × 3 — one per board (~€80 each)](#4-rtk-antenna--ardusimple-budget-survey-multiband)
- [ ] [SiK 915 MHz telemetry radio pair × 1 — for RTK correction link during recording sessions (~$25/pair)](#5-rtk-correction-radio-link--sik-915-mhz-recording--rfd900x-deployment)

**Standard GPS — Deployment Capsules (POC)**
- [ ] [u-blox NEO-M9N based GPS breakout × 3 — e.g. SparkFun #17285 (~$50 each), or SAM-M10Q module if power budget is tight (~$15)](#6-standard-gps--deployment-capsules-poc)

**Wind Protection**
- [ ] [Foam microphone windscreen (open-cell foam, lavalier/miniature mic size) — pack of 10, ~$10](#7-windscreen-deadcat)

**Long-Range Radio (post-POC deployment)**
- [ ] [RFD900x radio modem pair — for km-range RTK correction link in field deployment (~$200–250/pair)](#5-rtk-correction-radio-link--sik-915-mhz-recording--rfd900x-deployment)

---

**Priority order:** Buy microphones and barometers first — cheap, fast to arrive, lets you start software and audio testing immediately. Buy the RTK kit second — most expensive but essential before any serious data collection. Buy standard GPS modules last — not needed until past Track A.

**Estimated total:**
| Phase | Items | Approx. cost |
|---|---|---|
| Immediate (audio dev) | 4× ICS-43434 + 4× BMP388 | ~$72 |
| Before data collection | 3× simpleRTK2B + 3× antenna + radio pair | ~€720 + $25 |
| Before deployment POC | 3× NEO-M9N GPS + windscreens | ~$160 |
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
- The foam windscreen must fit snugly around the mic capsule without pressing on the membrane

**Sources:**
- [ICS-43434 datasheet — TDK InvenSense](https://invensense.tdk.com/products/ics-43434/)
- [Adafruit ICS-43434 breakout product page (#6049)](https://www.adafruit.com/product/6049)
- [INMP441 vs ICS-43434 comparison — AliExpress Wiki/community](https://www.aliexpress.com/s/wiki-ssr/article/inmp441-vs-ics-43434)

---

#### 2. Barometer — BMP388

The BMP388 was designed by Bosch specifically for drone altitude stabilization — it is in Bosch's own drone product line. Relative accuracy is ±0.08 hPa = **±0.66 m**, which satisfies the requirement for height difference measurement between mic and drone. Altitude noise as low as 0.1 m.

**Why BMP388 over MS5611:** The MS5611 is an older avionics barometer with excellent precision (sigma ~12 cm) but more complex calibration and less hobbyist documentation. For relative height measurement between two synchronized sensors at the same session, BMP388 is simpler to integrate and more than accurate enough.

**What to look out for when buying:**
- Buy the Adafruit breakout (#3966), not the bare chip — the BMP388 is an LGA package
- I2C address is configurable (0x76 or 0x77 via SDO pin) — useful if sharing a bus with other sensors
- Must zero both mic barometer and drone barometer relative to each other at the start of each recording session — do not use absolute altitude values, only relative height difference
- At deployment, the command center broadcasts its barometric reading; each capsule computes height relative to the command center to remove weather-driven pressure drift

**Sources:**
- [Bosch BMP388 — designed for drone altitude stabilization (Bosch Sensortec press release)](https://www.bosch-sensortec.com/news/barometric-pressure-sensor-bmp388-ce-drones.html)
- [Adafruit BMP388 breakout product page (#3966)](https://www.adafruit.com/product/3966)
- [Adafruit BMP388 guide — accuracy specs and I2C setup](https://learn.adafruit.com/adafruit-bmp388-bmp390-bmp3xx/overview)

---

#### 3. RTK GPS — ArduSimple simpleRTK2B Budget

The u-blox ZED-F9P is the industry-standard chip for budget RTK, achieving **±1–2 cm horizontal, ±2–3 cm vertical**. The ArduSimple simpleRTK2B Budget is the most popular ZED-F9P breakout board: the best-documented, active since 2018, extensive community support, and roughly half the price of the SparkFun GPS-RTK2 with better RTK-specific documentation.

You need **3 boards**: 1 configured as base station (placed on a fixed tripod at the recording site), 2 configured as rovers (one on the mic tripod, one mounted on the drone).

**What to look out for when buying:**
- **The board does not include an antenna** — budget for one multiband survey antenna per board (see below)
- Requires 5V or 3.3V power; communicates via USB or UART — UART is cleanest for Raspberry Pi integration
- Configure base vs rover mode using u-blox's free u-center software (Windows) or RTKLIB (cross-platform)
- For the drone-mounted rover, verify the board's weight is acceptable for your drone's payload capacity
- Verify RTK Fixed status (not just RTK Float) before any recording session — Float mode gives ±0.1–0.5 m, not centimeters

**Sources:**
- [ArduSimple simpleRTK2B product page](https://www.ardusimple.com/product/simplertk2b/)
- [u-blox ZED-F9P module datasheet](https://www.u-blox.com/en/product/zed-f9p-module)
- [SparkFun GPS-RTK2 board (ZED-F9P) — alternative breakout](https://www.sparkfun.com/products/15136)

---

#### 4. RTK Antenna — ArduSimple Budget Survey Multiband

Every ZED-F9P **requires a multiband (L1+L2) active antenna**. A standard single-band GPS patch antenna will not enable RTK — the F9P needs both frequencies to compute centimeter-level corrections. This is the most common purchasing mistake for first-time RTK setups.

ArduSimple's own comparison tests showed their budget survey antenna performs within millimeters of geodetic-grade antennas (which cost 10–50× more) in open-sky conditions — more than sufficient for open-field recording sessions.

**What to look out for when buying:**
- Connector must be SMA to match ArduSimple boards — verify before ordering
- For the **base station**: use the full survey antenna on a stable tripod — weight does not matter here
- For the **drone-mounted rover**: the survey antenna (~100–200g) may be too heavy. Use a lighter option like the u-blox ANN-MB-00 multiband antenna (~30g, ~$25) or Tallysman TW4721 (~40g) — slightly lower performance but flyable
- Keep antenna cables short (1–2 m ideal) — signal loss increases with cable length

**Sources:**
- [ArduSimple Budget Survey Multiband GNSS Antenna product page](https://www.ardusimple.com/product/survey-gnss-multiband-antenna/)
- [ArduSimple RTK antenna comparison: budget vs geodetic-grade](https://www.ardusimple.com/rtk-antenna-comparison-low-cost-vs-high-end-geodetic-antennas/)

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

For POC deployment capsules, standard GPS is sufficient — the triangulation target accuracy is ±20–40 m, and mic position errors of ±5 m are acceptable.

**u-blox NEO-M9N** (e.g. SparkFun breakout #17285, ~$50): supports L1+L5 dual-band, better accuracy near trees and buildings than single-band M8N, good documentation.

**u-blox SAM-M10Q** (~$15 bare module): highly integrated, very low power (1–2 mA), designed for IoT/battery devices. Better choice if power budget for the solar capsule is tight. Less hobbyist documentation but straightforward UART NMEA output.

**What to look out for when buying:**
- Both require an **active external antenna** — passive ceramic patch antennas give poor outdoor accuracy. Use a small active patch antenna with u.FL or SMA connector.
- Design the capsule so the antenna faces upward after landing — GPS needs clear sky view
- The SAM-M10Q is a bare module requiring soldering; the NEO-M9N on a SparkFun breakout is plug-and-play

**Sources:**
- [u-blox NEO-M9N module datasheet](https://www.u-blox.com/en/product/neo-m9n-module)
- [SparkFun GPS Breakout — NEO-M9N (#17285)](https://www.sparkfun.com/products/17285)

---

#### 7. Windscreen (Deadcat)

Buy open-cell foam windscreens sized for a miniature/lavalier microphone capsule (~10–15 mm diameter). Search "lavalier microphone foam windscreen" on Amazon — a pack of 10 costs ~$10.

Open-cell foam lets sound through while blocking wind turbulence against the membrane. Closed-cell foam (like craft foam) blocks both and should not be used.

**Test before any recording session:** put the windscreen on, record 10 seconds of moderate wind outdoors. Wind noise should be nearly inaudible. If you still hear strong low-frequency rumble, the foam is too thin or the fit is too loose.

**Sources:**
- [Gemini research plan (this project) — deadcat recommendation for microphone wind protection](gemini%20plan.md)
- Search "lavalier microphone foam windscreen" on Amazon — no single authoritative product page; verified by general audio field recording best practices.


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
