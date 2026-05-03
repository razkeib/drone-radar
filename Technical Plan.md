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
9. [Glossary](#glossary) — YAMNet, Model Backbone, Model Quantization, Transfer Learning, Fine-tuning, Domain Gap, SVM, MFCC, Mel Spectrogram, CNN, CNN-BiLSTM, BPF, TDOA, GCC-PHAT, MAE, Huber Loss, Inverse Square Law

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

1. Same microphone and deadcat setup as Phase A1.
2. Add GPS synchronization: GPS module on the microphone unit + GPS on the drone (or a phone with a GPS logging app held by the drone operator for early sessions).
3. Write a logger that saves: `timestamp | gps_mic_lat | gps_mic_lon | gps_drone_lat | gps_drone_lon | audio_file_path`. This becomes your ground-truth label database — the GPS coordinates let you compute the exact distance to the drone for each recording.
4. Do a short validation session: hover the drone at 4–5 measured distances (10 m, 30 m, 60 m, 100 m). Verify that the GPS-derived distances match reality within ±5 m.

**Checkpoint:** You can record 30-second clean drone audio clips and retrieve their GPS-derived distance to within ±5 m. Do not proceed until this works reliably.

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
  - `distance_m` — GPS-derived exact distance in meters (primary training label for regression)
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

**Checkpoint:** ~1,500 clips stored at `data/raw/{class_name}/{session_id}_{timestamp}.wav` with a matching `data/labels.csv`.

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

- Each microphone runs its own inference pipeline independently and reports to a central coordinator:
  - Track A: `{mic_id, gps_position, detected: bool, distance_bracket, timestamp}`
  - Track B: `{mic_id, gps_position, detected: bool, distance_m, distance_confidence, distance_bracket, timestamp}`
- **Triangulation (Track A — bracket-based):** if ≥2 mics detect simultaneously, compute intersection of distance circles using bracket midpoints. Gives a rough area hundreds of meters wide — sufficient to prove the concept.
- **Triangulation (Track B — regression-based):** use the continuous `distance_m` value from each mic as the circle radius. With ±20 m accuracy per mic, the intersection area shrinks to roughly 20–40 m radius — precise enough to meaningfully guide a response. Weight each mic's contribution by its `distance_confidence` (low-confidence readings expand their circle radius to reflect uncertainty).
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
| Starting Track B data collection | GPS distance labels accurate to ±5 m; logger is running reliably |
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
| Dataset management | Pandas (labels CSV) + structured file system |
| Experiment tracking | Weights & Biases (wandb) — free tier is enough |
| TDOA / [GCC-PHAT](#gcc-phat) | `pyroomacoustics` |
| Public datasets | HuggingFace `datasets`, direct GitHub download |

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
