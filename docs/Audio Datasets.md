# Audio Datasets for Drone Detection

> **License key**
> - NO COMMERCIAL USE — cannot be used in a commercial product without permission
> - MIT / CC BY 4.0 / Open — permissive, commercial use allowed
> - Research only — no explicit license; treat as research-use until confirmed

---

## Direct Drone Datasets

### saraalemadi/DroneAudioDataset
- **Source:** https://github.com/saraalemadi/DroneAudioDataset
- **Size:** Hundreds of clips
- **Format:** WAV
- **Labels:** Binary (drone / no-drone) and multiclass (Bebop, AR Drone, Phantom, background)
- **Notes:** Ground-based mic perspective — matches our detection task directly. Most-cited starting point in the literature. Mix of real drone recordings + ESC-50 and Speech Commands backgrounds.
- **License:** Research only (no explicit license stated)

### geronimobasso/drone-audio-detection-samples
- **Source:** https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
- **Size:** Self-described largest publicly available drone audio detection database
- **Format:** WAV, 16 kHz, 16-bit, mono; clips 500 ms to several minutes
- **Labels:** Binary — 0 (no drone), 1 (drone)
- **Notes:** Directly designed for binary drone detection. Ready to use as-is.
- **License:** Research only (no explicit license stated)

### 32-UAV Multiclass Acoustic Dataset
- **Source:** https://arxiv.org/abs/2509.04715 (data linked from paper)
- **Size:** 3,200 recordings / 16,000 s (~4.4 hrs)
- **Format:** WAV + spectrogram images + MFCC plots
- **Labels:** 32 drone models (collapse all to positive class for binary training)
- **Notes:** Best acoustic diversity for the positive class — 32 distinct UAV models.
- **License:** Open access (arXiv publication)

### Purdue Large-Scale UAV Audio Dataset
- **Source:** https://ieeexplore.ieee.org/document/10023792 | https://hammer.purdue.edu/articles/thesis/23696391
- **Size:** ~2.3 hrs / 15 distinct drones (toy to Class I UAVs)
- **Format:** WAV, 5-second clips
- **Labels:** 15-class by UAV model (use as positive class)
- **Notes:** Good size/type range coverage.
- **License:** Figshare institutional repository — research use

### DroneDetectionThesis / Drone-detection-dataset
- **Source:** https://github.com/DroneDetectionThesis/Drone-detection-dataset | https://zenodo.org/records/5500576
- **Size:** 90 audio clips + 650 videos
- **Format:** WAV
- **Labels:** Drone / Helicopter / Background (3-class)
- **Notes:** Useful because it includes helicopter as a separate class, making it easy to extract as a hard negative too.
- **License:** Zenodo — check record for specific terms

### DroneNoise Database — University of Salford
- **Source:** https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411
- **Size:** ~707 MB
- **Format:** WAV, field recordings (Edzell, Scotland, 2022)
- **Labels:** Positive class (drone overflight)
- **Notes:** Outdoor field recordings — complements indoor datasets.
- **License:** University of Salford Figshare — research use

### ahlab/DroneAudioSet (currently in use)
- **Source:** https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet
- **Size:** 42.6 GB / ~23.5 hrs
- **Format:** Parquet (WAV-sourced), 8-channel, 16 kHz
- **Labels:** drone-only / drone-with-source / source-only / ground-truth (4 configs)
- **Notes:** Drone-mounted mic perspective (inverse of our task) — useful for learning propeller acoustics but not ground-based detection. Split at recording level when using all 8 channels to avoid data leakage.
- **License:** MIT

### UaVirBASE
- **Source:** https://zenodo.org/records/15391924
- **Size:** Multi-channel field recordings
- **Format:** WAV, 8-channel, 96 kHz, 32-bit (downsample to 16 kHz for training)
- **Labels:** Spatial annotations (azimuth, distance, height) — use audio as positive class
- **Notes:** Spatially diverse — varied distances, altitudes, orientations.
- **License:** Open access (Zenodo)

### DCASE 2024 Task 2 — HoveringDrone Subset
- **Source:** https://zenodo.org/records/10902294
- **Size:** 6–10 s clips, single-channel
- **Format:** WAV, mono
- **Labels:** Normal / anomalous (machine condition monitoring framing — extract drone split as positive class)
- **Notes:** Short clips; anomaly framing means you only get "normal" drone sounds here.
- **License:** DCASE challenge — research use

### DREGON — Drone Ego-Noise and Localization Dataset
- **Source:** http://dregon.inria.fr/datasets/dregon/
- **Size:** Multi-channel WAV recordings
- **Format:** WAV, 8-channel array embedded in quadrotor
- **Labels:** Ego-noise + localization ground truth
- **Notes:** Similar to ahlab — drone-mounted perspective. Use for positive class acoustic diversity.
- **License:** INRIA — research use

### Sound-Based Drone Fault Classification Dataset
- **Source:** https://zenodo.org/records/7779574
- **Format:** WAV, anechoic chamber recordings
- **Labels:** Flight direction + fault conditions (propeller/motor faults)
- **Notes:** Useful for covering abnormal drone sounds that differ from standard propeller noise.
- **License:** Check Zenodo record

---

## Hard Negatives — Acoustically Confusable Sounds

These are the most important negatives to include. A model trained without them will false-positive on anything with a harmonic hum in the 150–1000 Hz range.

### DataSEC / DataSED — Environmental Noise Datasets
- **Source:** DataSEC: https://zenodo.org/records/15340689 | DataSED: https://zenodo.org/records/15346092
- **Size:** DataSEC — 5,024 clips; DataSED — 717 tracks; combined >35 hrs
- **Format:** WAV, 44.1 kHz, mono
- **Labels:** 22 environmental sound classes including Propeller aircraft and Helicopter
- **Notes:** Helicopter and propeller aircraft are the closest non-drone confusors. Priority hard negatives.
- **License:** Open access (Zenodo 2025)

### MAD — Military Audio Dataset
- **Source:** https://github.com/kaen2891/military_audio_dataset
- **Size:** 8,075 samples / ~12 hrs
- **Format:** WAV, 7 classes
- **Labels:** Communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter
- **Notes:** Extract the helicopter class — rotating blades + harmonic thumping directly confusable with drones.
- **License:** Open research access (Nature Scientific Data)

### FSD50K
- **Source:** https://zenodo.org/records/4060432
- **Size:** 51,197 clips / >100 hrs
- **Format:** WAV, 44.1 kHz
- **Labels:** 200 classes (AudioSet ontology)
- **Most relevant classes:** Helicopter, Fixed-wing aircraft, Propeller/airscrew, Electric motor, Fan, Mechanical fan, Engine, Buzz
- **Notes:** Best permissively-licensed general dataset. Pull the helicopter/propeller/fan/motor subsets as hard negatives; rest as general negatives.
- **License:** CC BY 4.0 — commercial use allowed

### AudioSet (Google) — targeted subsets
- **Source:** https://research.google.com/audioset/ | https://huggingface.co/datasets/agkphysics/AudioSet
- **Size:** 1.7M clips, 10s each, 527 classes (YouTube-sourced)
- **Format:** WAV / YouTube download required
- **Most relevant classes:** Helicopter (/m/0cmf2), Propeller+airscrew (/m/09ct_), Drone (/m/04rlf), Electric motor (/m/02jz0l), Fan (/m/04fgwm), Bee/wasp (/m/09b5t), Mosquito (/m/09kvh)
- **Notes:** Download only the relevant subsets. Individual YouTube clips may have restrictions beyond the CC BY 4.0 label license.
- **License:** CC BY 4.0 (labels) — individual audio from YouTube may vary

### HumBugDB — Large-Scale Acoustic Mosquito Dataset
- **Source:** https://zenodo.org/records/4904800
- **Size:** ~20 hrs mosquito + ~15 hrs background; 36 species
- **Format:** WAV
- **Notes:** Mosquito wingbeat frequency ~200–800 Hz with harmonics — spectrally overlaps small multi-rotor drones. Important hard negative for outdoor deployment.
- **License:** Zenodo — research use

### InsectSound1000
- **Source:** https://www.nature.com/articles/s41597-024-03301-4
- **Size:** >169,000 samples, 12 insect species
- **Format:** WAV, 16 kHz, 32-bit, 4-channel, 2,500 ms clips
- **Notes:** Anechoic chamber recordings. Bee/wasp wingbeats in 150–500 Hz range can confuse a drone classifier.
- **License:** Open access (Nature Scientific Data)

### InsectSet459 / InsectSet32
- **Source:** https://zenodo.org/records/7072196 (InsectSet32)
- **Size:** InsectSet32 — 32 species; InsectSet459 — 26,399 files
- **Format:** WAV/MP3
- **Notes:** Cicada stridulation produces loud broadband tonal buzzing with harmonics up to several kHz — relevant outdoor hard negative.
- **License:** Open access

### Bee Audio Dataset
- **Source:** https://zenodo.org/records/10359686
- **Notes:** Honey bee colony hum ~200–500 Hz. Relevant hard negative for agricultural or outdoor deployments.
- **License:** Zenodo open access

---

## General Background Negatives

> ⚠ NO COMMERCIAL USE — ESC-50 and UrbanSound8K are CC BY-NC. Do not use in commercial products. Use FSD50K (CC BY 4.0) as the permissive alternative.

### ESC-50
- **Source:** https://github.com/karolpiczak/ESC-50
- **Size:** 2,000 clips, 5s each, 40 clips per class
- **Format:** WAV, 44.1 kHz, mono
- **Labels:** 50 environmental sound classes
- **Notes:** Helicopter class = hard negative; all others = general negatives.
- **License:** ⚠ NO COMMERCIAL USE — CC BY-NC

### UrbanSound8K
- **Source:** https://urbansounddataset.weebly.com/urbansound8k.html
- **Size:** 8,732 clips, ≤4s each
- **Format:** Various (WAV/AIFF/OGG/FLAC), 44.1 kHz
- **Labels:** 10 urban sound classes (air conditioner, engine idling, drilling, jackhammer most relevant)
- **Notes:** No helicopter class, but mechanical sounds are useful general negatives.
- **License:** ⚠ NO COMMERCIAL USE — CC BY-NC 3.0

---

## Recommended Combination

| Role | Datasets |
|---|---|
| **Positive class (drone)** | saraalemadi + geronimobasso DADS + 32-UAV dataset + Purdue clips |
| **Hard negatives** | DataSEC/DataSED helicopter & propeller + MAD helicopter + FSD50K helicopter/fan/motor |
| **General negatives** | FSD50K remaining classes + UrbanSound8K + insect datasets (for outdoor deployment) |

---

## Key Notes for Future Dataset Additions

- **Split at recording level when a dataset has multiple channels** — all channels from one recording must stay in the same train/test split to avoid data leakage.
- **Hard negatives matter more than general negatives** — helicopters, propeller aircraft, bees, and fans will cause the most false positives in deployment. Prioritize them.
- **Task orientation matters** — some datasets are drone-mounted (ego-noise perspective, e.g. ahlab, DREGON). These are valid for positive-class acoustics but do not reflect how your microphone will hear a drone.
- **Check Zenodo/GitHub license pages before any commercial use** — "research only" datasets listed here do not have explicit commercial terms confirmed.
