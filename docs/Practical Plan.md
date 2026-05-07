# Practical Plan: Track A — Architecture B

Track A goal: working end-to-end demo — microphone captures audio, model detects drone, rough distance estimated, triangulation produces an area. Architecture B: ESP32-S3 capsules stream audio to a central laptop/PC running inference.

---

## Language & Framework Decisions

| Layer | Language | Reason |
|---|---|---|
| Fine-tuning, inference, preprocessing, triangulation | **Python** | Entire ML ecosystem lives here. Computation runs in C++/CUDA backends — Python speed is irrelevant for ML work |
| ESP32-S3 capsule firmware | **C/C++ (ESP-IDF)** | Python does not run on microcontrollers. ESP32-S3 only captures audio and streams it — it does not run the model |
| Central device (laptop / Raspberry Pi) | **Python** | TFLite or ONNX runtime handles inference; ~5 ms per window on RPi5 with INT8 quantization |

Fine-tuning uses TensorFlow Hub for YAMNet in Track A (fastest path — YAMNet is Google's model, TF Hub is its native home). Track B switches to PyTorch throughout.

---

## Project Structure

Create this inside the repo before writing any code:

```
drone-radar/
├── src/
│   ├── preprocessing/    # audio loading, filtering, feature extraction
│   ├── model/           # YAMNet fine-tuning
│   ├── detection/       # inference pipeline
│   ├── distance/        # intensity proxy
│   └── triangulation/   # multi-mic geometry
├── data/
│   ├── raw/             # DroneAudioset and any future datasets
│   └── processed/       # preprocessed tensors
├── notebooks/           # Jupyter for exploration and debugging
└── requirements.txt
```

---

## Step 1 — Environment Setup (~1 hour)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install tensorflow tensorflow-hub   # YAMNet for Track A
pip install torch torchaudio           # PyTorch, ready for Track B
pip install librosa scipy numpy pandas matplotlib
pip install datasets huggingface_hub   # download DroneAudioset
pip install sounddevice                # mic input (needed once hardware arrives)
pip install jupyter ipykernel          # exploration notebooks
pip install wandb                      # experiment tracking (free tier)
```

Save the installed versions:
```powershell
pip freeze > requirements.txt
```

---

## Step 2 — Download and Explore DroneAudioset (~2-4 hours)

```python
from datasets import load_dataset

dataset = load_dataset("ahlab-drone-project/DroneAudioSet", split="train")
print(dataset.features)
print(dataset[0])
```

Do this in a Jupyter notebook (`notebooks/01_explore_dataset.ipynb`) before touching model code:

1. Print the label distribution — how many drone clips vs non-drone?
2. Check sample rate — resample to 16 kHz if needed
3. Listen to 5-10 drone clips and 5-10 background clips
4. Plot mel spectrograms of both classes side by side

**What to look for in spectrograms:** drone clips must show clear horizontal bands below 1 kHz (the blade pass frequency and its harmonics). Background clips should show diffuse, unstructured energy. If they look similar, the preprocessing has a bug — catch it here, not during training.

---

## Step 3 — Preprocessing Pipeline (~3-5 hours)

Write `src/preprocessing/pipeline.py`.

Pipeline per audio clip:
1. Load `.wav`, resample to 16 kHz mono (`librosa.load`)
2. Band-pass filter: high-pass at 50 Hz, low-pass at 10 kHz (`scipy.signal.butter` + `sosfilt`)
3. Normalize RMS amplitude to a fixed level (removes mic gain variation between sessions)
4. Segment into 1-second windows with 500 ms hop (50% overlap). 1 second is the minimum reliable window for YAMNet — it processes audio in 960 ms internal patches, so shorter clips degrade embedding quality. Train and infer on the same window size to avoid a mismatch.
5. Compute mel spectrogram: `n_fft=1024`, `hop_length=512`, `n_mels=128`, convert to dB
6. Compute MFCC: 20 base coefficients + 20 delta (first derivative) = 40 total. Pure base MFCCs capture static spectral shape well for hovering drones, but real flight involves constant RPM adjustment, approach, recession, and passing — the delta captures the rate of spectral change across frames, which is informative for these dynamic cases. Delta-delta (second derivative) adds marginal gain here since the BiLSTM in Track B already models acceleration implicitly; omit it to keep dimensionality at 40. Compute delta with `librosa.feature.delta(mfcc, order=1)`.
7. Return both as tensors

Note for Track A: YAMNet takes raw waveform directly, not mel spectrogram. The mel spectrogram + MFCC pipeline is used in Track B. Write it now anyway — it will be needed and the preprocessing logic is the same.

**Checkpoint:** Run on 10 clips. Plot spectrograms. Drone clips must show harmonic bands. Wind/ambient clips must look different. If they look the same, debug before proceeding.

---

## Step 4 — Fine-Tune YAMNet (~1-2 days)

Write `src/model/train.py`.

YAMNet takes raw 16 kHz waveform and outputs a 1024-dim embedding per 960 ms frame. Add a binary classification head on top.

**Option A — Fast (do this first):** Pre-extract embeddings for all DroneAudioset clips, save them, then train a small classifier on the saved embeddings. Trains in minutes. Good enough for Track A.

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_embedding(waveform):
    # waveform: float32 tensor, 16 kHz mono, values in [-1, 1]
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # average across frames → one vector per clip
    return tf.reduce_mean(embeddings, axis=0).numpy()  # shape: (1024,)

# Run extract_embedding() on all clips, save embeddings + labels to disk
# Then train a small Dense classifier:
classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(1024,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
```

**Option B — Full fine-tuning:** Train YAMNet end-to-end with the classification head. Better accuracy but slower. Switch to this if Option A doesn't reach the checkpoint.

**Data split:** Split by recording session or clip group, not randomly by individual clip. Clips from the same session sound nearly identical — random splitting leaks information and gives falsely high accuracy.

**Class imbalance:** If non-drone clips outnumber drone clips, pass `class_weight` to `model.fit()` or use weighted loss. Prioritize recall — missing a drone is worse than a false alarm.

**Checkpoint:** On held-out DroneAudioset clips — recall >90% on drone clips, fewer than 1 false positive per minute of pure background audio. If not reached, add more non-drone clips or tune the decision threshold (lower than 0.5 increases recall at cost of more false alarms).

---

## Step 5 — Intensity Distance Proxy (~2-3 hours, can run in parallel with Step 4)

Write `src/distance/intensity_proxy.py`. Pure math — no hardware needed, fully testable.

```python
import math, json

BRACKETS = [
    (0,   50,  "Close"),
    (50,  150, "Medium"),
    (150, 300, "Far"),
    (300, float('inf'), "Likely not detected"),
]

def save_calibration(drone_type: str, d_ref: float, a_ref: float, path="calibration.json"):
    try:
        with open(path) as f:
            cal = json.load(f)
    except FileNotFoundError:
        cal = {}
    cal[drone_type] = {"d_ref": d_ref, "a_ref": a_ref}
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)

def estimate_distance(a_measured: float, drone_type: str, path="calibration.json") -> dict:
    with open(path) as f:
        cal = json.load(f)[drone_type]
    d_est = cal["d_ref"] * math.sqrt(cal["a_ref"] / a_measured)
    bracket = next(label for lo, hi, label in BRACKETS if lo <= d_est < hi)
    return {"distance_m": round(d_est, 1), "bracket": bracket}
```

Formula: `d_estimated = d_ref × sqrt(A_ref / A_measured)` (Inverse Square Law).

Test with synthetic values: if `a_ref=1.0` at `d_ref=10m` and you measure `a_measured=0.25`, result should be `d_est=20m`, bracket `Close`.

Calibration procedure (done once per drone type when hardware arrives):
1. Place mic on tripod
2. Hover drone at exactly 10 m
3. Record 30 seconds
4. Compute mean RMS of the recording → this is `A_ref`
5. Call `save_calibration(drone_type, d_ref=10.0, a_ref=A_ref)`

---

## Step 6 — Inference Loop (~3-4 hours)

Write `src/detection/inference.py`. Reads audio from a file in sliding 1-second windows for now; swap in mic input once hardware arrives — the detection logic does not change.

Latency profile: the buffer must fill for 1 second before the first prediction. After that, a new prediction fires every 500 ms (the hop). First detection latency ≈ 1 second; ongoing update rate = 500 ms.

```python
import soundfile as sf
import numpy as np
from src.preprocessing.pipeline import load_and_preprocess, segment, compute_rms
from src.distance.intensity_proxy import estimate_distance

def run_on_file(audio_path, model, calibration_path, drone_type):
    waveform, sr = load_and_preprocess(audio_path)
    windows = segment(waveform, window_s=1.0, hop_s=0.5, sr=sr)

    for i, window in enumerate(windows):
        prob = float(model.predict(window[np.newaxis, :])[0][0])
        detected = prob > 0.5

        if detected:
            rms = compute_rms(window)
            result = estimate_distance(rms, drone_type, calibration_path)
            print(f"t={i*0.5:.1f}s | DRONE ({prob:.2f}) | {result['bracket']} ({result['distance_m']} m)")
        else:
            print(f"t={i*0.5:.1f}s | no drone ({prob:.2f})")
```

When hardware arrives, add a real-time loop using `sounddevice.InputStream` feeding into the same `window → model.predict → estimate_distance` logic.

---

## Step 7 — Triangulation Logic (~3-4 hours)

Write `src/triangulation/triangulate.py`. Pure geometry — no hardware needed.

Input from each mic: `{mic_id, lat, lon, height_m, distance_bracket, timestamp}`

```python
import numpy as np

BRACKET_RADIUS = {
    "Close": 25.0,
    "Medium": 100.0,
    "Far": 225.0,
}

def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    # flat-earth approximation, valid for distances < 5 km
    x = (lon - ref_lon) * math.cos(math.radians(ref_lat)) * 111320
    y = (lat - ref_lat) * 111320
    return x, y

def triangulate(detections: list[dict]) -> dict:
    # detections: list of {mic_id, lat, lon, distance_bracket}
    # returns: {centroid_lat, centroid_lon, radius_m} or None
    if len(detections) < 2:
        return None

    ref_lat = detections[0]["lat"]
    ref_lon = detections[0]["lon"]

    circles = []
    for d in detections:
        x, y = latlon_to_xy(d["lat"], d["lon"], ref_lat, ref_lon)
        r = BRACKET_RADIUS[d["distance_bracket"]]
        circles.append((x, y, r))

    # grid search over candidate positions, find point minimizing distance-circle residuals
    xs = [c[0] for c in circles]
    ys = [c[1] for c in circles]
    cx, cy = np.mean(xs), np.mean(ys)
    span = max(c[2] for c in circles) * 2

    best_point = None
    best_score = float('inf')
    for dx in np.linspace(-span, span, 100):
        for dy in np.linspace(-span, span, 100):
            px, py = cx + dx, cy + dy
            score = sum((math.sqrt((px-x)**2 + (py-y)**2) - r)**2 for x, y, r in circles)
            if score < best_score:
                best_score = score
                best_point = (px, py)

    # convert best_point back to lat/lon
    result_lat = ref_lat + best_point[1] / 111320
    result_lon = ref_lon + best_point[0] / (111320 * math.cos(math.radians(ref_lat)))
    return {"lat": result_lat, "lon": result_lon, "radius_m": math.sqrt(best_score / len(circles))}
```

Test with synthetic mic positions (triangle with 100 m sides) and synthetic brackets. Verify the returned area contains the drone position you placed in the test.

---

## Step 8 — ESP32-S3 Firmware (C/C++, once hardware arrives)

Write in C/C++ using ESP-IDF. The ESP32-S3's only job:
1. Capture audio from ICS-43434 over I²S at 16 kHz, 16-bit mono
2. Stream raw PCM audio to the central laptop over WiFi (UDP for low latency)

The laptop receives the UDP stream and feeds it into the `run_realtime()` inference function. The model and all processing stay on the laptop — the ESP32-S3 never runs the model.

This step requires hardware. Do not start until the ESP32-S3 and ICS-43434 arrive.

---

## Suggested Timeline

| Day | Task |
|---|---|
| 1 | Environment setup, project structure, download DroneAudioset |
| 2 | Explore dataset (notebook), write and validate preprocessing pipeline |
| 3 | Preprocessing checkpoint: spectrograms look correct. Start YAMNet embedding extraction |
| 4 | Train binary classifier on embeddings, evaluate recall |
| 5 | If checkpoint passed: write inference loop + distance proxy. If not: debug model |
| 6–7 | Write triangulation, run end-to-end test on DroneAudioset audio files |
| Hardware arrival | Integration: connect ICS-43434, test live mic input, calibrate distance proxy |

---

## Checkpoints Summary

| Step | Checkpoint |
|---|---|
| Preprocessing | Drone spectrograms show horizontal harmonic bands below 1 kHz |
| YAMNet fine-tuning | Recall >90% on held-out DroneAudioset clips; <1 false positive per minute of background |
| Distance proxy | Synthetic test: known input → correct bracket output |
| Inference loop | Runs on a DroneAudioset drone clip and prints detections at correct timestamps |
| Triangulation | Synthetic 3-mic test: estimated area contains the placed drone position |
| Hardware integration | Live mic detects a hovering drone at 30 m with <1 false positive per minute |
