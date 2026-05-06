This is a fascinating project! Building a "sonic radar" for drones is a clever way to handle detection, especially in environments where visual line-of-sight is blocked. Since you’re starting from scratch with AI and audio, let's break this down into a roadmap that won't leave your head spinning.

---

## 1. Addressing Your Specific Questions

### One Microphone or Multiple?
If your goal is **classification** (identifying *if* it is a drone) and **distance estimation**, you can achieve this with **one high-quality microphone**. 

* **Single Microphone:** Relies on **Sound Pressure Level (SPL)**. As the drone moves away, the intensity decreases following the **Inverse Square Law**: 
    $$I \propto \frac{P}{4\pi r^2}$$
    (Where $I$ is intensity, $P$ is power, and $r$ is distance). Your model will learn to associate specific volume levels and frequency dampening with distance.
* **Multiple Microphones (Array):** This is better for **localization** (knowing *where* the drone is in 3D space). It uses time-difference-of-arrival (TDOA) to triangulate position. 
* **Recommendation:** Start with **one consistent microphone** type to reduce variables. If you want the model to work on *any* phone or device, you'll eventually need to record with multiple different microphones so the AI doesn't get "addicted" to the specific quirks of one sensor.

### Should you filter frequencies?
**Yes, but carefully.** Drones have a very specific "acoustic signature" created by their motors and propeller blades.
* **Low-pass/High-pass:** You should generally filter out very low-frequency "rumble" (wind noise, traffic) below **100Hz** and very high-frequency "hiss" above **15kHz-20kHz** that the human ear (and most drone motors) doesn't occupy.
* **The "Drone Signature":** Drones produce **harmonics**. If a motor spins at a certain RPM, it creates a fundamental frequency and several "multiples" above it. You want to keep these intact.

### How to "pass" audio to the AI?
You don't usually feed raw audio (.wav files) directly into a model. It’s too much data and very "noisy." Instead, you convert audio into an **image**.
* **The Spectrogram:** This is a visual representation of the spectrum of frequencies of a signal as it varies with time.
* **Mel-Spectrogram:** This is a version of a spectrogram that scales frequencies to match how humans actually hear. This is the "gold standard" input for audio AI.



### What type of Machine Learning model?
Since we are turning audio into "images" (spectrograms), the best model is a **CNN (Convolutional Neural Network)**. 
* **Why CNNs?** They are designed to find patterns in images. A CNN can "see" the horizontal lines of a drone's motor harmonics in a spectrogram and distinguish them from the chaotic "blobs" of bird chirps or wind.

---

## 2. The Macro Plan: Your Step-by-Step Roadmap

### Phase 1: Research & Hardware Setup
* **Understand the "Inverse Square Law":** Research how sound drops off over distance so you understand what your model is trying to learn.
* **Pick your Mic:** Use a "Measurement Microphone" or a high-quality omnidirectional mic if possible.
* **Sync Tooling:** Create a way to timestamp your audio recordings with a GPS or laser-rangefinder distance reading. Without accurate "ground truth" (knowing the *actual* distance during the recording), your model will just be guessing.

### Phase 2: Data Collection (The Most Important Part)
* **The "Clean" Set:** Record the drone at 1m, 5m, 10m, 50m, 100m in a quiet field.
* **The "Noisy" Set:** Record the same distances but with background noise (wind, cars, people talking). 
* **Variety:** Record different drones if possible. A DJI Mavic sounds very different from a high-pitched FPV racing drone.

### Phase 3: Preprocessing (The "Spectrogram Factory")
* **Segmenting:** Chop your long recordings into 1-second or 3-second "chunks."
* **Fourier Transform:** Use a Python library called `Librosa` to convert these chunks into **Mel-Spectrograms**.
* **Labeling:** Each image must be labeled with the drone type and the distance (e.g., `mavic_50meters.png`).

### Phase 4: Model Training
* **Architecture:** Use a pre-trained model like **ResNet** or **MobileNet** (which is lightweight and could run on a phone/drone controller).
* **Training:** Feed the images into the model. The model looks at the image and tries to predict the distance. If it's wrong, it adjusts its internal math and tries again.

### Phase 5: Testing & Deployment
* **Validation:** Test the model on audio it has **never heard before**. 
* **Latency:** Since drones move fast, your model needs to be able to process the audio in "real-time" (less than 100ms).

---

## 3. Things to Consider Before You Start
* **The Doppler Effect:** If a drone is flying toward the mic at high speed, the pitch will shift higher; as it passes, it shifts lower. Your model needs to be trained on "moving" samples, not just stationary ones.
* **Wind Noise:** Wind hitting a microphone membrane sounds like low-frequency explosions. You **must** use a "deadcat" (windjammer) over your mic, or your data will be useless.
* **Environmental Factors:** Humidity and temperature change how far sound travels. For a "precise" model, you might eventually need to include those as extra data points.

> **Pro Tip:** Start small. Don't try to predict the distance to the exact centimeter. Try to get the model to categorize distances into "buckets" first (e.g., Close, Medium, Far). Once that works, move to specific meter estimations.

How do you plan on measuring the "live distance" during your recordings—GPS, or are you thinking of using a fixed-distance setup?
