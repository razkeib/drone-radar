# Problem
As of today (29/04/2026), drones pose a threat on soldiers and civilians due to their cost/effect ratio and the difficulty to counter against them.
They are hard to counter for a number of reasons:
Detection:
1. Small Profile - Sonar-based radars have trouble detecting small objects, and even if they succeed, they frequently get false-positives. Image based detections have a lot of noise and have trouble detecting small objects (max 2x2 meter object at height in the sky).
2. No Heat Signature - Due to their electronic-based propellor rotation (the main propulsion), they create minimal heat and therefore are hard to detect with thermal radars.
3. Multiple RF Frequencies - Drones can operate on countless frequencies, making it hard to find the channel the detected drone is working on.
4. Fiber Optic Drones - Fiber optic wires can be attached to the drones, providing them a distance of up to 5-20 km, and sometimes even 50, while eliminating any detection or jamming based on RF.
5. Autonomus Drones - GPS-navigated drones can fly themselves without transmitting or receiving any signal, other than GPS, therefore eliminating RF-based detection.

Neutralization:
Due to their small profile and versatile movement, even if detected, they are hard to take down.
1. RF-Based Jamming - Requires knowing exactly on what frequency the drone is transmitting/receiving on and also requires the capability to jam on that frequency.
2. Electro-pulse - Requires the capability of sending an electro pulse in a specific direction (expensive equipment and requires high degree of engineering)
3. Physical - Hitting the drone physically requires precise aiming, whether it be by drone, net or bullets. The further the drone, the harder it gets.


# Solution
In this project, we'll take a look at multiple solutions to multiple problems.

Detection:
1. Sound-Based Detection Based On Classification And Triangulation
  Can work in bad weather, Cheap. Requires training a classification model for drones (the hard part). Creating a tower and RX/TX protocol to receive all microphones (may already exist).
2. Microphone Array
  Image which shows where in the image the sound is coming from while looking in the direction (gives only 1 vector. should be paired with other detectors, if used)


Neutralization:
1. M203 Shell Casing Net
  Good for close range, can be easily and quickly shot, and can be equipped on any soldier. Light-weight, cheap.
2. RF-Jamming
  Can take down drones from far away and probably be activated at all times. Requires knowing enemy drone frequencies and is limited to remote-controlled drones.


# Plan
Sound-Based Detection Based On Classification And Triangulation.
The easiest, fastest, cheapest and most effective product. POC can be done pretty quickly.
If doesn't succeed, can be great for resume and self development.

## POC Compromises
- Digital microphones instead of analog: signals already developed. Takes away the hassle of engineering a custom PCB
- Communication between microphones and control: Can be engineered after purchase by client to accustom to their needs
- Signal Encryption: Encryption can be done later, either by client or by us if needed.
- Drone neutralization: There are multiple ways and a huge competition to take down drones that work. There are minimal solutions to drone detection. Therefore the detection will be our main focus.

## Software
### Model Training (The Hard Part)
1. Gather as many examples of drone sounds as possible, ideally with classification of their type, and sounds that sound like drones (Tank, plane, bugs)
2. Create database to train model
3. Train model for classification (priority order: Drone or not, type of drone, type of propellors)
4. Test model

### Microphone Data
1. Plan structure of how data will be received from microphones
2. Create an environment to receive data and store it in a relevant database with microphone IDs, geo-locations and audio data for processing (should be low-latency)

### Triangulation
1. If multiple microphones classify drone detection, create triangulation between them and extract exact location (x, y and z - requires checking what's the format convention is for flying objects)
2. Make sure that if there are 2 or more drones, how to triangulate correctly

## Hardware
1. Purchase digital microphones, checking their consumption and attaching a relevant solar panel to them
2. Create signal to transmit and their format

## Military
1. Communication with the military to receive data about requirements and difficulties
2. Coordination to showcase POC and offer selling product



# Team
Raz - Team Leader
Ben - Electrical Engineer
Guy - Military Communicator, Cyber Specialist and Developer
Matan - Mathematician
Almog - Low-Level Software Engineer (Optional)
