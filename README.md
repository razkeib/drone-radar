# drone-radar

AI Prompt:
I want to create a drone classification and distance measurement model based on audio. My long term plan is to later process the data from multiple microphones and extract drone position based on triangulation, and maybe train the model with multiple drones if it will improve it (tell me what you think about this).
Currently my plan is to record as many samples of different drones, both stationary and moving, ideally using an omnidirectional measurement mic, while measuring live distance to the drone based on gps tracker on microphone and drone, and pass it to data training with the audio and distance together.

It is worth noting that I have little knowledge of machine learning and AI, and little-to-no knowledge of audio and sound.
The pinpoints and questions that I have in my mind:
I am aware of a Fourier Transform formula that I should use, but don't know what exactly it does and how to use it.
should I record samples with multiple microphones or just one? meaning, should I choose the best fit microphone for my use case and record the samples only with that microphone, or use multiple different ones? which option will give me the best, most precise classification and distance measurement model?
does the microphone membrane matter? should I buy an analog microphone and set it up or a digital one? if it's digital, then I understand that there are usually filters as part of it. do I want those? what should I filter out, and how?
what do I do with the audio in order to pass it on to the data training?
what type of machine learning model should I use? I understood there are different types, expecting different inputs and outputs, with different mathematical formulas.

Finally, I need a macro plan to understand how I should work, which things I should consider and check out before I act, and what I should do step by step.
