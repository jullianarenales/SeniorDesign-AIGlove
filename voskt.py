from vosk import Model, KaldiRecognizer
import pyaudio
import os.path
from pynput import keyboard

current_dir = os.getcwd()
model = Model("/home/team1/SeniorDesign-AIGlove/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

recognized_text = ""
is_listening = False

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def on_press(key):
    global is_listening
    if key == keyboard.KeyCode.from_char('f') and not is_listening:
        is_listening = True
        print("Listening...")

def on_release(key):
    global is_listening, recognized_text
    if key == keyboard.KeyCode.from_char('f') and is_listening:
        is_listening = False
        print("Processing...")
        
        # Process the recognized text and extract the object of interest
        if recognized_text:
            object_of_interest = ""
            words = recognized_text.strip().split()
            if len(words) > 0:
                object_of_interest = words[-1]  # Assume the last word is the object

            # Check if the object of interest is in the CLASSES list
            if object_of_interest in CLASSES:
                print("Object of interest:", object_of_interest)
            else:
                print("Unrecognized object:", object_of_interest)

            recognized_text = ""  # Reset the recognized text
            
    elif key == keyboard.Key.esc:
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    while True:
        if is_listening:
            data = stream.read(4096)
            if recognizer.AcceptWaveform(data):
                text = recognizer.Result()
                recognized_text += text[14:-3].strip() + " "  # Strip whitespace from the recognized text

        if not listener.running:
            break

# Close the audio stream and terminate the script
stream.stop_stream()
stream.close()
mic.terminate()