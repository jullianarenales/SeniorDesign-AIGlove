import pyaudio
import os.path
from vosk import Model, KaldiRecognizer

class SpeechRecognizer:
    def __init__(self, model_path):
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)

    def recognize_speech(self, audio_data):
        if self.recognizer.AcceptWaveform(audio_data):
            text = self.recognizer.Result()
            recognized_text = text[text.index('"text"') + 9: text.rindex('"')]  # Extract recognized text
            recognized_text = recognized_text.strip('"')  # Remove extra quotation marks
            return recognized_text
        else:
            return None
