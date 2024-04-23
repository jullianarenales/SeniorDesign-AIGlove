import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone with a timeout of 3 seconds
with sr.Microphone() as source:
    print("Please say something within 3 seconds...")
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
    try:
        audio = recognizer.listen(source)
    except sr.WaitTimeoutError:
        print("Timeout: No speech detected. Exiting...")
        exit()

# Recognize speech using Google Web Speech API
try:
    print("Recognizing...")
    text = recognizer.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Sorry, I couldn't understand what you said")
except sr.RequestError as e:
    print("Sorry, I couldn't request results from Google Speech Recognition service; {0}".format(e))
