import cv2
import numpy as np
import mediapipe as mp
import time
import spacy

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2)

# Flag to check if RealSense camera is connected
realsense_connected = False

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")


def extract_object(sentence):
    doc = nlp(sentence)
    for token in doc:
        if "obj" in token.dep_:
            return token.text
    return None


# Attempt to start RealSense pipeline
try:
    import pyrealsense2 as rs

    # Configure depth and color streams from Intel RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    realsense_connected = True

    # Load pre-trained model for object detection (modify paths as needed)
    net = cv2.dnn.readNetFromCaffe(r'C:\Users\korth\Downloads\MobileNetSSD_deploy.prototxt.txt',
                                   r'C:\Users\korth\Downloads\MobileNetSSD_deploy.caffemodel')
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

except Exception as e:
    print(f"Error: {e}")
    print("RealSense camera not connected. Switching to webcam...")

# Attempt to start webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Webcam not available. Please check the connection.")
else:
    print("Webcam started successfully.")

# Variables for performance tracking
start_time = time.time()
frame_count = 0


object_of_interest = extract_object("person")

while True:
    if realsense_connected:
        # Wait for the next set of frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
    else:
        # If RealSense is not connected, use webcam
        _, color_frame = cap.read()

    # Check if frames were successfully retrieved
    if color_frame is None:
        continue

    # Convert color frame to BGR format for OpenCV
    color_image = np.asanyarray(color_frame)

    # Process the frame with MediaPipe Hands
    results_hands = hands.process(color_image)

    # Draw the center point of the detected hand
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Calculate the middle point of the hand
            h, w, c = color_image.shape
            cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w), int(
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)
            cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)

    # Display the color image
    cv2.imshow('Image', color_image)

    # Increment frame count
    frame_count += 1

    # Calculate and display frame rate every 5 seconds
    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        frame_rate = frame_count / elapsed_time
        print(f"Frame rate: {frame_rate:.2f} fps")
        start_time = time.time()
        frame_count = 0

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
if realsense_connected:
    pipeline.stop()
cap.release()
cv2.destroyAllWindows()
