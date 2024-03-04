# This is the original file
import cv2
import numpy as np
import mediapipe as mp
import time
import spacy
import os
import pyrealsense2 as rs

# Enable OpenCV CUDA support if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available")
    cv2.dnn.cuda.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    cv2.dnn.cuda.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2)

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

def extract_object(sentence):
    doc = nlp(sentence)
    for token in doc:
        if "obj" in token.dep_:
            return token.text
    return None

# Specify the path to the bag file
bag_file_path = os.path.join("C:\\", "Users", "Luis", "Documents", "Capstone II", "SeniorDesign-AIGlove", "test.bag")

# Load pre-trained model for object detection (modify paths as needed)
prototxt_path = os.path.join("C:\\", "Users", "Luis", "Documents", "Capstone II", "SeniorDesign-AIGlove", "MobileNetSSD_deploy.prototxt.txt")
caffemodel_path = os.path.join("C:\\", "Users", "Luis", "Documents", "Capstone II", "SeniorDesign-AIGlove", "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
CLASSES = ["background", "bottle", "cat", "chair", "person", "tvmonitor"]

# Create RealSense pipeline and configure
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
config.enable_device_from_file(bag_file_path)
pipeline.start(config)

# Variables for performance tracking
start_time = time.time()
frame_count = 0

try:
    while True:
        # Wait for the next set of frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if frames were successfully retrieved
        if color_frame is None:
            continue

        # Convert color frame to BGR format for OpenCV
        color_image = np.asanyarray(color_frame.get_data())

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

        # Perform object detection
        blob = cv2.dnn.blobFromImage(color_image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections and draw bounding boxes
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #     if confidence > 0.2:
        #         class_id = int(detections[0, 0, i, 1])
        #         if 0 <= class_id < len(CLASSES) and CLASSES[class_id] == object_of_interest:
        #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #             (startX, startY, endX, endY) = box.astype("int")
        #             cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

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

finally:
    # Release the RealSense pipeline resources
    pipeline.stop()
    cv2.destroyAllWindows()
