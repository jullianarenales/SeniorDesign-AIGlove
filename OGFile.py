import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import speech_recognition as sr
import pyttsx3
import spacy

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2)

# Configure depth and color streams from Intel RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load pre-trained model for object detection (modify paths as needed)
net = cv2.dnn.readNetFromCaffe(r'MobileNetSSD_deploy.prototxt.txt',
                               r'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Speech recognition functions
def speak(text):
    engine.say(text)
    engine.runAndWait()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        audio = recognizer.listen(source)
        try:
            speech_text = recognizer.recognize_google(audio)
            print("You said: " + speech_text)
            return speech_text
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError:
            print("Could not request results from the service")


def extract_object(sentence):
    doc = nlp(sentence)
    for token in doc:
        if "obj" in token.dep_:
            return token.text
    return None


# Function to get 3D coordinates
def get_3d_coordinates(depth_frame, x, y):
    if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
        depth = depth_frame.get_distance(x, y)
        depth_point = rs.rs2_deproject_pixel_to_point(
            depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
            [x, y], depth)
        return np.array(depth_point)
    else:
        return None


# Function to get average depth, ignoring zero values
def get_average_depth(depth_frame, bbox):
    x, y, w, h = bbox
    depth_sum = 0
    count = 0
    depth_frame_width = depth_frame.get_width()
    depth_frame_height = depth_frame.get_height()

    # Define the size of the square (e.g., 5x5 pixels)
    square_size = 5
    half_square = square_size // 2

    for dx in range(-half_square, half_square + 1):
        for dy in range(-half_square, half_square + 1):
            sample_x = x + dx
            sample_y = y + dy

            # Ensure the sample points are within the frame boundaries
            if 0 <= sample_x < depth_frame_width and 0 <= sample_y < depth_frame_height:
                depth = depth_frame.get_distance(sample_x, sample_y)
                if depth > 0:  # Ignore zero values
                    depth_sum += depth
                    count += 1

    if count > 0:
        return depth_sum / count  # Return the average depth
    else:
        return None


# Define a square size for depth calculation
square_size = 5

# Recognize speech and extract object
spoken_text = recognize_speech()
object_of_interest = extract_object(spoken_text) if spoken_text else None
print("Object of interest:", object_of_interest)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Convert the frame to a blob and pass the blob through the network
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        object_center = None
        object_point = None

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == object_of_interest:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [color_image.shape[1], color_image.shape[0], color_image.shape[1],
                         color_image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    object_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
                    object_point = get_3d_coordinates(depth_frame, object_center[0], object_center[1])
                    object_avg_depth = get_average_depth(depth_frame, (startX, startY, endX - startX, endY - startY))
                    cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(color_image, f"{CLASSES[idx]}: {confidence:.2f}", (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Hand Tracking
        results_hands = hands.process(color_image)
        palm_center_depth = None

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Calculate the middle palm point
                palm_x = sum([hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]]) / 5
                palm_y = sum([hand_landmarks.landmark[i].y for i in [0, 5, 9, 13, 17]]) / 5
                palm_coord = (int(palm_x * color_image.shape[1]), int(palm_y * color_image.shape[0]))

                # Draw a circle at the middle palm point
                cv2.circle(color_image, palm_coord, 5, (0, 255, 0), -1)

                # Get 3D coordinates at the middle palm point
                palm_center_depth = get_3d_coordinates(depth_frame, palm_coord[0], palm_coord[1])

        # Calculate and display the vector
        if palm_center_depth is not None and object_point is not None:
            vector = np.array(object_point) - np.array(palm_center_depth)
            vector_label = f"Vector: X={vector[0]:.2f}, Y={vector[1]:.2f}, Z(depth)={vector[2]:.2f}"
            cv2.putText(color_image, vector_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(vector_label)

            # Draw the vector line from palm to object
            if object_center is not None and palm_coord is not None:
                cv2.line(color_image, palm_coord, object_center, (255, 0, 0), 2)

            # Display both color and depth images
        cv2.imshow('RealSense Color', color_image)
        cv2.imshow('RealSense Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()