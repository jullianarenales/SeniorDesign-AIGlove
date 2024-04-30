import time
import cv2
import pyrealsense2 as rs
import os.path
import pyaudio
import numpy as np

from object_detection import ObjectDetector
from glove_detection import GloveDetector
from speech_recognition import SpeechRecognizer

# Define constants and paths
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
caffemodel_path = "MobileNetSSD_deploy.caffemodel"
bag_file_path = "test3.bag"
voice_model_path = "vosk-model-small-en-us-0.15"

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
config.enable_device_from_file(bag_file_path)
pipeline.start(config)

spoken_text = "bottle"
align_to = rs.stream.color
align = rs.align(align_to)
last_object_center = None
last_object_point = None
object_point = None
object_center = None
box = None

def main():

    # Initialize Object Detector
    object_detector = ObjectDetector(prototxt_path, caffemodel_path)

    # Initialize Glove Detector
    glove_detector = GloveDetector()

    # Initialize Speech Recognizer
    speech_recognizer = SpeechRecognizer(voice_model_path)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image_bgr = np.asanyarray(color_frame.get_data())

            object_center, object_point = object_detector.detect_objects(color_image_bgr, CLASSES, "bottle")

            glove_center = glove_detector.find_glove_center(color_image_bgr)


            # Recognize speech and extract object
            audio_data = get_audio_data()
            spoken_text = speech_recognizer.recognize_speech(audio_data)
            object_of_interest = spoken_text

            # Process object detection and glove detection results
            # ...

            if object_center is not None:
                cv2.circle(color_image_bgr, object_center, 5, (0, 255, 0), -1)

            if glove_center is not None:
                glove_circle = cv2.circle(color_image_bgr, glove_center, 5, (0, 0, 255), -1)

                glove_point = glove_detector.get_3d_coordinates(depth_frame, glove_center[0], glove_center[1])
            else:
                glove_point = None
                glove_circle = None

            # Calculate and display the vector
            if object_point is not None and glove_point is not None:
                vector = np.array(object_point) - np.array(glove_point)

                # Transform the vector to account for the 45-degree angle
                transformed_vector = transform_vector(vector, 45)

                # Determine the finger direction based on the transformed vector
                finger_direction = get_finger_direction(transformed_vector)

                vector_label = f"Vector: X={vector[0]:.2f}, Y={vector[1]:.2f}, Z(depth)={vector[2]:.2f}"
                direction_label = f"Direction: {finger_direction}"
                cv2.putText(color_image_bgr, vector_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(color_image_bgr, direction_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            2)
                # print(vector_label)
                print(direction_label)

                # Draw the vector line from glove to object
                if object_center is not None and glove_center is not None:
                    cv2.line(color_image_bgr, glove_center, object_center, (255, 0, 0), 2)

            # Display the processed image
            cv2.imshow('RealSense Color', color_image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def get_finger_direction(vector):
    # Assuming the user is facing the positive Y direction
    x, y, _ = vector
    if x < -0.2:
        # GPIO.output(33, GPIO.HIGH)
        # GPIO.output(37, GPIO.LOW)
        # GPIO.output(36, GPIO.LOW)
        # GPIO.output(35, GPIO.LOW)
        # GPIO.output(38, GPIO.LOW)
        return 'thumb'
    elif -0.2 <= x < -0.1:
        # GPIO.output(38, GPIO.LOW)
        # GPIO.output(35, GPIO.HIGH)
        # GPIO.output(36, GPIO.LOW)
        # GPIO.output(37, GPIO.LOW)
        # GPIO.output(33, GPIO.LOW)
        return 'index'
    elif -0.1 <= x <= 0.1:
        # GPIO.output(38, GPIO.LOW)
        # GPIO.output(37, GPIO.LOW)
        # GPIO.output(36, GPIO.HIGH)
        # GPIO.output(35, GPIO.LOW)
        # GPIO.output(33, GPIO.LOW)
        return 'middle'
    elif 0.1 < x <= 0.2:
        # GPIO.output(38, GPIO.LOW)
        # GPIO.output(35, GPIO.LOW)
        # GPIO.output(36, GPIO.LOW)
        # GPIO.output(37, GPIO.HIGH)
        # GPIO.output(33, GPIO.LOW)
        return 'ring'
    else:
        # GPIO.output(33, GPIO.LOW)
        # GPIO.output(37, GPIO.LOW)
        # GPIO.output(36, GPIO.LOW)
        # GPIO.output(35, GPIO.LOW)
        # GPIO.output(38, GPIO.HIGH)
        return 'pinky'

def transform_vector(vector, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                [0, np.sin(angle_rad), np.cos(angle_rad)]])
    transformed_vector = np.dot(rotation_matrix, vector)
    return transformed_vector

def get_audio_data():
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    audio_data = stream.read(2 * 16000 * 2)  # Read 2 seconds of audio data
    stream.stop_stream()
    mic.terminate()
    return audio_data


main()
