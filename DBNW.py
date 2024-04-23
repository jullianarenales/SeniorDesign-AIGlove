import time
import os
import sys
# os.add_dll_directory("E:\\Users\\amade\\opencvGPU\\build\\bin")
# sys.path.append("E:\\Users\\amade\\anaconda3\\Lib\\site-packages")
import cv2
import numpy as np
#import mediapipe as mp
import pyrealsense2 as rs
import speech_recognition as sr
import pyttsx3
import spacy
import os.path
import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(38,GPIO.OUT)
GPIO.setup(37,GPIO.OUT)
GPIO.setup(36,GPIO.OUT)
GPIO.setup(35,GPIO.OUT)
GPIO.setup(33,GPIO.OUT)



# import simpleaudio as sa
print(cv2.cuda.getCudaEnabledDeviceCount())

# Check if CUDA is available
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

# Get current working directory
current_dir = os.getcwd()


# Specify the path to the bag file
# Configure depth and color streams from Intel RealSense
pipeline = rs.pipeline()
config = rs.config()

# Get the device product line
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))


# Enable depth stream with the same resolution as the color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

color_sensor = pipeline_profile.get_device().first_color_sensor()
current_brightness = color_sensor.get_option(rs.option.brightness)
print(f"Current brightness: {current_brightness}")
new_brightness = 8  # Adjust this value as needed (0-255)
color_sensor.set_option(rs.option.brightness, new_brightness)
print(f"Brightness set to: {new_brightness}")

# Start streaming
pipeline.start(config)


# Load pre-trained model for object detection (modify paths as needed)
prototxt_path = os.path.join(current_dir, "MobileNetSSD_deploy.prototxt.txt")
caffemodel_path = os.path.join(current_dir, "MobileNetSSD_deploy.caffemodel")

# Activate CUDA support
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
if cuda_available:
    print("CUDA is available. Enabling CUDA support in OpenCV.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("CUDA is not available. Using CPU for OpenCV operations.")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "scissors",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Speech recognition functions
print("def speak")
# def speak(text):
#     engine.say(text)
#     engine.runAndWait()


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        #speak("Listening...")
        
        audio = recognizer.listen(source)
        print("listen")
        try:
            speech_text = recognizer.recognize_google(audio)
            print("You said: " + speech_text)
            return speech_text
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError:
            print("Could not request results from the service")


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
def transform_vector(vector, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                [0, np.sin(angle_rad), np.cos(angle_rad)]])
    transformed_vector = np.dot(rotation_matrix, vector)
    return transformed_vector

def get_finger_direction(vector):
    # Assuming the user is facing the positive Y direction
    x, y, _ = vector
    if x < -0.2:
        GPIO.output(33, GPIO.HIGH)
        GPIO.output(37, GPIO.LOW)
        GPIO.output(36, GPIO.LOW)
        GPIO.output(35, GPIO.LOW)
        GPIO.output(38, GPIO.LOW)
        return 'thumb'
    elif -0.2 <= x < -0.1:
        GPIO.output(38, GPIO.LOW)
        GPIO.output(35, GPIO.HIGH)
        GPIO.output(36, GPIO.LOW)
        GPIO.output(37, GPIO.LOW)
        GPIO.output(33, GPIO.LOW)
        return 'index'
    elif -0.1 <= x <= 0.1:
        GPIO.output(38, GPIO.LOW)
        GPIO.output(37, GPIO.LOW)
        GPIO.output(36, GPIO.HIGH)
        GPIO.output(35, GPIO.LOW)
        GPIO.output(33, GPIO.LOW)
        return 'middle'
    elif 0.1 < x <= 0.2:
        GPIO.output(38, GPIO.LOW)
        GPIO.output(35, GPIO.LOW)
        GPIO.output(36, GPIO.LOW)
        GPIO.output(37, GPIO.HIGH)
        GPIO.output(33, GPIO.LOW)
        return 'ring'
    else:
        GPIO.output(33, GPIO.LOW)
        GPIO.output(37, GPIO.LOW)
        GPIO.output(36, GPIO.LOW)
        GPIO.output(35, GPIO.LOW)
        GPIO.output(38, GPIO.HIGH)
        return 'pinky'
# Function to find the center of the bright green glove
def find_glove_center(color_image_rgb):
    # Convert the color image from RGB to HSV color space
    hsv_image = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

    # Define the lower and upper bounds of the bright green to bright yellow color range in HSV
    lower_color = np.array([60, 75, 150])
    upper_color = np.array([90, 100, 255])

    # Create a binary mask based on the color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Apply morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assumed to be the glove)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)

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


# def calculate_beep_frequency(vector_length):
#     # Map vector length to beep frequency
#     # Adjust these parameters as needed
#     min_frequency = 1000  # Minimum beep frequency
#     max_frequency = 5000  # Maximum beep frequency
#     max_vector_length = 300  # Maximum length of vector for max frequency
#     min_vector_length = 50  # Minimum length of vector for min frequency

#     # Calculate frequency based on vector length
#     if vector_length < min_vector_length:
#         return max_frequency
#     elif vector_length > max_vector_length:
#         return min_frequency
#     else:
#         return max_frequency - ((vector_length - min_vector_length) / (max_vector_length - min_vector_length)) * (
#                 max_frequency - min_frequency)


# Define a square size for depth calculation
square_size = 5

# Recognize speech and extract object
spoken_text = recognize_speech()
object_of_interest = spoken_text
print("Object of interest:", spoken_text)

# Variables for performance tracking
start_time = time.time()
start_time_30 = time.time()
frame_count = 0
frame_count_30 = 0
fps_window = 500
fps_accumulator = 0
total_fps = 0

# align_to = rs.stream.color
# align = rs.align(align_to)

# try:
#     object_center = None
#     object_point = None
#     # palm_center_depth = None
#     # palm_coord = None
#     endX = 0
#     endY = 0
#     startX = 0
#     startY = 0
#     namedClass = "bottle"
#     confidence = 0.0
#     while True:
#         glove_center=None
#         glove_point=None
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         color_image = np.asanyarray(color_frame.get_data())

#         # Convert color space from RGB to BGR
#         color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

#         depth_image = np.asanyarray(depth_frame.get_data())
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         # Convert the frame to a blob and pass the blob through the network
#         blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 0.007843, (300, 300), 127.5)
#         net.setInput(blob)
#         detections = net.forward()

        

#         # Loop over the detections
#         for i in np.arange(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.7:
#                 idx = int(detections[0, 0, i, 1])
#                 if CLASSES[idx] == object_of_interest:
#                     box = detections[0, 0, i, 3:7] * np.array(
#                         [color_image.shape[1], color_image.shape[0], color_image.shape[1],
#                          color_image.shape[0]])
#                     (startX, startY, endX, endY) = box.astype("int")
#                     object_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
#                     object_point = get_3d_coordinates(depth_frame, object_center[0], object_center[1])
#                     object_avg_depth = get_average_depth(depth_frame, (startX, startY, endX - startX, endY - startY))
#                     namedClass = CLASSES[idx]
#         cv2.rectangle(color_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
#         cv2.putText(color_image, f"{namedClass}: {confidence:.2f}", (startX, startY - 15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


#         glove_center = find_glove_center(color_image_bgr)
            

#         if glove_center is not None:
#             cv2.circle(color_image_bgr, glove_center, 5, (0, 0, 255), -1)
#             glove_point = get_3d_coordinates(depth_frame, glove_center[0], glove_center[1])

       

#         #print(f"Object Point: {object_point}")  # Print the object point
#         #print(f"Glove Point: {glove_point}") 
        

#         # Calculate and display the vector
#         if object_point is not None and glove_point is not None:
#             vector = np.array(object_point) - np.array(glove_point)
            
#             # Transform the vector to account for the 45-degree angle
#             transformed_vector = transform_vector(vector, 45)
            
#             # Determine the finger direction based on the transformed vector
#             finger_direction = get_finger_direction(transformed_vector)
            
#             vector_label = f"Vector: X={vector[0]:.2f}, Y={vector[1]:.2f}, Z(depth)={vector[2]:.2f}"
#             direction_label = f"Direction: {finger_direction}"
#             cv2.putText(color_image_bgr, vector_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             cv2.putText(color_image_bgr, direction_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#             #print(vector_label)
#             print(direction_label)

#             # Draw the vector line from glove to object
#             if object_center is not None and glove_center is not None:
#                 cv2.line(color_image_bgr, glove_center, object_center, (255, 0, 0), 2)

#         # Display both color and depth images
#         cv2.imshow('RealSense Color', color_image_bgr)
#         cv2.imshow('RealSense Depth', depth_colormap)



#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# except RuntimeError as e:
#     print(f"Error: {e}")
GPIO.output(38,GPIO.LOW)
GPIO.output(37,GPIO.LOW)
GPIO.output(36,GPIO.LOW)
GPIO.output(35,GPIO.LOW)
GPIO.output(33,GPIO.LOW)
last_object_center = None
last_object_point = None
try:
    while True:
        

        object_point=None
        object_center=None
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert color space from RGB to BGR
        color_image_bgr = color_image#cv2.cvtColor(color_image, cv2.COLOR_RGB)
        
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Convert the frame to a blob and pass the blob through the network
        blob = cv2.dnn.blobFromImage(cv2.resize(color_image_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()


        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == object_of_interest:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [color_image_bgr.shape[1], color_image_bgr.shape[0], color_image_bgr.shape[1],
                         color_image_bgr.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    object_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
                    object_point = get_3d_coordinates(depth_frame, object_center[0], object_center[1])
                    #object_avg_depth = get_average_depth(depth_frame, (startX, startY, endX - startX, endY - startY))
                    cv2.rectangle(color_image_bgr, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(color_image_bgr, f"{CLASSES[idx]}: {confidence:.2f}", (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    last_object_center=object_center
                    last_object_point= object_point
        # Find the center of the bright green glove
        glove_center = find_glove_center(color_image_bgr)
        if object_center is None:
            object_center = last_object_center
            object_point = last_object_point
        
        if object_center is not None:
            cv2.circle(color_image_bgr, object_center,5,(0,255,0),-1)

        

        if glove_center is not None:
            glove_circle=cv2.circle(color_image_bgr, glove_center, 5, (0, 0, 255), -1)
            
            glove_point = get_3d_coordinates(depth_frame, glove_center[0], glove_center[1])
        else:
            glove_point=None
            glove_circle=None
        #print(f"Object Point: {object_point}")  # Print the object point
        #print(f"Glove Point: {glove_point}") 
        

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
            cv2.putText(color_image_bgr, direction_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #print(vector_label)
            print(direction_label)

            # Draw the vector line from glove to object
            if object_center is not None and glove_center is not None:
                cv2.line(color_image_bgr, glove_center, object_center, (255, 0, 0), 2)

        # Display both color and depth images
        cv2.imshow('RealSense Color', color_image_bgr)
        cv2.imshow('RealSense Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except RuntimeError as e:
    print(f"Error: {e}")
finally:
    pipeline.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()