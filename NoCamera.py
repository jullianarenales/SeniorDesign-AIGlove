import time
import cv2
import numpy as np
import pyrealsense2 as rs
from vosk import Model, KaldiRecognizer
import pyaudio
import os.path

# Check if CUDA is available
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

# Get current working directory
current_dir = os.getcwd()

# Specify the path to the bag file
# Configure depth and color streams from Intel RealSense
pipeline = rs.pipeline()
config = rs.config()

# Create RealSense pipeline and configure
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
bag_file_path = os.path.join(current_dir, "test3.bag")
config.enable_device_from_file(bag_file_path)

# Start streaming
pipeline.start(config)


# Load pre-trained model for object detection (modify paths as needed)
prototxt_path = os.path.join(current_dir, "MobileNetSSD_deploy.prototxt.txt")
caffemodel_path = os.path.join(current_dir, "MobileNetSSD_deploy.caffemodel")
voice_model = Model(current_dir + "/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(voice_model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Activate CUDA support
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
if cuda_available:
    print("CUDA is available. Enabling CUDA support in OpenCV.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("CUDA is not available. Using CPU for OpenCV operations.")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



def recognize_speech():
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    recognized_text = ""
    print("Listening for 2 seconds...")
    data = stream.read(2 * 16000 * 2)  # Read 3 seconds of audio data
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        recognized_text = text[text.index('"text"') + 9: text.rindex('"')]  # Extract recognized text
        recognized_text = recognized_text.strip('"')  # Remove extra quotation marks
        print("You said:", recognized_text)
    else:
        print("Could not understand the audio")
    stream.stop_stream()
    return recognized_text


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


# Haptic Functions
def gpio_pulse():
    print("pulsing")
    # for _ in range(2):
    #     GPIO.output(38, GPIO.HIGH)
    #     GPIO.output(35, GPIO.HIGH)
    #     GPIO.output(36, GPIO.HIGH)
    #     GPIO.output(37, GPIO.HIGH)
    #     GPIO.output(33, GPIO.HIGH)
    #     time.sleep(0.1)
    #     GPIO.output(38, GPIO.LOW)
    #     GPIO.output(35, GPIO.LOW)
    #     GPIO.output(36, GPIO.LOW)
    #     GPIO.output(37, GPIO.LOW)
    #     GPIO.output(33, GPIO.LOW)
    #     time.sleep(0.1)

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


def find_glove_center(color_image_rgb):
    # Convert the color image from RGB to HSV color space
    hsv_image = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

    # Define the lower and upper bounds of the bright green to bright yellow color range in HSV
    lower_color = np.array([55, 100, 100])
    upper_color = np.array([95, 255, 255])

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


def transform_vector(vector, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                [0, np.sin(angle_rad), np.cos(angle_rad)]])
    transformed_vector = np.dot(rotation_matrix, vector)
    return transformed_vector


# Define a square size for depth calculation
square_size = 5

# Recognize speech and extract object
spoken_text = "bottle"

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

align_to = rs.stream.color
align = rs.align(align_to)
last_object_center = None
last_object_point = None
object_point = None
object_center = None
box = None
try:
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Convert color space from RGB to BGR
        color_image_bgr = color_image  # cv2.cvtColor(color_image, cv2.COLOR_RGB)

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
                    # object_avg_depth = get_average_depth(depth_frame, (startX, startY, endX - startX, endY - startY))
                    cv2.rectangle(color_image_bgr, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(color_image_bgr, f"{CLASSES[idx]}: {confidence:.2f}", (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    last_object_center = object_center
                    last_object_point = object_point
        # Find the center of the bright green glove
        glove_center = find_glove_center(color_image_bgr)
        # if object_center is None:
        # object_center = last_object_center
        # object_point = last_object_point

        if object_center is not None:
            cv2.circle(color_image_bgr, object_center, 5, (0, 255, 0), -1)

        if glove_center is not None:
            glove_circle = cv2.circle(color_image_bgr, glove_center, 5, (0, 0, 255), -1)

            glove_point = get_3d_coordinates(depth_frame, glove_center[0], glove_center[1])
        else:
            glove_point = None
            glove_circle = None
        # print(f"Object Point: {object_point}")  # Print the object point
        # print(f"Glove Point: {glove_point}")

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
            # print(vector_label)
            print(direction_label)

            # Draw the vector line from glove to object
            if object_center is not None and glove_center is not None:
                cv2.line(color_image_bgr, glove_center, object_center, (255, 0, 0), 2)

        # Display both color and depth images
        cv2.imshow('RealSense Color', color_image_bgr)
        # cv2.imshow('RealSense Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except RuntimeError as e:
    print(f"Error: {e}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    os._exit(0)