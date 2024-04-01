import sys
import cv2
import numpy as np
import pyrealsense2 as rs

print(cv2.__version__)

# Configure depth and color streams from the bag file
pipeline = rs.pipeline()
config = rs.config()

try:
    # Specify the path to your bag file
    bag_file_path = r"C:\Users\korth\OneDrive\Documents\20240328_212449.bag"
    
    # Enable device from the bag file
    config.enable_device_from_file(bag_file_path)
    
    # Start streaming from the bag file
    pipeline.start(config)
except RuntimeError as e:
    print(f"Failed to start pipeline: {e}")
    sys.exit(1)

# Load the MobileNet SSD object detection model
net = cv2.dnn.readNetFromCaffe(r'C:\Users\korth\Downloads\MobileNetSSD_deploy.prototxt.txt',
                               r'C:\Users\korth\Downloads\MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Function to get 3D coordinates
# def get_3d_coordinates(depth_frame, x, y):
#     if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
#         depth = depth_frame.get_distance(x, y)
#         depth_point = rs.rs2_deproject_pixel_to_point(
#             depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
#             [x, y], depth)
#         return np.array(depth_point)
#     else:
#         return None
def get_3d_coordinates(depth_frame, x, y, region_size=50):
    if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
        points = []
        depth_frame_width = depth_frame.get_width()
        depth_frame_height = depth_frame.get_height()
        half_region = region_size // 2

        for dx in range(-half_region, half_region + 1):
            for dy in range(-half_region, half_region + 1):
                sample_x = x + dx
                sample_y = y + dy

                # Ensure the sample points are within the frame boundaries
            
                if 0 <= sample_x < depth_frame_width and 0 <= sample_y < depth_frame_height:
                    depth = depth_frame.get_distance(sample_x, sample_y)
                     # Ignore zero values
                    if depth >0:
                        depth_point = rs.rs2_deproject_pixel_to_point(
                            depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                            [sample_x, sample_y], depth)
                        points.append(depth_point)
         

                if points:
                    avg_point = np.mean(points, axis=0)
                    #print(f"3D Coordinates: {avg_point}")  # Print the average 3D coordinates
                    return avg_point
                return None

# Function to get average depth, ignoring zero values
def get_average_depth(depth_frame, bbox):
    x, y, w, h = bbox
    depth_sum = 0
    count = 0
    depth_frame_width = depth_frame.get_width()
    depth_frame_height = depth_frame.get_height()

    # Define the size of the square (e.g., 5x5 pixels)
    square_size = 50
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
        return 'thumb'  # Object is to the left
    elif -0.2 <= x < -0.1:
        return 'index'  # Object is at a 45-degree angle to the left
    elif -0.1 <= x <= 0.1:
        return 'middle'  # Object is straight ahead
    elif 0.1 < x <= 0.2:
        return 'ring'  # Object is at a 45-degree angle to the right
    else:
        return 'pinky'  # Object is to the right

# Function to find the center of the bright green glove
def find_glove_center(color_image_bgr):
    # Convert the color image from BGR to HSV color space
    hsv_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color in HSV
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([60, 255, 255])

    # Create a binary mask based on the green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

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
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert color space from RGB to BGR
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
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
                if CLASSES[idx] == 'bottle':
                    box = detections[0, 0, i, 3:7] * np.array(
                        [color_image_bgr.shape[1], color_image_bgr.shape[0], color_image_bgr.shape[1],
                         color_image_bgr.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    object_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)
                    object_point = get_3d_coordinates(depth_frame, object_center[0], object_center[1])
                    object_avg_depth = get_average_depth(depth_frame, (startX, startY, endX - startX, endY - startY))
                    cv2.rectangle(color_image_bgr, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(color_image_bgr, f"{CLASSES[idx]}: {confidence:.2f}", (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Find the center of the bright green glove
        glove_center = find_glove_center(color_image_bgr)
        

        if glove_center is not None:
            cv2.circle(color_image_bgr, glove_center, 5, (0, 0, 255), -1)
            glove_point = get_3d_coordinates(depth_frame, glove_center[0], glove_center[1])

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
    cv2.destroyAllWindows()