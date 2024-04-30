import cv2
import numpy as np
import pyrealsense2 as rs


class GloveDetector:
    def __init__(self):
        pass

    def find_glove_center(self, color_image_rgb):
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

    def get_3d_coordinates(self, depth_frame, x, y):
        if 0 <= x < depth_frame.get_width() and 0 <= y < depth_frame.get_height():
            depth = depth_frame.get_distance(x, y)
            depth_point = rs.rs2_deproject_pixel_to_point(
                depth_frame.get_profile().as_video_stream_profile().get_intrinsics(),
                [x, y], depth)
            return np.array(depth_point)
        else:
            return None
