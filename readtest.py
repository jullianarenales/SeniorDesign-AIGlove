import pyrealsense2 as rs
import numpy as np
import cv2
import pytesseract


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

try:
    # Wait for the 10th frame
    for i in range(10):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
    

    # Convert the color frame to a numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # Convert the color image from BGR to RGB format
    rgb_image = color_image

    # Perform text detection using pytesseract
    text = pytesseract.image_to_string(rgb_image)

    # Print the detected text
    print("Detected Text:")
    print(text)

    # Display the captured image
    cv2.imshow("Captured Image", rgb_image)
    cv2.waitKey(0)

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()



    ###api key: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