import pyrealsense2 as rs
import numpy as np
import cv2
import base64
from anthropic import Anthropic

# Configure the Intel RealSense D435i pipeline

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

# Anthropic API key
api_key = "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"

# Create an instance of the Anthropic client
client = Anthropic(api_key=api_key)

# Start the pipeline


try:
    for i in range(25):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Keep the last frame
        if i == 24:
            rgb_image = color_image

    # Encode the image as base64
    _, img_encoded = cv2.imencode(".jpg", rgb_image)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    # Send the request to the Claude API using the Anthropic library
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "describe how to model this in solidworks",
                    },
                ],
            },
        ],
    )

    # Extract the image description from the response
    description = response.content[0].text
    print("Image Description:")
    print(description)

finally:
    # Stop the pipeline
    pipeline.stop()
