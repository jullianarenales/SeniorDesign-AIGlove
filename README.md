# SeniorDesign-AIGlove
# VisionGuide Haptic Explorer

## About
The VisionGuide Haptic explorer is a collaboration between seniors from the Biomedical and Electrical Engineering departments and Computer Science departments for the University of Arkansas Spring 2024 Capstone. The goal of this project is to create a haptic glove which, with the addition of a depth sensing camera and Jetson Orin Nano, running Python-based software, will aid visually impaired users in small object acquisition.

This repository specifically contains the Python-based software for processing camera data, running computer vision processes to detect objects, passing feedback through the glove via GPIO, and controlling microphone input to allow users to specify objects.

### Awards
This project notably placed first in the University of Arkansas Spring 2024 Engineering Capstone Expo.

## Operation
### Note
This project was created with CUDA support available to optimize operations. To utilize this, OpenCV must be built from source with your specific GPU architecture in mind, and using Anaconda for Python is highly recommended.

For use with a real camera, a depth sensing camera is required. This program was tested with the Intel RealSense D435 Camera and Oak-D-Lite. Support for other models cannot be guaranteed. 

You will also require a microphone to give input to specify objects.

### Steps to Run
## Step 1 
Clone the repo:
`git clone https://github.com/jullianarenales/SeniorDesign-AIGlove.git` 

## Step 2
Open the repository and install all required dependencies. 
`pip install pyrealsense2 pyttsx3 spacy Jetson.GPIO vosk pyaudio'

## Step 3

Run either `VGHE.py` or `Desktop.py`

`VGHE.py` is for systems with RealSense Cameras and glove outputs.

`Desktop.py` is for simulating the glove and utilizes `.bag` files for simulation.

## Step 4

Microphone input will activate upon program start, and you will be prompted to say your target object.

The search function will then activate after getting input.

Press `q` to quit out of the window.

# Contributors
The code for this project was made possible by members from both the Biomedical Engineering and Computer Science teams.

## Biomedical
- Nicholas Korth
- Julianna Renales
- Goel Bindu

## Computer Science & Computer Engineering
- Amadeo Costaldi 
- Luis Guerra
- Drake Ford
- Pranav Natarajan
- Sandeep Chitturi







