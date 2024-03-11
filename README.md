# SeniorDesign-AIGlove
Repository for AI Glove Project

DualBoot is our main file
If the Oak-d-lite camera is not detected then you should load in a .bag recording of what one captured
- BE SURE TO UPDATE ALL THE PATHS TO REFLECT WHAT IS ON YOUR MACHINE
- Just comment out the Cuda enabling code out if it doesnt work (I couldnt get cuda to work on my machine so I couldnt test it)

Note for enabling CUDA support for OpenCV.

In order to utilize OpenCV with cuda support you must compile it from source using cmake.

You will need:
OpenCV: https://github.com/opencv/opencv
OpenCV Contrib: https://github.com/opencv/opencv_contrib
CMake: https://cmake.org/download/
Anaconda Python: https://www.anaconda.com/download Note:For ease of use ensure that Anaconda is used for all of your python dependencies and libraries and is your main interpreter. 
NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
NVIDIA cuDNN 8.9.7: https://developer.nvidia.com/rdp/cudnn-archive IMPORTANT: cuDNN 8.9.7 MUST be used as cuDNN 9.0(latest) is NOT currently supported by OpenCV and your build will fail.

Watch this youtube video for how to do it: https://www.youtube.com/watch?v=d8Jx6zO1yw0
and build with this command in an Anaconda prompt: ```cmake --build "E:\Users\yourname\opencvGPU\build" --target INSTALL --config Release```

Confirm that OpenCv is properly installed with:
(In an Anaconda Prompt)
```
python
import cv2
```
If this fails read the error logs carefully but it is likely an issue with your build. You cna confirm this by checking the OpenCv build and finding any red errors.

If you cannot resolve your build issues contact Amadeo Costaldi at amcostal@uark.edu 

The code will automatically use CUDA if available and you can benchmark your Cuda preformance vs your cpu with ```TestCUDA.py```.