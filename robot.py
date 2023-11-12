from jetbot import Robot
import time
import cv2
import numpy as np

#Jupiter widgets to display images
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg

import base64
import openai
import os
import requests
import json

image_widget = widgets.Image(format='jpeg')

display(image_widget)

daddyana = Robot()

def move_forward(duration, spd):
    daddyana.forward(spd)
    time.sleep(duration)
    daddyana.stop()

def move_right(duration,spd):
    daddyana.right(spd)
    time.sleep(duration)
    daddyana.stop()

def move_left(duration,spd):
    daddyana.left(spd)
    time.sleep(duration)
    daddyana.stop()

def move_backward(duration,spd):
    daddyana.backward(spd)
    time.sleep(duration)
    daddyana.stop()

def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=640, display_height=480, framerate=60, flip_method=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Start capturing from camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print("Cannot open camera")
    exit()

#state management variables
current_path = None  # The current path the robot is on
last_intersection = None  # The last intersection type encountered

def detect_intersection(thresholded_frame):
    pass

def decide_new_direction(current_path, intersection_type):  
    pass

while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #pass the by a filter to highlight the black and white parts of it, so we can see if the rectangular box exists
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresholded_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY_INV)  # Apply thresholding

    # Check for an intersection
    intersection_type, intersection_center = detect_intersection(thresholded_frame)

    if intersection_type is not None:
        # If we have detected an intersection, decide the new direction based on the rules
        new_direction = decide_new_direction(current_path, intersection_type)

        # Act on the new direction
        if new_direction == 'A':
            # Assuming 'A' means keep going straight
            move_forward(0.1, 0.3)
        elif new_direction == 'B':
            # Assuming 'B' means turn right
            move_right(0.1, 0.3)
        elif new_direction == 'C':
            # Assuming 'C' means turn left
            move_left(0.1, 0.3)
        elif new_direction == 'D':
            # Assuming 'D' means turn around or some other action
            # Implement the necessary action
            pass

        # Update the current path and last intersection
        current_path = new_direction
        last_intersection = intersection_type

    
    
    image_widget.value = bgr8_to_jpeg(frame)
    
    
    # Break the loop with 'q'
    if cv2.waitKey(1) == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()