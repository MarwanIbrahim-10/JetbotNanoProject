from jetbot import Robot
import time
import cv2

#Jupiter widgets to display images
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg

import base64
import openai
import os
import requests
import json

openai_key = "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"

openai.api_key = openai_key

# client = OpenAI()

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

while True:
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #process the image here!
    # Convert the frame to JPEG format
    _, buffer = cv2.imencode('.jpg', frame)

    # Encode the image as base64
    base64_image = base64.b64encode(buffer).decode()
    
    # Feed the image to the openai API
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {openai_key}"
#     }

#     payload = {
#         "model": "gpt-4-vision-preview",
#         "messages": [
#           {
#             "role": "user",
#             "content": [
#               {
#                 "type": "text",
#                 "text": "What’s in the main thing in this image? Respond with 1 word only"
#               },
#               {
#                 "type": "image_url",
#                 "image_url": {
#                   "url": f"data:image/jpeg;base64,{base64_image}"
#                 }
#               }
#             ]
#           }
#         ],
#         "max_tokens": 4
#     }

#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    # JSON data
#     json_data = response.json()

#     # Extracting the message content
#     message_content = json_data['choices'][0]['message']['content']

#     print(message_content)
    

    #make decisions on how the robot should move here
    
    image_widget.value = bgr8_to_jpeg(frame)
    
    
    # Break the loop with 'q'
    if cv2.waitKey(1) == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()