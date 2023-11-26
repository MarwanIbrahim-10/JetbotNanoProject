from jetbot import Robot
import time
import cv2
import numpy as np

#Jupiter widgets to display images
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg

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

def follow_line(frame, display_width):
    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #threshold the image to get the line in white
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    #find the contours of the line
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour, assumed to be the line
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the moments of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            # Calculate the x-coordinate of the centroid
            cX = int(M["m10"] / M["m00"])
            # Calculate the deviation from the center of the display width
            deviation = cX - (display_width // 2)
            return deviation
    return None  # No line detected

#modify the detect_intersection function to return more detailed information
def detect_intersection(hsv_frame):
    #define range for green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    #threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    #find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Assuming markers have a significant area
            # Additional logic to determine the type of intersection
            # This can be based on the position or number of green markers
            # For the sake of example, let's say you determine the direction based on the position
            x, y, w, h = cv2.boundingRect(contour)
            if x < hsv_frame.shape[1] // 2:
                return 'left'
            else:
                return 'right'
    return 'straight'  # Default to straight if no markers or specific conditions are not met


def decide_new_direction(intersection_type):
    # Implement logic based on the type of intersection
    if intersection_type == 'left':
        return 'C'  # Turn left
    elif intersection_type == 'right':
        return 'B'  # Turn right
    elif intersection_type == 'straight':
        return 'A'  # Continue straight
    # Add more conditions as necessary, for example 'uturn'


while True:
    #read from camera
    ret, frame = cap.read()

    #if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    print("Reading from camera...")

    #follow the line
    #assumes display width is 640, will have to adjust that accordingly
    deviation = follow_line(frame, 640)
    if deviation is not None:
        print(f"Line detected. Deviation: {deviation}")
        #adjust robot's direction based on deviation
        #proportional control of the robot
        speed = 0.3  #base speed
        k_p = 0.01  #proportional gain, adjust as necessary
        turn_speed = k_p * deviation
        daddyana.set_motors(speed + turn_speed, speed - turn_speed)
        time.sleep(0.1)
    else:
        print("No line detected, stopping the robot.")
        #no line detected, stop the robot or search for the line
        daddyana.stop()

    #intersection detection logic
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    intersection_type = detect_intersection(hsv_frame)
    if intersection_type:
        print(f"Intersection detected: {intersection_type}")
        new_direction = decide_new_direction(intersection_type)
        print(f"New direction: {new_direction}")
        # Implement the logic to handle the turn based on new_direction
        if new_direction == 'A':
            # Continue straight
            move_forward(0.1, speed)  # Example duration and speed
        elif new_direction == 'B':
            # Turn right
            move_right(0.1, speed)  # Example duration and speed
        elif new_direction == 'C':
            # Turn left
            move_left(0.1, speed)  # Example duration and speed
        # Add more actions as necessary
    else:
        print("No intersection detected.")

    #update the image widget
    image_widget.value = bgr8_to_jpeg(frame)

    #break the loop with 'q'
    if cv2.waitKey(1) == ord('q'):
        break
        
#when everything done, release the capture
cap.release()
cv2.destroyAllWindows()