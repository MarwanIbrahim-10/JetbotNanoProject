from jetbot import Robot
import cv2
import numpy as np
from IPython.display import display, Image
import io
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg
import time

def show_image_in_notebook(img):
    """ Function to display an image in a Jupyter notebook. """
    _, encoded_image = cv2.imencode('.png', img)
    i = Image(data=encoded_image)
    display(i)

# Create a button to stop the robot
stop_button = widgets.Button(description='Stop')
display(stop_button)

def stop_robot(b):
    global running
    running = False
    daddyana.stop()
    print("Robot stopped")

stop_button.on_click(stop_robot)

# Flag to control the loop
running = True

# Flag to check if the intersection has been handled
checked_intersection = False

image_widget = widgets.Image(format='jpeg')
binary_image_widget = widgets.Image(format='jpeg')
contour_image_widget = widgets.Image(format='jpeg')

display(image_widget, binary_image_widget, contour_image_widget)

daddyana = Robot()

def gstreamer_pipeline(capture_width=240, capture_height=540, display_width=160, display_height=240, framerate=30, flip_method=0):
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

def process_frame_for_contours(frame):
    global checked_intersection

    # Make a copy of the frame for drawing
    processed_frame = frame.copy()

    # Process for green contours (your existing line following code)
    # Convert to grayscale, threshold, find and draw green contours
    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 57, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    cv2.drawContours(processed_frame, large_contours, -1, (0, 255, 0), 3)

    lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2Lab)
    a_channel = lab[:,:,1]
    _, a_thresh = cv2.threshold(a_channel, 105, 255, cv2.THRESH_BINARY)
    show_image_in_notebook(a_thresh)
    contours, _ = cv2.findContours(a_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        print("contour area: ", cv2.contourArea(contour))
        if cv2.contourArea(contour) > 40:
            cv2.drawContours(processed_frame, [contour], -1, (0, 0, 255), 2)


    return processed_frame



try:
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame_for_contours(frame)

        # Update the image widget with the processed frame
        image_widget.value = bgr8_to_jpeg(processed_frame)


        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
        
finally:
    print("Cleaning up...")
    daddyana.stop()  # Ensure the robot is stopped
    cap.release()    # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

print("Program ended.")