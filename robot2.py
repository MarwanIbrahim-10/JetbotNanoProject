from jetbot import Robot
import cv2
import numpy as np
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg
import time

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

#ToDo: Function that draws red lines around the green color in the captured image
def draw_red_lines_around_green(contour_frame):
    print("I am inside the function that will draw the red lines")
    # Convert BGR to HSV
    hsv = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red lines around green contours
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Adjust the minimum area if needed
            cv2.drawContours(contour_frame, [contour], -1, (0, 0, 255), 2)

    return contour_frame

def follow_line(frame, robot):
    global checked_intersection
    # Make a copy of the frame to draw contours
    contour_frame = frame.copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 57, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to close gaps and remove noise
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    min_area = 100  # Minimum area to be considered a line
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw contours on the contour frame
    cv2.drawContours(contour_frame, large_contours, -1, (0, 255, 0), 3)

    if len(large_contours) > 0:
        c = max(contours, key=cv2.contourArea)

        # Find the extreme points
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])

        # Calculate the width of the line
        line_width = rightmost[0] - leftmost[0]
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Draw centroid on the original frame
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        center_x = frame.shape[1] / 2  # Calculate the center of the frame

        # Move the robot based on the centroid's position
        if cx < center_x - 10:
            print("Move left")
            # robot.left(0.1)
        elif cx > center_x + 10:
            print("Move right")
            # robot.right(0.1)
        else:
            print("Move forward")
            # robot.forward(0.15)

        print("Line Width:", line_width)
        if line_width > 40 and not checked_intersection:
            print("I am going to the function that draws the red lines")
            frame = draw_red_lines_around_green(contour_frame)
            robot.stop()
            time.sleep(3)
            checked_intersection = True
            print("Continue")
        elif line_width > 40 and checked_intersection:
            print("I am going to the function that draws the red lines")
            frame = draw_red_lines_around_green(contour_frame)
            print("Continue")
        else:
            checked_intersection = False
            print("Continue")

    else:
        print("No line detected")
        # robot.forward(0.2)

    return frame, contour_frame


try:
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()  

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame, contour_frame = follow_line(frame, daddyana)

        # Update the image widget
        image_widget.value = bgr8_to_jpeg(frame)
        contour_image_widget.value = bgr8_to_jpeg(contour_frame)


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