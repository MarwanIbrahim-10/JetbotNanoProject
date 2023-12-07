from jetbot import Robot
import cv2
import numpy as np
from IPython.display import display, Image
import io
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg
import time
import os

# Create a button to stop the robot
stop_button = widgets.Button(description='Stop')
display(stop_button)

screenshots_folder = "screenshots2"
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)


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

# Flag to delay the intersection delay logic
last_intersection_time = None
intersection_delay = 0.2  # Delay in seconds

image_widget = widgets.Image(format='jpeg')
binary_image_widget = widgets.Image(format='jpeg')
contour_image_widget = widgets.Image(format='jpeg')

display(image_widget, binary_image_widget, contour_image_widget)

daddyana = Robot()

#Low resolution but works
# def gstreamer_pipeline(capture_width=240, capture_height=540, display_width=160, display_height=240, framerate=10, flip_method=0):
#     return (
#         "nvarguscamerasrc ! "
#         "video/x-raw(memory:NVMM), "
#         "width=(int)%d, height=(int)%d, "
#         "format=(string)NV12, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )

#HD Resolution
def gstreamer_pipeline(
    capture_width=1280,  # Capture width
    capture_height=720,  # Capture height
    display_width=160,  # Same as capture width
    display_height=240,  # Same as capture height
    framerate=15,
    flip_method=0
):
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


# def gstreamer_pipeline(
#     capture_width=360,  # Increased capture width
#     capture_height=810,  # Increased capture height
#     display_width=160,  # Same display width
#     display_height=240,  # Same display height
#     framerate=30,
#     flip_method=0
# ):
#     return (
#         "nvarguscamerasrc ! "
#         "video/x-raw(memory:NVMM), "
#         "width=(int)%d, height=(int)%d, "
#         "format=(string)NV12, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )


def draw_midpoints(frame):
    print("I am inside the draw midpoints function")
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate the average of R, G, and B channels
    average_frame = np.mean(rgb_frame, axis=2).astype(np.uint8)

    # Threshold values (adjust as needed)
    threshold_value = 72
    mid_threshold = 20
    middle_gray_mask = np.logical_and(
        average_frame >= threshold_value, average_frame < (threshold_value + mid_threshold)
    )

    # Define cluster size thresholds
    min_cluster_width = 13
    min_cluster_height = 13

    # Find midpoints for each color
    middle_gray_midpoints = find_midpoints(middle_gray_mask, min_cluster_width, min_cluster_height)

    # Draw midpoints for each color on the frame
    draw_midpoints_on_frame(frame, middle_gray_midpoints, color=(0, 0, 255))  # Red for middle grey regions

def find_midpoints(mask, min_width, min_height):
    #here you can eliminate the areas that are not valid areas
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    midpoints = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            midpoint = (x + w//2, y + h//2)
            midpoints.append(midpoint)
    return midpoints

def draw_midpoints_on_frame(img, midpoints, color):
    for point in midpoints:
        cv2.circle(img, point, radius=3, color=color, thickness=-1)  # -1 thickness fills the circle


def follow_line(frame, robot):
    global checked_intersection, last_intersection_time
    # Make a copy of the frame to draw contours
    contour_frame = frame.copy()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)

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
#         cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        center_x = frame.shape[1] / 2  # Calculate the center of the frame

        # Move the robot based on the centroid's position
        if cx < center_x - 10:
            robot.left(0.1)
        elif cx > center_x + 10:
            robot.right(0.1)
        else:
            robot.forward(0.1)

        if line_width > 40:
            current_time = time.time()
            if last_intersection_time is None:
                # First time the line width exceeds the threshold
                last_intersection_time = current_time
                print("Intersection detected, starting timer")

            elapsed_time = current_time - last_intersection_time
            if elapsed_time > intersection_delay:
                if not checked_intersection:
                    robot.stop()
                    time.sleep(1)
                    #Draw the midpoints here
                    start_time = time.time()  # Start timing
                    screenshot_filename = os.path.join(screenshots_folder, f"screenshot_{int(time.time())}.jpg")
                    cv2.imwrite(screenshot_filename, frame)
                    print(f"Screenshot saved as {screenshot_filename}")
                    draw_midpoints(frame)  # Call the function to draw midpoints here
                    end_time = time.time()  # End timing
                    print(f"Time taken to draw midpoints: {end_time - start_time} seconds")
                    time.sleep(2)
                    checked_intersection = True
                    last_intersection_time = None  # Reset the timer
                    print("Continue after delay")
                else:
                    print("Continue")
            else:
                print(f"Waiting before stopping, elapsed time: {elapsed_time:.2f}")
        else:
            last_intersection_time = None  # Reset the timer if the line width is below threshold
            checked_intersection = False
            print("Continue")

    else:
        print("No line detected")
        robot.forward(0.2)

    return frame, contour_frame

try:
    print("Attempting to open camera")
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
        print("Camera opened")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            exit()
        else:
            print("Frame captured successfully")

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