from jetbot import Robot
import cv2
import numpy as np
from IPython.display import display, Image
import io
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg
import time
import os
import threading
import queue

#Image processing code
def decision_making(right,left,up,down):
    #need to take a left turn
    if right and up:
        return "left"
    #need to take a right turn
    elif left and up:
        return "right"
    #default decision is forward
    return "forward"

def check_contour_proximity(midpoint, rectangle, contours, frame, image_shape):
    up, down, left, right = False, False, False, False
    x, y = midpoint
    rect_height, rect_width = rectangle[3], rectangle[2]
    
    line_segments = {
        'up':    ((x, y), (x, y - rect_height)),
        'down':  ((x, y), (x, y + rect_height)),
        'left':  ((x, y), (x - rect_width, y)),
        'right': ((x, y), (x + rect_width, y))
    }

    line_img = np.zeros(image_shape[:2], dtype=np.uint8)  # Create once outside the loop
    for direction, (pt1, pt2) in line_segments.items():
        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)  # Directly use the color here
        cv2.line(line_img, pt1, pt2, 255, 1)  # Draw line segment
        intersect_img = cv2.bitwise_and(line_img, line_img, mask=cv2.drawContours(np.zeros_like(line_img), contours, -1, 255, -1))
        line_img.fill(0)  # Clear the image for the next iteration
        
        if np.any(intersect_img):
            if direction == 'up':
                up = True
            elif direction == 'down':
                down = True
            elif direction == 'left':
                left = True
            elif direction == 'right':
                right = True

    return up, down, left, right, frame


#cluster size thresholds
min_cluster_width = 13
min_cluster_height = 13

# Function to find midpoints of clusters in a mask and calculate their sizes
def find_midpoints_and_sizes(mask, min_width, min_height):
    min_area = 450
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    midpoints = []
    rectangles = []
    valid_contours = []
    
    for contour in contours:
        epsilon = 0.09 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) >= min_area:
            x, y, w, h = cv2.boundingRect(approx)
            midpoints.append((x + w // 2, y + h // 2))
            rectangles.append((x, y, w, h))
            valid_contours.append(approx)
    
    return midpoints, rectangles, valid_contours

# Function to process each image
def process_image(frame):
    frame_with_lines = frame.copy()

    # Convert the image to RGB mode
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate the average of R, G, and B channels
    average_image = np.mean(rgb_image, axis=2).astype(np.uint8)

    # Threshold the average image to create three shades of grey
    threshold_value = 50
    mid_threshold = 40
    black_mask = average_image < threshold_value
    white_mask = average_image >= (threshold_value + mid_threshold)
    middle_gray_mask = np.logical_and(
        average_image >= threshold_value, average_image < (threshold_value + mid_threshold)
    )

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological opening to the middle gray mask
    opened_mask = cv2.morphologyEx(middle_gray_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Calculate midpoints for middle gray regions, their sizes, rectangles, aspect ratios, and contours
    middle_gray_midpoints, middle_gray_rectangles, valid_contours = find_midpoints_and_sizes(opened_mask, min_cluster_width, min_cluster_height)
    
    #Only do this if the image has a rectangle with a midpoint
    if len(middle_gray_midpoints) > 0:
        # Find and draw contours for the black areas
        black_mask_uint8 = black_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(black_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_contours = contours

        if len(middle_gray_midpoints) == 2:
            return "turn"
        elif len(middle_gray_midpoints) == 1:
            for midpoint, rectangle in zip(middle_gray_midpoints, middle_gray_rectangles):
                up, down, left, right, frame_with_lines = check_contour_proximity(midpoint, rectangle, black_contours, frame, frame.shape)
                return decision_making(right, left, up, down)

        processed_image = cv2.imencode('.png', frame_with_lines)[1].tobytes()
        processed_image_widget.value = processed_image

    return "forward"

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=160,
    display_height=240,
    framerate=25,
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

def detect_intersection(frame):
    frame_with_lines = frame.copy()

    # Convert the image to RGB mode
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate the average of R, G, and B channels
    average_image = np.mean(rgb_image, axis=2).astype(np.uint8)

    # Threshold the average image to create three shades of grey
    threshold_value = 50
    mid_threshold = 40
    black_mask = average_image < threshold_value
    white_mask = average_image >= (threshold_value + mid_threshold)
    middle_gray_mask = np.logical_and(
        average_image >= threshold_value, average_image < (threshold_value + mid_threshold)
    )

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological opening to the middle gray mask
    opened_mask = cv2.morphologyEx(middle_gray_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Calculate midpoints for middle gray regions, their sizes, rectangles, aspect ratios, and contours
    middle_gray_midpoints, middle_gray_rectangles, valid_contours = find_midpoints_and_sizes(opened_mask, min_cluster_width, min_cluster_height)
    
    #Only do this if the image has a rectangle with a midpoint
    if len(middle_gray_midpoints) > 0:
        print("length of gray midpoints: ", len(middle_gray_midpoints))
        return True
    else:
        return False

def majority_vote(decisions):
    return max(set(decisions), key=decisions.count)

def follow_line(frame, robot, decision_queue, cap):
    global busy_handling_intersection
    # Make a copy of the frame to draw contours
    contour_frame = frame.copy()

    #Line following logic starts here
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

    #this means a line is currently detected, follow it
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

        center_x = frame.shape[1] / 2  # Calculate the center of the frame

        # Move the robot based on the centroid's position
        if cx < center_x - 10:
            robot.left(0.1)
        elif cx > center_x + 10:
            robot.right(0.1)
        else:
            robot.forward(0.125)
        #Line following logic ends here
        
        #detect if intersection exists
        intersection = False
        intersection = detect_intersection(frame)
        
        #if there is an intersection (the line is getting wider)
        if intersection and not busy_handling_intersection:
            print("Intersection found")

#             for i in range(3):
#                 ret, frame_copy = cap.read()  # Capture frame
#                 if ret:
            frame_copy = frame.copy()
            decision_queue.put(('process', frame_copy))
#             time.sleep(0.01)

            return frame

    else:
        print("No line detected")
        robot.forward(0.2)

    return frame

import threading

# Global variables and initialization
daddyana = Robot()
running = True

# Flag to give intersection dealing the priority
busy_handling_intersection = False

decision_queue = queue.Queue()
processed_decision_queue = queue.Queue()

image_widget = widgets.Image(format='jpeg')
processed_image_widget = widgets.Image(format='jpeg')

display(image_widget,processed_image_widget)

def move_robot(decision,robot):
    print("I am in the decision function and the decision is: ", decision)
    if decision == 'turn':
        robot.forward(0.1)
        time.sleep(0.5)
        turn(1.175,0.18)
    elif decision == 'left':
        robot.forward(0.1)
        time.sleep(0.8)
        robot.left(0.25)
        time.sleep(0.25)
        robot.forward(0.125)
        time.sleep(0.3)
    elif decision == "right":
        robot.forward(0.1)
        time.sleep(0.8)
        robot.right(0.25)
        time.sleep(0.5)
        robot.forward(0.125)
        time.sleep(0.3)
    else:
        print("Came in the decision function but don't need it")
    

def turn(duration, spd):
    daddyana.left(spd)
    time.sleep(duration)
    daddyana.stop()

def image_processing_thread(decision_queue, processed_decision_queue):
    global busy_handling_intersection
    decisions = []
    while running:
        try:
            task, frame = decision_queue.get(timeout=0.01)
            if task == 'process':
                busy_handling_intersection = True
                print("Processing image...")
                decision = process_image(frame)
                print("the decision is: ", decision)
#                 decisions.append(decision)
#                 if len(decisions) == 3:  # Wait for 3 decisions
#                     final_decision = majority_vote(decisions)
#                     processed_decision_queue.put(('decision', final_decision))
                if decision is not 'forward':
                    processed_decision_queue.put(('decision', decision))
                else:
                    busy_handling_intersection = False
#                     print("Final decision (majority vote): ", final_decision)
#                     decisions = []  # Reset for next set of decisions

        except queue.Empty:
            pass

def robot_control_thread(robot, processed_decision_queue):
    global busy_handling_intersection
    while running:
        try:
            task, decision = processed_decision_queue.get(timeout=0.01)
            if task == 'decision':
                print("Executing decision:", decision)
                move_robot(decision, robot)
                busy_handling_intersection = False
        except queue.Empty:
            pass

# Thread creation and starting
image_thread = threading.Thread(target=image_processing_thread, args=(decision_queue, processed_decision_queue))
control_thread = threading.Thread(target=robot_control_thread, args=(daddyana, processed_decision_queue))

image_thread.start()
control_thread.start()

# Main loop for capturing frames and performing line following
try:
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
            break

        _ = follow_line(frame, daddyana, decision_queue, cap)

        # Update the image widget
        image_widget.value = bgr8_to_jpeg(frame)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup
    running = False
    image_thread.join()
    control_thread.join()
    daddyana.stop()
    cap.release()
    cv2.destroyAllWindows()

print("Program ended.")