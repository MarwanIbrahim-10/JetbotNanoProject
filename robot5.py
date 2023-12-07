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

#######DIRECTING THE ROBOT
def make_a_decision(decision):
    print("I am in the decision function and the decision is: ", decision)
    if decision == 'turn':
        turning = True
        daddyana.forward(0.1)
        time.sleep(0.5)
        turn(1,0.18)
    elif decision == 'left':
        print("I am in the left condition")
        turning = True
        daddyana.forward(0.05)
        time.sleep(0.8)
        daddyana.left(0.2)
        time.sleep(0.18)
        daddyana.forward(0.1)
        time.sleep(0.1)
    elif decision == "right":
        print("I am in the right condition")
        turning = True
        daddyana.forward(0.1)
        time.sleep(0.5)
        daddyana.right(0.1)
        time.sleep(0.1)
        daddyana.forward(0.1)
        time.sleep(0.1)
    else:
        print("Came in the decision function but don't need it")

def turn(duration, spd):
    daddyana.left(spd)
    time.sleep(duration)
    daddyana.stop()

#######

#######IMAGE PROCESSING STARTS HERE
def decision_making(right,left,up,down):
    if right and up:
        decision = "left"
    elif left and up:
        decision = "right"
    elif left and down:
        decision = "forward"
    elif right and down:
        decision = "forward"
    else:
        decision = "forward"
    return decision

def check_contour_proximity(midpoint, rectangle, contours, frame, image_shape):
    up, down, left, right = False, False, False, False
    x, y = midpoint
    rect_height = rectangle[3]  # Assuming rectangle is (x, y, w, h)
    rect_width = rectangle[2]
    color_blue = (255, 0, 0)  # Blue color in BGR
    
    line_segments = {
        'up':    ((x, y), (x, y - rect_height)),
        'down':  ((x, y), (x, y + rect_height)),
        'left':  ((x, y), (x - rect_width, y)),
        'right': ((x, y), (x + rect_width, y))
    }


    # Draw line segments on the image
    for direction, (pt1, pt2) in line_segments.items():
        cv2.line(frame, pt1, pt2, color_blue, 1)  # Draw blue line on the image

    # Now check each contour for intersection with line segments
    for direction, (pt1, pt2) in line_segments.items():
        line_img = np.zeros(image_shape[:2], dtype=np.uint8)  # Create a blank single-channel image
        cv2.line(line_img, pt1, pt2, 255, 1)  # Draw line segment
        intersect_img = cv2.bitwise_and(line_img, line_img, mask=cv2.drawContours(np.zeros_like(line_img), contours, -1, 255, -1))
        
        if np.any(intersect_img):  # Check if there is any intersection
            if direction == 'up':
                up = True
            elif direction == 'down':
                down = True
            elif direction == 'left':
                left = True
            elif direction == 'right':
                right = True

    return up, down, left, right, frame  # Return the image for visualization


# Define cluster size thresholds
min_cluster_width = 13
min_cluster_height = 13

# Function to find midpoints of clusters in a mask and calculate their sizes
def find_midpoints_and_sizes(mask, min_width, min_height):
    min_area = 450
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    midpoints = []
    sizes = []
    aspect_ratios = []  # Store aspect ratios here
    rectangles = []  # Store rectangles here
    valid_contours = []  # To store contours that are likely to be rectangles
    contour_areas = []  # Store contour areas here
    
    for contour in contours:
        # Approximate the contour to simplify its shape
        epsilon = 0.09 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has about 4 vertices (rectangle-like)
        if len(approx) == 4:
            area = cv2.contourArea(approx)

            # Only proceed if the contour area is within the specified range
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Store area and other data in the lists
                contour_areas.append(area)
                midpoint = (x + w//2, y + h//2)
                midpoints.append(midpoint)
                sizes.append(w * h)
                aspect_ratios.append(aspect_ratio)
                rectangles.append((x, y, w, h))
                valid_contours.append(approx)  # Store the approximated contour
    
    # After processing all contours, print out the contour areas
#     print("Contour Areas:", contour_areas)
    return midpoints, sizes, rectangles, aspect_ratios, valid_contours

# Function to draw midpoints on the image
def draw_midpoints_and_contours(img, midpoints, contours, color):
    # Draw midpoints on each contour
    for point in midpoints:
        cv2.circle(img, point, radius=3, color=color, thickness=-1)  # -1 thickness fills the circle

# Function to process each image
def process_image(frame):
    frame_with_lines = frame.copy()
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not load image from {image_path}")
#         return

#     # Display the original image
#     display(Image(data=cv2.imencode('.png', image)[1].tobytes()))

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

    # Create an output image with three shades of grey
    output_image = np.zeros_like(rgb_image)
    output_image[black_mask] = [0, 0, 0]  # Black
    output_image[white_mask] = [255, 255, 255]  # White
    output_image[middle_gray_mask] = [128, 128, 128]  # Middle Grey

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological opening to the middle gray mask
    opened_mask = cv2.morphologyEx(middle_gray_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Calculate midpoints for middle gray regions, their sizes, rectangles, aspect ratios, and contours
    middle_gray_midpoints, middle_gray_sizes, middle_gray_rectangles, current_aspect_ratios, valid_contours = find_midpoints_and_sizes(opened_mask, min_cluster_width, min_cluster_height)

    # Sort the sizes and associated aspect ratios, rectangles, and contours in descending order
    sorted_data = sorted(zip(middle_gray_sizes, middle_gray_rectangles, current_aspect_ratios, valid_contours), reverse=True)
    sorted_sizes, sorted_rectangles, sorted_aspect_ratios, sorted_contours = zip(*sorted_data) if sorted_data else ([], [], [], [])
    
    # Draw midpoints and contours for middle gray regions on the output image
    draw_midpoints_and_contours(output_image, middle_gray_midpoints, sorted_contours, color=(0, 0, 255))
    
    decision = "nothing"
    
    #Only do this if the image has a rectangle with a midpoint
    if len(middle_gray_midpoints) > 0:
        # Find and draw contours for the black areas
        black_mask_uint8 = black_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(black_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_contours = contours
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)  # Drawing green contours
    
        if len(middle_gray_midpoints) == 2:
            decision = "turn"
        elif len(middle_gray_midpoints) == 1:    
            for midpoint, rectangle in zip(middle_gray_midpoints, middle_gray_rectangles):
                up, down, left, right, frame_with_lines = check_contour_proximity(midpoint, rectangle, black_contours, frame, frame.shape)
                decision = decision_making(right, left, up, down)
                
            
        processed_image = cv2.imencode('.png', frame_with_lines)[1].tobytes()
        processed_image_widget.value = processed_image
        
    return decision

    # Display the final image with midpoints and green contours
#     display(Image(data=cv2.imencode('.png', output_image)[1].tobytes()))

#######IMAGE PROCESSING END HERE

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

# Flag to check if the robot is turning
turning = False

# Flag to delay the intersection delay logic
last_intersection_time = None
# intersection_delay = 0.125  # Delay in seconds

image_widget = widgets.Image(format='jpeg')
contour_image_widget = widgets.Image(format='jpeg')
processed_image_widget = widgets.Image(format='jpeg')

display(image_widget, contour_image_widget,processed_image_widget)

daddyana = Robot()

#HD Resolution
def gstreamer_pipeline(
    capture_width=1280,  # Capture width
    capture_height=720,  # Capture height
    display_width=160,  # Same as capture width
    display_height=240,  # Same as capture height
    framerate=20,
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

def majority_vote(decisions):
    return max(set(decisions), key=decisions.count)

def process_image_thread(frame, result, index):
    result[index] = process_image(frame)

def draw_midpoints(frame):
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
    global checked_intersection, last_intersection_time, turning
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
#     cv2.drawContours(contour_frame, large_contours, -1, (0, 255, 0), 3)

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
        if cx < center_x - 10 and not turning:
            robot.left(0.1)
        elif cx > center_x + 10 and not turning:
            robot.right(0.1)
        elif not turning:
            robot.forward(0.1)

        if line_width > 40:
            current_time = time.time()
            if last_intersection_time is None:
                # First time the line width exceeds the threshold
                last_intersection_time = current_time

            elapsed_time = current_time - last_intersection_time
            if elapsed_time > intersection_delay:
                if not checked_intersection:
                    robot.stop()
                    time.sleep(0.1)
                    #ToDo: process the image here
                    # Place this inside the follow_line function, where you want to make the decision

                    # Array to store the results from each thread
                    results = [None] * 3

                    # Create and start threads
                    threads = []
                    for i in range(3):
                        frame_copy = frame.copy()  # Make a copy of the frame for each thread
                        thread = threading.Thread(target=process_image_thread, args=(frame_copy, results, i))
                        threads.append(thread)
                        thread.start()
                        time.sleep(0.01)  # Wait a bit before capturing the next frame

                    # Wait for all threads to complete
                    for thread in threads:
                        thread.join()

                    # Majority vote logic
                    print("Results is: ", results)
                    decision = majority_vote(results)
                    print("So the decision is: ", decision)
                    make_a_decision(decision)

                    #LATER: take 3 different screenshots here that are 0.01 seconds away from each other
                    #LATER: give the 3 different screenshots to the Image_processing code
#                     screenshot_filename = os.path.join(screenshots_folder, f"screenshot_{int(time.time())}.jpg")
#                     cv2.imwrite(screenshot_filename, frame)
                    #LATER: take the decision from the image processing code, and then we'll do something with it (TBD)
                    checked_intersection = True
                    last_intersection_time = None  # Reset the timer
#                 else:
#                     print("Continue")
#             else:
#                 print(f"Waiting before stopping, elapsed time: {elapsed_time:.2f}")
        else:
            last_intersection_time = None  # Reset the timer if the line width is below threshold
            checked_intersection = False

    else:
        print("No line detected")
        robot.forward(0.2)

    return frame, contour_frame

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
            exit()

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