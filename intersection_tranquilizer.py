import cv2
import numpy as np
from IPython.display import display, Image
import os

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

def check_contour_proximity(midpoint, rectangle, contours, image_shape):
    up, down, left, right = False, False, False, False
    x, y = midpoint
    rect_height, rect_width = rectangle[3], rectangle[2]  # Assuming rectangle is (x, y, w, h)

    # Define a tolerance for how close the contour needs to be to the line
    tolerance = 5  # pixels

    # Define max distance a contour can be from the midpoint to be considered
    max_distance = {
#         'up':    rect_height // 2,  # Half the height of the rectangle upwards
#         'down':  rect_height // 2,  # Half the height of the rectangle downwards
#         'left':  rect_width // 2,   # Half the width of the rectangle to the left
#         'right': rect_width // 2    # Half the width of the rectangle to the right
        #works
#         'up':    rect_height,  # Half the height of the rectangle upwards
#         'down':  rect_height,  # Half the height of the rectangle downwards
#         'left':  rect_height,   # Half the width of the rectangle to the left
#         'right': rect_height * 2    # Half the width of the rectangle to the right
        'up':    rect_width * 2,  # Half the height of the rectangle upwards
        'down':  rect_width * 2,  # Half the height of the rectangle downwards
        'left':  rect_width * 2,   # Half the width of the rectangle to the left
        'right': rect_width * 2    # Half the width of the rectangle to the right
    }

    # Check each contour
    for contour in contours:
        for point in contour:
            px, py = point.ravel()

            # Check for left and right within distance limits
            if y - tolerance <= py <= y + tolerance:
                if 0 < x - px <= max_distance['left']:
                    left = True
                elif 0 < px - x <= max_distance['right']:
                    right = True

            # Check for up and down within distance limits
            if x - tolerance <= px <= x + tolerance:
                if 0 < y - py <= max_distance['up']:
                    up = True
                elif 0 < py - y <= max_distance['down']:
                    down = True

            # Early exit if all directions are detected
            if up and down and left and right:
                return up, down, left, right

    print("Up: ", up)
    print("Down: ", down)
    print("Left: ", left)
    print("Right: ", right)
    return up, down, left, right

# Define cluster size thresholds
min_cluster_width = 13
min_cluster_height = 13

# Function to find midpoints of clusters in a mask and calculate their sizes
def find_midpoints_and_sizes(mask, min_width, min_height):
    min_area = 500
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
    # Draw all contours
#     cv2.drawContours(img, contours, -1, color, 1)

    # Draw midpoints on each contour
    for point in midpoints:
        cv2.circle(img, point, radius=3, color=color, thickness=-1)  # -1 thickness fills the circle

# Function to process each image
def process_image(image_path):
#     print("Image path is: ", image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Display the original image
#     print("Original Image:")
    display(Image(data=cv2.imencode('.png', image)[1].tobytes()))

    # Convert the image to RGB mode
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                up, down, left, right = check_contour_proximity(midpoint, rectangle, black_contours, image.shape)
                decision = decision_making(right,left,up,down)
        print("DECISION IS: ", decision)

    # Display the final image with midpoints and green contours
#     print("Processed image:")
    display(Image(data=cv2.imencode('.png', output_image)[1].tobytes()))

# Path to the screenshots folder
folder_path = 'screenshots2'

# List all files in the folder and process each image file
for file_name in os.listdir(folder_path):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    # Check if the file is an image (you can add more extensions if needed)
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#         print(f"Processing {file_path}...")
        process_image(file_path)