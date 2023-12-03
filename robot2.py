from jetbot import Robot
import cv2
import numpy as np
from IPython.display import display
import ipywidgets.widgets as widgets
from jetbot import bgr8_to_jpeg

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


try:
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Draw centroid on the image
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            center_x = frame.shape[1] / 2

            if cx < center_x - 10:
                print("Move left")
                daddyana.left(0.1)
            elif cx > center_x + 10:
                print("Move right")
                daddyana.right(0.1)
            else:
                print("Move forward")
                daddyana.forward(0.15)
        else:
            # No line found, stop
            daddyana.stop()

        # Update the image widget
        image_widget.value = bgr8_to_jpeg(frame)
        binary_image_widget.value = bgr8_to_jpeg(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

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