from jetbot import Robot
import time
import cv2

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

    #the format of the frame variable is a numpy 3d array

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #process the image here!

    #make decisions on how the robot should move here

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()