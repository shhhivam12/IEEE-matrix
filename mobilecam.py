# http://192.0.0.4:8080/video

import cv2

# Replace this URL with the URL of your IP Webcam
url = "http://100.69.185.11:8080/video"

# Start capturing the video stream
cap = cv2.VideoCapture(url)

# Check if the video stream is opened
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Streaming from IP Webcam...")

# Loop to capture each frame from the video stream
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # If a frame was successfully read
    if ret:
        # Display the frame
        cv2.imshow("IP Webcam Stream", frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Couldn't receive frame (stream end?). Exiting ...")
        break

# Release resources and close the window
cap.release()
cv2.destroyAllWindows()
