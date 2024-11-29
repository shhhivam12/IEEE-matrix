import cv2
import json

# Global variables to store points
parking_spots = []
current_spot = []

# Mouse callback function to select points
def select_parking_spots(event, x, y, flags, param):
    global current_spot

    if event == cv2.EVENT_LBUTTONDOWN:
        current_spot = [(x, y)]  # Start point

    elif event == cv2.EVENT_LBUTTONUP:
        current_spot.append((x, y))  # End point
        parking_spots.append(current_spot)
        current_spot = []

        # Draw rectangle on the image
        cv2.rectangle(frame, parking_spots[-1][0], parking_spots[-1][1], (0, 255, 0), 2)
        cv2.imshow('Define Parking Spots', frame)

# Start the video feed
cap = cv2.VideoCapture('http://172.16.71.142:8080/video')  # Replace with your camera URL

ret, frame = cap.read()
if not ret:
    print("Failed to capture video")
    cap.release()
    exit()

cv2.imshow('Define Parking Spots', frame)
cv2.setMouseCallback('Define Parking Spots', select_parking_spots)

print("Click and drag to define parking spots. Press 'q' to finish.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the parking spots to a JSON file
with open('parking_spots.json', 'w') as f:
    json.dump(parking_spots, f)

cap.release()
cv2.destroyAllWindows()

print("Parking spots defined and saved in 'parking_spots.json'.")
