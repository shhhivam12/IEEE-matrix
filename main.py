import cv2
import pickle
import json
import numpy as np

# Load the trained model
with open('parking_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the parking spots from JSON file
with open('parking_spots.json', 'r') as f:
    parking_spots = json.load(f)  # List of [[x1, y1], [x2, y2]] for each spot

# Start the video stream from IP Webcam (Replace URL with your IP Webcam URL)
cap = cv2.VideoCapture('http://172.16.71.142:8080/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    occupied_count = 0
    total_spots = len(parking_spots)

    # Process each parking spot
    for i, spot in enumerate(parking_spots):
        # Extract the top-left and bottom-right coordinates
        (x1, y1), (x2, y2) = spot

        # Extract and preprocess the region of interest (ROI)
        spot_roi = frame[y1:y2, x1:x2]
        spot_resized = cv2.resize(spot_roi, (64, 64))  # Resize to match training size
        spot_flattened = spot_resized.flatten().reshape(1, -1)  # Flatten for classifier

        # Predict if the spot is empty (0) or occupied (1)
        prediction = clf.predict(spot_flattened)
        label = "Occupied" if prediction == 1 else "Empty"

        # Update count of occupied spots
        if prediction == 1:
            occupied_count += 1

        # Draw rectangles and labels on the frame
        color = (0, 0, 255) if label == "Occupied" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Spot {i + 1}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Display total occupied and free spots
    free_count = total_spots - occupied_count
    status_text = f"Occupied: {occupied_count} | Free: {free_count}"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Parking Lot Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
