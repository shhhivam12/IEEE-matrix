import cv2
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, Response
import threading
import json

# Initialize Firebase Admin SDK
cred = credentials.Certificate('parking-map-36eb5-firebase-adminsdk-hq579-2af5708c88.json')  # Replace with your downloaded JSON file
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore client

# Load the trained model
with open('parking_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the parking spots from JSON
with open('parking_spots.json', 'r') as f:
    parking_spots = json.load(f)  # List of coordinates [[x1, y1], [x2, y2]]

# Start video stream (IP Webcam URL)
cap = cv2.VideoCapture('http://172.16.71.142:8080/video')  # Replace with your URL

app = Flask(__name__)  # Flask app initialization

frame = None  # Global frame variable to store the latest processed frame

def detect_parking():
    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, spot in enumerate(parking_spots):
            (x1, y1), (x2, y2) = spot
            spot_roi = frame[y1:y2, x1:x2]
            spot_resized = cv2.resize(spot_roi, (64, 64))
            spot_flattened = spot_resized.flatten().reshape(1, -1)

            prediction = clf.predict(spot_flattened)
            is_occupied = prediction[0] == 1
            label = "Occupied" if is_occupied else "Empty"
            color = (0, 0, 255) if is_occupied else (0, 255, 0)

            # Update Firestore
            doc_ref = (
            db.collection('parking_spots')
            .document('5wirlxKqyKn1U3i92tUU')
            .collection('spaces')
            .document(f'space_{i + 1}')
            )

            doc_ref.update({'isAvailable': not is_occupied})

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Spot {i + 1}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Display for debugging (remove if only server is needed)
        cv2.imshow('Parking Lot Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/parking_image')
def parking_image():
    global frame
    if frame is None:
        return "No frame available", 503  # Return if no frame is ready yet
    
    # Encode the frame with markings (already processed in detect_parking)
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/stream_spot/<spot_id>')
def stream_spot(spot_id):
    global frame
    if frame is None:
        return "No frame available", 503  # No frame captured yet

    # Find the index for the spot_id (e.g., space_1 -> index 0)
    spot_index = int(spot_id.split('_')[1]) - 1  # e.g., space_1 -> 0

    # Check if the spot_index is valid
    if spot_index < 0 or spot_index >= len(parking_spots):
        return "Invalid parking spot ID", 400

    # Extract the region of interest (ROI) for the given spot
    (x1, y1), (x2, y2) = parking_spots[spot_index]
    spot_frame = frame[y1:y2, x1:x2]  # Crop the frame to the spot area

    # Encode the cropped spot frame to JPEG
    _, buffer = cv2.imencode('.jpg', spot_frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    # Run the parking detection in a separate thread
    detection_thread = threading.Thread(target=detect_parking)
    detection_thread.start()
    
    # Run Flask app on port 5000
    app.run(port=5000)
