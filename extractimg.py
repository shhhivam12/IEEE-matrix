import cv2
import json
import os

# Load parking spot coordinates from JSON
with open('parking_spots.json', 'r') as f:
    parking_spots = json.load(f)

# Create directories to save images
os.makedirs('dataset/empty', exist_ok=True)
os.makedirs('dataset/occupied', exist_ok=True)

# Dataset dictionary for labeling
dataset = {'spots': []}

# Start video capture from the IP Webcam
cap = cv2.VideoCapture('http://100.69.185.11:8080/video')  # Replace with IP and port
frame_no=103

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Loop through each parking spot
    for i, (start, end) in enumerate(parking_spots):
        x1, y1 = start
        x2, y2 = end

        # Extract and display each parking spot region
        parking_region = frame[y1:y2, x1:x2]
        cv2.imshow(f'Spot {i+1}', parking_region)

        # Prompt user to label the spot as empty or occupied
        key = cv2.waitKey(0)  # Wait for key press
        if key == ord('e'):  # 'e' for empty
            label = 0
            save_path = f'dataset/empty/spot_{i}_frame_{frame_no}.png'
            cv2.imwrite(save_path, parking_region)
        elif key == ord('o'):  # 'o' for occupied
            label = 1
            save_path = f'dataset/occupied/spot_{i}_frame_{frame_no}.png'
            cv2.imwrite(save_path, parking_region)
        elif key == ord('q'):  # 'q' to quit
            cap.release()
            cv2.destroyAllWindows()
            # # Save the dataset as JSON before exiting
            # with open('dataset_labels.json', 'w') as json_file:
            #     json.dump(dataset, json_file)
            exit()

        # Append data to the dataset
        dataset['spots'].append({'spot_id': i, 'frame': save_path, 'label': label})

        # Close individual spot windows after capture
        cv2.destroyWindow(f'Spot {i+1}')

    # Display main frame for reference
    cv2.imshow('Parking Lot', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Save the dataset to JSON
# with open('dataset_labels.json', 'w') as json_file:
#     json.dump(dataset, json_file)

cap.release()
cv2.destroyAllWindows()
