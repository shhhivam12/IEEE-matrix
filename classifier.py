import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score
import pickle

# Paths to the data directories
empty_dir = 'dataset/empty'
occupied_dir = 'dataset/occupied'

# Load images and labels
X = []  # Feature vectors
y = []  # Labels: 0 for empty, 1 for occupied

def load_images_from_directory(directory, label):
    for file in os.listdir(directory):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            img = img.flatten()  # Flatten the image to 1D array
            X.append(img)
            y.append(label)

# Load empty and occupied images
load_images_from_directory(empty_dir, label=0)  # 0 for empty
load_images_from_directory(occupied_dir, label=1)  # 1 for occupied

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model using pickle
with open('parking_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved as 'parking_classifier.pkl'")
