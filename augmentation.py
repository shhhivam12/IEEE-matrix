import cv2
import numpy as np
import os

def augment_image(image, save_path, index):
    # Brightness Adjustment
    brightness_variations = [0.8, 1.0, 1.2]
    for brightness in brightness_variations:
        augmented = cv2.convertScaleAbs(image, alpha=brightness)
        cv2.imwrite(f"{save_path}/aug_brightness_{index}_{brightness}.jpg", augmented)

    # Rotation (Simulate Different Angles)
    for angle in [5, -5, 10, -10]:
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(f"{save_path}/aug_rotate_{index}_{angle}.jpg", rotated)

    # Cropping (Simulate Zoom)
    for crop_factor in [0.9, 0.95]:
        h, w = image.shape[:2]
        cropped = image[int(h * (1 - crop_factor)):int(h * crop_factor),
                        int(w * (1 - crop_factor)):int(w * crop_factor)]
        resized = cv2.resize(cropped, (w, h))
        cv2.imwrite(f"{save_path}/aug_crop_{index}_{crop_factor}.jpg", resized)

# Load your images and apply augmentations
input_path = "dataset/occupied"
save_path = "dataset/augocc"
os.makedirs(save_path, exist_ok=True)

for index, filename in enumerate(os.listdir(input_path)):
    img_path = os.path.join(input_path, filename)
    image = cv2.imread(img_path)
    augment_image(image, save_path, index)
