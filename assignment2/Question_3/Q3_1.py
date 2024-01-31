import cv2
import os
import csv
import numpy as np
from skimage import feature

def extract_lbp_features(image_path, radius=1, n_points=8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    lbp = feature.local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    return hist

def load_images_and_extract_features(directory):
    feature_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            features = extract_lbp_features(image_path)
            feature_list.append(features)
    return feature_list

def save_features_to_csv(data, labels, csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for label, features in zip(labels, data):
            writer.writerow([label] + list(features))

face_dir = "./data/face/train/"
face_csv = "train_face.csv"
face_features = load_images_and_extract_features(face_dir)
face_labels = [1] * len(face_features)  # Label 1 for face
save_features_to_csv(face_features, face_labels, face_csv)
print(f"LBP features saved to {face_csv}")

val_dir = "./data/face/val/"
val_csv = "val_face.csv"
val_features = load_images_and_extract_features(val_dir)
val_labels = [1] * len(val_features)  # Label 1 for face
save_features_to_csv(val_features, val_labels, val_csv)
print(f"LBP features saved to {val_csv}")

test_dir = "./data/face/test/"
test_csv = "test_face.csv"
test_features = load_images_and_extract_features(test_dir)
test_labels = [1] * len(test_features)  # Label 1 for face
save_features_to_csv(test_features, test_labels, test_csv)
print(f"LBP features saved to {test_csv}")