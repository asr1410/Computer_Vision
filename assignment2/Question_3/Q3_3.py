import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import csv

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

non_face_dir = "./data/non-face/"
non_face_csv = "train_non_face.csv"
non_face_features = load_images_and_extract_features(non_face_dir)
non_face_labels = [0] * len(non_face_features)

save_features_to_csv(non_face_features, non_face_labels, non_face_csv)
print(f"LBP features saved to {non_face_csv}")

X_face = np.genfromtxt("train_face.csv", delimiter=',')
X_non_face = np.genfromtxt("train_non_face.csv", delimiter=',')

y_face = np.ones(X_face.shape[0])
y_non_face = np.zeros(X_non_face.shape[0])

X = np.vstack((X_face, X_non_face))
y = np.concatenate((y_face, y_non_face))

np.random.seed(42)
n_samples = X.shape[0]
n_test = int(n_samples * 0.2)
test_indices = np.random.choice(n_samples, n_test, replace=False)
train_indices = np.array(list(set(range(n_samples)) - set(test_indices)))
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)