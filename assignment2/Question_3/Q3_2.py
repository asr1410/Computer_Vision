import numpy as np

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            data.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return data

train_data = read_csv("train_face.csv")
val_data = read_csv("val_face.csv")
test_data = read_csv("test_face.csv")

X_train = np.array([row[1:] for row in train_data])
y_train = np.array([row[0] for row in train_data])
X_val = np.array([row[1:] for row in val_data])
y_val = np.array([row[0] for row in val_data])
X_test = np.array([row[1:] for row in test_data])
y_test = np.array([row[0] for row in test_data])

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict_knn(X_train, y_train, x, k=3):
    distances = [euclidean_distance(x, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = np.bincount(k_nearest_labels).argmax()
    return most_common

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y[idx] * (np.dot(x, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x, self.w))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

best_knn_accuracy = 0
best_knn_model = None
best_svm_accuracy = 0
best_svm_model = None

for k in [1, 3, 5, 7]:
    knn_predictions = [predict_knn(X_train, y_train, x, k) for x in X_val]
    knn_accuracy = (np.array(knn_predictions) == y_val).mean()

    if knn_accuracy > best_knn_accuracy:
        best_knn_accuracy = knn_accuracy
        best_knn_model = k

svm = LinearSVM()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_val)
svm_accuracy = (np.array(svm_predictions) == y_val).mean()

if svm_accuracy > best_svm_accuracy:
    best_svm_accuracy = svm_accuracy
    best_svm_model = svm

knn_test_predictions = [predict_knn(X_train, y_train, x, best_knn_model) for x in X_test]
svm_test_predictions = best_svm_model.predict(X_test)

knn_test_accuracy = (np.array(knn_test_predictions) == y_test).mean()
svm_test_accuracy = (np.array(svm_test_predictions) == y_test).mean()

print(f"Best KNN Model Accuracy on Val.csv: {best_knn_accuracy:.2f}")
print(f"Accuracy on Test.csv using Best KNN Model: {knn_test_accuracy:.2f}")

print(f"Best SVM Model Accuracy on Val.csv: {best_svm_accuracy:.2f}")
print(f"Accuracy on Test.csv using Best SVM Model: {svm_test_accuracy:.2f}")