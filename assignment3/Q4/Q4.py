import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_sample_image
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from sklearn.cluster import MeanShift
import cv2
import networkx as nx

## Load the image using cv2
china = cv2.imread('./n01440764_tench.JPEG')

# Convert the image from BGR to RGB
china = cv2.cvtColor(china, cv2.COLOR_BGR2RGB)

# Normalize the image
data = china / 255.0

# Flatten the image to 2D array
data_shape = data.shape
data_2d = data.reshape(data_shape[0] * data_shape[1], data_shape[2])

# Standardize the data
data_2d_standardized = StandardScaler().fit_transform(data_2d)

# Reshape the standardized data back to image shape
data_standardized = data_2d_standardized.reshape(data_shape)

# Convert the image to grayscale for Mean-Shift
gray_china = cv2.cvtColor((data * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

# Flatten the grayscale image to 2D array
gray_china_2d = gray_china.reshape(-1, 1)

# K-Means clustering
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans_labels = kmeans.fit_predict(data_2d_standardized)
kmeans_segmented = kmeans_labels.reshape(data_shape[0], data_shape[1])

# GMM clustering
gmm = GaussianMixture(n_components=8, random_state=42)
gmm_labels = gmm.fit_predict(data_2d_standardized)
gmm_segmented = gmm_labels.reshape(data_shape[0], data_shape[1])

# Mean-Shift clustering
ms = MeanShift(bandwidth=0.1, bin_seeding=True)
ms.fit(gray_china_2d)
ms_labels = ms.labels_

ms_segmented = ms_labels.reshape(data_shape[0], data_shape[1])

# Create graphs for Mean-Shift and GMM
ms_graph = image.img_to_graph(ms_segmented)
gmm_graph = image.img_to_graph(gmm_segmented)

# Create k-means graph using KMeans++ initialization
connectivity = image.grid_to_graph(n_x=data_shape[0], n_y=data_shape[1])
kmeans_graph = connectivity

# Plotting the results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(data)
axes[0, 0].set_title('Original Image')

# Continue plotting the results
axes[0, 1].imshow(kmeans_segmented, cmap='nipy_spectral')
axes[0, 1].set_title('K-Means Segmentation')

axes[0, 2].imshow(gmm_segmented, cmap='nipy_spectral')
axes[0, 2].set_title('GMM Segmentation')

axes[1, 0].imshow(ms_segmented, cmap='nipy_spectral')
axes[1, 0].set_title('Mean-Shift Segmentation')

# Plot the graphs
nx.draw(ms_graph, ax=axes[1, 1], node_size=1, edge_color='b')
axes[1, 1].set_title('Mean-Shift Graph')

nx.draw(gmm_graph, ax=axes[1, 2], node_size=1, edge_color='r')
axes[1, 2].set_title('GMM Graph')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift
from sklearn.feature_extraction import image
from scipy import ndimage as ndi
import networkx as nx
from sklearn.datasets import load_sample_image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load sample image from ImageNet dataset
china = mpimg.imread("./n01440764_tench.JPEG")
data = china / 255.0  # Scale pixel values to [0, 1]

# Convert the image to grayscale
gray_china = cv2.cvtColor((data * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

# Feature vectors for different algorithms
felzenszwalb_features = felzenszwalb(data, scale=100, sigma=0.5, min_size=50)
slic_features = slic(data, n_segments=100, compactness=10, sigma=1)
quickshift_features = quickshift(data, kernel_size=3, max_dist=6, ratio=0.5)

# Function to apply Min-Cut segmentation
def min_cut_segmentation(feature_vector):
    # Here you should implement the Min-Cut algorithm
    # For now, let's just return the input feature vector
    return feature_vector

# Apply Min-Cut on different feature vectors
min_cut_segmented_felzenszwalb = min_cut_segmentation(felzenszwalb_features)
min_cut_segmented_slic = min_cut_segmentation(slic_features)
min_cut_segmented_quickshift = min_cut_segmentation(quickshift_features)

# Plotting the results
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

axes[0, 0].imshow(data)
axes[0, 0].set_title('Original Image')

axes[0, 1].imshow(felzenszwalb_features, cmap='nipy_spectral')
axes[0, 1].set_title('Felzenszwalb Segmentation')

axes[1, 0].imshow(slic_features, cmap='nipy_spectral')
axes[1, 0].set_title('SLIC Segmentation')

axes[1, 1].imshow(quickshift_features, cmap='nipy_spectral')
axes[1, 1].set_title('QuickShift Segmentation')

# Compare Min-Cut segmentation on different feature vectors
axes[2, 0].imshow(min_cut_segmented_felzenszwalb, cmap='nipy_spectral')
axes[2, 0].set_title('Min-Cut on Felzenszwalb')

axes[2, 1].imshow(min_cut_segmented_slic, cmap='nipy_spectral')
axes[2, 1].set_title('Min-Cut on SLIC')

plt.tight_layout()
plt.show()