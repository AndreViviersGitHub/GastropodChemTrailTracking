import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

# Define the directory containing your images
image_dir = 'E:\\Snail Images\\RandomBS'

# Number of clusters (adjust as needed)
num_clusters = 8


# Function to load and preprocess images
def load_images(image_dir):
    images = []
    image_files = [filename for filename in os.listdir(image_dir) if
                   filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Get the total number of images that match the supported extensions
    total_images = len(image_files)

    for count, filename in enumerate(image_files, start=1):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)

        # Print progress
        print(f"Loading image {count}/{total_images}: {filename}")

    return images


# Function to cluster images using K-Means
def cluster_images(images, num_clusters):
    # Reshape and normalize the pixel values
    flattened_images = np.array([img.reshape(-1) / 255.0 for img in images])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(flattened_images)

    # Print progress
    print("Clustering images complete.")

    return cluster_labels


# Function to organize images into folders based on clusters
def organize_images(image_dir, cluster_labels):
    for i in range(num_clusters):
        cluster_folder = os.path.join(image_dir, f'Cluster_{i}')
        os.makedirs(cluster_folder, exist_ok=True)

    for filename, label in zip(os.listdir(image_dir), cluster_labels):
        src_path = os.path.join(image_dir, filename)
        dest_folder = os.path.join(image_dir, f'Cluster_{label}')
        dest_path = os.path.join(dest_folder, filename)
        os.rename(src_path, dest_path)


if __name__ == '__main__':
    print("Loading and preprocessing images...")
    images = load_images(image_dir)

    if not images:
        print("No valid images found in the directory.")
    else:
        print(f"Clustering {len(images)} images...")
        cluster_labels = cluster_images(images, num_clusters)
        print("Organizing images into folders...")
        organize_images(image_dir, cluster_labels)
        print("Images have been clustered and organized into folders.")
