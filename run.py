import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
import subprocess
import shutil
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyntcloud import PyntCloud
import open3d as o3d
import pandas as pd

# Load VGG16 Model
model = VGG16(weights='imagenet')

def extract_features(folder_path, output_folder):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
            # Step 2: Load and Preprocess Images
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)

            # Step 4: Extract Features
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            features = model.predict(img_array)

            # Step 5: Save Features (Optional)
            output_path = os.path.join(output_folder, f'features_{filename}.npy')
            np.save(output_path, features)


# Image folders:
source_path = "replace /path to images"
results_path = "replace /path to output"

# Process the images and store results to results_path folder

#Clear stable diffusion output and backs up content to a different path

# Generate a timestamp for the current time
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

# Create the destination folder with the timestamp
output_folder = os.path.join(results_path, f"result_{timestamp}")

if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

# Copy the source folder to the destination
shutil.copytree(source_path, output_folder)

# Clear the contents of the source folder
#for root, dirs, files in os.walk(source_path):
#    for file in files:
#        file_path = os.path.join(root, file)
#        os.remove(file_path)
#    for dir in dirs:
#        dir_path = os.path.join(root, dir)
#        os.rmdir(dir_path)

# Process each image in the folder
extract_features(source_path, output_folder)

# Step 1: Set the path to the folder containing the saved features
features_folder = output_folder

# Step 2: Initialize arrays to store features and filenames
all_features = []
all_filenames = []

# Step 3: Load features from files
for filename in os.listdir(features_folder):
    if filename.endswith('.npy'):
        features_path = os.path.join(features_folder, filename)
        features = np.load(features_path)
        
        # Assuming features are 1D or 2D, reshape them into a flat 1D array
        features = features.flatten()

        all_features.append(features)
        all_filenames.append(filename)

# Step 4: Apply PCA for dimensionality reduction (optional but recommended)
num_components = min(len(all_features), 2)  # Use the minimum of the number of samples and features
pca = PCA(n_components=num_components)
reduced_features = pca.fit_transform(all_features)

# Step 5: Visualize the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point in the point cloud
ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c='b', marker='o')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# Mesh generation

# Step 4: Create synthetic coordinates for the point cloud
num_points = len(all_features)
synthetic_coordinates = np.random.rand(num_points, 3)  # Random 3D coordinates

# Step 5: Create a PointCloud object with Open3D
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(synthetic_coordinates)

# Step 6: Estimate normals for the point cloud
cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Step 7: Poisson surface reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud)

# Step 6: Save the mesh to a file (e.g., in STL format)
mesh_filename = 'mesh.ply'
mesh_path = os.path.join(output_folder, mesh_filename)
o3d.io.write_triangle_mesh(mesh_path, mesh)

plt.show()


