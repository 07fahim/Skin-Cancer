import os
import random
import shutil
from math import floor

# Define paths
dataset_dir = 'train'
output_dir = 'skinsegmentation'

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Create the output directories if they don't exist
train_output_dir = os.path.join(output_dir, 'train')
val_output_dir = os.path.join(output_dir, 'val')
test_output_dir = os.path.join(output_dir, 'test')

for dir_path in [train_output_dir, val_output_dir, test_output_dir]:
    os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

# List all image files and corresponding label files
images = sorted([f for f in os.listdir(os.path.join(dataset_dir, 'images')) if f.endswith('.jpg')])
labels = sorted([f for f in os.listdir(os.path.join(dataset_dir, 'labels')) if f.endswith('.txt')])

# Ensure image-label file pairs match
assert len(images) == len(labels), "Number of images and labels do not match!"

# Shuffle the dataset
data = list(zip(images, labels))
random.shuffle(data)

# Split the dataset
num_images = len(images)
train_count = floor(num_images * train_ratio)
val_count = floor(num_images * val_ratio)
test_count = num_images - train_count - val_count

train_data = data[:train_count]
val_data = data[train_count:train_count + val_count]
test_data = data[train_count + val_count:]

# Function to copy files to the destination
def copy_files(data, destination):
    for image_file, label_file in data:
        shutil.copy(os.path.join(dataset_dir, 'images', image_file), os.path.join(destination, 'images', image_file))
        shutil.copy(os.path.join(dataset_dir, 'labels', label_file), os.path.join(destination, 'labels', label_file))

# Copy files to their respective directories
copy_files(train_data, train_output_dir)
copy_files(val_data, val_output_dir)
copy_files(test_data, test_output_dir)

print(f"Training set: {train_count} images")
print(f"Validation set: {val_count} images")
print(f"Test set: {test_count} images")
