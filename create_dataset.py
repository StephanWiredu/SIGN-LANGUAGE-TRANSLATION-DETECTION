import os
import pickle
import cv2
import numpy as np

# Define the directory where your image data is stored
DATA_DIR = './data'

data = []
labels = []

# Define the number of classes and dataset size
num_classes = 26
dataset_size = 384

# Loop through each class
for class_label in range(num_classes):
    class_dir = os.path.join(DATA_DIR, str(class_label))
    
    # Ensure the class directory exists
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Class directory '{class_dir}' not found.")
    
    print(f'Collecting data for class {class_label}')
    
    # Loop through images in the class directory
    for img_path in os.listdir(class_dir)[:dataset_size]:
        # Load and preprocess the image
        img = cv2.imread(os.path.join(class_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = cv2.resize(img, (64, 64))  # Resize to match your model's input shape
        img = img / 255.0  # Normalize pixel values to [0, 1]
        
        # Append the preprocessed image to the 'data' list
        data.append(img)
        
        # Append the class label to the 'labels' list
        labels.append(class_label)

# Convert 'data' and 'labels' lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Verify the shape of 'data' and 'labels' arrays
print("Shape of 'data' array:", data.shape)
print("Shape of 'labels' array:", labels.shape)

# Save 'data' and 'labels' arrays to a pickle file
with open('data.pickle', 'wb') as file:
    pickle.dump({'data': data, 'labels': labels}, file)
