import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Define constants
DATA_DIR = './data'  
LOG_DIR = './logs'
NUM_CLASSES = 26  
DATASET_SIZE = 384
IMAGE_SIZE = (64, 64)
EPOCHS = 100  
BATCH_SIZE = 32

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory '{DATA_DIR}' not found.")

# Initialize lists to store image data and labels
images = []
labels = []

# Loop through each class
for class_label in range(NUM_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(class_label))
    
    # Ensure the class directory exists
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Class directory '{class_dir}' not found.")
    
    print(f'Collecting data for class {class_label}')
    
    # Loop through images in the class directory
    for img_path in os.listdir(class_dir)[:DATASET_SIZE]:
        # Load and preprocess the image
        img = cv2.imread(os.path.join(class_dir, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = cv2.resize(img, IMAGE_SIZE)  # Resize to match your model's input shape
        img = img / 255.0  # Normalize pixel values to [0, 1]
        
        # Append the preprocessed image to the 'images' list
        images.append(img)
        
        # Append the class label to the 'labels' list
        labels.append(class_label)

# Convert 'images' and 'labels' lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Verify the shape of 'images' and 'labels' arrays
print("Shape of 'images' array:", images.shape)
print("Shape of 'labels' array:", labels.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True, stratify=labels)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Create the model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')  # Update the number of classes here

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping criteria
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the best weights when early stopping is triggered
)

# Define a ModelCheckpoint callback to save the best model during training
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

# Convert class_weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Train the model with the TensorBoard callback, early stopping, and model checkpoint callbacks
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict  # Use the calculated class weights
)

# Load the best model (optional)
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate the best model on the test data
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Save the trained model to a native Keras format
best_model.save('your_new_model.h5')
print("Model saved as 'your_new_model.h5'")
