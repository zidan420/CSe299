import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, label, img_size=(64, 64)):
    images = []
    labels = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load image: {path}")
            continue
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize pixel values
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load image paths
car_images = ['datasets/cars/' + f for f in os.listdir('datasets/cars/') if f.endswith('.jpg')]
non_car_images = ['datasets/trees/' + f for f in os.listdir('datasets/trees/') if f.endswith('.jpg')]

# Load and preprocess data
car_data, car_labels = load_and_preprocess_images(car_images, 1)
non_car_data, non_car_labels = load_and_preprocess_images(non_car_images, 0)

# Combine datasets
X = np.concatenate((car_data, non_car_data), axis=0)
y = np.concatenate((car_labels, non_car_labels), axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build simpler CNN model for small dataset
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Function to predict on a new image
def predict_image(image_path, model, img_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image"
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return "Car" if prediction[0][0] > 0.5 else "Not a Car"

# Example usage (uncomment to test a new image)
result = predict_image('image_60.jpg', model)
print(f"Prediction: {result}")