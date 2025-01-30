import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('/Users/gk/Documents/GitHub/Plant-Leaf-Disease-Detection/notebooks/alexnet_5.keras')

# Path to your training dataset
train_data_dir = '/Users/gk/Desktop/data/train'

# Dynamically generate class names from subdirectory names
class_names = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))])

# Function to preprocess the image
def preprocess_image(image_path, img_size=(227, 227)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, img_size)  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict disease from an image
def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Get predictions
    predicted_class_index = np.argmax(prediction)  # Get index of highest probability
    confidence = np.max(prediction)  # Get confidence score
    predicted_class_name = class_names[predicted_class_index]  # Get disease name

    # Display the image with prediction
    plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class_name} (Class {predicted_class_index}, {confidence:.2f})")
    plt.show()

    return predicted_class_index, predicted_class_name, confidence

# Test with a new image (replace with actual path)
image_path = "/Users/gk/Desktop/data/valid/Peach___Bacterial_spot/image (566).JPG"# Change to actual image path
predicted_class_index, predicted_class_name, confidence = predict_disease(image_path)

# Print result
print(f"Predicted Class Index: {predicted_class_index}, Disease: {predicted_class_name}, Confidence: {confidence:.2f}")
