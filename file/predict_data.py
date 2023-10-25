import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

import pickle

# Load the class_indices dictionary from the saved file
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Load the saved model
model = keras.models.load_model('your_model.h5')

# Define the image file path
image_path = '/Users/dheemankumar/Desktop/FILE/ST lab/plant project/test/TomatoYellowCurlVirus3.JPG'  # Replace with the path to your image

# Load and preprocess the image
img = image.load_img(image_path, target_size=(200, 200))  # Adjust target_size as needed
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Apply the same rescaling used during training

# Make a prediction
predictions = model.predict(img)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)

# Map the class index to the class label using validation_dataset.class_indices
class_label = [class_label for class_label, index in class_indices.items() if index == predicted_class_index[0]]

# Print the predicted class label
print("Predicted class:", class_label[0])