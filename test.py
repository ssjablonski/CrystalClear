import os
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model('models/cnn2AA21kNowyKod5tren76.keras')
# C:\Users\48573\Desktop\projekty\CrystalClear\models\cnn2NowyKod2tren3.keras
# C:\Users\48573\Desktop\projekty\CrystalClear\models\cnn2AA21kNowyKod.keras
# C:\Users\48573\Desktop\projekty\CrystalClear\models\cnn2AA21kNowyKod5tren76.keras

# Define the directory
data_dir = 'aa'

# Get the list of all subdirectories
subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Define class names
class_names = sorted(os.listdir(data_dir))

# Initialize counter for correct predictions
correct_predictions = 0

# Create a figure for subplots
fig = plt.figure(figsize=(20, 20))

# For each subdirectory
for i, subdir in enumerate(subdirs, start=1):
    # Get a list of all images in the subdirectory
    images = [os.path.join(subdir, f) for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]

    # Select a random image
    random_image_path = random.choice(images)

    # Load and preprocess the image
    image = cv2.imread(random_image_path)
    image_resized = cv2.resize(image, (96, 96))  # replace with the size your model expects
    image_normalized = image_resized / 255.0  # normalize pixel values if your model expects it
    image_batch = np.expand_dims(image_normalized, axis=0)  # add an extra dimension because the model expects batches

    # Make prediction
    prediction = model.predict(image_batch)

    # Get the indices that would sort the array, then take the last three
    top_three_indices = np.argsort(prediction[0])[-3:]

    # Map the predicted class indices to the class names
    predicted_classes = [class_names[i] for i in top_three_indices]

    # Check if actual class is in top three predictions
    actual_class = os.path.basename(subdir)
    if actual_class in predicted_classes:
        correct_predictions += 1

    # Print the top three predicted class names
    print(f'Actual class: {actual_class}, Top three predicted classes: {predicted_classes}')

    # Add subplot for the image
    # ax = fig.add_subplot(5, 5, i)  # adjust numbers according to the number of subdirectories
    # ax.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    # ax.title.set_text(f'Actual: {actual_class}, Predicted: {predicted_classes}')

# Show all images at once

# Calculate and print accuracy
accuracy = correct_predictions / len(subdirs)
print(f'Accuracy: {accuracy}')