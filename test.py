import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import matplotlib.pyplot as plt
import glob

def test_model(model):
    test_dir = 'test'
    class_names = sorted(os.listdir(test_dir))

    correct_predictions = 0
    total_predictions = 0

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        image_files = glob.glob(os.path.join(class_dir, '*.jpg'))  # adjust the file extension if needed

        for image_file in image_files:  # iterate over all images
            image = cv2.imread(image_file)
            image_resized = cv2.resize(image, (96, 96))  # replace with the size your model expects
            image_normalized = image_resized / 255.0  # normalize pixel values if your model expects it
            image_batch = np.expand_dims(image_normalized, axis=0)  # add an extra dimension because the model expects batches

            prediction = model.predict(image_batch)
            top_three_indices = np.argsort(prediction[0])[-3:]  # get top 3 predictions
            predicted_classes = [class_names[i] for i in top_three_indices]

            # new code to display percentages
            predicted_percentages = [prediction[0][i] for i in top_three_indices]
            predicted_with_percentages = list(zip(predicted_classes, predicted_percentages))

            print(f"File: {image_file}")
            print(f"Actual: {class_name}")
            print(f"Predicted: {predicted_with_percentages}")

            total_predictions += 1
            if class_name in predicted_classes:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy}")



def nice(model):
    data_dir = 'test'

    # Load data
    data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(96, 96), batch_size=16)
    class_names = data.class_names
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    # Scale data
    data = data.map(lambda x, y: (x/255, tf.one_hot(y, depth=15)))
    print(len(data))

    # Split data
    # test_size = int(len(data) * 0.5)
    test_size = int(len(data) * 0.75)
    test = data.take(test_size)

    top3_acc = TopKCategoricalAccuracy(k=3)

    for batch in test.as_numpy_iterator():
        X, y_true = batch
        y_pred = model.predict(X)
        
        # Update Top-3 accuracy
        top3_acc.update_state(y_true, y_pred)

        # Optionally print the predicted class names for the first few images in the batch
        for i in range(len(y_pred)-1):
            top3_indices = np.argsort(y_pred[i])[-3:]
            print(f"Top-3 Predicted: {[class_names[j] for j in top3_indices]}, Actual: {class_names[np.argmax(y_true[i])]}")

    print(f"Test Top-3 Accuracy: {top3_acc.result().numpy()}")

# model = load_model('models/cnn5RMS.keras') # 96% accuracy
# model = load_model('models/ADAM.keras') # 100% accuracy


# test_model(model)
nice(model)
