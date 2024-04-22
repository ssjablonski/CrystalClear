import os
import cv2
import numpy as np
import imghdr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,  BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Setup and configure GPU to avoid out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Remove dodgy images
data_dir = 'ds/train'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, image_class)
    for image in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'Image not in ext list {image_path}')
                os.remove(image_path)
        except Exception as e:
            print(f'Issue with image {image_path}')
            os.remove(image_path)

# Load data
data = tf.keras.utils.image_dataset_from_directory(data_dir)
class_names = data.class_names
# print(class_names)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Scale data
data = data.map(lambda x, y: (x/255, tf.one_hot(y, depth=12)))
print(len(data))

# Split data
train_size = int(len(data) * 0.8)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Conv2D(128, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(12, activation='softmax')
# ])

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(12, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
callbacks = [earlystop, learning_rate_reduction, tensorboard_callback]
hist = model.fit(train, epochs=25, validation_data=val, callbacks=callbacks)

# Plot performance
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
axs[0].plot(hist.history['loss'], color='teal', label='loss')
axs[0].plot(hist.history['val_loss'], color='orange', label='val_loss')
axs[0].legend(loc="upper left")
axs[0].set_title('Loss')

axs[1].plot(hist.history['accuracy'], color='teal', label='accuracy')
axs[1].plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
axs[1].legend(loc="upper left")
axs[1].set_title('Accuracy')

plt.show()


# Evaluate
acc = CategoricalAccuracy()
for batch in test.as_numpy_iterator():
    X, y_true = batch
    y_pred = model.predict(X)
    # Convert predictions from one-hot vectors to class indices
    y_pred_labels = np.argmax(y_pred, axis=1)
    # Convert true labels from one-hot vectors to class indices
    y_true_labels = np.argmax(y_true, axis=1)
    # Update accuracy with true labels
    acc.update_state(y_true_labels, y_pred_labels)

    # Optionally print the predicted class names for the first few images in the batch
    for i in range(len(y_pred_labels)-1):
        print(f"Predicted: {class_names[y_pred_labels[i]]}, Actual: {class_names[y_true_labels[i]]}")

print(f"Test Accuracy: {acc.result().numpy()}")


# Save the Model
model.save(os.path.join('models', 'crystalClear3.h5'))
