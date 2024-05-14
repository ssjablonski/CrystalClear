import os
import cv2
import numpy as np
import imghdr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,  BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Setup and configure GPU to avoid out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Remove dodgy images
data_dir = 'new'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Load data
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(96, 96), batch_size=32)
class_names = data.class_names
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Scale data
data = data.map(lambda x, y: (x/255, tf.one_hot(y, depth=15)))
print(len(data))

# Split data
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

model = load_model('models/ADAM.keras')  

# model = Sequential([
#     Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 3)),
#     BatchNormalization(),
#     Conv2D(32, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(4, 4)),  # increased pool_size
#     Dropout(0.25),

#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(4, 4)),  # increased pool_size
#     Dropout(0.25),

#     Conv2D(128, (3, 3), padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(128, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(4, 4)),  # increased pool_size
#     Dropout(0.25),

#     Flatten(),
#     Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     Dropout(0.5),
#     Dense(15, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()


earlystop = EarlyStopping(patience=10)
modelCheck = ModelCheckpoint('models/1.keras', verbose=1, save_best_only=True),

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.001)

# Train
# logdir = 'logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
callbacks = [earlystop, modelCheck, learning_rate_reduction]
hist = model.fit(train, epochs=30, validation_data=val, callbacks=callbacks)

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


# Initialize the Top-3 accuracy metric
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


# Save the Model
model.save(os.path.join('models', 'ADAM2.keras'))
