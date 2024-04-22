import os
import cv2
import numpy as np
import imghdr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

# Setup and configure GPU to avoid out of memory errors
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


# Data preparation and augmentation
def prepare_data(data_dir, target_size=(96, 96), batch_size=64):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator

def load_model_directly(model_path):
    try:
        print(f"Loading model from {model_path}")
        model = load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load the model from {model_path}. Error: {e}")



# Model configuration
def build_model(input_shape=(96, 96, 3), num_classes=21):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
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
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    return model


adam = tf.keras.optimizers.Adam(learning_rate=0.001)
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001)

def train_model(model, train_generator, val_generator, class_weight_dict, epochs=50):
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    callbacks = [
        EarlyStopping(patience=10),
        ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.001),
        ModelCheckpoint('models/cnn2AA80k_{epoch}.keras', verbose=1, save_best_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    return history

def display_image_predictions(generator, model, num_images=5):
    for i in range(num_images):
        x, y = next(generator)
        image = x[0]
        label = y[0]
        prediction = model.predict(np.expand_dims(image, axis=0))
        predicted_label = np.argmax(prediction, axis=1)

        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f'Actual: {np.argmax(label)}, Predicted: {predicted_label[0]}')
        plt.axis('off')
        plt.show()


def main():
    configure_gpu()
    data_dir = 'aa'
    train_generator, val_generator = prepare_data(data_dir)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))
    # model = build_model(input_shape=(96, 96, 3), num_classes=train_generator.num_classes)
    model = load_model_directly('models/cnn2AA21kNowyKod.keras')
    model.summary()
    history = train_model(model, train_generator, val_generator, class_weight_dict, epochs=50)
    model.save('models/cnn2AA21kNowyKod2tren.keras')
    # Add additional model evaluation or further training steps here if needed
    display_image_predictions(val_generator, model)

if __name__ == '__main__':
    main()



