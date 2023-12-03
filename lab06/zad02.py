from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import itertools


FAST_RUN = True
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

directory = "dogs-cats-mini/"
filenames = os.listdir(directory)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax'),
])

model2 = Sequential([
    Conv2D(32, (3, 3), activation='sigmoid', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='sigmoid'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='sigmoid'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='sigmoid'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax'),
])

model3 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(2, activation='softmax'),
])

model4 = Sequential([
    Conv2D(32, (3, 3), activation='sigmoid', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='sigmoid'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='sigmoid'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='sigmoid'),
    BatchNormalization(),
    Dense(2, activation='softmax'),
])

models = [model1, model2, model3, model4]
optimizers = ['rmsprop', 'adam']
titles = [
    ('relu + dropout, rmsprop optimizer', 'relu_dropout_rmsprop'),
    ('relu + dropout, adam optimizer', 'relu_dropout_adam'),
    ('sigmoid + dropout, rmsprop optimizer', 'sigmoid_dropout_rmsprop'),
    ('sigmoid + dropout, adam optimizer', 'sigmoid_dropout_adam'),
    ('relu + no dropout, rmsprop optimizer', 'relu_rmsprop'),
    ('relu + no dropout, adam optimizer', 'relu_adam'),
    ('sigmoid + no dropout, rmsprop optimizer', 'sigmoid_rmsprop'),
    ('sigmoid + no dropout, adam optimizer', 'sigmoid_adam'),
]

plots_dir = 'plots/'

for (model, optimizer), (title, file) in zip(itertools.product(models, optimizers), titles):
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    train_df, test_val = train_test_split(df, test_size=0.30, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df, validate_df = train_test_split(df, test_size=0.50, random_state=42)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    batch_size = 15

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        directory,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        directory,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    epochs = 1 if FAST_RUN else 50
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )

    print(history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    ax1.legend(loc='best', shadow=True)
    ax2.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(plots_dir + file + '.png')
