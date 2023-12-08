from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import itertools
from pathlib import Path


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

models_dir = 'models/'

i = 0
for title, file in titles:
    print(f"Testing model number {i}: {title}")

    model: Sequential = keras.saving.load_model(models_dir + file + ".keras")

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    _, test_df = train_test_split(df, test_size=0.30, random_state=42)
    test_df = test_df.reset_index(drop=True)

    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        directory,
        x_col='filename',
        y_col='category',
        class_mode=None,
        target_size=IMAGE_SIZE,
        shuffle=False
    )

    prediction = model.predict(test_generator)
    test_df['predicted_cat'] = np.argmax(prediction, axis=-1)

    count_all = 0
    count_correct = 0
    for ind in test_df.index:
        count_all += 1
        if test_df['category'][ind] == test_df['predicted_cat'][ind]:
            count_correct += 1

    print(f"Score: {count_correct}/{count_all}")

    i += 1
