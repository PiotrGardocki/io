from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os


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

i = 1
for title, file in titles:
    try:
        model: Sequential = keras.saving.load_model(models_dir + file + ".keras")
    except:
        break

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    _, test_df = train_test_split(df, test_size=0.30, random_state=42)
    test_df: pd.DataFrame = test_df.reset_index(drop=True)

    print(f"Testing model number {i}: {title}")
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

    prediction = model.predict(test_generator, verbose=0)
    test_df['predicted_cat'] = np.argmax(prediction, axis=-1)
    test_df['predicted_cat'] = test_df['predicted_cat'].replace({1: 'dog', 0: 'cat'})

    count_all = 0
    count_correct = 0
    incorrect_filenames = []
    for ind in test_df.index:
        count_all += 1
        if test_df['category'][ind] == test_df['predicted_cat'][ind]:
            count_correct += 1
        else:
            incorrect_filenames.append(test_df['filename'][ind])

    print(f"Some incorrect guesses: {incorrect_filenames[:10]}")

    #print(f"Score: {count_correct}/{count_all}, {count_correct * 100 / count_all:.2f}%")
    #print(confusion_matrix(test_df['category'], test_df['predicted_cat']))

    i += 1
