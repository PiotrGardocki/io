from keras.models import Sequential
from keras import layers
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

optimizers = [
    keras.optimizers.Adam(learning_rate=0.001)
]
activations = [
    'relu',
]

print("Zadanie 4")

df = pd.read_csv("diabetes.csv")
df['class'] = df['class'].apply(lambda x: 1 if x == 'tested_positive' else 0)
df = df.sample(frac=1)

divide_point = int(len(df) * 0.7)
train_set = df[:divide_point]
test_set = df[divide_point:]

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("class")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_set)
test_ds = dataframe_to_dataset(test_set)

train_ds = train_ds.batch(32)
test_ds = test_ds.batch(32)

def encode_feature(feature, name, dataset):
    normalizer = layers.Normalization()

    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature

pregnant_times = keras.Input(shape=(1,), name="pregnant-times")
glucose_concentr = keras.Input(shape=(1,), name="glucose-concentr")
blood_pressure = keras.Input(shape=(1,), name="blood-pressure")
skin_thickness = keras.Input(shape=(1,), name="skin-thickness")
insulin = keras.Input(shape=(1,), name="insulin")
mass_index = keras.Input(shape=(1,), name="mass-index")
pedigree_func = keras.Input(shape=(1,), name="pedigree-func")
age = keras.Input(shape=(1,), name="age")

all_inputs = [
    pregnant_times,
    glucose_concentr,
    blood_pressure,
    skin_thickness,
    insulin,
    mass_index,
    pedigree_func,
    age,
]

pregnant_times_encoded = encode_feature(pregnant_times, "pregnant-times", train_ds)
glucose_concentr_encoded = encode_feature(glucose_concentr, "glucose-concentr", train_ds)
blood_pressure_encoded = encode_feature(blood_pressure, "blood-pressure", train_ds)
skin_thickness_encoded = encode_feature(skin_thickness, "skin-thickness", train_ds)
insulin_encoded = encode_feature(insulin, "insulin", train_ds)
mass_index_encoded = encode_feature(mass_index, "mass-index", train_ds)
pedigree_func_encoded = encode_feature(pedigree_func, "pedigree-func", train_ds)
age_encoded = encode_feature(age, "age", train_ds)

all_features = layers.concatenate(
    [
        pregnant_times,
        glucose_concentr,
        blood_pressure,
        skin_thickness,
        insulin,
        mass_index,
        pedigree_func,
        age,
    ]
)

x = layers.Dense(6, activation="relu")(all_features)
x = layers.Dense(3, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
#model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='mean_absolute_error',
    metrics=['mae'],
)

history = model.fit(train_ds, epochs=500, validation_data=test_ds)

def round_array(array):
    return np.array([
        np.array([round(row[0])]) for row in array
    ])

from sklearn.metrics import confusion_matrix
def print_result(test_set, results):
    total = len(test_set.index)
    correct = 0

    for row, res in zip(test_set.iterrows(), results):
        if row[1]['class'] == res:
            correct += 1

    print(f"Final result: {correct}/{total}")
    print(confusion_matrix(results, test_set['class']))

predicted_results = round_array(model.predict(test_ds))
print_result(test_set, predicted_results)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
