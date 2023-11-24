from keras.models import Sequential
from keras import layers
import keras
import numpy as np

import functools
import operator
import random

input_numbers = 3
results = 2
max_int = 100
train_size = 100000
test_size = 1000

model = Sequential(
    [
        #keras.Input(shape=(input_numbers,)),
        #layers.Dense(10),
        #layers.Dense(results),
        keras.Input(shape=(input_numbers,), name='inp'),
        layers.Dense(results, activation='linear'),
    ]
)

def get_random_tuples(num):
    return np.array([
        np.array([random.randint(0, max_int) for _ in range(3)]) for _ in range(num)
    ])

def get_ref_answers(numbers):
    return np.array([
        np.array([functools.reduce(operator.add, row),
                  functools.reduce(operator.mul, row)]) for row in numbers
    ])

def round_array(array):
    return np.array([
        np.array([round(row[0]), round(row[1])]) for row in array
    ])

def diff_arrays(arr1, arr2):
    return np.array([
        np.array([row[0] - arr2[i][0], row[1] - arr2[i][1]]) for i, row in enumerate(arr1)
    ])

train_data = get_random_tuples(train_size)
train_result = get_ref_answers(train_data)
test_data = get_random_tuples(test_size)
test_result = get_ref_answers(test_data)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mae'],
)

history = model.fit(train_data, train_result, epochs=1000, validation_data=(test_data, test_result))
print(history.history)

print(test_data)
predicted_results = round_array(model.predict(test_data))
#print(test_result[0], predicted_results[0])
print(diff_arrays(predicted_results, test_result))
#print(round_array(model.predict(test_data)))
#print(test_result)