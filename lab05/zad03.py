from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd

print("Zadanie 3")

df = pd.read_csv("diabetes.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

train_data = train_set[:, 0:8]
train_labels = train_set[:, 8]
test_data = test_set[:, 0:8]
test_labels = test_set[:, 8]

mlp = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=50000, activation='relu', alpha=0.00005)

mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
print(confusion_matrix(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))
print(confusion_matrix(predictions_test, test_labels))
