from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=120)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs, train_classes)
score = clf.score(test_inputs, test_classes)
print("\nDecisionTreeClassifier")
print("Accuracy:", score * 100, "%")

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_inputs, train_classes)
score = clf.score(test_inputs, test_classes)
print("\nk-NN 3")
print("Accuracy:", score * 100, "%")

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(train_inputs, train_classes)
score = clf.score(test_inputs, test_classes)
print("\nk-NN 5")
print("Accuracy:", score * 100, "%")

clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(train_inputs, train_classes)
score = clf.score(test_inputs, test_classes)
print("\nk-NN 11")
print("Accuracy:", score * 100, "%")

clf = GaussianNB()
clf.fit(train_inputs, train_classes)
score = clf.score(test_inputs, test_classes)
print("\nNaive Bayes")
print("Accuracy:", score * 100, "%")

