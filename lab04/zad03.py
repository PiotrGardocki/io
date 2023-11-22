from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
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

tree.plot_tree(clf)
plt.show()

score = clf.score(test_inputs, test_classes)
print("Accuracy:", score * 100, "%")

disp = ConfusionMatrixDisplay.from_estimator(clf,
                                             test_inputs,
                                             test_classes)
plt.show()
