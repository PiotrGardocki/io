from sklearn.model_selection import train_test_split
import pandas as pd

def classify_iris(sl, sw, pl, pw):
    if pw < 1.0:
        return("Setosa")
    elif pl > 4.8:
        return("Virginica")
    else:
        return("Versicolor")

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=120)

good_predictions = 0
len = test_set.shape[0]

for iris in test_set:
    if classify_iris(*iris[:4]) == iris[4]:
        good_predictions += 1

print(f"{good_predictions}/{len}")
print(good_predictions / len * 100, "%")
