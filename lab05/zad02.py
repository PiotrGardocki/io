from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

print("Zadanie 2")
iris = load_iris()
datasets = train_test_split(iris.data, iris.target,
                            train_size=0.7)
train_data, test_data, train_labels, test_labels = datasets

scaler = StandardScaler()
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

hidden_layers = (
    (2,),
    (3,),
    (3, 3),
)

for size in hidden_layers:
    print(f"Result for size {size}")
    mlp = MLPClassifier(hidden_layer_sizes=size, max_iter=3000)

    mlp.fit(train_data, train_labels)

    predictions_train = mlp.predict(train_data)
    print(accuracy_score(predictions_train, train_labels))
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels))
