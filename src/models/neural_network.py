from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_neural_network(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"Neural Network - Accuracy Train: {accuracy_score(y_train, y_pred_train)}")
    print(f"Neural Network - Accuracy Test: {accuracy_score(y_test, y_pred_test)}")
