from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"Random Forest - MSE Train: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Random Forest - MSE Test: {mean_squared_error(y_test, y_pred_test)}")
