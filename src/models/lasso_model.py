from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def train_lasso_regression(X_train, y_train, X_test, y_test):
    # Lasso model
    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X_train, y_train)

    print("\n--- Lasso Regression ---")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    # Match coefficients with feature names
    print("\nFeature Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"{feature}: {coef}")

    # Calculate and print errors
    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, model.predict(X_test))
    print(f"\nMean Squared Error (Train): {mse_train}")
    print(f"Mean Squared Error (Test): {mse_test}")
