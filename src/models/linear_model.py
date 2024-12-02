from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_linear_regression(X_train, y_train, X_test, y_test):
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Print results
    print("\n--- Linear Regression ---")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    # Match coefficients with feature names
    print("\nFeature Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"{feature}: {coef}")

    # Calculate and print errors
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    print(f"\nMean Squared Error (Train): {mse_train}")
    print(f"Mean Squared Error (Test): {mse_test}")
