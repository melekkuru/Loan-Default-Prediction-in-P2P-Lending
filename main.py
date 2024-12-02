from src.preprocess import preprocess_data
from src.models.linear_model import train_linear_regression
from src.models.ridge_model import train_ridge_regression
from src.models.lasso_model import train_lasso_regression
from src.models.random_forest import train_random_forest
from src.models.neural_network import train_neural_network

def main():
    # File paths
    TRAIN_DATA_PATH = "data/trainData.csv"
    TEST_DATA_PATH = "data/testData.csv"

    # Preprocess datasets
    print("Preprocessing training and testing datasets...")
    train_data = preprocess_data("C:/Users/melek/OneDrive/Masaüstü/DATA SCIENCE ESSEX MASTER/TERM 2/Big Data and Financial Computing/ASSIGNMENT/Assignment 2 - Data-20240325/trainData.csv")
    test_data = preprocess_data("C:/Users/melek/OneDrive/Masaüstü/DATA SCIENCE ESSEX MASTER/TERM 2/Big Data and Financial Computing/ASSIGNMENT/Assignment 2 - Data-20240325/testData.csv")

    # Separate features (X) and target (y)
    X_train = train_data.drop("y", axis=1)
    y_train = train_data["y"]
    X_test = test_data.drop("y", axis=1)
    y_test = test_data["y"]

    # Train and evaluate models
    print("\n--- Linear Regression ---")
    train_linear_regression(X_train, y_train, X_test, y_test)

    print("\n--- Ridge Regression ---")
    train_ridge_regression(X_train, y_train, X_test, y_test)

    print("\n--- Lasso Regression ---")
    train_lasso_regression(X_train, y_train, X_test, y_test)

    print("\n--- Random Forest ---")
    train_random_forest(X_train, y_train, X_test, y_test)

    print("\n--- Neural Network ---")
    train_neural_network(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
