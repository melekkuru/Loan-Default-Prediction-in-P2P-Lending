# Loan Default Prediction in P2P Lending

Predicting loan defaults on a peer-to-peer lending platform using multiple machine learning models. The project compares Linear Regression, Ridge, Lasso, Random Forest, and Neural Network approaches to identify the most effective method for credit risk assessment.

>  Developed as part of the **Big Data for Computational Finance (CF969)** module at the University of Essex.

---

## Overview

P2P lending platforms connect borrowers and lenders directly, but loan defaults pose significant financial risk. This project builds and evaluates five predictive models to classify whether a borrower is likely to default, using real-world-style loan data with features such as income, loan grade, interest rate, and payment history.

**Pipeline:**
1. Data preprocessing — handling missing values, feature engineering, normalization
2. Model training — five models with hyperparameter tuning
3. Evaluation — MSE comparison, feature importance analysis, model selection

---

## Models & Results

| Model | Train MSE | Test MSE | Notes |
|-------|-----------|----------|-------|
| Linear Regression | 0.0669 | 3820.7 | Severe overfitting due to multicollinearity |
| Ridge Regression (λ=0.01) | 0.0668 | 0.0678 | Regularization resolves overfitting |
| Lasso Regression (λ=0.01) | 0.0962 | 0.0967 | Sparse model; selects `grade` and `total_rec_prncp` |
| **Random Forest (100 trees)** | **0.0028** | **0.0223** | **Best overall — high accuracy + interpretability** |
| Neural Network (MLP) | 96.64% acc | 95.93% acc | Highest accuracy but lower interpretability |

**Best model:** Random Forest — chosen for its balance of accuracy, generalization, and interpretability. Key features: `recoveries`, `int_rate`, `installment`.

---

## Project structure

```
Loan-Default-Prediction-in-P2P-Lending/
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   └── models/
│       ├── linear_model.py
│       ├── ridge_model.py
│       ├── lasso_model.py
│       ├── random_forest.py
│       └── neural_network.py
├── notebook/
│   └── loan_default_prediction.ipynb
├── main.py                   # Run all models
├── requirements.txt
└── README.md
```

---

## Getting started

```bash
git clone https://github.com/melekkuru/Loan-Default-Prediction-in-P2P-Lending.git
cd Loan-Default-Prediction-in-P2P-Lending

python -m venv venv
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

**Note:** The dataset is not included due to privacy constraints. Place your `trainData.csv` and `testData.csv` files in the project root and update paths in `main.py`.

```bash
python main.py
```

---

## Technologies

Python · scikit-learn · TensorFlow/Keras · pandas · NumPy · Matplotlib · Seaborn

---

## License

This project is for educational purposes. Feel free to use it as a reference.
