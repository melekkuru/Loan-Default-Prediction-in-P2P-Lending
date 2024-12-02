# Loan-Default-Prediction-in-P2P-Lending


This project aims to predict loan default probabilities in peer-to-peer (P2P) lending using various machine learning models. By analyzing loan data, the project provides insights into the factors contributing to loan defaults, which can help stakeholders make better financial decisions.

---

## **Project Overview**

Peer-to-peer lending platforms enable individuals to lend and borrow money without traditional financial intermediaries. However, one key challenge is the risk of loan defaults. This project employs multiple machine learning algorithms to analyze and predict loan default risks.

### **Key Features**
- Data preprocessing: Cleaning and encoding of loan datasets.
- Model training and evaluation:
  - **Linear Regression**
  - **Ridge Regression**
  - **Lasso Regression**
  - **Random Forest**
  - **Neural Network**
- Model comparison based on performance metrics.
- Scalable and modular project structure.

---

## **Project Structure**

```plaintext
Loan Default Prediction in P2P Lending/
├── src/                      # Source folder
│   ├── preprocess.py         # Preprocessing functions
│   ├── models/               # Machine learning models
│   │   ├── linear_model.py
│   │   ├── ridge_model.py
│   │   ├── lasso_model.py
│   │   ├── random_forest.py
│   │   ├── neural_network.py
│   └── utils.py              # Utility functions
├── main.py                   # Main script to run the project
├── requirements.txt          # Python dependencies
├── notebook.ipynb            # Jupyter Notebook for exploratory analysis and experimentation
├── README.md                 # Project documentation
```

---

## **How to Run**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction
```

### **2. Set Up the Environment**
Create and activate a virtual environment:
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### **3. Update Dataset Paths**
Replace the dataset paths in `main.py` with the full paths to your local datasets:
```python
train_data = preprocess_data("<full-path-to-trainData.csv>")
test_data = preprocess_data("<full-path-to-testData.csv>")
```

### **4. Run the Project**
```bash
python main.py
```

The outputs (model coefficients, metrics, etc.) will be displayed in the console.

---

## **Results**

The project compares the performance of different machine learning models, including:
- Mean Squared Error (MSE) for both training and test datasets.
- Feature importance and coefficient analysis.
- Predictive performance of the neural network.

---

## **Requirements**

Ensure the following dependencies are installed (available in `requirements.txt`):
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorflow

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## **Data Privacy**

Due to privacy concerns, the datasets used in this project are not included in the repository. Replace the dataset paths with your own files as described in the **How to Run** section.

---


## **Acknowledgments**

- This project was developed as part of a data science coursework.



