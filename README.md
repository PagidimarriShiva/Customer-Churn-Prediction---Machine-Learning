# Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to churn based on customer demographics, account information, and banking activity.

## Project Overview

Customer churn is an important business problem for banks because identifying customers who are likely to leave can help the business take retention actions.

In this project, I built a supervised machine learning classification pipeline to predict the `Exited` target variable.

The project covers:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Train-test split
* Feature scaling
* Categorical feature encoding
* Model training
* Model evaluation
* Model comparison
* Hyperparameter tuning
* Model serialization
* Streamlit application

## Dataset

The dataset contains customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card status
* Active Member status
* Estimated Salary

### Target Variable

`Exited`

```text
0 → Customer did not churn
1 → Customer churned
```

The dataset contains:

```text
Total records: 10,000

Stayed: 7,963
Churned: 2,037
```

## Exploratory Data Analysis

The following analysis was performed:

* Dataset shape and structure
* Data types and information
* Missing-value check
* Duplicate-value check
* Descriptive statistics
* Categorical feature distributions
* Gender vs. churn analysis
* Geography vs. churn analysis
* Target distribution
* Separation of input features and target

## Data Preprocessing

The features were divided into numerical and categorical columns.

### Numerical Features

```text
CreditScore
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
```

### Categorical Features

```text
Geography
Gender
```

A `ColumnTransformer` was used to apply different preprocessing steps:

* `StandardScaler` for numerical features
* `OneHotEncoder` for categorical features

The preprocessing and model were combined using a Scikit-learn `Pipeline`.

## Train-Test Split

The dataset was divided into:

```text
Training set: 80%
Testing set: 20%
```

Stratified splitting was used because the target classes are imbalanced.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Models Tested

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Comparison

| Model               | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |   ROC-AUC |
| ------------------- | -------: | ----------------: | -------------: | ---------: | --------: |
| Logistic Regression |    80.8% |             58.9% |          18.7% |      28.4% |         — |
| Decision Tree       |    86.0% |             69.7% |          51.3% |  **59.0%** |     0.832 |
| Random Forest       |    85.8% |         **76.0%** |          43.5% |      55.0% | **0.846** |

## Final Model

The **Decision Tree Classifier** was selected as the final model.

The model was selected based primarily on the performance of the positive class (`Exited = 1`), particularly:

* Recall
* F1-score

The Decision Tree achieved:

```text
Accuracy:        86.0%
Churn Precision: 69.7%
Churn Recall:    51.3%
Churn F1-score:  59.0%
ROC-AUC:         0.832
```

The Decision Tree was tuned using `max_depth` and cross-validation on the training data.

## Confusion Matrix

Final Decision Tree results:

```text
                 Predicted
                 0       1

Actual 0       1502     91
Actual 1        198    209
```

This means the model correctly identified:

* 1,502 customers who stayed
* 209 customers who churned

It incorrectly classified:

* 91 customers as churners when they stayed
* 198 customers as non-churners when they actually churned

## Streamlit Application

A Streamlit web application was created to allow users to enter customer information and receive a churn prediction.

The application accepts:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card status
* Active Member status
* Estimated Salary

The application then returns:

```text
Prediction
Churn probability
```

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── Churn_Modelling.csv
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Key Learning Outcomes

Through this project, I practiced:

* Exploratory Data Analysis
* Classification problems
* Handling numerical and categorical features
* Feature scaling
* One-hot encoding
* Scikit-learn pipelines
* Decision Tree classification
* Hyperparameter tuning
* Cross-validation
* Classification metrics
* Confusion matrix analysis
* ROC-AUC evaluation
* Model serialization with Joblib
* Building an ML web application with Streamlit

## Future Improvements

Possible improvements include:

* Handling class imbalance using techniques such as class weights or resampling
* Hyperparameter optimization
* Threshold tuning based on business requirements
* Feature importance analysis
* Model explainability
* Improved Streamlit UI
* Cloud deployment

