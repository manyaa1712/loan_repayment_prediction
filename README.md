# Loan Repayment Prediction Using Ensemble Learning Methods

## Objective
The goal of this project is to predict whether a bank should approve a loan for an applicant based on their financial profile, utilizing ensemble learning methods to enhance predictive accuracy.

## Project Overview
This project involves the implementation of several machine learning techniques to predict loan repayment. The dataset used contains various financial attributes of loan applicants. The project includes data preprocessing, exploratory data analysis, feature engineering, and modeling using multiple ensemble learning methods.

## Key Steps

### 1. Importing Libraries
Key libraries used include:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

### 2. Data Reading and Preprocessing
- Read data from `loan_data.csv`.
- Handle missing values and encode categorical variables.
- Perform exploratory data analysis to understand the dataset.

### 3. Data Visualization
- Visualize the distribution of `fico` scores.
- Plot the relationship between `fico` scores and other attributes.
- Use `seaborn` and `matplotlib` for visualizations.

### 4. Train-Test Split
- Split the dataset into training and testing sets using `train_test_split`.

### 5. Modeling
Several models were trained and evaluated, including:
- **Decision Tree:** Initial model with grid search for hyperparameter tuning.
- **Bagging:** Applied bagging to the decision tree model.
- **AdaBoost:** Implemented AdaBoost with decision tree and random forest classifiers.
- **Random Forest:** Trained a random forest classifier and evaluated its performance.
- **AdaBoost with Random Forest:** Combined AdaBoost with random forest for improved performance.

### 6. Evaluation
- Confusion matrix and classification reports were generated to evaluate model performance.
- Accuracy, precision, recall, and F1-score were calculated for different models.

## Results
- **Decision Tree Classifier:** Achieved an accuracy of 84.58%.
- **Bagging with Decision Tree:** Mean score of 73.10%.
- **AdaBoost with Decision Tree:** Same result as decision tree with 84%.
- **Random Forest Classifier:** Achieved an accuracy of 84.7%.
- **AdaBoost with Random Forest:** Achieved the highest accuracy among the models.

## Conclusion
The ensemble learning methods, particularly the combination of AdaBoost with Random Forest, provided significant improvements in prediction accuracy. The project demonstrates the effectiveness of ensemble techniques in improving predictive performance for loan repayment prediction tasks.

## References
- Dataset: `loan_data.csv`
