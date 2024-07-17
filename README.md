# Loan Repayment Prediction Using Ensemble Learning Methods

## Objective
The goal of this project is to predict whether a bank should approve a loan for an applicant based on their financial profile, utilizing ensemble learning methods to enhance predictive accuracy. Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this and thus connect people who need money (borrowers) with people who have money (investors).

## Project Overview
This project involves the implementation of several machine learning techniques to predict loan repayment. The dataset used contains various financial attributes of loan applicants. The project includes data preprocessing, exploratory data analysis, feature engineering, and modeling using multiple ensemble learning methods.

We will use old lending data and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from the csv already provided (It has been cleaned of NA values).

Here are what the columns represent:

- credit.policy: 1 if the customer meets the credit underwriting criteria and 0 otherwise.
- purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", "home_improvement" and "all_other"). 
- int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers assessed as more risky are assigned higher interest rates.
- installment: The monthly installments owed by the borrower if the loan is funded.
- log.annual.inc: The natural log of the self-reported annual income of the borrower.
- dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
- fico: The FICO credit score of the borrower.
- days.with.cr.line: The number of days the borrower has had a credit line.
- revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
- revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
- inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
- delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
- not.fully.paid: It is the target variable we are trying to predict whether the borrower has fully paid off the loan or not. 1 means the borrower has not fully paid off the loan otherwise 0.
 
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
- **Random Forest Classifier:** Achieved an accuracy of 84.75%.
- **AdaBoost with Random Forest:** Achieved an accuracy of 84.69%.
- While computing different ensemble learning technologies, we found that most of the bagging and boosting algorithms provided similar results with minimal differences in accuracy. However, among all these ensemble methods, the best model for this dataset was the Random Forest, achieving an accuracy of almost 85%.

## Conclusion
The ensemble learning methods, particularly the Random Forest Classifier, provided significant improvements in prediction accuracy. The project demonstrates the effectiveness of ensemble techniques in improving predictive performance for loan repayment prediction tasks.

## References
- Dataset: `loan_data.csv`
