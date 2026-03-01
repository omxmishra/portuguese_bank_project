# Portuguese Bank Term Deposit Prediction App
## Overview

- This project is an end-to-end machine learning application that predicts whether a bank customer is likely to subscribe to a term deposit.
- The main goal was to go beyond a notebook-based model and build something usable — where preprocessing, prediction, and basic interpretation are handled in one consistent pipeline.
- The application currently runs locally using Streamlit.

## Why This Project?

In the dataset used, only about 11% of customers actually subscribe to a term deposit.
That means blindly contacting everyone is inefficient.
This project focuses on:

- Estimating subscription probability
- Identifying above-average leads
- Supporting better targeting decisions

Instead of treating it as a simple classification problem, probabilities are interpreted relative to the dataset baseline.

## Dataset
- Portuguese Bank Marketing dataset
- Target variable: subscription (yes/no)
- Positive class ratio: ~11%

The features include:
- Demographics (age, job, marital status, education)
- Campaign interaction details (contacts, previous outcomes)
- Macroeconomic indicators (Euribor rate, CPI, employment variation)

## Model Approach

Several models were evaluated, including:
1)Logistic Regression
2)KNN
3)SVM
4)Decision Tree
5)Random Forest
6)Gradient Boosting
7)XGBoost

Based on ROC-AUC comparison, Gradient Boosting was selected for deployment.
The final model is built using a Scikit-learn Pipeline that combines:
- ColumnTransformer
   OneHotEncoder (categorical features)
   StandardScaler (numerical features)
- GradientBoostingClassifier

This ensures consistent preprocessing during both training and inference.

## Model Performance

Evaluation Metric: ROC-AUC
Best ROC-AUC: ~0.81

Given the imbalanced nature of the dataset (~11% positive), a custom decision threshold is used instead of relying strictly on 0.5.

## Application Features

The Streamlit app includes:
- Interactive input form
- Subscription probability output
- Adjustable decision threshold
- Business recommendation (prioritize / lower priority)
- Top 10 feature importance visualization
 
The focus was on building something functional and explainable, not just achieving a score

## Project Structure

PORTUGUESE_BANK_PROJECT

├── app.py

├── bank_model.pkl

├── requirements.txt

├── Portuguese_Bank_Marketing.ipynb

├── data/

## Key Learnings

- Handling imbalanced classification problems
- Importance of maintaining preprocessing consistency
- Aligning model evaluation with deployment decisions
- Building an ML project beyond a notebook

# Screenshots

<img width="1917" height="931" alt="image" src="https://github.com/user-attachments/assets/e9f6c951-522c-418b-8e3c-7a664bf95d40" />

<img width="1916" height="952" alt="image" src="https://github.com/user-attachments/assets/c6be99f7-ef44-40e2-b0ce-c0914c66fd2e" />

<img width="1917" height="776" alt="image" src="https://github.com/user-attachments/assets/defa515c-cbbe-4732-8662-c89e23857c1e" />

<img width="1892" height="947" alt="image" src="https://github.com/user-attachments/assets/f9e5f6a7-d44c-43cc-9e23-87c49d6cb116" />

<img width="1872" height="772" alt="image" src="https://github.com/user-attachments/assets/61438a9e-dbe5-462f-8c4e-43e1ae292cec" />

