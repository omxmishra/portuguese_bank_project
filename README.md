# Portuguese Bank Term Deposit Prediction App
## Overview

This project is an end-to-end machine learning application that predicts whether a bank customer is likely to subscribe to a term deposit.

The main goal was to go beyond a notebook-based model and build something usable — where preprocessing, prediction, and basic interpretation are handled in one consistent pipeline.

The application currently runs locally using Streamlit.

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
-Demographics (age, job, marital status, education)
-Campaign interaction details (contacts, previous outcomes)
-Macroeconomic indicators (Euribor rate, CPI, employment variation)

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

The focus was on building something functional and explainable, not just achieving a score.

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

## Screenshots

<img width="1910" height="946" alt="image" src="https://github.com/user-attachments/assets/f93df99f-771b-4b3e-9f2c-5251b7aa8cf1" />

<img width="1910" height="950" alt="image" src="https://github.com/user-attachments/assets/b3b5a1bc-bf7c-462c-a4c0-74c9f743501d" />

<img width="1917" height="895" alt="image" src="https://github.com/user-attachments/assets/02e06881-ac27-419b-b9b6-fe37bc7dc80c" />

<img width="1918" height="910" alt="image" src="https://github.com/user-attachments/assets/52d78287-a763-4210-aa2f-66b569361613" />

<img width="1913" height="960" alt="image" src="https://github.com/user-attachments/assets/f590bbda-2d16-4ce8-918d-36258b7411fb" />


