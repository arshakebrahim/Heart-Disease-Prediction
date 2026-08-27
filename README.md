Heart Disease Prediction & Deployment Pipeline
An end-to-end clinical machine learning pipeline analyzing 50,000 patient records to assess cardiovascular disease risk, benchmark multiple classification algorithms, and serve real-time predictions via an interactive Gradio web application.

Project Overview

This repository implements a modular machine learning pipeline designed to perform clinical decision support. The system handles raw patient telemetry, applies standardized missing-value imputation and feature scaling, benchmarks three ML models, and exposes an intuitive UI for real-time risk classification.

Key Data Architecture & Preprocessing

Cohort Size: 50,000 patient records featuring 20 clinical and lifestyle predictors (14 numerical, 6 categorical).

Missing Data Imputation: Handled 20,109 missing records in Alcohol_Intake by categorizing them as 'None'.

Preprocessing Pipeline: Implemented ColumnTransformer with StandardScaler for numerical measures (Age, BP, Cholesterol, BMI, Heart Rate) and OneHotEncoder for categorical inputs.

Top Predictors: Random Forest Gini importance shows that the top 5 features account for >92% of total predictive power (Age: 25.1%, Total Cholesterol: 24.5%, Hypertension: 20.6%, Diabetes: 14.3%, Previous Heart Attack: 8.0%).

Model Performance Benchmark

All models were evaluated using an 80/20 stratified train-test split:

Model Algorithm	Accuracy	Precision	Recall	F1-Score	ROC-AUC	Status
Logistic Regression	92.38%	91.78%	91.78%	0.9178	0.9822	Baseline
Random Forest Classifier	100.0%	100.0%	100.0%	1.0000	1.0000	Selected
HistGradientBoosting	100.0%	100.0%	100.0%	1.0000	1.0000	Top Performer
4-Stage Workflow Pipeline

Data Prep: Clean 50,000 rows and impute missing values.

Feature Engineering: Apply StandardScaler and OneHotEncoder transformations.

Model Evaluation: Perform stratified cross-validation and multi-metric comparison.

UI Deployment: Serve the inference engine using a Gradio web interface.

Tech Stack

Language: Python

ML Libraries: Scikit-learn, Pandas, NumPy

Deployment: Gradio

Data Source: Kaggle Clinical Dataset
