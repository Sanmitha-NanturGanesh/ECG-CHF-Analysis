# ECG-CHF-Analysis
ECG-based Congestive Heart Failure pattern analysis using HRV features and machine learning

## Project Overview
This project analyzes ECG signals to identify patterns associated with Congestive Heart Failure (CHF). It transforms raw biomedical signals into meaningful heart rate variability (HRV) features and applies exploratory analysis, clustering, anomaly detection, and machine learning to study abnormal cardiac patterns.

## Dataset
The project is based on the BIDMC Congestive Heart Failure Database from PhysioNet.

- 15 patients diagnosed with severe CHF
- Approximately 20-hour ECG recordings per patient
- 250 Hz sampling rate

## Objective
The goal of this project is to process raw ECG signals, extract HRV-based features, identify abnormal signal behavior, and build predictive models to understand CHF-related trends.

## Workflow
1. ECG signal preprocessing using bandpass filtering
2. R-peak detection and RR interval extraction
3. HRV feature engineering
4. Exploratory data analysis
5. K-means clustering
6. Isolation Forest anomaly detection
7. Classification modeling

## Key Features
- Mean RR
- Standard Deviation of RR Intervals
- RMSSD
- pNN50
- Variance of RR Intervals

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## Key Insights
- HRV features revealed clear differences in heart rhythm behavior
- Clustering separated distinct groups of variability patterns
- Anomaly detection highlighted unusual cardiac signals
- XGBoost achieved the best classification performance

## Tools and Technologies
Python, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Jupyter Notebook

## Business / Real-World Relevance
This project demonstrates how raw healthcare signal data can be converted into actionable insights that support early risk detection and more effective patient monitoring.
