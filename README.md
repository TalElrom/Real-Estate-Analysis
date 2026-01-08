# Real-Estate-Analysis
Predicting and Understanding Property Over-Performance Using Machine Learning
## Project Overview

This project explores the real estate market through a dual analytical approach combining supervised learning and unsupervised learning techniques.

The primary objective is to classify whether a residential property is sold above or below its appraised value, and to uncover hidden market structures that explain this behavior.
Rather than predicting an exact sale price, the project focuses on identifying over-performing transactions, which is often more actionable from a business and investment perspective.

## Problem Definition

Given historical real estate transaction data, we aim to answer two key questions:

Supervised task:
Can we predict whether a property will sell above its appraised value?

Unsupervised task:
Can we identify natural, data-driven market segments that characterize different types of properties and transaction behaviors?

## Dataset

Source: Real estate sales transaction data

Initial size: 7,410 transactions × 23 features

After cleaning and preprocessing: 6,894 valid transactions

The dataset includes:

Sale price and appraised value

Property type

Location (neighborhood identifiers)

Transaction timing (year, quarter)

Ownership-related attributes

The target variable is defined as:

SaleAboveAppraisedValue = (SalePrice > TotalAppraisedValue)

## Data Preprocessing & Feature Engineering

The preprocessing pipeline included:

Handling missing values and removing invalid records

Encoding categorical variables using One-Hot Encoding

Creating time-based features (sale year, quarter)

Engineering ownership-related signals (e.g., owner turnover rate)

Feature selection using Recursive Feature Elimination (RFE) with XGBoost

The final feature set was reduced to 14 informative features.

## Supervised Learning Models

The task was framed as a binary classification problem.
We evaluated multiple models to capture both linear and non-linear relationships between features and the target variable:

Logistic Regression (baseline, interpretable)

Decision Tree

Support Vector Machine (SVM)

Random Forest

XGBoost (Extreme Gradient Boosting)

### Evaluation Strategy

Primary metric: Recall
(Missing an over-performing property is more costly than a false positive.)

Additional metrics:

Accuracy

Precision

F1-score

ROC-AUC

Model performance was analyzed using:

Confusion matrices

ROC curves

Precision–Recall curves

### Final Model

XGBoost emerged as the best-performing model.

Hyperparameter tuning was performed using RandomizedSearchCV, optimizing for Recall.

Final Recall achieved: 0.614

## Model Interpretation & Insights

Feature importance analysis revealed that over-performance is driven primarily by:

**Market timing** (sale year and quarter)

**Location** (neighborhood identifiers)

**Ownership patterns** (e.g., turnover rate)

These findings align with trends observed during the exploratory data analysis (EDA).

## Unsupervised Learning: Market Segmentation

To further understand market structure, we applied clustering techniques:

**K-Means Clustering**

**DBSCAN**

**Key Findings**

K-Means revealed a clear separation into two primary market archetypes:

Compact, high-density properties (apartments and condominiums)

Spacious, private homes

Additional density-based clustering identified six finer-grained sub-segments

Cluster quality was evaluated using the Silhouette Score and visual inspection

This unsupervised analysis complements the classification task by providing explanatory and structural insights into the market.

## Requirements

All required Python libraries are listed in requirements.txt.

Main dependencies include:

pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

## Summary

This project delivers:

A predictive classification model for identifying over-performing property sales

A clear evaluation strategy aligned with business objectives

Interpretable insights into the factors driving market behavior

A data-driven segmentation of the real estate market

Together, these results demonstrate how machine learning can be used not only for prediction, but also for understanding complex market dynamics.
