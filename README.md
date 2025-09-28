# Algerian Forest Fire Prediction using Regression
*A complete data science project in Python to predict the Fire Weather Index (FWI) using Scikit-learn and Statsmodels, with a focus on rigorous data cleaning and feature selection.*


## Project Goal

The objective of this project was to develop a regression model capable of predicting the Fire Weather Index (FWI) using meteorological data from two regions in Algeria. This notebook walks through the entire process, from initial data wrangling and cleaning to model training and evaluation.


## Dataset Insights

### Data Source: Algerian Forest Fires Dataset from the UCI Machine learning Repository.

### Content: The dataset comprises 244 records of daily weather observations from the Bejaia and Sidi Bel-Abbes regions during the 2012 fire season (June-September).

### Challenge: The raw data was particularly challenging, arriving as a single file with embedded headers, inconsistent data types, missing values, and formatting issues like extra whitespace in column names and categorical values.


## My Approach: A Step-by-Step Breakdown

This project followed a structured machine learning workflow to ensure robust and reliable results.

### 1. Data Cleaning & Preprocessing

This was the most intensive part of the project. Key steps included:

Structuring the Data: I first parsed the single file and programmatically split it into two distinct datasets, one for each Algerian region.

Tidying Up: I corrected column names by stripping extra whitespace and dropped irrelevant header rows that were mixed in with the data.

Handling Missing & Inconsistent Data: After identifying nulls, I used .dropna() for removal. I also cleaned up the categorical Classes feature (e.g., "fire   " vs. "fire") and encoded it into a binary (1/0) format.

Correcting Data Types: I converted all feature columns from their initial object type to the correct numerical int or float types to prepare for analysis.

### 2. Exploratory Data Analysis (EDA)

With a clean dataset, I performed EDA to uncover patterns:

Feature Distributions: I plotted histograms for each variable to understand its distribution and identify any skewness or outliers.

Correlation Analysis: A correlation heatmap was crucial in revealing the relationships between the features. It clearly showed strong multicollinearity between several of the fire indices.

### 3. Feature Selection & Modeling

Statistical Feature Selection (with OLS): Before finalizing the model, I used the statsmodels.OLS method to perform a regression on the training data. This provided a detailed statistical summary, including the p-value for each feature. Based on this analysis, I removed the first four features (Temperature, RH, Ws, Rain) as their p-values were greater than the 0.05 significance threshold, indicating they were not statistically significant predictors of FWI.

#### Managing Multicollinearity: Based on the heatmap, I also removed features with a correlation greater than 0.85 (specifically DMC and DC) to create a more stable model.

#### Data Splitting: The data was split into a 75/25 train-test ratio.

#### Scaling: I applied StandardScaler to the training and testing sets to standardize the features, a necessary step for the regularization models.

#### Model Comparison: I trained and evaluated three different regression models on the selected features:

Linear Regression (as a baseline)

Ridge Regression (to manage any remaining multicollinearity)

Lasso Regression (for its feature selection capabilities)


## Results & Key Findings

| Model | Test MSE | Test R² Score |
| :--- | :--- | :--- |
| **Linear Regression** | 0.67 | ~0.985 |
| **Ridge Regression** | 0.68 | ~0.985 |
| **Lasso Regression** | 2.25 | ~0.949 |

## Conclusion:

Both the standard Linear Regression and Ridge Regression models were outstanding, explaining over 98% of the variance in the FWI with very low error. The Lasso model, while still strong, had a slightly lower R² score. This suggests that after manually removing the highly correlated and statistically insignificant features, Lasso's automatic feature selection didn't offer an additional benefit.
