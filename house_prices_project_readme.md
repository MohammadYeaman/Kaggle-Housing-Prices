# House Prices Prediction - Advanced Regression Techniques

## Project Overview

This project is a machine learning pipeline designed to predict continuous housing prices for the Ames Housing dataset. It uses Python, `pandas`, and `xgboost` to clean the data, handle missing values, and generate highly accurate predictions. 

**Final Kaggle Score (RMSE): 14132.38565**
**Rank:** Top 300 (~290)

## Data Processing & Feature Engineering

To achieve a high-scoring prediction, this project goes beyond basic data processing and implements several advanced techniques:

* **Outlier Removal:** Drops extreme anomalies from the training data (e.g., houses with over 4,000 sq. ft. of living area selling for unusually low prices) to prevent the model from learning incorrect patterns.
* **Neighborhood-Based Imputation:** Instead of filling missing `LotFrontage` values with a global average or zero, this model intelligently groups the data by `Neighborhood` and fills missing values with the localized median.
* **One-Hot Encoding Alignment:** Utilizes `pd.get_dummies` for categorical variables and ensures the test dataset perfectly mirrors the training dataset columns using `.reindex()` to prevent feature mismatches.

## Model Architecture

The predictions are powered by an **XGBoost Regressor**, optimized for structured tabular data.

* **Algorithm:** `XGBRegressor`
* **Estimators (`n_estimators`):** 500 decision trees
* **Learning Rate:** 0.05 (to ensure steady, precise convergence without overfitting)
* **Max Depth:** 5 (limits tree complexity to capture patterns without memorizing noise)

## Files in this Repository

* `model.py`: The core Python script that handles data loading, preprocessing, model training, and prediction generation.
* `train.csv`: The training dataset containing housing features and the actual sale prices.
* `test.csv`: The evaluation dataset containing housing features without the sale prices.
* `Submission.csv`: The final generated output file formatted for Kaggle scoring.

## How to Run the Code

1. **Install Dependencies:**
   Ensure you have Python installed along with the required libraries.
   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```

2. **Execute the Script:**
   Run the model script from your terminal. Ensure that `train.csv` and `test.csv` are placed in the same directory.
   ```bash
   python model.py
   ```

3. **Results:**
   The script will train the XGBoost model and automatically output a `Submission.csv` file mapping each house's `Id` to its predicted `SalePrice`.