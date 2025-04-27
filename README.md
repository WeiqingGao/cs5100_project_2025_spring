# Overview
This project participates in the Kaggle competition Child Mind Institute - Problematic Internet Use, aiming to predict the 
level of problematic internet usage exhibited by children and adolescents based on their physical activity and fitness data.
Details and dataset used can be seen https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview.

Problematic internet use among children is an increasingly concerning issue, often associated with mental health problems 
like depression and anxiety. However, directly measuring problematic internet use is challenging and often inaccessible, 
while physical and fitness measurements are widely available and easy to collect. Thus, the goal of this project is to 
develop a predictive model that analyzes children's physical activity data to identify early signs of problematic internet
use, enabling earlier interventions and healthier digital habits. 


# Structure

## Data Preparation
### dynamic data (time-series data)
- Loaded .parquet format time series data.
- Extracted statistical features (mean, std, min, max, etc.) for downstream modeling.
- Standardized features and imputed missing values using feature means.
- Applied Principal Component Analysis (PCA) to reduce dimensionality:
  - Selected top 15 principal components capturing over 80% of the variance.
### static data
- Cleaned outliers and implausible measurements (e.g., unrealistic body fat percentages).
- Deleted irrelevant features and grouped others by age to normalize age-related biases.
- Standardized features to ensure consistency across different feature scales.
- Performed quantile-based binning to handle skewness and extreme values.
- Imputed missing values using:
  - Lasso-based regression models when feasible.
  - Mean imputation for cases with high missingness.
### merging
- Combined processed static features and dynamic PCA components based on the unique `id`.

## Modeling and Evaluation
### model training
- Trained multiple models including:
  - LightGBM
  - XGBoost
  - CatBoost
- Used Stratified K-Fold cross validation to evaluate models and obtain stable estimates.
- Optimized thresholds to map predictions into discrete categories (0-3) maximizing quadratic Cohen's Kappa.
- Applied Optuna to search optimal hyperparameters for boosting models.
- Users can choose to use predefined tuned hyperparameters if rerunning tuning was not desired.
### meta-model (ensembling)
- Built a second-layer Ridge regression meta-model on top of individual model outputs.
- Evaluated different meta-learner (Ridge, Lasso, Linear Regression, Random Forest, GBDT, MLP).
- Selected the best-performing meta-model based on cross-validated $R^2$ scores and their characteristics.

## Visualization and Interpretation
### model prediction analysis
- Scatter plots of OOF predictions vs. true labels.
- Ensemble prediction visualization.
- Residual distribution plots.
### feature importance
- Visualized built-in feature importance for LightGBM, XGBoost, and CatBoost.
- Used ELI5 permutation importance for more robust feature contribution comparisons.
### model correlation
- Analyzed correlations between different model predictions to evaluate diversity
### threshold analysis
- Heatmaps showing optimized thresholds derived from cross-validation.
