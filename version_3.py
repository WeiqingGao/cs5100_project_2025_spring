import os
import logging
import warnings
from typing import Tuple, List, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import cohen_kappa_score
from sklearn.base import clone
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global settings
SEED = 8234
N_SPLITS = 10
OPTIMIZE_PARAMS = False
N_TRIALS = 25  # Number of iterations for Optuna
VOTING = True
BASE_THRESHOLDS = [30, 50, 80]


# Helper Functions & Classes
def apply_pca(
    train: pd.DataFrame, 
    test: pd.DataFrame,
    n_components: int = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Apply PCA to reduce dimensionality of features.

    Parameters:
        train (pd.DataFrame): Training data (standardized).
        test (pd.DataFrame): Test data (standardized).
        n_components (int): Number of components to retain. If None, keep all.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, PCA]: PCA-transformed training data,
        test data, and the PCA model.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)

    logging.info("Explained variance ratio: %s", 
                 pca.explained_variance_ratio_)
    logging.info("Total explained variance: %f", 
                 np.sum(pca.explained_variance_ratio_))

    train_pca_df = pd.DataFrame(
        train_pca,
        columns=[f"PC_{i+1}" for i in range(train_pca.shape[1])]
    )
    test_pca_df = pd.DataFrame(
        test_pca,
        columns=[f"PC_{i+1}" for i in range(test_pca.shape[1])]
    )
    return train_pca_df, test_pca_df, pca


def extract_time_features(df: pd.DataFrame) -> List[float]:
    """
    Extract statistical features from a time series dataframe.

    Parameters:
        df (pd.DataFrame): Raw time series data.

    Returns:
        List[float]: Extracted feature vector.
    """
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum()
    ]

    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    full_masks = np.ones(len(df), dtype=bool)

    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [full_masks, night, day]

    def extract_stats(data: pd.Series) -> List[float]:
        return [
            data.mean(),
            data.std(),
            data.max(),
            data.min(),
            data.diff().mean(),
            data.diff().std()
        ]

    for key in keys:
        for mask in masks:
            filtered = df.loc[mask, key]
            features.extend(extract_stats(filtered))

    return features


def process_file(filename: str, dirname: str) -> Tuple[List[float], str]:
    """
    Process a single parquet file to extract time series features.

    Parameters:
        file_path (str): Name of the subdirectory containing the parquet file.
        dirname (str): Directory where the parquet files are stored.

    Returns:
        Tuple[List[float], str]: Extracted feature vector and sample id.
    """
    filepath = os.path.join(dirname, filename, "part-0.parquet")
    df = pd.read_parquet(filepath)
    if "step" in df.columns:
        df.drop("step", axis=1, inplace=True)
    sample_id = filename.split("=")[1]
    return extract_time_features(df), sample_id


def collect_time_features(dirname: str) -> pd.DataFrame:
    """
    Load and process time series data from a directory containing parquet
    files. Filters out hidden files and non-directory files.

    Parameters:
        dirname (str): Directory containing subdirectories of parquet files.

    Returns:
        pd.DataFrame: DataFrame of extracted features with an 'id' column.
    """
    ids = [
        d for d in os.listdir(dirname)
        if not d.startswith(".") and os.path.isdir(os.path.join(dirname, d))
    ]
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda f: process_file(f, dirname), ids
                ),
                total=len(ids)
            )
        )
    features, labels = zip(*results)
    df = pd.DataFrame(
        features, columns=[f"stat_{i}" for i in range(len(features[0]))]
    )
    df["id"] = labels
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features by clipping or replacing implausible/extreme values.

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    target_cols = ["FGC-FGC_GSND", "FGC-FGC_GSD"]
    for col in target_cols:
        assert col in df.columns, (
            f"Column {col} not found in dataframe during cleaning."
        )
    df[target_cols] = df[target_cols].clip(lower=9, upper=60)

    df["BIA-BIA_Fat"] = np.where(
        df["BIA-BIA_Fat"] < 5, np.nan, df["BIA-BIA_Fat"]
    )
    df["BIA-BIA_Fat"] = np.where(
        df["BIA-BIA_Fat"] > 60, np.nan, df["BIA-BIA_Fat"]
    )
    df["BIA-BIA_BMR"] = np.where(
        df["BIA-BIA_BMR"] > 4000, np.nan, df["BIA-BIA_BMR"]
    )
    df["BIA-BIA_DEE"] = np.where(
        df["BIA-BIA_DEE"] > 8000, np.nan, df["BIA-BIA_DEE"]
    )
    df["BIA-BIA_BMC"] = np.where(
        df["BIA-BIA_BMC"] <= 0, np.nan, df["BIA-BIA_BMC"]
    )
    df["BIA-BIA_BMC"] = np.where(
        df["BIA-BIA_BMC"] > 10, np.nan, df["BIA-BIA_BMC"]
    )
    df["BIA-BIA_FFM"] = np.where(
        df["BIA-BIA_FFM"] <= 0, np.nan, df["BIA-BIA_FFM"]
    )
    df["BIA-BIA_FFM"] = np.where(
        df["BIA-BIA_FFM"] > 300, np.nan, df["BIA-BIA_FFM"]
    )
    df["BIA-BIA_FMI"] = np.where(
        df["BIA-BIA_FMI"] < 0, np.nan, df["BIA-BIA_FMI"]
    )
    df["BIA-BIA_ECW"] = np.where(
        df["BIA-BIA_ECW"] > 100, np.nan, df["BIA-BIA_ECW"]
    )
    df["BIA-BIA_LDM"] = np.where(
        df["BIA-BIA_LDM"] > 100, np.nan, df["BIA-BIA_LDM"]
    )
    df["BIA-BIA_LST"] = np.where(
        df["BIA-BIA_LST"] > 300, np.nan, df["BIA-BIA_LST"]
    )
    df["BIA-BIA_SMM"] = np.where(
        df["BIA-BIA_SMM"] > 300, np.nan, df["BIA-BIA_SMM"]
    )
    df["BIA-BIA_TBW"] = np.where(
        df["BIA-BIA_TBW"] > 300, np.nan, df["BIA-BIA_TBW"]
    )

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering to transform raw static data into
    standardized and interpretable features.

    Parameters:
        df (pd.DataFrame): Input static data.

    Returns:
        pd.DataFrame: DataFrame with newly engineered features.
    """
    # Drop Season-related columns.
    season_cols = [col for col in df.columns if "Season" in col]
    df = df.drop(season_cols, axis=1)

    def get_age_group(age: float) -> Any:
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan

    df["group"] = df["Basic_Demos-Age"].apply(get_age_group)

    BMI_map = {
        0: 16.3, 1: 15.9, 2: 16.1, 3: 16.8, 4: 17.3,
        5: 19.2, 6: 20.2, 7: 22.3, 8: 23.6
    }
    df["BMI_mean_norm"] = (
        df[["Physical-BMI", "BIA-BIA_BMI"]].mean(axis=1) /
        df["group"].map(BMI_map)
    )

    zones = [
        "FGC-FGC_CU_Zone", "FGC-FGC_GSND_Zone",
        "FGC-FGC_GSD_Zone", "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL_Zone", "FGC-FGC_SRR_Zone",
        "FGC-FGC_TL_Zone"
    ]
    df["FGC_Zones_mean"] = df[zones].mean(axis=1)
    df["FGC_Zones_min"] = df[zones].min(axis=1)
    df["FGC_Zones_max"] = df[zones].max(axis=1)

    GSD_max_map = {
        0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2,
        5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4
    }
    GSD_min_map = {
        0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4,
        5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1
    }
    df["GS_max"] = (
        df[["FGC-FGC_GSND", "FGC-FGC_GSD"]].max(axis=1) /
        df["group"].map(GSD_max_map)
    )
    df["GS_min"] = (
        df[["FGC-FGC_GSND", "FGC-FGC_GSD"]].min(axis=1) /
        df["group"].map(GSD_min_map)
    )

    cu_map = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0,
              4: 10.0, 5: 14.0, 6: 20.0, 7: 20.0, 8: 20.0}
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0,
              4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {0: 8.0, 1: 8.0, 2: 8.0, 3: 9.0,
              4: 9.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0}

    df["CU_norm"] = df["FGC-FGC_CU"] / df["group"].map(cu_map)
    df["PU_norm"] = df["FGC-FGC_PU"] / df["group"].map(pu_map)
    df["TL_norm"] = df["FGC-FGC_TL"] / df["group"].map(tl_map)

    df["SR_min"] = df[["FGC-FGC_SRL", "FGC-FGC_SRR"]].min(axis=1)
    df["SR_max"] = df[["FGC-FGC_SRL", "FGC-FGC_SRR"]].max(axis=1)

    bmr_map = {
        0: 934.0, 1: 941.0, 2: 999.0, 3: 1048.0,
        4: 1283.0, 5: 1255.0, 6: 1481.0, 7: 1519.0, 8: 1650.0
    }
    dee_map = {
        0: 1471.0, 1: 1508.0, 2: 1640.0, 3: 1735.0,
        4: 2132.0, 5: 2121.0, 6: 2528.0, 7: 2566.0, 8: 2793.0
    }
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map)
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]

    ffm_map = {
        0: 42.0, 1: 43.0, 2: 49.0, 3: 54.0,
        4: 60.0, 5: 76.0, 6: 94.0, 7: 104.0, 8: 111.0
    }
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df["group"].map(ffm_map)

    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]

    drop_features = [
        "FGC-FGC_GSND", "FGC-FGC_GSD", "FGC-FGC_CU_Zone",
        "FGC-FGC_GSND_Zone", "FGC-FGC_GSD_Zone", "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL_Zone", "FGC-FGC_SRR_Zone", "FGC-FGC_TL_Zone",
        "Physical-BMI", "BIA-BIA_BMI", "FGC-FGC_CU", "FGC-FGC_PU",
        "FGC-FGC_TL", "FGC-FGC_SRL", "FGC-FGC_SRR", "BIA-BIA_BMR",
        "BIA-BIA_DEE", "BIA-BIA_Frame_num", "BIA-BIA_FFM"
    ]
    df = df.drop(drop_features, axis=1)
    return df


def binning(
        train: pd.DataFrame, 
        test: pd.DataFrame,
        columns: List[str], 
        n_bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin continuous variables into discrete categories using quantile-based
    binning.

    Parameters:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        columns (List[str]): List of column names to bin.
        n_bins (int): Number of bins.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Modified train and test data with
        binned columns.
    """
    combined = pd.concat([train, test], axis=0)
    bin_boundaries: Dict[str, Any] = {}

    for c in columns:
        edges = pd.qcut(
            combined[c],
            n_bins,
            retbins=True,
            labels=range(n_bins),
            duplicates="drop"
        )[1]
        bin_boundaries[c] = edges

    for c, edges in bin_boundaries.items():
        train[c] = pd.cut(
            train[c],
            bins=edges,
            labels=range(len(edges) - 1),
            include_lowest=True
        ).astype(float)
        test[c] = pd.cut(
            test[c],
            bins=edges,
            labels=range(len(edges) - 1),
            include_lowest=True
        ).astype(float)

    return train, test


class Impute_With_Model:
    """
    Custom imputer that uses a regression model to predict missing values.
    Falls back to mean imputation if there are not enough auxiliary features
    or samples.
    """
    def __init__(
        self,
        na_frac: float = 0.5,
        min_samples: int = 0
    ) -> None:
        self.model_dict: Dict[str, Any] = {}
        self.mean_dict: Dict[str, float] = {}
        self.features: List[str] = []
        self.na_frac = na_frac
        self.min_samples = min_samples

    def find_features(
        self,
        data: pd.DataFrame,
        feature: str,
        tmp_features: List[str]
    ) -> np.ndarray:
        missing_rows = data[feature].isna()
        na_fraction = data[missing_rows][tmp_features].isna().mean(axis=0)
        valid_features = np.array(tmp_features)[na_fraction <= self.na_frac]
        return valid_features

    def build_fillers(
        self,
        model: Any,
        data: pd.DataFrame,
        features: List[str]
    ) -> None:
        self.features = features
        for feature in features:
            self.mean_dict[feature] = np.mean(data[feature])

        for feature in tqdm(features, desc="Fitting imputation models"):
            if data[feature].isna().sum() > 0:
                model_clone = clone(model)
                X = data[data[feature].notna()].copy()
                tmp_features = [f for f in features if f != feature]
                valid_features = self.find_features(
                    data, feature, tmp_features
                )

                if len(valid_features) >= 1 and X.shape[0] > self.min_samples:
                    for f in valid_features:
                        X[f] = X[f].fillna(self.mean_dict[f])
                    model_clone.fit(X[valid_features], X[feature])
                    self.model_dict[feature] = (
                        model_clone,
                        valid_features.copy()
                    )
                else:
                    self.model_dict[feature] = (
                        "mean",
                        np.mean(data[feature])
                    )

    def apply_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        imputed_data = data.copy()
        for feature, model in self.model_dict.items():
            missing_rows = imputed_data[feature].isna()
            if missing_rows.any():
                if model[0] == "mean":
                    imputed_data[feature].fillna(model[1], inplace=True)
                else:
                    tmp_features = [
                        f for f in self.features if f != feature
                    ]
                    X_missing = data.loc[
                        missing_rows, tmp_features
                    ].copy()
                    for f in tmp_features:
                        X_missing[f] = X_missing[f].fillna(
                            self.mean_dict[f]
                        )
                    imputed_data.loc[
                        missing_rows, feature
                    ] = model[0].predict(X_missing[model[1]])
        return imputed_data


def categorize_predictions(
    raw_preds: np.ndarray,
    thresholds: List[float]
) -> np.ndarray:
    """
    Map continuous predictions to 4 categories (0, 1, 2, 3) based on given
    thresholds.

    Parameters:
        raw_preds (np.ndarray): Continuous model predictions.
        thresholds (List[float]): Thresholds to divide the predictions.

    Returns:
        np.ndarray: Categorized predictions.
    """
    return np.where(
        raw_preds < thresholds[0],
        0,
        np.where(
            raw_preds < thresholds[1],
            1,
            np.where(raw_preds < thresholds[2], 2, 3)
        )
    )


def find_optimal_thresholds(
    y_true: np.ndarray,
    raw_preds: np.ndarray,
    start_vals: List[float] = [0.5, 1.5, 2.5]
) -> np.ndarray:
    """
    Optimize thresholds for mapping predictions by maximizing the Cohen
    Kappa score.

    Parameters:
        y_true (np.ndarray): True target values.
        raw_preds (np.ndarray): Continuous predictions.
        start_vals (List[float]): Initial threshold values.

    Returns:
        np.ndarray: Optimized thresholds.
    """
    def kappa_loss(thresholds, y_true, raw_preds):
        preds = categorize_predictions(raw_preds, thresholds)
        return -cohen_kappa_score(
            y_true, preds, weights='quadratic'
        )

    res = minimize(
        kappa_loss,
        x0=start_vals, args=(y_true, raw_preds),
        method='Powell'
    )
    assert res.success, "Threshold optimization failed."
    return res.x


def compute_sample_weights(series: pd.Series) -> pd.Series:
    """
    Calculate sample weights by binning the target variable into 10 bins.
    Less populated bins get higher weights to mitigate imbalance.

    Parameters:
        series (pd.Series): Target variable.

    Returns:
        pd.Series: Normalized sample weights.
    """
    bins = pd.cut(series, bins=10, labels=False)
    weights_df = bins.value_counts().reset_index()
    weights_df.columns = ['target_bins', 'count']
    weights_df['count'] = 1 / weights_df['count']
    weight_map = weights_df.set_index('target_bins')['count'].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()


# Data Processing Functions
def load_static_data(
    train_path: str,
    test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load static (CSV) datasets.

    Parameters:
        train_path (str): File path for training CSV.
        test_path (str): File path for test CSV.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded training and test data.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info(
        f"Static train shape: {train.shape}, test shape: {test.shape}"
    )
    return train, test


def process_static_data(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process static data by performing cleaning, feature engineering,
    binning, and missing value imputation. The processing order is:
    first cleaning, then feature engineering.

    Parameters:
        train (pd.DataFrame): Raw training data.
        test (pd.DataFrame): Raw test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed training and test data.
    """
    # Data Cleaning
    logging.info("Cleaning static data...")
    train = clean_features(train)
    test = clean_features(test)

    # Feature Engineering
    logging.info("Performing feature engineering on static data...")
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Drop missing target rows
    if "sii" in train.columns:
        train = train[train["sii"].notna()]
        logging.info(
            f"After removing missing target, train shape: {train.shape}"
        )

    # Binning
    bin_cols = [
        "PAQ_A-PAQ_A_Total", "BMR_norm", "DEE_norm", "GS_min", "GS_max",
        "BIA-BIA_FFMI", "BIA-BIA_BMC", "Physical-HeartRate", "BIA-BIA_ICW",
        "Fitness_Endurance-Time_Sec", "BIA-BIA_LDM", "BIA-BIA_SMM",
        "BIA-BIA_TBW", "DEE_BMR", "ICW_ECW"
    ]
    logging.info("Binning selected features...")
    train, test = binning(train, test, bin_cols, n_bins=10)

    # Save the target, discrete target, and id columns
    target = train["PCIAT-PCIAT_Total"].copy() \
        if "PCIAT-PCIAT_Total" in train.columns else None
    discrete_target = train["sii"].copy() \
        if "sii" in train.columns else None
    train_id = train["id"].copy() if "id" in train.columns else None
    test_id = test["id"].copy() if "id" in test.columns else None

    # Exclude features not in test
    exclude = [
        'PCIAT-Season', 'PCIAT-PCIAT_01', 'PCIAT-PCIAT_02', 'PCIAT-PCIAT_03',
        'PCIAT-PCIAT_04', 'PCIAT-PCIAT_05', 'PCIAT-PCIAT_06', 'PCIAT-PCIAT_07',
        'PCIAT-PCIAT_08', 'PCIAT-PCIAT_09', 'PCIAT-PCIAT_10', 'PCIAT-PCIAT_11',
        'PCIAT-PCIAT_12', 'PCIAT-PCIAT_13', 'PCIAT-PCIAT_14', 'PCIAT-PCIAT_15',
        'PCIAT-PCIAT_16', 'PCIAT-PCIAT_17', 'PCIAT-PCIAT_18', 'PCIAT-PCIAT_19',
        'PCIAT-PCIAT_20', 'PCIAT-PCIAT_Total', 'sii', 'id'
    ]
    features = [f for f in train.columns if f not in exclude]
    # Test set has no target, exclude it when selecting features
    train_features = train[features].copy()
    test_features = test[features].copy()

    # Feature set for imputation, excluding id and discrete target
    features_impute = [f for f in features if f not in ("id", "sii")]

    # Missing value imputation
    logging.info("Performing missing value imputation on static data...")
    model = LassoCV(cv=5, random_state=SEED)
    imputer = Impute_With_Model(na_frac=0.4)
    imputer.build_fillers(model, train_features, features_impute)
    train_imputed = imputer.apply_imputation(train_features)
    test_imputed = imputer.apply_imputation(test_features)

    # Add back target, discrete target, and id to train; id to test
    if target is not None:
        train_imputed["PCIAT-PCIAT_Total"] = target
    if discrete_target is not None:
        train_imputed["sii"] = discrete_target
    if train_id is not None:
        train_imputed["id"] = train_id
    if test_id is not None:
        test_imputed["id"] = test_id

    return train_imputed, test_imputed

def process_dynamic_data(
    train_dir: str,
    test_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process dynamic (time series) data from parquet files:
      - Load data and extract statistical features.
      - Standardize features.
      - Fill missing values.
      - Apply PCA for dimensionality reduction.

    Parameters:
        train_dir (str): Directory path for training time series data.
        test_dir (str): Directory path for test time series data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed dynamic training and
        test data (with PCA features and 'id').
    """
    logging.info("Loading dynamic time series data...")
    train_time_series = collect_time_features(train_dir)
    test_time_series = collect_time_features(test_dir)

    # Remove 'id' column for feature processing.
    train_dynamic_features = train_time_series.drop('id', axis=1)
    test_dynamic_features = test_time_series.drop('id', axis=1)

    # Standardize features.
    scaler = StandardScaler()
    train_dynamic_features = pd.DataFrame(
        scaler.fit_transform(train_dynamic_features), columns=train_dynamic_features.columns
    )
    test_dynamic_features = pd.DataFrame(
        scaler.transform(test_dynamic_features), columns=test_dynamic_features.columns
    )

    # Fill missing values using training set means.
    for c in train_dynamic_features.columns:
        m = np.mean(train_dynamic_features[c])
        train_dynamic_features[c].fillna(m, inplace=True)
        test_dynamic_features[c].fillna(m, inplace=True)

    # Apply PCA.
    logging.info("Applying PCA to dynamic features...")
    train_dynamic_pca, test_dynamic_pca, pca = apply_pca(
        train_dynamic_features, test_dynamic_features, n_components=15, random_state=SEED
    )
    train_dynamic_pca['id'] = train_time_series['id']
    test_dynamic_pca['id'] = test_time_series['id']

    return train_dynamic_pca, test_dynamic_pca


def merge_static_dynamic_features(
    static_train: pd.DataFrame,
    static_test: pd.DataFrame,
    dynamic_train: pd.DataFrame,
    dynamic_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge processed static data with dynamic (PCA-transformed) data
    based on the 'id' field.

    Parameters:
        static_train (pd.DataFrame): Processed static training data.
        static_test (pd.DataFrame): Processed static test data.
        dynamic_train (pd.DataFrame): Processed dynamic training data.
        dynamic_test (pd.DataFrame): Processed dynamic test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Final merged training and
        test datasets.
    """
    logging.info("Merging static and dynamic data...")
    final_train = pd.merge(static_train, dynamic_train, how="left", on='id')
    final_test = pd.merge(static_test, dynamic_test, how="left", on='id')
    logging.info(
        f"Final train shape: {final_train.shape}, "
        f"final test shape: {final_test.shape}"
    )
    return final_train, final_test


# Modeling Functions
def cross_val_kappa(
    model_, 
    data, 
    features, 
    score_col, 
    index_col, 
    cv,
    sample_weights=False, 
    verbose=False
):
    """
    Perform cross-validation and compute Cohen's Kappa.
    """
    kappa_scores = []
    oof_preds = np.zeros(len(data))
    thresholds_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(
            data, data[index_col])):
        train_x = data[features].iloc[train_idx]
        val_x = data[features].iloc[val_idx]
        train_score_y = data[score_col].iloc[train_idx]
        train_idx_y = data[index_col].iloc[train_idx]
        val_score_y = data[score_col].iloc[val_idx]
        val_idx_y = data[index_col].iloc[val_idx]

        if sample_weights:
            weights = compute_sample_weights(train_score_y)
            model_.fit(train_x, train_score_y, sample_weight=weights)
        else:
            model_.fit(train_x, train_score_y)

        predict_train_y = model_.predict(train_x)
        predict_val_y = model_.predict(val_x)

        oof_preds[val_idx] = predict_val_y

        threshold_opt = find_optimal_thresholds(
            train_idx_y.to_numpy(), predict_train_y,
            start_vals=BASE_THRESHOLDS
        )
        thresholds_list.append(threshold_opt)
        predict_val_cat_y = categorize_predictions(predict_val_y, threshold_opt)
        kappa = cohen_kappa_score(val_idx_y, predict_val_cat_y,
                                  weights="quadratic")
        kappa_scores.append(kappa)

        if verbose:
            print(f"Fold {fold_idx}: Kappa = {kappa}")

    if verbose:
        print("Mean CV Kappa: %f", np.mean(kappa_scores))
        print("Std CV: %f", np.std(kappa_scores))
    return np.mean(kappa_scores), oof_preds, thresholds_list


def multi_cv_kappa(model_, data, features, score_col, index_col, cv,
                     seeds, sample_weights=False, verbose=False):
    """
    Perform cross-validation multiple times with different seeds.
    """
    scores = []
    oof_preds = np.zeros(len(data))
    for s in seeds:
        cv.random_state = s
        score, oof, _ = cross_val_kappa(
            model_, data, features, score_col, index_col, cv,
            sample_weights=sample_weights, verbose=False
        )
        scores.append(score)
        oof_preds += oof
    oof_preds /= len(seeds)
    return np.mean(scores), oof_preds


def optuna_objective(
    trial, 
    model_name, 
    X, 
    features, 
    score_col, 
    index_col, 
    cv,
    sample_weights=False
):
    """
    Optuna objective function for hyperparameter tuning.
    """
    def sample_params():
        shared_params = {
            "random_state": SEED,
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 0.01, 0.05
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8)
        }
        if model_name == "xgboost":
            model_params = {
                **shared_params,
                "objective": trial.suggest_categorical(
                    "objective", ["reg:tweedie", "reg:pseudohubererror"]
                ),
                "num_parallel_tree": trial.suggest_int(
                    "num_parallel_tree", 2, 30
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 0.8
                ),
                "reg_alpha": trial.suggest_loguniform(
                    "reg_alpha", 1e-5, 1e-1
                ),
                "reg_lambda": trial.suggest_loguniform(
                    "reg_lambda", 1e-5, 1e-1
                )
            }
            if model_params["objective"] == "reg:tweedie":
                model_params["tweedie_variance_power"] = trial.suggest_float(
                    "tweedie_variance_power", 1, 2
                )
        elif model_name == "lightgbm":
            model_params = {
                **shared_params,
                "objective": trial.suggest_categorical(
                    "objective", ["poisson", "tweedie", "regression"]
                ),
                "verbosity": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 0.8
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", 20, 100
                )
            }
            if model_params["objective"] == "tweedie":
                model_params["tweedie_variance_power"] = trial.suggest_float(
                    "tweedie_variance_power", 1, 2
                )
        elif model_name == "catboost":
            model_params = {
                **shared_params,
                "loss_function": trial.suggest_categorical(
                    "objective", ["Tweedie:variance_power=1.5",
                                  "Poisson", "RMSE"]
                ),
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 2, 4),
                "l2_leaf_reg": trial.suggest_loguniform(
                    "l2_leaf_reg", 1e-3, 1e-1
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 1.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-3, 10.0
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", 20, 60
                )
            }
        else:
            raise ValueError(f"Unsupported model_type: {model_name}")
        return model_params
      
    def build_model(params: dict) -> Any:
        if model_name == "xgboost":
            return XGBRegressor(**params, use_label_encoder=False)
        elif model_name == "lightgbm":
            return LGBMRegressor(**params)
        elif model_name == "catboost":
            return CatBoostRegressor(**params, verbose=0)

    params = sample_params()
    model = build_model(params)
    seeds = [random.randint(1, 10000) for _ in range(20)]
    score, _ = multi_cv_kappa(
        model, X, features, score_col, index_col, cv,
        seeds, sample_weights=sample_weights, verbose=True
    )
    return score


def tune_model(X, features, score_col, index_col, model_type,
                     n_trials: int = 30, cv=None,
                     sample_weights: bool = False) -> dict:
    """
    Run hyperparameter optimization using Optuna.
    """
    optuna_study = optuna.create_study(direction="maximize")
    optuna_study.optimize(
        lambda trial: optuna_objective(trial, model_type, X, features, score_col,
                                index_col, cv, sample_weights),
        n_trials=n_trials
    )
    print(f"Best params for {model_type}: {optuna_study.best_params}")
    print(f"Best score: {optuna_study.best_value}")
    return optuna_study.best_params


def create_cv_splitter(strategy: str = "stratified", n_splits: int = 5,
           seed: int = 42) -> Any:
    """
    Return a cross-validation splitter based on the strategy.
    """
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    elif strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        raise ValueError("Unsupported CV strategy")


# Main Execution Functions
def prepare_data(
    static_train_path: str, 
    static_test_path: str,
    dynamic_train_dir: str,
    dynamic_test_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load static and dynamic data, process and merge them.
    """
    logging.info("Loading static data...")
    static_train, static_test = load_static_data(static_train_path,
                                                 static_test_path)
    static_train, static_test =  process_static_data(static_train,
                                                    static_test)
    logging.info("Processing dynamic data...")
    dynamic_train, dynamic_test = process_dynamic_data(
        dynamic_train_dir, dynamic_test_dir
    )
    final_train, final_test = merge_static_dynamic_features(
        static_train, static_test, dynamic_train, dynamic_test
    )
    return final_train, final_test


def train_models(
    final_train: pd.DataFrame, 
    final_test: pd.DataFrame,
    cv: Any
) -> None:
    """
    Run model training, cross-validation and prediction.
    """
    # Define features (exclude id, target and 'sii')
    features = [col for col in final_train.columns
                if col not in ["id", "PCIAT-PCIAT_Total", "sii"]]
    # Exclude features that are not needed
    exclude_set = {"PC_9", "PC_12", "Fitness_Endurance-Max_Stage",
                   "Basic_Demos-Sex", "BMI_mean_norm", "PC_11",
                   "PC_8", "FGC_Zones_min", "Physical-Systolic_BP",
                   "PC_4", "BIA-BIA_FMI", "BIA-BIA_LST",
                   "Physical-Diastolic_BP", "BIA-BIA_ECW",
                   "Fitness_Endurance-Time_Mins", "PAQ_C-PAQ_C_Total",
                   "PC_10", "BIA-BIA_Fat", "FFM_norm", "PC_14", "PC_7"}
    reduced_features = [f for f in features if f not in exclude_set]
    logging.info("Total reduced features: %d", len(reduced_features))
    # Apply the same feature set to all models
    lgb_feats = xgb_feats = cat_feats = reduced_features

    # Model hyperparameters
    model_params = {
        "lgb_params": {
            "objective": "poisson",
            "n_estimators": 295,
            "max_depth": 4,
            "learning_rate": 0.04505693066482616,
            "subsample": 0.6042489155604022,
            "colsample_bytree": 0.5021876720502726,
            "min_data_in_leaf": 100
        },
        "xgb_params": {
            "objective": "reg:tweedie",
            "num_parallel_tree": 12,
            "n_estimators": 236,
            "max_depth": 3,
            "learning_rate": 0.04223740904479563,
            "subsample": 0.7157264603586825,
            "colsample_bytree": 0.7897918901977528,
            "reg_alpha": 0.005335705058190553,
            "reg_lambda": 0.0001897435318347022,
            "tweedie_variance_power": 1.1393958601390142
        },
        "xgb_params_2": {
            "objective": "reg:tweedie",
            "num_parallel_tree": 18,
            "n_estimators": 175,
            "max_depth": 3,
            "learning_rate": 0.032620453423049305,
            "subsample": 0.6155579670568023,
            "colsample_bytree": 0.5988773292417443,
            "reg_alpha": 0.0028895066837627205,
            "reg_lambda": 0.002232531512636924,
            "tweedie_variance_power": 1.1708678482038286
        },
        "cat_params": {
            "objective": "RMSE",
            "iterations": 238,
            "depth": 4,
            "learning_rate": 0.044523361750173816,
            "l2_leaf_reg": 0.09301285673435761,
            "subsample": 0.6902492783438681,
            "bagging_temperature": 0.3007304771330199,
            "random_strength": 3.562201626987314,
            "min_data_in_leaf": 60
        },
        "xtrees_params": {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_leaf": 20,
            "bootstrap": False
        }
    }
    lgb_params = model_params["lgb_params"]
    xgb_params = model_params["xgb_params"]
    xgb_params_2 = model_params["xgb_params_2"]
    cat_params = model_params["cat_params"]
    xtrees_params = model_params["xtrees_params"]

    models = {
        "LGBM": (LGBMRegressor(**lgb_params, random_state=SEED,
                               verbosity=-1), lgb_feats),
        "XGB": (XGBRegressor(**xgb_params, random_state=SEED,
                             verbosity=0), xgb_feats),
        "XGB_2": (XGBRegressor(**xgb_params_2, random_state=SEED,
                               verbosity=0), xgb_feats),
        "CatBoost": (CatBoostRegressor(**cat_params, random_state=SEED,
                                       verbose=0), cat_feats),
        "ExtraTrees": (ExtraTreesRegressor(**xtrees_params,
                                           random_state=SEED),
                       reduced_features)
    }
    weights = compute_sample_weights(final_train["PCIAT-PCIAT_Total"])

    scores = {}
    oof_preds = {}
    test_preds = {}
    thresholds_dict = {}

    for name, (model, feats) in models.items():
        score, oof, thrs = cross_val_kappa(
            model, final_train, feats, "PCIAT-PCIAT_Total", "sii",
            cv, verbose=True, sample_weights=True
        )
        model.fit(final_train[feats],
                  final_train["PCIAT-PCIAT_Total"],
                  sample_weight=weights)
        pred_test = model.predict(final_test[feats])
        scores[name] = score
        oof_preds[name] = oof
        test_preds[name] = pred_test
        thresholds_dict[name] = thrs

    logging.info("Overall Mean Kappa: %f", 
                 np.mean(list(scores.values())))
    if "PCIAT-PCIAT_Total" in final_train.columns:
        sns.set_theme(style="whitegrid")
        plt.hist(final_train["PCIAT-PCIAT_Total"], bins=50,
                 color="darkorange")
        plt.title("Score Distribution")
        plt.show()


# Main Function
def main() -> None:
    """
    Main function to run preprocessing and modeling.
    """
    static_train_path = (
        "/Users/weiqinggao/Documents/obsidian/markdown_files/courses/"
        "2025_spring/cs5100/project/child-mind-institute-problematic-"
        "internet-use/train.csv"
    )
    static_test_path = (
        "/Users/weiqinggao/Documents/obsidian/markdown_files/courses/"
        "2025_spring/cs5100/project/child-mind-institute-problematic-"
        "internet-use/test.csv"
    )
    dynamic_train_dir = (
        "/Users/weiqinggao/Documents/obsidian/markdown_files/courses/"
        "2025_spring/cs5100/project/child-mind-institute-problematic-"
        "internet-use/series_train.parquet"
    )
    dynamic_test_dir = (
        "/Users/weiqinggao/Documents/obsidian/markdown_files/courses/"
        "2025_spring/cs5100/project/child-mind-institute-problematic-"
        "internet-use/series_test.parquet"
    )
    final_train, final_test = prepare_data(
        static_train_path, static_test_path,
        dynamic_train_dir, dynamic_test_dir
    )
    cv = create_cv_splitter(strategy="stratified", n_splits=N_SPLITS, seed=SEED)
    train_models(final_train, final_test, cv)


if __name__ == "__main__":
    main()
