"""
================================================================
  MODEL.PY — Machine Learning Pipeline
  Random Forest & Linear Regression
  Dashboard Produksi Tanaman Perkebunan Indonesia
================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan fitur sebelum training:
    - Handle missing values
    - Pastikan semua kolom numerik
    - Drop kolom dengan variance nol
    """
    X = X.copy()

    # Drop kolom dengan variance nol (tidak informatif)
    variance = X.var()
    zero_var_cols = variance[variance == 0].index
    if len(zero_var_cols) > 0:
        X = X.drop(columns=zero_var_cols)

    # Handle missing values dengan median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
):
    """
    Melatih model Random Forest Regressor.

    Returns:
        tuple: (model, metrics_dict)
            - model: RandomForestRegressor yang sudah di-train
            - metrics_dict: {'mae', 'rmse', 'r2'}
    """
    # Bersihkan fitur
    X_clean = _prepare_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state
    )

    # Inisialisasi model dengan hyperparameter yang robust
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    return model, metrics


def train_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Melatih model Linear Regression dengan StandardScaler.

    Returns:
        tuple: (pipeline, metrics_dict)
            - pipeline: sklearn Pipeline (scaler + LR)
            - metrics_dict: {'mae', 'rmse', 'r2'}
    """
    # Bersihkan fitur
    X_clean = _prepare_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=test_size, random_state=random_state
    )

    # Pipeline: Scaler + Linear Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])

    # Training
    pipeline.fit(X_train, y_train)

    # Prediksi
    y_pred = pipeline.predict(X_test)

    # Metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    return pipeline, metrics


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """
    Mengekstrak feature importance dari model Random Forest.

    Args:
        model: Trained RandomForestRegressor
        feature_names: List atau array nama fitur

    Returns:
        DataFrame dengan kolom 'feature' dan 'importance', diurutkan descending
    """
    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    feat_df = feat_df.sort_values("importance", ascending=False).reset_index(drop=True)

    return feat_df


def predict_new_data(model, X_new: pd.DataFrame) -> np.ndarray:
    """
    Melakukan prediksi pada data baru.

    Args:
        model: Trained model (RF atau Pipeline LR)
        X_new: DataFrame fitur baru

    Returns:
        Array hasil prediksi
    """
    X_clean = _prepare_features(X_new)
    return model.predict(X_clean)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluasi model pada dataset tertentu.

    Returns:
        Dict dengan keys: mae, rmse, r2, predictions
    """
    X_clean = _prepare_features(X)
    y_pred = model.predict(X_clean)

    return {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred),
        "predictions": y_pred,
    }
