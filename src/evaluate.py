from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(
    model: Any,
    test_data: pd.DataFrame,
    *,
    full_df: Optional[pd.DataFrame] = None,
    demographics: Optional[pd.DataFrame] = None,
    model_features: Optional[List[str]] = None,
    zipcode_col: str = "zipcode",
    placeholder_col: str = "hous_val_amt",
    setup_needed: bool = True,
    target_column: str = "price"
) -> Dict[str, float]:
    """
    Evaluates the model and returns key performance metrics.

    This will automatically handle two cases:
    1. XGBRegressor (or anything else): predict on X_test directly.
    2. Pipeline wrapping KNeighborsRegressor: requires merging demographics.

    Parameters
    ----------
    model : Any
        A fitted model or sklearn Pipeline.
    X_test : pd.DataFrame
        Test feature DataFrame (including 'zipcode' as needed).
    y_test : pd.Series
        True targets.
    full_df : pd.DataFrame, optional
        The original full DataFrame from which X_test was derived. Required
        if evaluating a KNeighbors pipeline.
    demographics : pd.DataFrame, optional
        The demographics DataFrame to merge in. Required for KNeighbors pipeline.
    model_features : List[str], optional
        The final feature list that the KNeighborsRegressor expects.
        Required when using a KNeighbors pipeline.
    zipcode_col : str, default='zipcode'
        Column name to join on.
    placeholder_col : str, default='hous_val_amt'
        Column to drop before merging demographics.

    Returns
    -------
    Dict[str, float]
        A dict with keys 'MSE', 'MAE', and 'MAPE'.
    """
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    # Determine if this is a KNeighbors pipeline:
    is_knn_pipeline = (
        isinstance(model, Pipeline)
        and any(isinstance(step, KNeighborsRegressor) for _, step in model.steps)
    )
    if setup_needed:
        if is_knn_pipeline:
            if full_df is None or demographics is None or model_features is None:
                raise ValueError(
                    "full_df, demographics, and model_features must be provided "
                    "when evaluating a KNeighbors pipeline."
                )

            # Reconstruct the exact rows & merge demographics
            # 1) Take only the rows in X_test (by id) from the full_df
            df_sub = (
                full_df
                #.drop(columns=[placeholder_col])
                .loc[full_df.id.isin(test_data.id)]
                .merge(demographics, on=zipcode_col, how="left", suffixes=("_full_df", None))
            )
            # 2) Select only the features the model expects
            X_eval = df_sub[model_features]
            y_eval = df_sub[target_column]

        else:
            X_eval = X_test[model_features]
            y_eval = y_test
    else:
        X_eval = X_test[model_features]
        y_eval = y_test

    # Standard predictions & metrics
    y_pred = model.predict(X_eval)

    metrics = {
        'MSE':  mean_squared_error(y_eval, y_pred),
        'MAE':  mean_absolute_error(y_eval, y_pred),
        'MAPE': mean_absolute_percentage_error(y_eval, y_pred),
    }
    return metrics


def feature_importances(model, X_train, output_dir):
    """
    Returns a Plot of feature importances.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    feats = X_train.columns[idx]
    imps = importances[idx]

    plt.figure(figsize=(12, 8))
    plt.barh(feats, imps)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()

    fi_path = os.path.join(output_dir, "challenger_model_feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
