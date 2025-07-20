import os
import logging
from typing import Any, Dict, Tuple
import pickle

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json

from src.evaluate import evaluate_model, feature_importances

# ——— Logging setup —————————————————————————————————————————————
log_file_path = "logs/model_training.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


def load_train_test_data(train_path: str, test_path: str):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def prepare_data(data: pd.DataFrame,
                 cols_to_drop: list, 
                 target_column: str,
                 ) -> Tuple[pd.DataFrame, pd.Series]:

    X = data.drop(columns=cols_to_drop + [target_column])
    y = data[target_column]

    return X, y


def get_search_space() -> Dict[str, Any]:
    """
    Define the Hyperopt search space for XGBRegressor hyperparameters.

    Returns
    -------
    Dict[str, Any]
        A dictionary of hyperparameter search distributions.
    """
    return {
        'n_estimators':      hp.choice('n_estimators',     [100, 200, 300, 500]),
        'max_depth':         hp.quniform('max_depth',     3, 10, 1),
        'learning_rate':     hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'subsample':         hp.uniform('subsample',       0.6, 1.0),
        'colsample_bytree':  hp.uniform('colsample_bytree',0.6, 1.0),
        'gamma':             hp.uniform('gamma',           0,   5),
        'min_child_weight':  hp.quniform('min_child_weight', 1, 10, 1),
        'reg_alpha':         hp.loguniform('reg_alpha',     np.log(1e-3), np.log(10)),
        'reg_lambda':        hp.loguniform('reg_lambda',    np.log(1e-3), np.log(10)),
    }


def objective(
    params: Dict[str, Any],
    X_train: Any,
    y_train: Any,
    cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Objective function for Hyperopt: 5 fold CV mean absolute percentage error.

    Parameters
    ----------
    params : dict
        Hyperparameter values drawn from the search space.
    X_train : array like
        Training features.
    y_train : array like
        Training targets.
    cv_splits : int, default=5
        Number of folds for cross validation.

    Returns
    -------
    Dict[str, Any]
        - loss: the MAPE to minimize
        - status: always STATUS_OK for Hyperopt
    """
    # ensure ints where required
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **params
    )

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    neg_mape_scores = cross_val_score(
        model,
        X_train, y_train,
        scoring='neg_mean_absolute_percentage_error',
        cv=cv,
        n_jobs=-1
    )
    mape = -neg_mape_scores.mean()
    logger.info(f"Tested params {params} → MAPE={mape:.5f}")
    return {'loss': mape, 'status': STATUS_OK}


def run_optimization(
    X_train: Any,
    y_train: Any,
    max_evals: int = 30
) -> Dict[str, Any]:
    """
    Run Hyperopt optimization to find the best hyperparameters.

    Parameters
    ----------
    X_train : array like
        Training features.
    y_train : array like
        Training targets.
    max_evals : int, default=50
        Maximum number of Hyperopt evaluations.

    Returns
    -------
    best_params : dict
        The best hyperparameters (converted to the correct types/values).
    trials : hyperopt.Trials
        Object containing full trial history.
    """
    space = get_search_space()
    trials = Trials()

    logger.info("Starting hyperparameter optimization...")
    best = fmin(
        fn=lambda p: objective(p, X_train, y_train),
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Convert HP choice indices back to actual values
    best['n_estimators'] = [100, 200, 300, 500][best['n_estimators']]
    best['max_depth'] = int(best['max_depth'])
    best['min_child_weight'] = int(best['min_child_weight'])

    logger.info(f"Optimization complete. Best params: {best}")
    return best


def train_best_model(
    best_params: Dict[str, Any],
    X_train: Any,
    y_train: Any
) -> XGBRegressor:
    """
    Train an XGBRegressor with the best hyperparameters on the full training set.

    Parameters
    ----------
    best_params : dict
        Hyperparameters to use for training.
    X_train : array like
        Training features.
    y_train : array like
        Training targets.

    Returns
    -------
    model : XGBRegressor
        The trained XGBRegressor instance.
    """
    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params
    )
    logger.info("Training final model with best hyperparameters...")
    model.fit(X_train, y_train)
    logger.info("Final model training complete.")
    return model


def save_challenger_model(model, metrics, X_train, output_dir: str = "model") -> None:
    """
    Save the trained model to the specified output directory.

    Parameters
    ----------
    model : XGBRegressor
        The trained model to save.
    output_dir : str, default="model"
        Directory where the model will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save model as pickle
    model_path = os.path.join(output_dir, "challenger_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Pickled model saved to {model_path}")

    metrics_path = os.path.join(output_dir, "challenger_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    feature_importances(model, X_train, output_dir)


def launch() -> Tuple[XGBRegressor, Dict[str, Any], Trials]:
    """
    Full tuning pipeline: optimize hyperparameters, then train final model.

    Parameters
    ----------
    X_train : array like
        Training features.
    y_train : array like
        Training targets.
    max_evals : int, default=50
        Number of hyperparameter evaluations.

    Returns
    -------
    model : XGBRegressor
        Trained model with optimized hyperparameters.
    best_params : dict
        The best-found hyperparameters.
    trials : hyperopt.Trials
        Full Hyperopt trial history.
    """

    train_data_path = "data/golden/train_data.csv"
    test_data_path = "data/golden/test_data.csv"

    logger.info("Loading training data...")
    train_data, test_data = load_train_test_data(train_data_path, test_data_path)

    X_train, y_train = prepare_data(train_data, ['id', 'date'], 'price')
    X_test, y_test = prepare_data(test_data, ['id', 'date'], 'price')

    logger.info("Starting model training pipeline...")
    best_params = run_optimization(X_train, y_train)
    model = train_best_model(best_params, X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Model evaluation metrics: {metrics}")

    save_challenger_model(model, metrics, X_train)
