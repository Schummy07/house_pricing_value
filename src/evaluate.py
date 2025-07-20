from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns key performance metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
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
