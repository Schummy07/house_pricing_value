from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


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