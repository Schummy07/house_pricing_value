import pandas as pd


def make_prediction(input_data: pd.DataFrame, model):
    """
    Make predictions with the loaded model.
    """
    prediction = model.predict(input_data)
    return prediction
