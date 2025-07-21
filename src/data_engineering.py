import os
import logging
from datetime import timedelta
from typing import Tuple

import pandas as pd

# ——— Logging setup —————————————————————————————————————————————
log_file_path = "logs/data_engineering.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)


def load_data(sales_path: str, demographics_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sales and demographics data from CSV files.

    Parameters
    ----------
    sales_path : str
        Path to the sales CSV file.
    demographics_path : str
        Path to the demographics CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - df: DataFrame of sales records.
        - demographics: DataFrame of zipcode demographics.
    """
    df = pd.read_csv(sales_path)
    demographics = pd.read_csv(demographics_path)
    return df, demographics


def add_trend(
    df: pd.DataFrame,
    date_col: str = 'date',
    trend_col: str = 'trend',
    quad_col: str = 'trend_sq'
) -> pd.DataFrame:
    """
    Add a linear and quadratic trend column based on months since the earliest date.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a date column.
    date_col : str, default 'date'
        Column name for dates (must be datetime dtype).
    trend_col : str, default 'trend'
        Name of the new linear trend column (in months).
    quad_col : str, default 'trend_sq'
        Name of the new quadratic trend column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with the two new columns added.
    """
    df2 = df.copy()
    first = df2[date_col].min()
    y0, m0 = first.year, first.month

    df2[trend_col] = (
        (df2[date_col].dt.year - y0) * 12 +
        (df2[date_col].dt.month - m0)
    )
    df2[quad_col] = df2[trend_col] ** 2
    return df2


def add_house_age(
    df: pd.DataFrame,
    date_col: str = 'date',
    built_col: str = 'yr_built',
    age_col: str = 'house_age'
) -> pd.DataFrame:
    """
    Calculate each house’s age in full years as of the observation date.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing build year and date columns.
    date_col : str, default 'date'
        Column name for the observation date (datetime dtype).
    built_col : str, default 'yr_built'
        Column name for the year the house was built (int).
    age_col : str, default 'house_age'
        Name of the new age column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with `age_col` added and `built_col` dropped.
    """
    df2 = df.copy()
    df2[age_col] = df2[date_col].dt.year - df2[built_col]
    df2.drop(columns=[built_col], inplace=True)
    df2[age_col] = df2[age_col].clip(lower=0)
    return df2


def add_renovation_flag(
    df: pd.DataFrame,
    renovated_col: str = 'yr_renovated',
    flag_col: str = 'was_renovated'
) -> pd.DataFrame:
    """
    Flag houses that have ever been renovated.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a renovation-year column.
    renovated_col : str, default 'yr_renovated'
        Column name for the year of last renovation (0 if never).
    flag_col : str, default 'was_renovated'
        Name of the new boolean flag column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with the boolean `flag_col` added.
    """
    df2 = df.copy()
    df2[flag_col] = df2[renovated_col] > 0
    return df2


def fix_outliers(
    df: pd.DataFrame,
    bedrooms_col: str = 'bedrooms',
    bathrooms_col: str = 'bathrooms'
) -> pd.DataFrame:
    """
    Cap bedrooms and bathrooms at reasonable maximums.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with bedroom and bathroom counts.
    bedrooms_col : str, default 'bedrooms'
        Column name for number of bedrooms.
    bathrooms_col : str, default 'bathrooms'
        Column name for number of bathrooms.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with outliers capped and bathrooms cast to int.
    """
    df2 = df.copy()
    df2.loc[df2[bedrooms_col] > 6, bedrooms_col] = 6
    df2.loc[df2[bathrooms_col] > 4, bathrooms_col] = 4
    df2[bathrooms_col] = df2[bathrooms_col].round(0).astype(int)
    return df2


def split_data(data: pd.DataFrame) -> None:
    """
    Split data into train, validation, and test sets based on the last 30-day windows.

    The function writes three CSVs into `data/golden/`:
      - train_data.csv
      - val_data.csv
      - test_data.csv

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset, already sorted or indexed by date.

    Returns
    -------
    None
    """
    out_dir = "data/golden"
    os.makedirs(out_dir, exist_ok=True)

    def _save_data(df: pd.DataFrame, filename: str) -> None:
        df.to_csv(filename, index=False)

    last_date = data.date.max()
    val_data = data[data.date > (last_date - timedelta(days=30))]
    data.drop(val_data.index, inplace=True)

    last_date = data.date.max()
    test_data = data[data.date > (last_date - timedelta(days=30))]
    data.drop(test_data.index, inplace=True)

    _save_data(data,      os.path.join(out_dir, "train_data.csv"))
    _save_data(val_data,  os.path.join(out_dir, "val_data.csv"))
    _save_data(test_data, os.path.join(out_dir, "test_data.csv"))


def launch() -> None:
    """
    Run the full data preparation pipeline:
      1. Load raw CSVs
      2. Feature engineering (date parsing, merges, trend, age, flags, outlier capping)
      3. Split into train/val/test sets

    Returns
    -------
    pd.DataFrame
        The final prepared DataFrame (before splitting).
    """
    sales_path = "data/kc_house_data.csv"
    demographics_path = "data/zipcode_demographics.csv"

    logger.info("Loading data...")
    df, demographics = load_data(sales_path, demographics_path)

    logger.info("Doing feature engineering...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(demographics[["zipcode", "hous_val_amt"]], on='zipcode', how='left')
    cols_to_drop = ["zipcode", "sqft_living15", "sqft_lot15"]
    data = df.drop(columns=cols_to_drop)

    data = add_trend(data)
    data = add_house_age(data)
    data = add_renovation_flag(data)
    data = fix_outliers(data)

    logger.info("Splitting data into train, validation, and test sets...")
    split_data(data)

    logger.info("Data preparation complete. Files written to 'data/golden/'.")
