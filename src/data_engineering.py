def add_trend(df, date_col='date', trend_col='trend', quad_col='trend_sq'):
    """
    Add a linear and quadratic trend column to df based on monthly steps from the earliest date.

    Parameters
    ----------
    df : pd.DataFrame
        Your data. Must contain a column with dates.
    date_col : str, default 'date'
        Name of the column in df with date or datetime values.
    trend_col : str, default 'trend'
        Name for the new linear trend column.
    quad_col : str, default 'trend_sq'
        Name for the new quadratic trend column.

    Returns
    -------
    df2 : pd.DataFrame
        A copy of df with trend_col and quad_col added.
    """
    df2 = df.copy()

    first = df2[date_col].min()
    y0, m0 = first.year, first.month

    df2[trend_col] = (
        (df2[date_col].dt.year  - y0) * 12 +
        (df2[date_col].dt.month - m0)
    )

    df2[quad_col] = df2[trend_col] ** 2

    return df2


def add_house_age(df, date_col='date', built_col='yr_built', age_col='house_age'):
    """
    Calculate the age of each house (in full years) as of the date in `date_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Your data. Must contain `date_col` and `built_col`.
    date_col : str, default 'date'
        Name of the column with the observation date (datetime or parseable string).
    built_col : str, default 'yr_built'
        Name of the column with the year the house was built (int).
    age_col : str, default 'house_age'
        Name for the new age column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with `age_col` added.
    """
    df2 = df.copy()

    df2[age_col] = df2[date_col].dt.year - df2[built_col]
    df2.drop(columns=[built_col], inplace=True)
    df2[age_col] = df2[age_col].clip(lower=0)
    return df2


def add_renovation_flag(df, renovated_col='yr_renovated', flag_col='was_renovated'):
    """
    Create a boolean flag indicating whether a house has ever been renovated.

    Parameters
    ----------
    df : pd.DataFrame
        Your data. Must contain `renovated_col`.
    renovated_col : str, default 'yr_renovated'
        Name of the column with the year of last renovation (0 if never).
    flag_col : str, default 'was_renovated'
        Name for the new boolean flag column.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with `flag_col` added.
    """
    df2 = df.copy()

    df2[flag_col] = df2[renovated_col] > 0
    return df2
