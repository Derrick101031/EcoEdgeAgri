from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def fit_scaler(df_sub: pd.DataFrame) -> MinMaxScaler:
    """Create a 0-1 MinMaxScaler for a single plant."""
    sc = MinMaxScaler()
    sc.fit(df_sub)
    return sc


def apply_scaler(df_sub: pd.DataFrame, sc: MinMaxScaler) -> pd.DataFrame:
    """Return a copy of df_sub but scaled."""
    df_scaled = df_sub.copy()
    df_scaled[:] = sc.transform(df_sub)
    return df_scaled

