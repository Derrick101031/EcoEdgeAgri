import pandas as pd

FEATURES = ["Soil_Moisture", "Ambient_Temperature", "Humidity"]

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Read ThingSpeak CSV (or any tabular log) and tidy the columns.
    Ensures Timestamp is a datetime index and sorted.
    """
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.sort_values(["Plant_ID", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_sequences(df: pd.DataFrame, seq_len: int):
    """
    Slide a (seq_len × 3) window across each plant_id.
    Returns X (N, seq_len, 3) and y (N, 3) *normalised to 0-1*.
    """
    from .preprocess import fit_scaler, apply_scaler   # local import to avoid cycle

    X_all, y_all = [], []
    scalers = {}                                       # one scaler per plant

    for pid, sub in df.groupby("Plant_ID"):
        sc = fit_scaler(sub[FEATURES])
        scalers[pid] = sc
        vals = apply_scaler(sub[FEATURES], sc).values

        for i in range(len(vals) - seq_len):
            X_all.append(vals[i:i+seq_len])
            y_all.append(vals[i+seq_len])

    import numpy as np
    return np.asarray(X_all, dtype="float32"), np.asarray(y_all, dtype="float32"), scalers

