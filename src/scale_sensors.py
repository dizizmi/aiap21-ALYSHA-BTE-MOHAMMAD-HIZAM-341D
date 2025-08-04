from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_sensor_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
   
    # fill missing values with median
    for col in sensor_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            raise ValueError(f"[ERROR] Missing expected sensor column: {col}")

    # scale
    scaler = StandardScaler()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

    return df