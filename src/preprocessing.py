import pandas as pd

def clean_code(df: pd.DataFrame) -> pd.DataFrame:

    #these col are TEXT columns 

    old_col = [
        'Time of Day',
        'CO_GasSensor',
        'HVAC Operation Mode',
        'Ambient Light Level',
        'Activity Level'
    ]

    for col in old_col:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df = pd.get_dummies(df, columns=old_col, drop_first=False)
    
    
    return df