import pandas as pd
from sklearn.preprocessing import LabelEncoder

def normalize_activity_label(label: str) -> str:
   #because the activity levels for all was repeated twice eg. 'Activity Level_low' and 'Activity Level_low activity'
    label = label.replace("_", " ").replace("-", " ").lower().strip()
    if "low" in label:
        return "low activity"
    elif "moderate" in label:
        return "moderate activity"
    elif "high" in label:
        return "high activity"
    else:
        return label  # fallback

def extract_target_and_features(df: pd.DataFrame, original_col="Activity Level") -> tuple:
   
    if original_col not in df.columns:
        # Attempt to reconstruct it from one-hot columns
        activity_cols = [col for col in df.columns if col.lower().startswith("activity level_")]
        if not activity_cols:
            raise ValueError("No activity level columns found.")

        # Recover original labels from one-hot columns
        df[original_col] = df[activity_cols].idxmax(axis=1).str.replace("Activity Level_", "")

    # Normalize activity level labels
    df[original_col] = df[original_col].apply(normalize_activity_label)

    # Encode labels numerically
    le = LabelEncoder()
    df['ActivityLevelEncoded'] = le.fit_transform(df[original_col])

    # Drop one-hot activity level columns if they exist
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("activity level_")])

    # Drop Session ID if present
    if "Session ID" in df.columns:
        df = df.drop(columns=["Session ID"])

    # Separate X and y
    X = df.drop(columns=['Activity Level', 'ActivityLevelEncoded'])
    y = df['ActivityLevelEncoded']

    return X, y, le

