import pandas as pd
import sqlite3
from preprocessing import clean_code
from scale_sensors import scale_sensor_features
from sklearn.preprocessing import StandardScaler
from extract_target import extract_target_and_features
from train_model import train_and_evaluate_models, tune_all_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def load_the_data(db_path: str, table_name: str) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from '{table_name}'")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data from {table_name}: {e}")
        return pd.DataFrame()


def main():

    #Load Data, successfully cleaned
    db_path = "data/gas_monitoring.db"
    table_name = "gas_monitoring"
    
    df = load_the_data(db_path, table_name)

    if df.empty:
        return
    df = clean_code(df)


    '''
        displays 37 col with encoding eg. 'CO_GasSensor_extremely high', 'CO_GasSensor_extremely low',
       'CO_GasSensor_high', 'CO_GasSensor_low', 'CO_GasSensor_medium',
       'CO_GasSensor_none'
    '''
    #these columns are REAL values columns that have null values
    sensor_cols = [
        'Temperature', 'Humidity',
        'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor',
        'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2',
        'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4'
    ]
    df = scale_sensor_features(df, sensor_cols)
    
    '''
    displays scaled val of all columns where the NULL values are replaced w median accordingly and scaled.
    eg. temperature > -0.308477 etc.
    '''
    X, y, label_encoder = extract_target_and_features(df)
    print(label_encoder.classes_)

    '''
    displays the features X and target y
    X: 
       Temperature, Humidity, CO2_InfraredSensor, CO2_ElectroChemicalSensor,
       MetalOxideSensor_Unit1, MetalOxideSensor_Unit2, MetalOxideSensor_Unit3,
       MetalOxideSensor_Unit4, Time of Day_afternoon, Time of Day_evening,
       Time of Day_morning, HVAC Operation Mode_cooling, HVAC Operation Mode_heating,
       HVAC Operation Mode_off, Ambient Light Level_dark, Ambient Light Level_light,
       Activity Level_low activity, Activity Level_moderate activity,
       Activity Level_high activity
       y: 
       [0 1 2]
       '''

    results = train_and_evaluate_models(X, y, feature_names=X.columns, label_encoder=label_encoder)

    print("parameter tuning")
    tuned_models = tune_all_models(X, y)


if __name__ == "__main__":
    main()