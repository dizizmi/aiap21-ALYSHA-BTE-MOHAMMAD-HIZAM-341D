ALYSHA BTE MOHAMMAD HIZAM
alyshabm000@gmail.com

## Indoor Quality Monitoring for Elderly ML Pipeline

The project builds a ML pipeline to predict the activity level of elderly individuals living alone, based on indoor environmental sensor data. It includes preprocessing, modeling, evaluation, and interpretation of the key contributing features to activity-level classification.

The project is structured the following:

aiap21-ALYSHA-BTE-MOHAMMAD-HIZAM-341D/
├── data/ gas_monitoring.db                 
├── src/
├─────extract_target.py (extract and normalizes target label, convert numeric using LabelEncoder, returns X,y and the encoder)   
├─────load_data.py (connects to database and loads main())
├─────preprocessing.py (cleans raw categorical col w one hot encoding)
├─────scale_sensors.py (handles 'sensors' value preprocessing and fills NaN with meadian and scale feature with StandardScaler, normalizes inconsistent text)
├─────train_model.py (train baseline models, evaluates accuracy/precision/f1/confusion matrix, plot results)   
├── eda.ipynb                
├── requirements.txt 
├── run.sh           
└── README.md              


# The logical flow 
| Stage                | Details                                                                 |
|---------------------|------------------------------------------------------------------------- |
| Data Loading         | Structured import from `.db` using SQLite and Pandas                    |
| Preprocessing        | Cleaned and encoded all inputs, scaled continuous variables             |
| Feature Engineering  | Normalized inconsistent text values, dropped irrelevant columns         |
| Target Engineering   | Encoded `"Activity Level"` as multiclass target                         |
| Model Training       | Applied multiple classifiers with hyperparameter tuning                 |
| Evaluation           | Used F1 and confusion matrix to assess health-critical misclassifications |
| Deployment Thinking  | Considered class imbalance, real-time applicability, and feature insights |


To execute: 
```bash
# 1. Load and preprocess
python data/load_data.py

# 2. Optionally, run analysis notebook for EDA
jupyter eda.ipynb

# 3. Train models
# From within a script or notebook:
from train_models import train_and_evaluate_models
results = train_and_evaluate_models(X, y, feature_names=X.columns, label_encoder=label_encoder)
```

To change test size or model parameters, edit:
- `train_models.py`: model hyperparameters
- `train_test_split()`: test size


# EDA Findings
Refer to eda.ipynb for eda.

# Feature processing summary
| Feature                          | Type         | Processing                               |
|----------------------------------|--------------|------------------------------------------|
| Temperature, Humidity            | Continuous   | Fill missing + z-score scaling           |
| CO₂ & Metal Oxide Sensors        | Continuous   | Fill missing + z-score scaling           |
| HVAC Mode, Ambient Light, CO Gas | Categorical  | Clean + one-hot encode                   |
| Activity Level (target)          | Categorical  | Normalized, label encoded (3 classes)    |

- Cleaned and normalized all categorical columns (e.g., `"Activity Level"`, `"HVAC Operation Mode"`)
- One-hot encoded all text-based categorical features
- Median-filled missing values in sensor data (noted could be due to equipment interruptions)
- Applied z-score normalization to continuous features (temperature, humidity, CO₂ sensors)
- Dropped `Session ID` as a non-informative identifier
- Extracted and label-encoded `"Activity Level"` for supervised classification

Environmental domain insights supported these decisions:
> - High CO and humidity levels were flagged as concerning for respiratory risk
> - RH between 40–60% was considered optimal
> - Dim light for prolonged periods at night suggested inactivity or fall risk

# Model used
| Model               | Rationale                                         |
|--------------------|--------------------------------------------------- |
| Random Forest       | Handles non-linear data, gives feature importance |
| Logistic Regression | Interpretable baseline                            |
| SVM                 | Robust in high-dimension, kernel learning         |
| Naive Bayes         | Fast, baseline model -assumes feature independence|

- Random Forest is robust to noise and ideal for ensemble learning over environmental variables
- SVM and Logistic Regression are strong baselines for classification
- Naive Bayes included for interpretability and comparison
**Optimization:**
- Used GridSearchCV with 5-fold Stratified Cross-Validation
- Hyperparameters tuned: `n_estimators`, `max_depth`, `C`, `kernel`, etc.
- Scoring metric: **weighted F1-score** to handle imbalanced classes

# Evaluation metrics
- **Accuracy**: overall correctness
- **Precision**: reliability of each predicted class
- **Recall**: ability to capture each true class
- **F1 Score (weighted)**: balanced metric that accounts for class imbalance
- **Confusion Matrix**: visual inspection of misclassifications\

Example Insight: Poor recall on "high activity" could represent a missed detection of important behavior change or fall risk in an elderly individual.

## Model Evaluation
To assess how well each model could predict elderly residents’ activity levels based on indoor environmental conditions, I trained and evaluated four classification models using a holdout test set (30% of the data). The models were selected to reflect a range of complexity and interpretability.
Random Forest performed the strongest overall, achieving an accuracy of 67.7% and a weighted F1 score of 0.657. It showed robust generalization, particularly in identifying low and moderate activity levels. However, it struggled with the high activity class (F1: 0.19), likely due to class imbalance. Feature importance analysis highlighted MetalOxideSensor_Unit4, CO2_ElectroChemicalSensor, and Temperature as the most influential predictors—aligning with literature on indoor air quality impacts.

Logistic Regression reached 62.6% accuracy with a weighted F1 score of 0.582, but it failed to classify high activity instances altogether. This suggests the model's linear assumptions may be too limiting for the nonlinear patterns in this data. Despite this, it offered high interpretability through its coefficients, with CO_GasSensor_low, MetalOxideSensor_Unit4, and Ambient Light Level_dim emerging as key contributors.

Support Vector Machine (SVM) achieved a balance between the two with 64.3% accuracy and 0.612 weighted F1, but similar to logistic regression, it struggled to predict high activity cases. The RBF kernel captured moderate nonlinearities, yet the class imbalance again limited performance on minority classes.

Naive Bayes was the least effective, with only 25.7% accuracy and a weighted F1 score of 0.277. While it showed artificially high precision due to overfitting to dominant classes, its recall was poor across the board. This suggests that the independence assumption among features (e.g., sensor readings) may not hold for this dataset.

To improve generalization, I applied grid search with 5-fold cross-validation for Random Forest, Logistic Regression, and SVM using a weighted F1 score as the scoring metric.
Random Forest tuned best with n_estimators=100 and default max depth, reaching an F1 of 0.67 on cross-validation — consistent with its test performance.
Logistic Regression performed best with C=10, though it remained limited by class imbalance.
SVM tuned with C=1 and an RBF kernel, showing modest improvements over defaults.


# Deployment considerations
- Sensor data must be cleaned and normalized before inference
- Class imbalance may skew predictions (can apply SMOTE or weighted loss)
- Random Forest provides interpretability for alert systems
- Future: integrate temporal analysis such as LSTM or anomaly detection for early risk