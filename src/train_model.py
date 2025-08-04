
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(name, model, X_test, y_test, y_pred, label_encoder, feature_names):
    print(f"\n--- Evaluation: {name} ---")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Feature importance if available
    if name in ["Random Forest", "Logistic Regression"]:
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            else:
                importance = model.coef_[0]
            feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            return feature_importance
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            return None
    return None

def train_and_evaluate_models(X, y, feature_names, label_encoder):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf', probability=True),
        "Naive Bayes": GaussianNB()
    }

    results = {}

    print("Beginning model training and evaluation...")
    for name, model in models.items():
        print(f"\n=== Training: {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        feature_importance = evaluate_model(name, model, X_test, y_test, y_pred, label_encoder, feature_names)
        results[name] = {
            "model": model,
            "predictions": y_pred,
            "feature_importance": feature_importance
        }

    return results


def tune_model(model, param_grid, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='weighted')

    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
    grid_search.fit(X, y)

    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")
    print("\nClassification Report on Full Set:")
    y_pred = grid_search.best_estimator_.predict(X)
    print(classification_report(y, y_pred))

    return grid_search.best_estimator_

def tune_all_models(X, y):
    print("Running hyperparameter tuning for all models...")

    best_models = {}

    #Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    print("Tuning Random Forest")
    best_models['Random Forest'] = tune_model(rf, rf_params, X, y)

    #Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg_params = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    print("Tuning Logistic Regression")
    best_models['Logistic Regression'] = tune_model(logreg, logreg_params, X, y)

    #SVM
    svm = SVC(probability=True)
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    print("Tuning SVM")
    best_models['SVM'] = tune_model(svm, svm_params, X, y)

    #Naive Bayes (No tuning needed)
    print("Naive Bayes uses default parameters (no tuning required).")
    best_models['Naive Bayes'] = GaussianNB().fit(X, y)

    return best_models
