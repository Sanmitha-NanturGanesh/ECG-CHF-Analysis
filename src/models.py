import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def prepare_features(df, feature_columns):
    """
    Prepare and scale feature matrix for modeling.

    Returns:
        X_scaled, scaler
    """
    X = df[feature_columns].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_kmeans(df, feature_columns, n_clusters=3, random_state=42):
    """
    Run KMeans clustering on selected HRV features.
    """
    X_scaled, _ = prepare_features(df, feature_columns)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)

    result_df = df.copy()
    result_df["cluster"] = labels

    return model, result_df


def run_isolation_forest(df, feature_columns, contamination=0.15, random_state=42):
    """
    Detect anomalies in HRV feature space using Isolation Forest.
    """
    X_scaled, _ = prepare_features(df, feature_columns)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    anomaly_labels = model.fit_predict(X_scaled)

    result_df = df.copy()
    result_df["anomaly"] = anomaly_labels  # -1 = anomaly, 1 = normal

    return model, result_df


def evaluate_model(model, X_test, y_test):
    """
    Compute evaluation metrics for a trained classifier.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }


def train_classification_models(df, feature_columns, target_column, test_size=0.2, random_state=42):
    """
    Train multiple classification models and return evaluation metrics.

    Returns:
        metrics_df, trained_models
    """
    X = df[feature_columns]
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=False, random_state=random_state)
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state
        )

    metrics = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        scores = evaluate_model(model, X_test, y_test)
        scores["model"] = model_name
        metrics.append(scores)
        trained_models[model_name] = model

    metrics_df = pd.DataFrame(metrics)[["model", "accuracy", "precision", "recall", "f1_score"]]
    metrics_df = metrics_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    return metrics_df, trained_models
