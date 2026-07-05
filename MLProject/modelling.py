"""
modelling.py  (versi MLProject — Workflow-CI)
----------------------------------------------
Versi modelling yang kompatibel dengan MLflow Projects.
Menerima argumen dari MLProject entry point.

Melatih RandomForestClassifier dengan manual logging
agar parameter bisa dikontrol dari CLI/MLProject.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn


TARGET_COL = "loan_status"


def load_data(data_path: str, test_size: float, random_state: int):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


def save_classification_report(y_true, y_pred, save_path="classification_report.txt"):
    report = classification_report(y_true, y_pred, target_names=["Rejected", "Approved"])
    with open(save_path, "w") as f:
        f.write("Classification Report\n" + "=" * 40 + "\n" + report)
    return save_path


def train(args):
    import os
    # Gunakan tracking URI dari environment (di-set oleh mlflow run)
    # Fallback ke SQLite jika tidak ada
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Loan_Approval_CI")

    X_train, X_test, y_train, y_test = load_data(args.data_path, args.test_size, args.random_state)
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Gunakan run yang sudah dibuat oleh mlflow run jika ada
    active_run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_id=active_run_id, run_name="CI_RandomForest" if not active_run_id else None):
        # Log params
        mlflow.log_param("n_estimators",  args.n_estimators)
        mlflow.log_param("max_depth",     args.max_depth)
        mlflow.log_param("test_size",     args.test_size)
        mlflow.log_param("random_state",  args.random_state)

        # Train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")

        # Artefak
        cm_path     = plot_confusion_matrix(y_test, y_pred)
        report_path = save_classification_report(y_test, y_pred)

        mlflow.log_artifact(cm_path,     artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="LoanApproval_CI",
        )

        print("[INFO] Run berhasil. Artefak tersimpan di mlruns/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    type=str,   default="loan_approval_dataset_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int,   default=100)
    parser.add_argument("--max_depth",    type=int,   default=10)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--random_state", type=int,   default=42)
    args = parser.parse_args()
    train(args)
