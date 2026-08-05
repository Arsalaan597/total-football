import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate():
    data_path = os.path.join("data", "processed", "epl_processed_v01.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run build_dataset.py first.")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Define features and target label
    feature_cols = [
        "home_points_last_5",
        "away_points_last_5",
        "home_goals_for_last_5",
        "away_goals_for_last_5",
        "home_goals_against_last_5",
        "away_goals_against_last_5",
        "home_home_points_last_5",
        "away_away_points_last_5",
    ]
    target_col = "FTR"

    # Time-based split: Older matches for training, recent season for testing
    split_date = pd.to_datetime("2023-08-01")
    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n")

    # 1. Baseline Model (Always predict most frequent outcome in training set)
    most_frequent = y_train.mode()[0]
    baseline_preds = [most_frequent] * len(y_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds, average="macro")

    print(f"--- Baseline Model (Always '{most_frequent}') ---")
    print(f"Accuracy: {baseline_acc:.4f}")
    print(f"Macro F1: {baseline_f1:.4f}\n")

    # 2. Logistic Regression Pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Evaluation Metrics
    model_acc = accuracy_score(y_test, y_pred)
    model_f1 = f1_score(y_test, y_pred, average="macro")

    print("--- Multinomial Logistic Regression Model ---")
    print(f"Accuracy: {model_acc:.4f}")
    print(f"Macro F1: {model_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away (A)", "Draw (D)", "Home (H)"]))

    print("Confusion Matrix:")
    labels = ["A", "D", "H"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df)
    print("\n")

    # Sample Match Prediction with Probabilities
    sample_idx = 0
    sample_match = test_df.iloc[sample_idx]
    classes = pipeline.named_steps["logisticregression"].classes_
    proba_dict = dict(zip(classes, y_proba[sample_idx]))

    print("--- Sample Held-Out Match Prediction ---")
    print(f"Match: {sample_match['HomeTeam']} vs {sample_match['AwayTeam']} on {sample_match['Date'].strftime('%Y-%m-%d')}")
    print(f"Actual Result: {sample_match['FTR']}")
    print(f"Predicted Result: {y_pred[sample_idx]}")
    print(f"Predicted Probabilities: Home Win: {proba_dict.get('H', 0):.2%}, Draw: {proba_dict.get('D', 0):.2%}, Away Win: {proba_dict.get('A', 0):.2%}")

    # Save trained model artifact
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "winner_model_v01.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nModel successfully saved to: {model_path}")


if __name__ == "__main__":
    train_and_evaluate()