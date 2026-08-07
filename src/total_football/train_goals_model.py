import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_goals():
    data_path = os.path.join("data", "processed", "epl_processed_v01.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed dataset missing at {data_path}. Run build_dataset.py first."
        )

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Create target variables derived directly from raw scores
    df["total_goals"] = df["FTHG"] + df["FTAG"]
    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
    df["btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)

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

    # Time-based split matching Winner Model v0.1
    split_date = pd.to_datetime("2023-08-01")
    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    # Train separate Poisson models for Home and Away Goals
    model_home = make_pipeline(
        StandardScaler(), PoissonRegressor(max_iter=1000)
    )
    model_away = make_pipeline(
        StandardScaler(), PoissonRegressor(max_iter=1000)
    )

    model_home.fit(X_train, train_df["FTHG"])
    model_away.fit(X_train, train_df["FTAG"])

    # Predict expected goals (lambda parameters) on test set
    exp_home_goals = model_home.predict(X_test)
    exp_away_goals = model_away.predict(X_test)
    exp_total_goals = exp_home_goals + exp_away_goals

    # Derive Over/Under 2.5 prediction (Total Expected Goals > 2.5)
    pred_over_2_5 = (exp_total_goals > 2.5).astype(int)

    # Calculate Evaluation Metrics
    mae_goals = mean_absolute_error(test_df["total_goals"], exp_total_goals)
    acc_over_2_5 = accuracy_score(test_df["over_2_5"], pred_over_2_5)

    # Naive Baseline for Over 2.5 (Predict majority class in train)
    majority_over_2_5 = train_df["over_2_5"].mode()[0]
    baseline_acc = accuracy_score(
        test_df["over_2_5"], [majority_over_2_5] * len(test_df)
    )

    print("--- Goals & Over/Under 2.5 Model Results ---")
    print(f"Total Goals Mean Absolute Error (MAE): {mae_goals:.3f} goals")
    print(f"Over/Under 2.5 Accuracy: {acc_over_2_5:.2%}")
    print(f"Baseline Over/Under 2.5 Accuracy: {baseline_acc:.2%}\n")

    # Sample Match Output
    sample_match = test_df.iloc[0]
    sh_exp = exp_home_goals[0]
    sa_exp = exp_away_goals[0]
    print("--- Sample Match Goal Expectations ---")
    print(
        f"Match: {sample_match['HomeTeam']} vs {sample_match['AwayTeam']} on {sample_match['Date'].strftime('%Y-%m-%d')}"
    )
    print(
        f"Actual Score: {int(sample_match['FTHG'])} - {int(sample_match['FTAG'])} (Total: {int(sample_match['total_goals'])})"
    )
    print(
        f"Expected Score: {sh_exp:.2f} - {sa_exp:.2f} (Expected Total: {sh_exp + sa_exp:.2f})"
    )
    print(
        f"Predicted Over 2.5: {'Yes' if (sh_exp + sa_exp) > 2.5 else 'No'} | Actual Over 2.5: {'Yes' if sample_match['over_2_5'] == 1 else 'No'}"
    )

    # Save artifact containing both fitted Poisson models
    models_dir = os.path.join("models")
    os.makedirs(models_dir, exist_ok=True)
    goals_model_path = os.path.join(models_dir, "goals_model_v01.joblib")
    joblib.dump(
        {"home_model": model_home, "away_model": model_away}, goals_model_path
    )
    print(f"\nGoals model artifact successfully saved to: {goals_model_path}")


if __name__ == "__main__":
    train_and_evaluate_goals()