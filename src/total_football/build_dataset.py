import glob
import os
import pandas as pd


def load_and_merge_data(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "EPL_*.csv")))
    if not files:
        raise FileNotFoundError(f"No EPL CSV files found in {raw_dir}")

    dfs = []
    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

    for f in files:
        df = pd.read_csv(f)
        df = df[required_cols].copy()
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    combined = combined.dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(
        drop=True
    )
    return combined


def compute_rolling_features(df):
    home_df = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
    home_df.columns = ["Date", "Team", "GF", "GA", "FTR"]
    home_df["IsHome"] = 1
    home_df["Points"] = home_df["FTR"].map({"H": 3, "D": 1, "A": 0})

    away_df = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR"]].copy()
    away_df.columns = ["Date", "Team", "GF", "GA", "FTR"]
    away_df["IsHome"] = 0
    away_df["Points"] = away_df["FTR"].map({"A": 3, "D": 1, "H": 0})

    team_matches = pd.concat([home_df, away_df], ignore_index=True)
    team_matches = team_matches.sort_values(["Date"]).reset_index(drop=True)

    for stat in ["Points", "GF", "GA"]:
        team_matches[f"{stat.lower()}_last_5"] = team_matches.groupby("Team")[
            stat
        ].transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())

    team_matches["venue_points_last_5"] = team_matches.groupby(
        ["Team", "IsHome"]
    )["Points"].transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())

    home_stats = team_matches[team_matches["IsHome"] == 1].copy()
    away_stats = team_matches[team_matches["IsHome"] == 0].copy()

    rename_home = {
        "points_last_5": "home_points_last_5",
        "gf_last_5": "home_goals_for_last_5",
        "ga_last_5": "home_goals_against_last_5",
        "venue_points_last_5": "home_home_points_last_5",
    }
    rename_away = {
        "points_last_5": "away_points_last_5",
        "gf_last_5": "away_goals_for_last_5",
        "ga_last_5": "away_goals_against_last_5",
        "venue_points_last_5": "away_away_points_last_5",
    }

    df_merged = (
        df.merge(
            home_stats[["Date", "Team"] + list(rename_home.keys())],
            left_on=["Date", "HomeTeam"],
            right_on=["Date", "Team"],
            how="left",
        )
        .drop(columns=["Team"])
        .rename(columns=rename_home)
    )

    df_merged = (
        df_merged.merge(
            away_stats[["Date", "Team"] + list(rename_away.keys())],
            left_on=["Date", "AwayTeam"],
            right_on=["Date", "Team"],
            how="left",
        )
        .drop(columns=["Team"])
        .rename(columns=rename_away)
    )

    df_merged = df_merged.dropna().reset_index(drop=True)
    return df_merged


if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw files...")
    raw_data = load_and_merge_data(raw_dir)
    print(f"Total raw matches: {len(raw_data)}")

    print("Building rolling features (leakage-safe)...")
    processed_data = compute_rolling_features(raw_data)
    print(
        f"Total valid training rows with full 5-match history: {len(processed_data)}"
    )

    out_path = os.path.join(processed_dir, "epl_processed_v01.csv")
    processed_data.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to: {out_path}")