import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Total Football — Match Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Total Football — Winner Model v0.1")
st.markdown("Predict English Premier League match outcomes using leakage-safe historical rolling stats.")

# 1. Load Data & Saved Model Pipeline
@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "epl_processed_v01.csv")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "winner_model_v01.joblib")
    return joblib.load(model_path)

try:
    df = load_data()
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Get unique team list sorted
teams = sorted(df["HomeTeam"].unique())

st.divider()
st.subheader("Select Match Fixture")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", teams, index=0)
with col2:
    away_teams = [t for t in teams if t != home_team]
    away_team = st.selectbox("Away Team", away_teams, index=0)

if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    # Fetch latest rolling form for selected teams
    latest_home = df[df["HomeTeam"] == home_team].sort_values("Date").iloc[-1]
    latest_away = df[df["AwayTeam"] == away_team].sort_values("Date").iloc[-1]

    # Prepare feature input
    input_data = pd.DataFrame([{
        "home_points_last_5": latest_home["home_points_last_5"],
        "away_points_last_5": latest_away["away_points_last_5"],
        "home_goals_for_last_5": latest_home["home_goals_for_last_5"],
        "away_goals_for_last_5": latest_away["away_goals_for_last_5"],
        "home_goals_against_last_5": latest_home["home_goals_against_last_5"],
        "away_goals_against_last_5": latest_away["away_goals_against_last_5"],
        "home_home_points_last_5": latest_home["home_home_points_last_5"],
        "away_away_points_last_5": latest_away["away_away_points_last_5"],
    }])

    # Generate probabilities
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    classes = pipeline.named_steps["logisticregression"].classes_
    proba_dict = dict(zip(classes, probabilities))

    label_map = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}
    
    st.divider()
    st.subheader(f"Prediction: **{label_map.get(prediction, prediction)}**")

    # Display Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric(f"🏠 {home_team} Win", f"{proba_dict.get('H', 0):.1%}")
    m2.metric("🤝 Draw", f"{proba_dict.get('D', 0):.1%}")
    m3.metric(f"🚀 {away_team} Win", f"{proba_dict.get('A', 0):.1%}")

    # Display Recent Rolling Form Context
    st.write("---")
    st.subheader("Recent 5-Match Form Inputs")
    st.json({
        f"{home_team} (Home)": {
            "Points Last 5": int(latest_home["home_points_last_5"]),
            "Goals For Last 5": int(latest_home["home_goals_for_last_5"]),
            "Goals Against Last 5": int(latest_home["home_goals_against_last_5"]),
            "Home Form Points": int(latest_home["home_home_points_last_5"]),
        },
        f"{away_team} (Away)": {
            "Points Last 5": int(latest_away["away_points_last_5"]),
            "Goals For Last 5": int(latest_away["away_goals_for_last_5"]),
            "Goals Against Last 5": int(latest_away["away_goals_against_last_5"]),
            "Away Form Points": int(latest_away["away_away_points_last_5"]),
        }
    })