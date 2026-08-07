import os

import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Total Football — Decision Engine v1.0",
    page_icon="⚽",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #31354a;
    }
    .decision-box {
        background-color: #15202b;
        border-left: 5px solid #1da1f2;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚽ Total Football — Central Decision Engine (v1.0)")
st.caption(
    "Multi-model match analytics combining Winner Class Probabilities with Poisson Goal Expectations."
)


# 1. Cache Data & Models
@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "epl_processed_v01.csv")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource
def load_models():
    winner_path = os.path.join("models", "winner_model_v01.joblib")
    goals_path = os.path.join("models", "goals_model_v01.joblib")

    winner_model = joblib.load(winner_path)
    goals_model = joblib.load(goals_path)

    return winner_model, goals_model


try:
    df = load_data()
    winner_pipeline, goals_pipeline = load_models()
except Exception as e:
    st.error(
        f"Error loading models or dataset. Ensure both winner and goals models are trained: {e}"
    )
    st.stop()

teams = sorted(df["HomeTeam"].unique())

st.divider()
st.subheader("Select Fixture")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox(
        "🏠 Home Team",
        teams,
        index=teams.index("Arsenal") if "Arsenal" in teams else 0,
    )
with col2:
    away_options = [t for t in teams if t != home_team]
    away_team = st.selectbox(
        "🚀 Away Team",
        away_options,
        index=away_options.index("Aston Villa")
        if "Aston Villa" in away_options
        else 0,
    )

if st.button(
    "⚡ Generate Multi-Model Analysis", type="primary", use_container_width=True
):
    # Fetch latest rolling form for selected teams
    latest_home = df[df["HomeTeam"] == home_team].sort_values("Date").iloc[-1]
    latest_away = df[df["AwayTeam"] == away_team].sort_values("Date").iloc[-1]

    input_data = pd.DataFrame(
        [
            {
                "home_points_last_5": latest_home["home_points_last_5"],
                "away_points_last_5": latest_away["away_points_last_5"],
                "home_goals_for_last_5": latest_home["home_goals_for_last_5"],
                "away_goals_for_last_5": latest_away["away_goals_for_last_5"],
                "home_goals_against_last_5": latest_home[
                    "home_goals_against_last_5"
                ],
                "away_goals_against_last_5": latest_away[
                    "away_goals_against_last_5"
                ],
                "home_home_points_last_5": latest_home[
                    "home_home_points_last_5"
                ],
                "away_away_points_last_5": latest_away[
                    "away_away_points_last_5"
                ],
            }
        ]
    )

    # Model 1 Inference: Winner Probabilities
    winner_pred = winner_pipeline.predict(input_data)[0]
    winner_proba = winner_pipeline.predict_proba(input_data)[0]
    classes = winner_pipeline.named_steps["logisticregression"].classes_
    proba_dict = dict(zip(classes, winner_proba))

    p_home = proba_dict.get("H", 0)
    p_draw = proba_dict.get("D", 0)
    p_away = proba_dict.get("A", 0)

    # Model 2 Inference: Expected Goals
    exp_home = goals_pipeline["home_model"].predict(input_data)[0]
    exp_away = goals_pipeline["away_model"].predict(input_data)[0]
    exp_total = exp_home + exp_away
    over_2_5 = exp_total > 2.5

    # -------------------------------------------------------------
    # CENTRAL DECISION ENGINE (Multi-Model Synthesis Logic)
    # -------------------------------------------------------------
    st.divider()
    st.header("🎯 Central Decision Engine Recommendation")

    # Determine confidence level and match narrative
    max_prob = max(p_home, p_draw, p_away)

    if max_prob >= 0.55:
        confidence = "HIGH CONFIDENCE"
        conf_color = "🟢"
    elif max_prob >= 0.42:
        confidence = "MODERATE CONFIDENCE"
        conf_color = "🟡"
    else:
        confidence = "LOW CONFIDENCE / UNPREDICTABLE"
        conf_color = "🔴"

    # Synthesis rules
    if winner_pred == "H" and over_2_5:
        narrative = f"Strong home advantage for **{home_team}** in an open, high-scoring match."
        betting_insight = f"Primary: **{home_team} Win** | Secondary: **Over 2.5 Goals**"
    elif winner_pred == "H" and not over_2_5:
        narrative = f"Narrow home win expected for **{home_team}** in a tight, low-scoring fixture."
        betting_insight = f"Primary: **{home_team} Win** | Secondary: **Under 2.5 Goals**"
    elif winner_pred == "A" and over_2_5:
        narrative = f"Away dominance expected for **{away_team}** with strong offensive output."
        betting_insight = f"Primary: **{away_team} Win** | Secondary: **Over 2.5 Goals**"
    elif winner_pred == "A" and not over_2_5:
        narrative = f"Low-margin away victory projected for **{away_team}**."
        betting_insight = f"Primary: **{away_team} Win** | Secondary: **Under 2.5 Goals**"
    else:
        narrative = f"Tight tactical battle with significant probability of a draw."
        betting_insight = (
            f"Primary: **Draw / Double Chance** | Secondary: **Under 2.5 Goals**"
        )

    # Display Decision Summary Cards
    d1, d2, d3 = st.columns(3)
    d1.metric("Overall Prediction", f"{winner_pred} ({confidence})")
    d2.metric("Projected Scoreline", f"{exp_home:.1f} - {exp_away:.1f}")
    d3.metric(
        "Over/Under 2.5 Goals", "OVER 2.5" if over_2_5 else "UNDER 2.5"
    )

    st.markdown(
        f"""
        <div class="decision-box">
            <h4>{conf_color} <b>Engine Insight:</b> {narrative}</h4>
            <p><b>Recommended Market Strategy:</b> {betting_insight}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # SECTION 1: Winner Model Breakdown
    # -------------------------------------------------------------
    st.header("1. Winner Model Analysis (Logistic Regression)")
    col_h, col_d, col_a = st.columns(3)
    col_h.metric(label=f"🏠 {home_team} Win", value=f"{p_home:.1%}")
    col_d.metric(label="🤝 Draw", value=f"{p_draw:.1%}")
    col_a.metric(label=f"🚀 {away_team} Win", value=f"{p_away:.1%}")

    st.progress(p_home, text=f"{home_team} Win Probability Gauge")

    # -------------------------------------------------------------
    # SECTION 2: Goals Model Breakdown
    # -------------------------------------------------------------
    st.header("2. Expected Goals Analysis (Poisson Model)")
    g1, g2, g3 = st.columns(3)
    g1.metric(f"🏠 Expected {home_team} Goals", f"{exp_home:.2f}")
    g2.metric(f"🚀 Expected {away_team} Goals", f"{exp_away:.2f}")
    g3.metric("🎯 Total Expected Goals", f"{exp_total:.2f}")

    # -------------------------------------------------------------
    # SECTION 3: Underlying Input Data
    # -------------------------------------------------------------
    st.divider()
    st.subheader("📊 5-Match Rolling Form Inputs (Leakage-Safe)")
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Points (Last 5)",
                "Goals Scored (Last 5)",
                "Goals Conceded (Last 5)",
                "Venue Form Points",
            ],
            f"🏠 {home_team}": [
                int(latest_home["home_points_last_5"]),
                int(latest_home["home_goals_for_last_5"]),
                int(latest_home["home_goals_against_last_5"]),
                int(latest_home["home_home_points_last_5"]),
            ],
            f"🚀 {away_team}": [
                int(latest_away["away_points_last_5"]),
                int(latest_away["away_goals_for_last_5"]),
                int(latest_away["away_goals_against_last_5"]),
                int(latest_away["away_away_points_last_5"]),
            ],
        }
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)