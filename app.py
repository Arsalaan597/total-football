import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Total Football — Decision Engine",
    page_icon="⚽",
    layout="wide",
)

# Custom styling for metric cards
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #31354a;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚽ Total Football — Decision Engine (v1.0)")
st.caption(
    "Multi-model match analytics platform integrating Winner Prediction and Expected Goals models."
)


# 1. Load Data & Saved Model Artifacts
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

    # 1. Winner Model Inference
    winner_pred = winner_pipeline.predict(input_data)[0]
    winner_proba = winner_pipeline.predict_proba(input_data)[0]
    classes = winner_pipeline.named_steps["logisticregression"].classes_
    proba_dict = dict(zip(classes, winner_proba))

    # 2. Goals Model Inference
    exp_home = goals_pipeline["home_model"].predict(input_data)[0]
    exp_away = goals_pipeline["away_model"].predict(input_data)[0]
    exp_total = exp_home + exp_away
    over_2_5_label = "OVER 2.5 GOALS" if exp_total > 2.5 else "UNDER 2.5 GOALS"

    st.divider()

    # SECTION 1: Winner Predictions
    st.header(f"1. Match Winner Analysis")
    col_h, col_d, col_a = st.columns(3)
    col_h.metric(
        label=f"🏠 {home_team} Win", value=f"{proba_dict.get('H', 0):.1%}"
    )
    col_d.metric(label="🤝 Draw", value=f"{proba_dict.get('D', 0):.1%}")
    col_a.metric(
        label=f"🚀 {away_team} Win", value=f"{proba_dict.get('A', 0):.1%}"
    )

    st.progress(
        proba_dict.get("H", 0), text=f"{home_team} Win Probability Gauge"
    )

    st.write("")

    # SECTION 2: Goals & Total Expectations
    st.header(f"2. Goal Expectation Analysis (Poisson Model)")
    g1, g2, g3 = st.columns(3)
    g1.metric(f"🏠 Expected {home_team} Goals", f"{exp_home:.2f}")
    g2.metric(f"🚀 Expected {away_team} Goals", f"{exp_away:.2f}")
    g3.metric("🎯 Projected Total Goals", f"{exp_total:.2f}")

    if exp_total > 2.5:
        st.success(
            f"🔥 **Over/Under 2.5 Projection:** {over_2_5_label} ({exp_total:.2f} expected goals)"
        )
    else:
        st.warning(
            f"🔒 **Over/Under 2.5 Projection:** {over_2_5_label} ({exp_total:.2f} expected goals)"
        )

    st.divider()

    # SECTION 3: Head-to-Head Form Summary
    st.subheader("📊 5-Match Rolling Form Inputs")
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