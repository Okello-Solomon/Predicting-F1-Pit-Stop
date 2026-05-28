import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="F1 Pit Strategy Prediction System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: red;
}

/* Sidebar radio labels */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: red !important;
}

/* Selected option */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
    color: darkred !important;
    font-weight: 800 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
pipeline = joblib.load("xgb_pipeline.pkl")
race_te_mapping = joblib.load("race_te.pkl")
laptime_delta_min = joblib.load("laptime_delta_min.pkl")

# =========================================================
# SIDEBAR MENU (ONLY 2 OPTIONS)
# =========================================================
st.sidebar.markdown("## Navigation")

menu = st.sidebar.radio(
    "",
    [
        "Prediction System",
        "Power BI Visualization"
    ]
)

# =========================================================
# ================= PREDICTION SYSTEM =====================
# =========================================================
if menu == "Prediction System":

    st.markdown(
        "<h1 style='color:red'>🏎️ Formula 1 Next-Lap Pit Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    Predict whether a Formula 1 driver will pit on the next lap using telemetry and race strategy data.
    
    **Target:**
    - 0 = No Pit Stop  
    - 1 = Pit Stop
    """)

    # =====================================================
    # SIDEBAR INPUTS
    # =====================================================
    st.sidebar.header("Race Telemetry Inputs")

    year = st.sidebar.selectbox("Season", [2022, 2023, 2024, 2025])

    pitstop = st.sidebar.selectbox(
        "Current Lap Pit Stop",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    position_change = st.sidebar.slider("Position Change", -10, 10, 0)
    stint = st.sidebar.slider("Stint", 1, 8, 2)
    tyrelife = st.sidebar.slider("Tyre Life", 1, 80, 15)
    lapnumber = st.sidebar.slider("Lap Number", 1, 78, 20)

    laptime_delta = st.sidebar.number_input("Lap Time Delta", value=0.0)

    # =====================================================
    # MAIN INPUTS
    # =====================================================
    st.subheader("Race & Tyre Information")

    col1, col2 = st.columns(2)

    with col1:
        race = st.selectbox("Race", sorted(list(race_te_mapping.keys())))
        race_te = race_te_mapping.get(race, 0.0)

    with col2:
        compound_encoded = st.selectbox(
            "Tyre Compound",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "HARD",
                1: "MEDIUM",
                2: "SOFT",
                3: "INTERMEDIATE"
            }[x]
        )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    stint_log = np.log1p(stint)
    tyrelife_log = np.log1p(tyrelife)
    lapnumber_log = np.log1p(lapnumber)

    laptime_delta_shifted = laptime_delta - laptime_delta_min + 1
    laptime_delta_log = np.log1p(laptime_delta_shifted)

    # =====================================================
    # INPUT DATAFRAME
    # =====================================================
    input_df = pd.DataFrame([{
        "Year": year,
        "PitStop": pitstop,
        "Position_Change": position_change,
        "Stint_log": stint_log,
        "TyreLife_log": tyrelife_log,
        "LapNumber_log": lapnumber_log,
        "LapTime_Delta_shifted": laptime_delta_shifted,
        "LapTime_Delta_log": laptime_delta_log,
        "Race_TE": race_te,
        "Compound_encoded": compound_encoded
    }])

    # =====================================================
    # PREDICTION
    # =====================================================
    if st.button("🏁 Predict Pit Decision"):

        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        pred_class = prediction[0]

        status_map = {0: "No Pit Stop", 1: "Pit Stop"}
        color_map = {0: "#2ecc71", 1: "#e74c3c"}

        st.subheader("Prediction Result")

        st.markdown(f"""
        <div style='
            background-color:{color_map[pred_class]};
            padding:20px;
            border-radius:10px;
            color:white;
            font-size:22px;
            font-weight:bold;
            text-align:center;
        '>
        {status_map[pred_class]}
        </div>
        """, unsafe_allow_html=True)

        prob_no_pit = prediction_proba[0][0]
        prob_pit = prediction_proba[0][1]

        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("No Pit Stop", f"{prob_no_pit*100:.2f}%")

        with col2:
            st.metric("Pit Stop", f"{prob_pit*100:.2f}%")

        st.progress(int(prob_pit * 100))

        if prob_pit > 0.65:
            st.error("High Pit Probability")
        elif prob_pit > 0.35:
            st.warning("Moderate Pit Probability")
        else:
            st.success("Low Pit Probability")
# =========================================================
# ================= POWER BI PAGE =========================
# =========================================================
# =========================
# --- POWER BI VISUALIZATION TAB ---
# =========================
elif menu == "Power BI Visualization":

    st.markdown(
        "<h1 style='color:red'>Formula 1 Pit Strategy Dashboard</h1>",
        unsafe_allow_html=True
    )

    # =========================================================
    # PROJECT OVERVIEW
    # =========================================================
    st.markdown("## Project Overview")

    st.markdown(
        "This interactive Power BI dashboard provides a data-driven analysis of Formula 1 "
        "telemetry and race strategy data to examine the factors influencing next-lap pit "
        "decisions (PitNextLap). The project explores how tire degradation, race progression, "
        "seasonal trends, and tire compounds impact pit timing decisions across multiple "
        "Formula 1 seasons."
    )

    # =========================================================
    # DETAILS SECTION
    # =========================================================
    with st.expander("View Full Dashboard Insights"):

        st.markdown(
            "The dashboard delivers strategic insights into race management, tire performance, "
            "pit timing behavior, and seasonal race patterns through interactive telemetry "
            "analytics and Formula 1 race segmentation. The dataset contains over 439,000 "
            "telemetry records collected across multiple Grand Prix events, racing seasons, "
            "drivers, and tire compounds."
        )

        st.markdown(
            "The analysis highlights critical Formula 1 strategy patterns related to tire wear, "
            "optimal pit windows, compound durability, and race progression, all of which "
            "directly influence race outcomes, team strategy execution, and overall performance."
        )

        # =========================================================
        st.markdown("## Race Strategy & Telemetry Insights")

        st.markdown(
            "Pit stop decisions in Formula 1 are strongly influenced by tire degradation, "
            "lap progression, weather conditions, and race interruptions such as safety cars. "
            "Teams continuously balance tire performance, race pace, and track position when "
            "determining optimal pit timing strategies."
        )

        # =========================================================
        st.markdown("## Key Performance Indicators (KPIs)")

        st.markdown("- Total Laps Analyzed: 439,140")
        st.markdown("- Grand Prix Events: 26 Circuits")
        st.markdown("- Driver Profiles: 887")
        st.markdown("- Average Lap Time: 90.95 Seconds")
        st.markdown("- Average Tyre Life: 14.16 Laps")
        st.markdown("- Average Stint Length: 1.79")
        st.markdown("- Current Pit Stop Rate: 14%")
        st.markdown("- Next-Lap Pit Probability: 20%")

        # =========================================================
        st.markdown("### Pit Stop vs No Pit Stop Distribution (Donut Chart)")

        st.markdown(
            "The telemetry analysis shows that most racing laps do not result in immediate "
            "pit decisions. Approximately 80.1% of laps are classified as No Pit Stop "
            "scenarios, while 19.9% represent laps where drivers are predicted to pit "
            "on the next lap."
        )

        st.markdown(
            "This class imbalance reflects real-world Formula 1 race strategy behavior, "
            "where pit windows occur only during specific phases of the race."
        )

        # =========================================================
        st.markdown("### Next-Lap Pit Decisions by Tire Compound (Treemap)")

        st.markdown(
            "The treemap visualization analyzes how pit stop behavior varies across "
            "different tire compounds including HARD, MEDIUM, SOFT, and INTERMEDIATE tires."
        )

        st.markdown("**Key Insights:**")

        st.markdown(
            "- HARD compounds recorded the highest number of pit decisions at 63.9%, "
            "reflecting their use during long race stints and gradual degradation."
        )

        st.markdown(
            "- MEDIUM tires accounted for 24.4% of pit decisions, supporting balanced "
            "race strategies between durability and pace."
        )

        st.markdown(
            "- SOFT tires represented 8.6% of pit decisions due to their aggressive "
            "short-stint race usage."
        )

        st.markdown(
            "- INTERMEDIATE tires contributed only 3.1% of pit activity and were "
            "primarily associated with wet-weather race conditions."
        )

        st.markdown(
            "Overall, tire durability and degradation remain the dominant factors "
            "driving Formula 1 pit timing decisions."
        )

        # =========================================================
        st.markdown("### Next-Lap Pit Decisions Over Race Progression (Line Chart)")

        st.markdown(
            "This line chart tracks pit stop frequency throughout race progression "
            "using lap numbers to identify strategic pit windows."
        )

        st.markdown(
            "The visualization reveals that pit activity is heavily concentrated "
            "during mid-race phases where tire degradation becomes increasingly "
            "significant."
        )

        st.markdown(
            "Early race laps show minimal pit activity, while late-race pit stops "
            "decline as teams prioritize track position and race completion."
        )

        st.markdown(
            "Additional pit spikes outside normal windows are often triggered by "
            "safety cars, weather changes, and unexpected race incidents."
        )

        # =========================================================
        st.markdown("### Pit Decisions vs Tyre Life (Scatter Plot)")

        st.markdown(
            "The scatter plot examines the relationship between tyre age and "
            "next-lap pit decisions."
        )

        st.markdown(
            "Pit stop activity peaks between 10 and 25 laps of tyre life, "
            "which represents the most strategically optimal pit window "
            "across many race conditions."
        )

        st.markdown(
            "Fresh tires show very low pit probability, while heavily worn tires "
            "demonstrate declining pit density due to significant performance loss."
        )

        st.markdown(
            "The analysis confirms that tire degradation is one of the strongest "
            "drivers of Formula 1 pit strategy decisions."
        )

        # =========================================================
        st.markdown("### Seasonal Pit Decision Trends (Donut Chart)")

        st.markdown(
            "This section evaluates how pit stop behavior changes across Formula 1 "
            "seasons from 2022 to 2025."
        )

        st.markdown("**Seasonal Distribution:**")

        st.markdown("- 2024: 37,538 pit decisions (42.96%)")
        st.markdown("- 2025: 26,418 pit decisions (30.23%)")
        st.markdown("- 2022: 22,117 pit decisions (25.31%)")
        st.markdown("- 2023: 1,308 pit decisions (1.50%)")

        st.markdown(
            "The 2024 season recorded the highest pit stop activity, while 2023 "
            "showed unusually low pit frequency compared to other seasons."
        )

        st.markdown(
            "Seasonal differences likely reflect changes in race conditions, tire "
            "behavior, regulation adjustments, and team strategy execution."
        )

        # =========================================================
        st.markdown("### Pit Decisions Over Laps by Season (Area Chart)")

        st.markdown(
            "This area chart visualizes how pit stop activity evolves throughout "
            "race progression across different Formula 1 seasons."
        )

        st.markdown(
            "The analysis shows that strategic pit windows remain relatively "
            "consistent across seasons, with most pit activity clustering around "
            "similar lap ranges."
        )

        st.markdown(
            "Minor variations between seasons are largely influenced by race-specific "
            "factors such as weather, safety cars, track characteristics, and "
            "tire strategy changes."
        )

        st.markdown(
            "Overall, the findings suggest that Formula 1 pit timing patterns "
            "remain strategically stable despite seasonal variability."
        )

        # =========================================================
        st.markdown("### Dynamic Dashboard Capabilities")

        st.markdown(
            "The dashboard incorporates interactive Power BI features including "
            "dynamic filtering, telemetry segmentation, and DAX-driven calculations "
            "to support flexible race strategy exploration."
        )

        st.markdown("#### Dynamic Metrics")

        st.markdown("Users can dynamically analyze multiple telemetry metrics:")

        st.markdown("- Pit Stop Counts")
        st.markdown("- Average Tyre Life")
        st.markdown("- Average Lap Time")
        st.markdown("- Average Stint Length")
        st.markdown("- Pit Stop Probability")
        st.markdown("- Driver-Level Performance")
        st.markdown("- Race-Level Strategy Trends")

        st.markdown("#### Dynamic Dimensions")

        st.markdown("Users can segment analysis across:")

        st.markdown("- Grand Prix")
        st.markdown("- Racing Season")
        st.markdown("- Tire Compound")
        st.markdown("- Driver")
        st.markdown("- Pit Decision")
        st.markdown("- Lap Number")
        st.markdown("- Stint")
        st.markdown("- Position Change")

        st.markdown(
            "This enables multidimensional Formula 1 race strategy analysis "
            "within a compact and highly interactive dashboard environment."
        )

        # =========================================================
        st.markdown("### Lap Time Distribution Analysis (Histogram)")

        st.markdown(
            "The histogram visualizes the distribution of lap times across all "
            "telemetry records in the dataset."
        )

        st.markdown(
            "Most lap times are concentrated around the global average of "
            "90.95 seconds, although variability exists due to differences "
            "in circuit layouts, tire conditions, weather, and race strategy."
        )

        st.markdown(
            "The distribution is moderately right-skewed, indicating occasional "
            "slower laps caused by pit stops, traffic, incidents, and changing "
            "track conditions."
        )

        # =========================================================
        st.markdown("## Technical Stack")

        st.markdown("- Power BI Desktop")
        st.markdown("- DAX (Dynamic KPIs & Interactive Analytics)")
        st.markdown("- Power Query")
        st.markdown("- Formula 1 Telemetry Analytics")
        st.markdown("- Race Strategy Visualization")

        # =========================================================
        st.markdown("## Data Source")

        st.markdown(
            "The analysis is based on the Formula 1 Strategy Playground Series dataset."
        )

        st.markdown(
            "**Source:** [F1 Strategy Playground Series Dataset on Kaggle]"
            "(https://www.kaggle.com/competitions/playground-series-s6e5/data)"
        )

    # =========================================================
    # DASHBOARD IMAGES
    # =========================================================
    st.subheader("Dashboard Preview")

    st.image("Dashboard 1.png", use_container_width=True)
    st.image("Dashboard 2.png", use_container_width=True)

    st.markdown(
        "**[Power BI Portfolio](https://github.com/Okello-Solomon/powerbi-dashboards/blob/main/README.md)**"
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Formula 1 Next-Lap Pit Decision Prediction System | Power BI & Machine Learning Analytics 🏎️"
)