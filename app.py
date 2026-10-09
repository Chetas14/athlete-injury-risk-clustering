import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Athlete Injury Risk Analytics",
    page_icon="🏃",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 20px;
}

.login-box {
    max-width: 450px;
    margin: auto;
    padding: 35px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.10);
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATABASE
# ==========================================================

def create_database():

    conn = sqlite3.connect("users.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)

    password = hashlib.sha256(
        "admin123".encode()
    ).hexdigest()

    cursor.execute("""
        INSERT OR IGNORE INTO users
        (username, password)
        VALUES (?, ?)
    """, ("admin", password))

    conn.commit()
    conn.close()


def verify_login(username, password):

    conn = sqlite3.connect("users.db")

    cursor = conn.cursor()

    hashed_password = hashlib.sha256(
        password.encode()
    ).hexdigest()

    cursor.execute("""
        SELECT * FROM users
        WHERE username = ? AND password = ?
    """, (username, hashed_password))

    result = cursor.fetchone()

    conn.close()

    return result is not None


create_database()


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    athletes = pd.read_csv(
        "clustered_athletes.csv"
    )

    raw_data = pd.read_csv(
        "multimodal_sports_injury_dataset.csv"
    )

    risk_profiles = pd.read_csv(
        "cluster_risk_profiles.csv"
    )

    return athletes, raw_data, risk_profiles


athletes, raw_data, risk_profiles = load_data()


# ==========================================================
# SESSION STATE
# ==========================================================

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


# ==========================================================
# LOGIN PAGE
# ==========================================================

if not st.session_state.logged_in:

    st.markdown(
        "<h1 style='text-align:center;'>🏃 Athlete Injury Risk Analytics</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#666;'>"
        "Hierarchical Clustering of Athlete Biomechanical Load Data"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    with st.container():

        st.markdown(
            "<div class='login-box'>",
            unsafe_allow_html=True
        )

        st.subheader("🔐 Login")

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login_button = st.button(
            "Login",
            use_container_width=True
        )

        if login_button:

            if verify_login(
                username,
                password
            ):

                st.session_state.logged_in = True

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🏃 Athlete Analytics")

st.sidebar.write(
    "Hierarchical Clustering System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Athlete Data",
        "Clustering",
        "Risk Profiles",
        "Injury Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Project ID: SH26066"
)

if st.sidebar.button(
    "Logout",
    use_container_width=True
):

    st.session_state.logged_in = False

    st.rerun()


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.markdown(
        "<div class='dashboard-title'>🏃 Athlete Injury Risk Analytics</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='dashboard-subtitle'>"
        "Hierarchical Clustering of Athlete Biomechanical Load Data "
        "to Flag Distinct Injury-Risk Profiles"
        "</div>",
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    total_athletes = athletes["athlete_id"].nunique()

    total_sessions = len(raw_data)

    cluster_1_count = len(
        athletes[athletes["cluster"] == 1]
    )

    cluster_2_count = len(
        athletes[athletes["cluster"] == 2]
    )

    injury_rate_1 = risk_profiles.loc[
        risk_profiles["cluster"] == 1,
        "injury_rate"
    ].iloc[0] * 100

    injury_rate_2 = risk_profiles.loc[
        risk_profiles["cluster"] == 2,
        "injury_rate"
    ].iloc[0] * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Athletes",
            total_athletes
        )

    with col2:

        st.metric(
            "Total Sessions",
            f"{total_sessions:,}"
        )

    with col3:

        st.metric(
            "Cluster 1",
            cluster_1_count
        )

    with col4:

        st.metric(
            "Cluster 2",
            cluster_2_count
        )

    st.divider()

    # ------------------------------------------------------
    # PROJECT OVERVIEW
    # ------------------------------------------------------

    st.subheader("📌 Project Overview")

    st.write(
        """
        This system groups athletes into distinct profiles based on
        biomechanical load, movement characteristics, training volume,
        fatigue and recovery-related features.

        Hierarchical clustering is used to discover natural similarities
        between athletes. Historical injury information is then used to
        validate whether the discovered clusters show different injury
        patterns.
        """
    )

    # ------------------------------------------------------
    # CLUSTER DISTRIBUTION
    # ------------------------------------------------------

    st.subheader("📊 Cluster Distribution")

    cluster_counts = athletes["cluster"].value_counts().sort_index()

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            cluster_counts
        )

    with col2:

        cluster_table = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Athletes": cluster_counts.values,
            "Percentage": (
                cluster_counts.values /
                total_athletes * 100
            ).round(2)
        })

        st.dataframe(
            cluster_table,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ------------------------------------------------------
    # INJURY RATE
    # ------------------------------------------------------

    st.subheader("🩹 Historical Injury Rate")

    injury_data = pd.DataFrame({
        "Cluster 1": [injury_rate_1],
        "Cluster 2": [injury_rate_2]
    }, index=["Injury Rate (%)"])

    st.bar_chart(
        injury_data.T
    )

    higher_cluster = (
        "Cluster 1"
        if injury_rate_1 > injury_rate_2
        else "Cluster 2"
    )

    st.info(
        f"{higher_cluster} shows the higher historical injury rate."
    )


# ==========================================================
# ATHLETE DATA
# ==========================================================

elif page == "Athlete Data":

    st.title("👤 Athlete Data")

    st.write(
        "Explore individual athlete-level features and cluster assignments."
    )

    # ------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------

    search_id = st.text_input(
        "Search Athlete ID"
    )

    if search_id:

        filtered = athletes[
            athletes["athlete_id"]
            .astype(str)
            .str.contains(
                search_id,
                case=False,
                na=False
            )
        ]

    else:

        filtered = athletes

    st.write(
        f"Showing {len(filtered)} athlete(s)"
    )

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500
    )

    # ------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Athlete Data",
        data=csv_data,
        file_name="athlete_data.csv",
        mime="text/csv"
    )


# ==========================================================
# CLUSTERING
# ==========================================================

elif page == "Clustering":

    st.title("🔗 Hierarchical Clustering")

    st.write(
        "Explore hierarchical clustering using different linkage methods."
    )

    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    features = [
        "avg_ground_reaction_force",
        "avg_muscle_activity",
        "avg_jump_height",
        "avg_range_of_motion",
        "joint_angle_variability",
        "rom_variability",
        "avg_gait_speed",
        "avg_cadence",
        "total_steps",
        "avg_training_intensity",
        "total_training_duration",
        "avg_training_load",
        "avg_recovery_score",
        "avg_stress_level"
    ]

    X = athletes[features].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------
    # LINKAGE
    # ------------------------------------------------------

    linkage_method = st.selectbox(
        "Select Linkage Method",
        [
            "Single",
            "Complete",
            "Average",
            "Ward"
        ]
    )

    method_map = {
        "Single": "single",
        "Complete": "complete",
        "Average": "average",
        "Ward": "ward"
    }

    method = method_map[
        linkage_method
    ]

    Z = linkage(
        X_scaled,
        method=method
    )

    # ------------------------------------------------------
    # DENDROGRAM
    # ------------------------------------------------------

    st.subheader(
        "🌳 Hierarchical Clustering Dendrogram"
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=20,
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax
    )

    ax.set_title(
        f"{linkage_method} Linkage Dendrogram"
    )

    ax.set_xlabel(
        "Cluster / Sample"
    )

    ax.set_ylabel(
        "Distance"
    )

    st.pyplot(
        fig
    )

    plt.close(fig)

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    st.subheader(
        "📍 PCA Cluster Visualization"
    )

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X_scaled
    )

    pca_df = pd.DataFrame(
        X_pca,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    pca_df["Cluster"] = (
        athletes["cluster"]
        .astype(str)
        .values
    )

    st.scatter_chart(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster"
    )

    # ------------------------------------------------------
    # METHOD INFORMATION
    # ------------------------------------------------------

    st.subheader(
        "📋 Linkage Method"
    )

    method_descriptions = {

        "Single":
            "Uses the minimum distance between observations.",

        "Complete":
            "Uses the maximum distance between observations.",

        "Average":
            "Uses the average pairwise distance between observations.",

        "Ward":
            "Minimizes the increase in within-cluster variance."
    }

    st.info(
        method_descriptions[
            linkage_method
        ]
    )


# ==========================================================
# RISK PROFILES
# ==========================================================

elif page == "Risk Profiles":

    st.title("⚠️ Cluster Risk Profiles")

    st.write(
        "Comparison of biomechanical, movement, training "
        "and recovery characteristics across clusters."
    )

    cluster1 = risk_profiles[
        risk_profiles["cluster"] == 1
    ].iloc[0]

    cluster2 = risk_profiles[
        risk_profiles["cluster"] == 2
    ].iloc[0]

    # ------------------------------------------------------
    # CLUSTER CARDS
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔴 Cluster 1")

        st.metric(
            "Athletes",
            int(cluster1["athletes"])
        )

        st.metric(
            "Historical Injury Rate",
            f"{cluster1['injury_rate'] * 100:.2f}%"
        )

        st.metric(
            "Training Load",
            f"{cluster1['avg_training_load']:.2f}"
        )

        st.metric(
            "Fatigue Index",
            f"{cluster1['avg_fatigue_index']:.2f}"
        )

    with col2:

        st.subheader("🔵 Cluster 2")

        st.metric(
            "Athletes",
            int(cluster2["athletes"])
        )

        st.metric(
            "Historical Injury Rate",
            f"{cluster2['injury_rate'] * 100:.2f}%"
        )

        st.metric(
            "Training Load",
            f"{cluster2['avg_training_load']:.2f}"
        )

        st.metric(
            "Fatigue Index",
            f"{cluster2['avg_fatigue_index']:.2f}"
        )

    st.divider()

    # ------------------------------------------------------
    # FEATURE COMPARISON
    # ------------------------------------------------------

    st.subheader(
        "📊 Cluster Feature Comparison"
    )

    comparison = pd.DataFrame({

        "Feature": [

            "Ground Reaction Force",
            "Muscle Activity",
            "Jump Height",
            "Range of Motion",
            "Joint Angle Variability",
            "ROM Variability",
            "Gait Speed",
            "Cadence",
            "Training Intensity",
            "Training Duration",
            "Training Load",
            "Fatigue Index",
            "Recovery Score",
            "Sleep Quality",
            "Stress Level"
        ],

        "Cluster 1": [

            cluster1["avg_ground_reaction_force"],
            cluster1["avg_muscle_activity"],
            cluster1["avg_jump_height"],
            cluster1["avg_range_of_motion"],
            cluster1["joint_angle_variability"],
            cluster1["rom_variability"],
            cluster1["avg_gait_speed"],
            cluster1["avg_cadence"],
            cluster1["avg_training_intensity"],
            cluster1["total_training_duration"],
            cluster1["avg_training_load"],
            cluster1["avg_fatigue_index"],
            cluster1["avg_recovery_score"],
            cluster1["avg_sleep_quality"],
            cluster1["avg_stress_level"]
        ],

        "Cluster 2": [

            cluster2["avg_ground_reaction_force"],
            cluster2["avg_muscle_activity"],
            cluster2["avg_jump_height"],
            cluster2["avg_range_of_motion"],
            cluster2["joint_angle_variability"],
            cluster2["rom_variability"],
            cluster2["avg_gait_speed"],
            cluster2["avg_cadence"],
            cluster2["avg_training_intensity"],
            cluster2["total_training_duration"],
            cluster2["avg_training_load"],
            cluster2["avg_fatigue_index"],
            cluster2["avg_recovery_score"],
            cluster2["avg_sleep_quality"],
            cluster2["avg_stress_level"]
        ]
    })

    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ------------------------------------------------------
    # TRAINING & FATIGUE
    # ------------------------------------------------------

    st.subheader(
        "🏋️ Training & Fatigue Comparison"
    )

    training_data = pd.DataFrame({

        "Cluster 1": [

            cluster1["avg_training_load"],
            cluster1["avg_fatigue_index"],
            cluster1["avg_training_intensity"]
        ],

        "Cluster 2": [

            cluster2["avg_training_load"],
            cluster2["avg_fatigue_index"],
            cluster2["avg_training_intensity"]
        ]

    }, index=[

        "Training Load",
        "Fatigue Index",
        "Training Intensity"

    ])

    st.bar_chart(
        training_data
    )

    # ------------------------------------------------------
    # RECOVERY
    # ------------------------------------------------------

    st.subheader(
        "😴 Recovery Comparison"
    )

    recovery_data = pd.DataFrame({

        "Cluster 1": [

            cluster1["avg_recovery_score"],
            cluster1["avg_sleep_quality"],
            cluster1["avg_stress_level"]
        ],

        "Cluster 2": [

            cluster2["avg_recovery_score"],
            cluster2["avg_sleep_quality"],
            cluster2["avg_stress_level"]
        ]

    }, index=[

        "Recovery Score",
        "Sleep Quality",
        "Stress Level"

    ])

    st.bar_chart(
        recovery_data
    )

    # ------------------------------------------------------
    # INJURY RATE
    # ------------------------------------------------------

    st.subheader(
        "🩹 Historical Injury Rate"
    )

    injury_data = pd.DataFrame({

        "Injury Rate (%)": [

            cluster1["injury_rate"] * 100,
            cluster2["injury_rate"] * 100

        ]

    }, index=[

        "Cluster 1",
        "Cluster 2"

    ])

    st.bar_chart(
        injury_data
    )

    st.divider()

    # ------------------------------------------------------
    # HIGHER INJURY RATE
    # ------------------------------------------------------

    if cluster1["injury_rate"] > cluster2["injury_rate"]:

        st.warning(

            f"Cluster 1 shows the higher historical injury rate "
            f"({cluster1['injury_rate'] * 100:.2f}%) compared with "
            f"Cluster 2 ({cluster2['injury_rate'] * 100:.2f}%)."

        )

    else:

        st.info(

            f"Cluster 2 shows the higher historical injury rate "
            f"({cluster2['injury_rate'] * 100:.2f}%) compared with "
            f"Cluster 1 ({cluster1['injury_rate'] * 100:.2f}%)."

        )


# ==========================================================
# INJURY ANALYSIS
# ==========================================================

elif page == "Injury Analysis":

    st.title("🩹 Injury Analysis")

    st.write(
        "Historical injury information used to validate the discovered clusters."
    )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    total_injury_sessions = int(
        risk_profiles["injury_sessions"].sum()
    )

    average_injury_rate = (
        athletes["injury_rate"].mean() * 100
    )

    max_injury_level = int(
        athletes["max_injury_level"].max()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Injury Sessions",
            f"{total_injury_sessions:,}"
        )

    with col2:

        st.metric(
            "Average Athlete Injury Rate",
            f"{average_injury_rate:.2f}%"
        )

    with col3:

        st.metric(
            "Maximum Injury Level",
            max_injury_level
        )

    st.divider()

    # ------------------------------------------------------
    # CLUSTER VALIDATION TABLE
    # ------------------------------------------------------

    st.subheader(
        "📊 Injury Validation by Cluster"
    )

    validation = pd.DataFrame({

        "Cluster": risk_profiles["cluster"],

        "Athletes": risk_profiles["athletes"],

        "Total Injury Sessions":
            risk_profiles["injury_sessions"],

        "Average Injury Rate (%)":
            risk_profiles["injury_rate"] * 100,

        "Maximum Injury Level":
            risk_profiles["max_injury_level"]

    })

    validation[
        "Average Injury Rate (%)"
    ] = validation[
        "Average Injury Rate (%)"
    ].round(2)

    st.dataframe(
        validation,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # INJURY RATE CHART
    # ------------------------------------------------------

    st.subheader(
        "📈 Injury Rate by Cluster"
    )

    injury_chart = pd.DataFrame({

        "Injury Rate (%)": [

            risk_profiles.loc[
                risk_profiles["cluster"] == 1,
                "injury_rate"
            ].iloc[0] * 100,

            risk_profiles.loc[
                risk_profiles["cluster"] == 2,
                "injury_rate"
            ].iloc[0] * 100

        ]

    }, index=[

        "Cluster 1",
        "Cluster 2"

    ])

    st.bar_chart(
        injury_chart
    )

    # ------------------------------------------------------
    # TOTAL INJURY SESSIONS
    # ------------------------------------------------------

    st.subheader(
        "🩹 Total Injury Sessions"
    )

    injury_sessions = pd.DataFrame({

        "Injury Sessions": [

            risk_profiles.loc[
                risk_profiles["cluster"] == 1,
                "injury_sessions"
            ].iloc[0],

            risk_profiles.loc[
                risk_profiles["cluster"] == 2,
                "injury_sessions"
            ].iloc[0]

        ]

    }, index=[

        "Cluster 1",
        "Cluster 2"

    ])

    st.bar_chart(
        injury_sessions
    )

    # ------------------------------------------------------
    # HIGHER INJURY RATE
    # ------------------------------------------------------

    cluster1_rate = risk_profiles.loc[
        risk_profiles["cluster"] == 1,
        "injury_rate"
    ].iloc[0]

    cluster2_rate = risk_profiles.loc[
        risk_profiles["cluster"] == 2,
        "injury_rate"
    ].iloc[0]

    difference = abs(
        cluster1_rate - cluster2_rate
    ) * 100

    if cluster1_rate > cluster2_rate:

        st.warning(

            f"Cluster 1 has the higher historical injury rate: "
            f"{cluster1_rate * 100:.2f}% vs "
            f"{cluster2_rate * 100:.2f}%. "
            f"The difference is {difference:.2f} percentage points."

        )

    else:

        st.info(

            f"Cluster 2 has the higher historical injury rate: "
            f"{cluster2_rate * 100:.2f}% vs "
            f"{cluster1_rate * 100:.2f}%. "
            f"The difference is {difference:.2f} percentage points."

        )

    st.caption(
        "Historical injury records are used for post-clustering validation "
        "and are not used as clustering input features."
    )

    # ------------------------------------------------------
    # DOWNLOAD
    # ------------------------------------------------------

    st.divider()

    csv_data = validation.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Injury Analysis",
        data=csv_data,
        file_name="injury_validation.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Athlete Injury Risk Analytics | Hierarchical Clustering | Project SH26066"
)