import pandas as pd


# Load final clustered data
df = pd.read_csv("clustered_athletes.csv")


# Features for risk profiling
profile_features = [
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
    "avg_fatigue_index",
    "avg_recovery_score",
    "avg_sleep_quality",
    "avg_stress_level"
]


# Calculate average feature values for each cluster
profile = df.groupby("cluster")[profile_features].mean()


# Add injury information
injury_profile = df.groupby("cluster").agg(
    athletes=("athlete_id", "count"),
    injury_sessions=("injury_sessions", "sum"),
    injury_rate=("injury_rate", "mean"),
    max_injury_level=("max_injury_level", "mean")
)


# Combine both tables
profile = profile.join(injury_profile)


# Display profile
print("\nCLUSTER RISK PROFILES")
print(profile.round(3).to_string())


# Save profile
profile.to_csv("cluster_risk_profiles.csv")


print("\nRisk profile saved as:")
print("cluster_risk_profiles.csv")