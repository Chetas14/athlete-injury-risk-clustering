import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster


# Load athlete-level dataset
df = pd.read_csv("athlete_level_features.csv")


# Features used for clustering
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


# Select features
X = df[features]


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Complete linkage
linkage_matrix = linkage(
    X_scaled,
    method="complete"
)


# Create 2 clusters
clusters = fcluster(
    linkage_matrix,
    t=2,
    criterion="maxclust"
)


# Add cluster labels
df["cluster"] = clusters


# Display cluster sizes
print("\nFINAL CLUSTER SIZES")
print(df["cluster"].value_counts().sort_index())


# Save final dataset
df.to_csv(
    "clustered_athletes.csv",
    index=False
)


print("\nFinal clustered dataset saved as:")
print("clustered_athletes.csv")