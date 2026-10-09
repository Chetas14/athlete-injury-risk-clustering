import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Load dataset
df = pd.read_csv("athlete_level_features.csv")

# Final features for clustering
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

# Create linkage matrices
single_linkage = linkage(X_scaled, method="single")
complete_linkage = linkage(X_scaled, method="complete")
average_linkage = linkage(X_scaled, method="average")
ward_linkage = linkage(X_scaled, method="ward")

# Create dendrograms
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
dendrogram(single_linkage, no_labels=True)
plt.title("Single Linkage")
plt.xlabel("Athletes")
plt.ylabel("Distance")

plt.subplot(2, 2, 2)
dendrogram(complete_linkage, no_labels=True)
plt.title("Complete Linkage")
plt.xlabel("Athletes")
plt.ylabel("Distance")

plt.subplot(2, 2, 3)
dendrogram(average_linkage, no_labels=True)
plt.title("Average Linkage")
plt.xlabel("Athletes")
plt.ylabel("Distance")

plt.subplot(2, 2, 4)
dendrogram(ward_linkage, no_labels=True)
plt.title("Ward Linkage")
plt.xlabel("Athletes")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()