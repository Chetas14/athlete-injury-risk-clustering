import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    "avg_fatigue_index",
    "avg_recovery_score",
    "avg_sleep_quality",
    "avg_stress_level"
]

# Create correlation matrix
correlation = df[features].corr()

# Display correlation matrix
print(correlation)

# Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Clustering Features")
plt.tight_layout()
plt.show()