import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster


# Load dataset
df = pd.read_csv("athlete_level_features.csv")


# Final clustering features
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


# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Create linkage matrices
linkage_methods = {
    "Single": linkage(X_scaled, method="single"),
    "Complete": linkage(X_scaled, method="complete"),
    "Average": linkage(X_scaled, method="average"),
    "Ward": linkage(X_scaled, method="ward")
}


# Store results
results = []


# Test different numbers of clusters
for method_name, linkage_matrix in linkage_methods.items():

    for k in range(2, 9):

        labels = fcluster(
            linkage_matrix,
            t=k,
            criterion="maxclust"
        )

        score = silhouette_score(X_scaled, labels)

        results.append({
            "Linkage": method_name,
            "Clusters": k,
            "Silhouette Score": score
        })


# Convert results to DataFrame
results_df = pd.DataFrame(results)


# Display all results
print("\nSILHOUETTE SCORES")
print(results_df.to_string(index=False))


# Find best result for each linkage method
best_results = results_df.loc[
    results_df.groupby("Linkage")["Silhouette Score"].idxmax()
]


print("\nBEST RESULT FOR EACH LINKAGE METHOD")
print(best_results.to_string(index=False))


# Plot silhouette scores
plt.figure(figsize=(10, 6))

for method in results_df["Linkage"].unique():

    method_data = results_df[
        results_df["Linkage"] == method
    ]

    plt.plot(
        method_data["Clusters"],
        method_data["Silhouette Score"],
        marker="o",
        label=method
    )


plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.xticks(range(2, 9))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# Compare cluster sizes for the best candidate configurations

best_configs = {
    "Single": 2,
    "Complete": 2,
    "Average": 2,
    "Ward": 8
}

print("\nCLUSTER SIZE COMPARISON")

for method_name, k in best_configs.items():

    linkage_matrix = linkage_methods[method_name]

    labels = fcluster(
        linkage_matrix,
        t=k,
        criterion="maxclust"
    )

    print(f"\n{method_name} Linkage - {k} Clusters")

    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    print(cluster_sizes)