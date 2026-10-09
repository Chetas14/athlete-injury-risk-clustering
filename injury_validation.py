import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load clustered athlete dataset
df = pd.read_csv("clustered_athletes.csv")


# Compare injury statistics by cluster
summary = df.groupby("cluster").agg(
    athletes=("athlete_id", "count"),
    total_injury_sessions=("injury_sessions", "sum"),
    average_injury_rate=("injury_rate", "mean"),
    average_max_injury_level=("max_injury_level", "mean")
).reset_index()


# Calculate percentage of athletes with at least one injury session
injured_athletes = (
    df.groupby("cluster")["injury_sessions"]
    .apply(lambda x: (x > 0).sum())
    .reset_index(name="athletes_with_injury")
)

summary = summary.merge(
    injured_athletes,
    on="cluster"
)

summary["injured_athlete_percentage"] = (
    summary["athletes_with_injury"]
    / summary["athletes"]
    * 100
)


# Display results
print("\nINJURY VALIDATION BY CLUSTER")
print(summary.to_string(index=False))


# Identify higher-risk cluster based on average injury rate
high_risk_cluster = summary.loc[
    summary["average_injury_rate"].idxmax(),
    "cluster"
]

print("\nHIGHER INJURY-RATE CLUSTER:")
print(f"Cluster {int(high_risk_cluster)}")


# Plot average injury rate
plt.figure(figsize=(8, 5))

sns.barplot(
    data=summary,
    x="cluster",
    y="average_injury_rate"
)

plt.title("Average Injury Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Injury Rate")

plt.tight_layout()
plt.show()


# Plot injured athlete percentage
plt.figure(figsize=(8, 5))

sns.barplot(
    data=summary,
    x="cluster",
    y="injured_athlete_percentage"
)

plt.title("Percentage of Athletes With Injury Sessions")
plt.xlabel("Cluster")
plt.ylabel("Athletes With Injury (%)")

plt.tight_layout()
plt.show()