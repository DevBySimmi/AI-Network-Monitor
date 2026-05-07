import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -----------------------------
# Generate Sample Network Data
# -----------------------------
np.random.seed(42)

# Normal latency values
normal_data = np.random.normal(loc=50, scale=5, size=200)

# Abnormal latency values
anomalies = np.array([90, 95, 100, 110])

# Combine data
data = np.concatenate([normal_data, anomalies])

# Reshape for model
X = data.reshape(-1, 1)

# -----------------------------
# Train Isolation Forest Model
# -----------------------------
model = IsolationForest(
    contamination=0.02,
    random_state=42
)

predictions = model.fit_predict(X)

# -----------------------------
# Separate Normal & Anomalies
# -----------------------------
normal_points = X[predictions == 1]
anomaly_points = X[predictions == -1]

normal_index = np.where(predictions == 1)[0]
anomaly_index = np.where(predictions == -1)[0]

# -----------------------------
# Plot Graph
# -----------------------------
plt.figure(figsize=(10, 5))

# Normal points
plt.scatter(
    normal_index,
    normal_points,
    color='green',
    label='Normal Data'
)

# Anomaly points
plt.scatter(
    anomaly_index,
    anomaly_points,
    color='red',
    s=100,
    label='Anomalies'
)

# Labels & Title
plt.title("Network Latency Anomaly Detection")
plt.xlabel("Data Point Index")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)

# Show graph
plt.show()