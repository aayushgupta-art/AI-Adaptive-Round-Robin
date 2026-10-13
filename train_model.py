import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the final training dataset
data = pd.read_csv("C:/Users/ayush/Downloads/Adaptive_RR_Scheduling (1)_with_dataset/Adaptive_RR_Scheduling/training_data.csv")

print("Training data loaded successfully. Total rows:", len(data))

# Input features (X)
X = data[
    [
        "ready_queue_length",
        "average_cpu_burst",
        "io_ratio"
    ]
]

# Target label (y)
y = data["optimal_quantum"]

# Create and train the ML model (Random Forest CLASSIFIER)
# n_estimators=50 means it uses 50 decision trees to vote on the best quantum.
# max_depth=5 keeps the model extremely lightweight and fast for OS operations.
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)

print("\nML model trained successfully!")

# Test prediction using the latest system state
latest = X.iloc[[-1]]

prediction = model.predict(latest)[0]

print("Predicted optimal time quantum:", prediction)

# CRITICAL STEP: Export the trained model to a file
joblib.dump(model, "adaptive_quantum_model.pkl")
print("Model saved to disk as 'adaptive_quantum_model.pkl'. Ready for OS Integration.")