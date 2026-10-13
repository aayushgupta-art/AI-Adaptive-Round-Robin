from process import Process
from round_robin import RoundRobinSimulator
from train_model import model, X
import pandas as pd


# Create test processes
processes = [
    Process("P1", 0, 8),
    Process("P2", 1, 14),
    Process("P3", 2, 2),
    Process("P4", 4, 19)
]


# Create Round Robin simulator
# Get the latest workload features
latest = pd.DataFrame([{
    "ready_queue_length": len(processes),
    "average_cpu_burst": sum(p.burst_time for p in processes) / len(processes),
    "io_ratio": 0.5
}])

# Predict the time quantum using ML
predicted_quantum = model.predict(latest)[0]

# Keep the quantum within a safe range
predicted_quantum = max(1, min(20, round(predicted_quantum)))

print("ML Predicted Time Quantum:", predicted_quantum)

# Create Round Robin simulator using ML prediction
simulator = RoundRobinSimulator(
    time_quantum=predicted_quantum,
    context_switch_penalty=1
)


# Run the simulation
results = simulator.run(processes)


# Display results
print("Round Robin Scheduling Results")
print("-" * 100)

for p in results:
    print(
        f"{p.pid:<5}"
        f"Arrival={p.arrival_time:<5}"
        f"Burst={p.burst_time:<5}"
        f"Start={p.start_time:<5}"
        f"Completion={p.completion_time:<5}"
        f"Waiting={p.waiting_time:<5}"
        f"Turnaround={p.turnaround_time:<5}"
        f"Response={p.response_time:<5}"
    )

print("-" * 100)
print("Total Context Switches:", simulator.context_switches)
print("\nNumber of ML feature records:", len(simulator.feature_data))
import csv

with open("feature_data.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
      fieldnames=[
    "ready_queue_length",
    "average_cpu_burst",
    "recent_workload_behavior"
]
    )
    writer.writeheader()
    writer.writerows(simulator.feature_data)

print("feature_data.csv created successfully!")