import csv


# Read the features we already generated
with open("feature_data.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    features = list(reader)


# Read the optimal quantum found by generate_labels.py
# For now, use the optimal quantum from the look-ahead result.
# We will improve this later to generate state-specific labels.
print("Loaded feature records:", len(features))

optimal_quantum = int(input("Enter the Optimal Quantum from generate_labels.py: "))


# Create final training dataset
with open("training_data.csv", "w", newline="") as file:
    fieldnames = [
        "ready_queue_length",
        "average_cpu_burst",
        "recent_workload_behavior",
        "optimal_quantum"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    for row in features:
        writer.writerow({
            "ready_queue_length": row["ready_queue_length"],
            "average_cpu_burst": row["average_cpu_burst"],
            "recent_workload_behavior": row["recent_workload_behavior"],
            "optimal_quantum": optimal_quantum
        })


print("training_data.csv created successfully!")