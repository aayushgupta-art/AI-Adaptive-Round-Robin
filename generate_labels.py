import csv
import copy

QUANTUMS = [2, 5, 10, 20]


def load_processes(filename):
    processes = []

    with open(filename, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            processes.append({
                "process_id": row["process_id"],
                "arrival_time": int(row["arrival_time"]),
                "burst_time": int(row["burst_time"]),
            })

    return processes


def simulate(processes, quantum):
    processes = copy.deepcopy(processes)

    current_time = 0
    completed = 0
    context_switches = 0
    ready_queue = []

    for p in processes:
        p["remaining"] = p["burst_time"]
        p["completion"] = 0

    next_process = 0
    total = len(processes)

    while completed < total:

        while (
            next_process < total
            and processes[next_process]["arrival_time"] <= current_time
        ):
            ready_queue.append(processes[next_process])
            next_process += 1

        if not ready_queue:
            if next_process < total:
                current_time = processes[next_process]["arrival_time"]
                continue

        current = ready_queue.pop(0)

        execution_time = min(
            quantum,
            current["remaining"]
        )

        current_time += execution_time
        current["remaining"] -= execution_time

        while (
            next_process < total
            and processes[next_process]["arrival_time"] <= current_time
        ):
            ready_queue.append(processes[next_process])
            next_process += 1

        if current["remaining"] == 0:
            current["completion"] = current_time
            completed += 1
        else:
            ready_queue.append(current)

        if ready_queue:
            context_switches += 1

    total_waiting = 0

    for p in processes:
        turnaround = p["completion"] - p["arrival_time"]
        waiting = turnaround - p["burst_time"]
        total_waiting += waiting

    return total_waiting, context_switches


def calculate_score(waiting, context_switches):
    return waiting + (context_switches * 2)


def find_best_quantum(processes):
    results = []

    for quantum in QUANTUMS:
        waiting, switches = simulate(processes, quantum)
        score = calculate_score(waiting, switches)

        results.append({
            "quantum": quantum,
            "waiting": waiting,
            "context_switches": switches,
            "score": score
        })

        print(
            f"Q={quantum} | "
            f"Waiting={waiting} | "
            f"Context Switches={switches} | "
            f"Score={score}"
        )

    best = min(results, key=lambda x: x["score"])

    return best["quantum"]


# Load workloads
workloads = [
    "cpu_workload.csv",
    "io_workload.csv",
    "mixed_workload.csv"
]

all_rows = []

for filename in workloads:

    print("\nProcessing:", filename)

    processes = load_processes(filename)

    best_quantum = find_best_quantum(processes)

    print("Optimal Quantum =", best_quantum)

    # Calculate workload features
    ready_queue_length = len(processes)

    average_cpu_burst = (
        sum(p["burst_time"] for p in processes)
        / len(processes)
    )

    io_count = sum(
        1 for p in processes
        if p["burst_time"] <= 5
    )

    recent_workload_behavior = io_count / len(processes)

    all_rows.append({
        "ready_queue_length": ready_queue_length,
        "average_cpu_burst": round(average_cpu_burst, 2),
        "recent_workload_behavior": round(
            recent_workload_behavior, 2
        ),
        "optimal_quantum": best_quantum
    })


# Save training dataset
with open("training_data.csv", "w", newline="") as file:

    fieldnames = [
        "ready_queue_length",
        "average_cpu_burst",
        "recent_workload_behavior",
        "optimal_quantum"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_rows)


print("\ntraining_data.csv created successfully!")
print("Total training records:", len(all_rows))