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
                "burst_time": int(row["burst_time"])
            })

    return processes


def simulate_state(processes, quantum):
    processes = copy.deepcopy(processes)

    current_time = 0
    completed = 0
    context_switches = 0
    ready_queue = []
    next_process = 0

    for p in processes:
        p["remaining"] = p["burst_time"]
        p["completion"] = 0

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
        pid = current["process_id"]

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

    average_waiting = total_waiting / total

    return average_waiting, context_switches


def get_best_quantum(processes):
    best_quantum = None
    best_score = float("inf")

    for quantum in QUANTUMS:

        waiting, switches = simulate_state(
            processes,
            quantum
        )

        score = waiting + (switches * 2)

        if score < best_score:
            best_score = score
            best_quantum = quantum

    return best_quantum


def create_state_record(processes, index):
    state = processes[:index]

    if not state:
        return None

    ready_queue_length = len(state)

    average_cpu_burst = (
        sum(p["burst_time"] for p in state)
        / ready_queue_length
    )

    io_count = sum(
        1 for p in state
        if p["burst_time"] <= 5
    )

    recent_workload_behavior = (
        io_count / ready_queue_length
    )

    optimal_quantum = get_best_quantum(state)

    return {
        "ready_queue_length": ready_queue_length,
        "average_cpu_burst": round(average_cpu_burst, 2),
        "recent_workload_behavior": round(
            recent_workload_behavior,
            2
        ),
        "optimal_quantum": optimal_quantum
    }


# Workload files
workload_files = [
    "cpu_workload.csv",
    "io_workload.csv",
    "mixed_workload.csv"
]

training_rows = []

for filename in workload_files:

    print("Processing:", filename)

    processes = load_processes(filename)

    # Generate several different system states
    step = max(1, len(processes) // 10)

    for index in range(
        step,
        len(processes) + 1,
        step
    ):
        record = create_state_record(
            processes,
            index
        )

        if record:
            training_rows.append(record)

    print(
        "States generated:",
        len(training_rows)
    )


# Save final dataset
with open(
    "training_data.csv",
    "w",
    newline=""
) as file:

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
    writer.writerows(training_rows)


print("\ntraining_data.csv created successfully!")
print("Total state records:", len(training_rows))