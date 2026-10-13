import csv
import random


def generate_workload(filename, workload_type, number_of_processes=1000):
    processes = []

    for i in range(1, number_of_processes + 1):

        # Arrival time
        arrival_time = random.randint(0, 1000)

        # Choose burst time based on workload type
        if workload_type == "cpu":
            burst_time = random.randint(20, 50)

        elif workload_type == "io":
            burst_time = random.randint(1, 5)

        elif workload_type == "mixed":
            if random.choice([True, False]):
                burst_time = random.randint(20, 50)
            else:
                burst_time = random.randint(1, 5)

        processes.append([
            f"P{i}",
            arrival_time,
            burst_time
        ])

    # Write data to CSV
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "process_id",
            "arrival_time",
            "burst_time"
        ])

        writer.writerows(processes)

    print(f"{filename} created with {number_of_processes} processes.")


# Generate the three required workloads
generate_workload("cpu_workload.csv", "cpu", 1000)
generate_workload("io_workload.csv", "io", 1000)
generate_workload("mixed_workload.csv", "mixed", 1000)