from process import Process
from round_robin import RoundRobinSimulator


def run_test(quantum):
    processes = [
        Process("P1", 0, 5),
        Process("P2", 1, 4),
        Process("P3", 2, 2),
        Process("P4", 4, 3)
    ]

    simulator = RoundRobinSimulator(
        time_quantum=quantum,
        context_switch_penalty=1
    )

    results = simulator.run(processes)

    avg_waiting = sum(p.waiting_time for p in results) / len(results)

    avg_turnaround = (
        sum(p.turnaround_time for p in results) / len(results)
    )

    avg_response = (
        sum(p.response_time for p in results) / len(results)
    )

    return avg_waiting, avg_turnaround, avg_response, simulator.context_switches


print("Round Robin Comparison")
print("=" * 60)

for q in [2, 5, 10, 20]:
    waiting, turnaround, response, switches = run_test(q)

    print(
        f"Q={q:<3} "
        f"Waiting={waiting:.2f} | "
        f"Turnaround={turnaround:.2f} | "
        f"Response={response:.2f} | "
        f"Context Switches={switches}"
    )