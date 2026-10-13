from collections import deque
from process import Process


class RoundRobinSimulator:
    def __init__(self, time_quantum, context_switch_penalty):
        self.time_quantum = time_quantum
        self.context_switch_penalty = context_switch_penalty

        self.current_time = 0
        self.ready_queue = deque()
        self.context_switches = 0
        self.timeline = []

        # ML feature storage
        self.feature_data = []

        # Recent workload history
        self.recent_processes = []
        self.window_size = 10

    def run(self, processes):
        # Sort processes by arrival time
        processes = sorted(processes, key=lambda p: p.arrival_time)

        next_process = 0
        completed = 0
        total_processes = len(processes)

        while completed < total_processes:

            # Add all processes that have arrived
            while (
                next_process < total_processes
                and processes[next_process].arrival_time <= self.current_time
            ):
                self.ready_queue.append(processes[next_process])
                next_process += 1

            # If no process is ready, move time forward
            if not self.ready_queue:
                if next_process < total_processes:
                    if self.current_time < processes[next_process].arrival_time:
                        self.timeline.append({
                            "start": self.current_time,
                            "end": processes[next_process].arrival_time,
                            "label": "Idle",
                            "kind": "idle",
                        })
                    self.current_time = processes[next_process].arrival_time
                    continue
            # Capture current ready queue statistics
            ready_queue_length = len(self.ready_queue)

            average_cpu_burst = (
                sum(p.remaining_burst for p in self.ready_queue)
                / ready_queue_length
                if ready_queue_length > 0
                else 0
            )
            # Calculate recent workload behavior
            if len(self.recent_processes) > 0:
                io_count = sum(
                    1 for p in self.recent_processes
                    if p.burst_time <= 5
                )
                recent_workload_behavior = io_count / len(self.recent_processes)
            else:
                recent_workload_behavior = 0

            self.feature_data.append({
    "ready_queue_length": ready_queue_length,
    "average_cpu_burst": average_cpu_burst,
    "recent_workload_behavior": recent_workload_behavior
})
            # Take the first process from the ready queue
            current_process = self.ready_queue.popleft()
# Record first start time and response time
            if current_process.start_time == -1:
                current_process.start_time = self.current_time
                current_process.response_time = (
                    self.current_time - current_process.arrival_time
                )

            # Execute the process for one time quantum
            execution_time = min(
                self.time_quantum,
                current_process.remaining_burst
            )

            execution_start = self.current_time
            self.current_time += execution_time
            current_process.remaining_burst -= execution_time
            self.timeline.append({
                "start": execution_start,
                "end": self.current_time,
                "label": current_process.pid,
                "kind": "execution",
            })

            # Add processes that arrived during execution
            while (
                next_process < total_processes
                and processes[next_process].arrival_time <= self.current_time
            ):
                self.ready_queue.append(processes[next_process])
                next_process += 1

            # Check whether the current process has finished
            if current_process.remaining_burst == 0:
                current_process.completion_time = self.current_time

                current_process.turnaround_time = (
                    current_process.completion_time
                    - current_process.arrival_time
                )

                current_process.waiting_time = (
                    current_process.turnaround_time
                    - current_process.burst_time
                )

                completed += 1

            else:
                # Process is not finished.
                self.ready_queue.append(current_process)

            # Context switch penalty
            if self.ready_queue:
                switch_start = self.current_time
                self.current_time += self.context_switch_penalty
                self.context_switches += 1
                if self.context_switch_penalty > 0:
                    self.timeline.append({
                        "start": switch_start,
                        "end": self.current_time,
                        "label": "CS",
                        "kind": "switch",
                    })
                if len(self.recent_processes) > 0:
                    io_count = sum(
                        1 for p in self.recent_processes
                        if p.burst_time <= 5
                    )
                    recent_workload_behavior = io_count / len(self.recent_processes)
                else:
                    recent_workload_behavior = 0
                self.recent_processes.append(current_process)

                if len(self.recent_processes) > self.window_size:
                    self.recent_processes.pop(0)

                # Add processes that arrived during context switch
                while (
                    next_process < total_processes
                    and processes[next_process].arrival_time <= self.current_time
                ):
                    self.ready_queue.append(processes[next_process])
                    next_process += 1

        return processes