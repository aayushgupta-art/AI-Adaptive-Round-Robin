class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time

        # Remaining burst initially equals the full burst time
        self.remaining_burst = burst_time

        # -1 means the process has not started yet
        self.start_time = -1

        # OS scheduling metrics
        self.completion_time = -1
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = 0
