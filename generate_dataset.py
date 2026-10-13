import pandas as pd
import numpy as np

# Set seed so the random data is reproducible if you run it multiple times
np.random.seed(42)

# Define dataset size
n_samples = 5000

# 1. Generate Randomized Features (The "X" variables)
# Real OS queues rarely hit 100+. 1 to 50 is a realistic snapshot.
queue_lengths = np.random.randint(1, 51, size=n_samples)

# Average CPU burst in milliseconds (1ms to 100ms)
avg_bursts = np.random.uniform(1.0, 100.0, size=n_samples)

# IO Ratio: 0.0 (100% CPU bound) to 1.0 (100% I/O bound)
io_ratios = np.random.uniform(0.0, 1.0, size=n_samples)

# 2. Heuristic Rule Engine (The Brute-Force Logic)
# This simulates evaluating the perfect quantum for a given system state.
def determine_optimal_quantum(queue_len, avg_burst, io_ratio):
    # Rule 1: If the queue is mostly I/O-bound (interactive), keep the quantum tiny 
    # to guarantee fast response times for users.
    if io_ratio > 0.6:
        return 2 
        
    # Rule 2: If it's mostly CPU-bound (heavy computation), use a massive quantum 
    # to avoid wasting CPU cycles on constant context switching.
    elif io_ratio < 0.4 and avg_burst > 30:
        return 20 
        
    # Rule 3: If the queue is dangerously long, drop the quantum to a medium-low 
    # level so the processes at the back don't starve to death.
    elif queue_len > 30:
        return 5
        
    # Default: Mixed or normal workloads get a standard quantum.
    else:
        return 10

# Generate the target labels using the heuristic
optimal_quantums = [
    determine_optimal_quantum(q, b, i) 
    for q, b, i in zip(queue_lengths, avg_bursts, io_ratios)
]

# 3. Inject Real-World Noise
# Real OS data is never mathematically perfect. We randomly scramble 5% of the 
# answers to force the ML model to learn the trend, not just memorize the exact rules.
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
for idx in noise_indices:
    optimal_quantums[idx] = np.random.choice([2, 5, 10, 20])

# 4. Build and Export the Dataset
df = pd.DataFrame({
    'ready_queue_length': queue_lengths,
    'average_cpu_burst': np.round(avg_bursts, 2),
    'io_ratio': np.round(io_ratios, 2),
    'optimal_quantum': optimal_quantums
})

df.to_csv('training_data.csv', index=False)

# 5. Print Validation Metrics
print("Dataset generated successfully: 'training_data.csv'")
print(f"Total Rows: {len(df)}\n")
print("Class Distribution (Check for bias):")
print(df['optimal_quantum'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))