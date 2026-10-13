import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Raw Data ---
# These are the exact numbers from your Week 4 evaluation output
labels_baseline = 'Baseline (Fixed Q=5)'
labels_ai = 'AI-Adaptive'

context_switches = [666, 308]
wait_times = [1460.88, 1433.89]
turnaround_times = [1492.01, 1465.02]
response_times = [205.82, 494.89]

colors = ['#1f77b4', '#ff7f0e'] # Professional Blue and Orange

# --- 2. Chart Configuration ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Performance Comparison: Baseline vs. AI-Adaptive Round Robin', fontsize=16, fontweight='bold', y=1.02)

def add_value_labels(ax, bars):
    """Utility function to attach a text label above each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}' if isinstance(height, float) else f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

# --- Panel 1: Context Switches (The Big Win) ---
bars_cs = axes[0].bar([labels_baseline, labels_ai], context_switches, color=colors, width=0.5)
axes[0].set_title('Total Context Switches\n(Lower is Better)', fontsize=12)
axes[0].set_ylabel('Count')
add_value_labels(axes[0], bars_cs)

# --- Panel 2: Wait & Turnaround Times (The Marginal Win) ---
x = np.arange(2)  # the label locations
width = 0.35      # the width of the bars

bars_wait = axes[1].bar(x - width/2, wait_times, width, label='Wait Time', color=['#41b6c4', '#225ea8'])
bars_turn = axes[1].bar(x + width/2, turnaround_times, width, label='Turnaround Time', color=['#a1dab4', '#41b6c4'])

axes[1].set_title('Waiting & Turnaround Time\n(Lower is Better)', fontsize=12)
axes[1].set_ylabel('Milliseconds (ms)')
axes[1].set_xticks(x)
axes[1].set_xticklabels([labels_baseline, labels_ai])
add_value_labels(axes[1], bars_wait)
add_value_labels(axes[1], bars_turn)

# --- Panel 3: Response Time (The Trade-Off) ---
bars_resp = axes[2].bar([labels_baseline, labels_ai], response_times, color=colors, width=0.5)
axes[2].set_title('Initial Response Time\n(Trade-off metric)', fontsize=12)
axes[2].set_ylabel('Milliseconds (ms)')
add_value_labels(axes[2], bars_resp)

# --- 3. Final Formatting and Export ---
plt.tight_layout()
plt.savefig('scheduler_evaluation_charts.png', dpi=300, bbox_inches='tight')
print("Charts generated successfully and saved as 'scheduler_evaluation_charts.png'")
plt.show()