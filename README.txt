Adaptive_RR_Scheduling
│
├── __pycache__       ← Python-generated cache folder
│
├── process.py        → Process / PCB Data Structure
├── round_robin.py    → Baseline Round Robin Simulator
├── train_model.py    → Trains the adaptive quantum prediction model
├── main.py           → Testing & Output
├── app.py            → Streamlit dashboard for visual demonstration
├── requirements.txt  → Python dependencies for dashboard execution
└── README.txt        → Project overview and usage instructions

To run the CLI simulation:
    python main.py

To run the Streamlit dashboard:
    streamlit run app.py

The dashboard displays:
- Dark and light appearance modes, switchable from the sidebar
- Process workload input and queue metrics
- AI predicted quantum based on queue conditions
- Fixed-quantum (no-AI) vs model-selected-quantum (AI) simulation results
- Simulated elapsed-time difference, percentage change, and context-switch overhead
- A breakdown of process execution, context-switch, and CPU idle time
- Waiting, turnaround, response, and context switch comparison
- Model distribution and feature importance insights
- Round Robin Gantt chart with CPU events positioned left-to-right by simulation time
- Idle periods and context-switch overhead shown as separate chart blocks
- CSV downloads for per-process results, execution timeline, and training data

The computing-time comparison is based on simulated schedule time, not real hardware
runtime. Both schedulers run the same process workload, so required process execution
time stays constant; differences in elapsed time come from context-switch overhead
and idle periods. The selected fixed quantum is the no-AI baseline.