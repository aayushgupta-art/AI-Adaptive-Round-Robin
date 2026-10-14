import os

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st

from process import Process
from round_robin import RoundRobinSimulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaptive_quantum_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "training_data.csv")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}.")
    return pd.read_csv(DATA_PATH)


def parse_processes(raw_text):
    rows = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace(",", " ").split()]
        if len(parts) < 3:
            raise ValueError(
                "Each process line must follow: PID arrival burst. Example: P1 0 8"
            )
        pid = parts[0]
        arrival = int(parts[1])
        burst = int(parts[2])
        if arrival < 0 or burst <= 0:
            raise ValueError("Arrival times must be zero or greater and burst times must be positive.")
        rows.append(Process(pid, arrival, burst))
    if not rows:
        raise ValueError("Enter at least one process.")
    return rows


def generate_random_workload(count, max_arrival=30, max_burst=30):
    rng = np.random.default_rng(42)
    workload = []
    for i in range(1, count + 1):
        arrival = int(rng.integers(0, max_arrival + 1))
        burst = int(rng.integers(1, max_burst + 1))
        workload.append(Process(f"P{i}", arrival, burst))
    return workload


def get_default_processes():
    return [
        Process("P1", 0, 8),
        Process("P2", 1, 14),
        Process("P3", 2, 2),
        Process("P4", 4, 19),
        Process("P5", 8, 6),
    ]


def build_feature_frame(processes):
    if not processes:
        return pd.DataFrame([{"ready_queue_length": 0, "average_cpu_burst": 0.0, "io_ratio": 0.0}])

    ready_queue_length = len(processes)
    average_cpu_burst = sum(p.burst_time for p in processes) / ready_queue_length
    io_ratio = sum(1 for p in processes if p.burst_time <= 5) / ready_queue_length

    return pd.DataFrame(
        [{
            "ready_queue_length": ready_queue_length,
            "average_cpu_burst": average_cpu_burst,
            "io_ratio": io_ratio,
        }]
    )


def run_round_robin(processes, quantum, context_switch_penalty):
    simulator = RoundRobinSimulator(time_quantum=quantum, context_switch_penalty=context_switch_penalty)
    results = simulator.run(processes)
    avg_waiting = sum(p.waiting_time for p in results) / len(results)
    avg_turnaround = sum(p.turnaround_time for p in results) / len(results)
    avg_response = sum(p.response_time for p in results) / len(results)
    execution_time = sum(
        item["end"] - item["start"]
        for item in simulator.timeline
        if item["kind"] == "execution"
    )
    context_switch_time = sum(
        item["end"] - item["start"]
        for item in simulator.timeline
        if item["kind"] == "switch"
    )
    idle_time = sum(
        item["end"] - item["start"]
        for item in simulator.timeline
        if item["kind"] == "idle"
    )

    return {
        "results": results,
        "avg_waiting": avg_waiting,
        "avg_turnaround": avg_turnaround,
        "avg_response": avg_response,
        "context_switches": simulator.context_switches,
        "execution_time": execution_time,
        "context_switch_time": context_switch_time,
        "idle_time": idle_time,
        "elapsed_time": execution_time + context_switch_time + idle_time,
        "feature_data": simulator.feature_data,
        "timeline": simulator.timeline,
    }


def copy_processes(processes):
    return [Process(p.pid, p.arrival_time, p.burst_time) for p in processes]


def draw_timeline(timeline, focus_index=None, light_mode=False):
    colors = {"execution": "#31c48d", "switch": "#f59e0b", "idle": "#64748b"}
    fig, ax = plt.subplots(figsize=(12, 2.8))
    background = "#ffffff" if light_mode else "#0b1220"
    foreground = "#172033" if light_mode else "#e2e8f0"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    for index, item in enumerate(timeline):
        duration = item["end"] - item["start"]
        selected = focus_index == index
        ax.barh(
            0,
            duration,
            left=item["start"],
            height=0.58,
            color=colors[item["kind"]],
            edgecolor=foreground if selected else background,
            linewidth=2.2 if selected else 0.8,
            alpha=1.0 if selected or focus_index is None else 0.62,
            zorder=3,
        )
        if duration >= 1.2:
            ax.text(
                item["start"] + duration / 2,
                0,
                item["label"],
                ha="center",
                va="center",
                color="#07111f",
                fontsize=8,
                fontweight="bold",
                clip_on=True,
                zorder=4,
            )
    ax.set_yticks([0])
    ax.set_yticklabels(["CPU"])
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel("Simulation time")
    ax.set_title("Round Robin Gantt chart")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.set_axisbelow(True)
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.title.set_color(foreground)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=foreground, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1" if light_mode else "#334155")
    fig.tight_layout()
    return fig


def format_results(results):
    rows = []
    for p in results:
        rows.append(
            {
                "PID": p.pid,
                "Arrival": p.arrival_time,
                "Burst": p.burst_time,
                "Start": p.start_time,
                "Completion": p.completion_time,
                "Waiting": p.waiting_time,
                "Turnaround": p.turnaround_time,
                "Response": p.response_time,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Adaptive RR Scheduler", page_icon="⚙️", layout="wide")

appearance = st.radio("Appearance", ["Dark", "Light"], horizontal=True, index=0)
light_mode = appearance == "Light"

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(ellipse at 12% 0%, #14243a 0, #0b1220 40%, #080d17 100%); }
    [data-testid="stSidebar"] { background: #0c1523; border-right: 1px solid #26364a; }
    [data-testid="stMetric"] { background: #111d2d; border: 1px solid #26364a; padding: 14px 16px; border-radius: 12px; }
    [data-testid="stMetricValue"] { color: #67e8bd; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .hero { padding: 22px 26px; border: 1px solid #26364a; border-radius: 16px; background: linear-gradient(115deg, #13243a, #101b2b 60%, #12332f); margin-bottom: 18px; }
    .hero h1 { color: #f8fafc; }
    .hero p { color: #b8c6d8; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if light_mode:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background: #f3f6fb !important;
            color: #172033 !important;
        }
        [data-testid="stSidebar"] {
            background: #e8eef7 !important;
            border-right: 1px solid #cbd5e1 !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"] {
            color: #172033;
        }
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #d7e0ec !important;
        }
        [data-testid="stMetricValue"] { color: #087f5b !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
            color: #475569 !important;
        }
        .hero {
            background: linear-gradient(115deg, #e5edf8, #f8fafc 60%, #dff5ec) !important;
            border-color: #cbd5e1 !important;
        }
        .hero h1 { color: #102033 !important; }
        .hero p { color: #334155 !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div,
        .stApp [data-baseweb="input"] > div {
            background-color: #ffffff !important;
            color: #172033 !important;
        }
        .stApp [data-testid="stDataFrame"], .stApp [data-testid="stTable"] {
            background: #ffffff;
        }
        .stApp [data-testid="stTabs"] button { color: #334155 !important; }
        .stApp [data-testid="stAlert"] { background: #eaf2fb; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hero"><h1>Adaptive Round Robin Scheduler</h1>'
    '<p>Explore how machine learning selects a time quantum and changes CPU scheduling outcomes.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation controls")
    workload_mode = st.selectbox("Workload type", ["Preset example", "Custom workload", "Random workload"])

    quantum = st.slider("No-AI baseline quantum", min_value=1, max_value=20, value=5)
    context_penalty = st.slider("Context switch penalty", min_value=0, max_value=3, value=1)

    if workload_mode == "Preset example":
        process_list = get_default_processes()
    elif workload_mode == "Custom workload":
        default_text = "P1 0 8\nP2 1 14\nP3 2 2\nP4 4 19\nP5 8 6"
        custom_text = st.text_area("Processes (PID arrival burst)", value=default_text, height=200)
        try:
            process_list = parse_processes(custom_text)
        except ValueError as exc:
            st.error(str(exc))
            process_list = get_default_processes()
    else:
        workload_count = st.slider("Random workload size", min_value=4, max_value=20, value=8)
        process_list = generate_random_workload(workload_count)

    st.caption("Results update automatically when controls or workload change.")

# Unified scheduler selector — drives the whole-page view
active_scheduler = st.radio(
    "Active scheduler view",
    ["AI-Adaptive Scheduler", "No-AI Baseline Scheduler"],
    horizontal=True,
    key="global_scheduler_selector"
)

if process_list:
    model = load_model()
    features = build_feature_frame(process_list)
    predicted_quantum = model.predict(features)[0]
    predicted_quantum = int(max(1, min(20, round(float(predicted_quantum)))))

    baseline_metrics = run_round_robin(copy_processes(process_list), quantum, context_penalty)
    adaptive_metrics = run_round_robin(copy_processes(process_list), predicted_quantum, context_penalty)
    input_io_ratio = sum(1 for p in process_list if p.burst_time <= 5) / len(process_list)

    # Metrics adapt to the selected scheduler mode
    if active_scheduler == "No-AI Baseline Scheduler":
        display_quantum = quantum
        metrics_header = "No-AI Baseline Scheduler"
        schedule_source = "Fixed baseline quantum"
    else:
        display_quantum = predicted_quantum
        metrics_header = "AI-Adaptive Scheduler"
        schedule_source = "AI-predicted quantum"

    st.subheader("Model recommendation" if active_scheduler == "AI-Adaptive Scheduler" else "Baseline recommendation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Time Quantum", display_quantum)
    col2.metric("Queue length", len(process_list))
    col3.metric("Avg CPU burst", round(sum(p.burst_time for p in process_list) / len(process_list), 2) if process_list else 0.0)
    st.caption(f"{schedule_source} · {len(process_list)} processes · average burst "
               f"{features.iloc[0]['average_cpu_burst']:.2f} · I/O-bound ratio {input_io_ratio:.0%}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["AI vs no-AI comparison", "Execution detail", "Model insights"])

    with tab1:
        st.subheader("How much simulated time does the AI save?")
        baseline_elapsed = baseline_metrics["elapsed_time"]
        adaptive_elapsed = adaptive_metrics["elapsed_time"]
        elapsed_saved = baseline_elapsed - adaptive_elapsed
        saved_percent = elapsed_saved / baseline_elapsed * 100 if baseline_elapsed else 0.0
        if elapsed_saved > 0:
            headline = (
                f"With AI, this workload finishes {elapsed_saved:.2f} simulation time units "
                f"earlier ({saved_percent:.1f}% less elapsed time) than fixed Q={quantum}."
            )
        elif elapsed_saved < 0:
            headline = (
                f"With AI, this workload takes {abs(elapsed_saved):.2f} more simulation time units "
                f"({abs(saved_percent):.1f}% more elapsed time) than fixed Q={quantum}."
            )
        else:
            headline = "Both schedulers finish this workload in the same simulated elapsed time."
        st.markdown(f"**{headline}**")
        st.info(
            "This is simulated schedule time, not measured real-world computer runtime. "
            "Both schedulers execute the same total process work; any elapsed-time difference "
            "comes from context-switch overhead and idle time, not from the AI removing CPU work."
        )

        time1, time2, time3, time4 = st.columns(4)
        time1.metric(f"Without AI · fixed Q={quantum}", f"{baseline_elapsed:.2f} units")
        time2.metric(f"With AI · selected Q={predicted_quantum}", f"{adaptive_elapsed:.2f} units")
        time3.metric("Elapsed time saved", f"{elapsed_saved:+.2f} units")
        time4.metric("Change vs fixed quantum", f"{saved_percent:+.1f}%")

        st.caption(
            "Elapsed time = process execution + context-switch overhead + idle time. "
            "Positive savings mean the AI schedule finishes sooner; a negative value means it takes longer."
        )

        comparison_df = pd.DataFrame(
            {
                f"Without AI (fixed Q={quantum})": [
                    baseline_metrics["elapsed_time"],
                    baseline_metrics["execution_time"],
                    baseline_metrics["context_switch_time"],
                    baseline_metrics["idle_time"],
                    baseline_metrics["context_switches"],
                    baseline_metrics["avg_waiting"],
                    baseline_metrics["avg_turnaround"],
                    baseline_metrics["avg_response"],
                ],
                f"With AI (selected Q={predicted_quantum})": [
                    adaptive_metrics["elapsed_time"],
                    adaptive_metrics["execution_time"],
                    adaptive_metrics["context_switch_time"],
                    adaptive_metrics["idle_time"],
                    adaptive_metrics["context_switches"],
                    adaptive_metrics["avg_waiting"],
                    adaptive_metrics["avg_turnaround"],
                    adaptive_metrics["avg_response"],
                ],
            },
            index=[
                "Total elapsed time (simulation units)",
                "Process execution time (simulation units)",
                "Context-switch overhead (simulation units)",
                "CPU idle time (simulation units)",
                "Context switches (count)",
                "Average waiting time (simulation units)",
                "Average turnaround time (simulation units)",
                "Average response time (simulation units)",
            ],
        )
        comparison_df["AI change (with AI - without AI)"] = (
            comparison_df.iloc[:, 1] - comparison_df.iloc[:, 0]
        )
        st.subheader("Side-by-side results")
        st.dataframe(comparison_df.round(2), width="stretch")
        st.caption(
            "For the time rows, a negative AI change means less time with AI. "
            "Process execution time is expected to match because both schedulers run the same workload."
        )
        st.bar_chart(
            comparison_df.loc[
                [
                    "Total elapsed time (simulation units)",
                    "Context-switch overhead (simulation units)",
                    "Average waiting time (simulation units)",
                    "Average turnaround time (simulation units)",
                    "Average response time (simulation units)",
                ],
                [f"Without AI (fixed Q={quantum})", f"With AI (selected Q={predicted_quantum})"],
            ]
        )
        st.download_button(
            "Download AI vs no-AI comparison (CSV)",
            data=comparison_df.to_csv().encode("utf-8"),
            file_name="scheduler_performance_comparison.csv",
            mime="text/csv",
        )

        st.subheader("Workload profile")
        workload_df = pd.DataFrame(
            [{"Process": p.pid, "Arrival": p.arrival_time, "Burst": p.burst_time}
             for p in process_list]
        ).set_index("Process")
        st.bar_chart(workload_df[["Burst"]], color="#31c48d")

    with tab2:
        # Derive active mode from the global unified selector (not a separate tab-level radio)
        if active_scheduler == "AI-Adaptive Scheduler":
            current_metrics = adaptive_metrics
            current_quantum = predicted_quantum
            schedule_label = "AI-Adaptive"
            results_df = format_results(current_metrics["results"])
            timeline = current_metrics["timeline"]
            feature_data = current_metrics["feature_data"]
            file_prefix = "adaptive_scheduler"
        else:  # No-AI Baseline Scheduler
            current_metrics = baseline_metrics
            current_quantum = quantum
            schedule_label = "No-AI Baseline"
            results_df = format_results(current_metrics["results"])
            timeline = current_metrics["timeline"]
            feature_data = current_metrics["feature_data"]
            file_prefix = "baseline_scheduler"

        st.subheader(f"{schedule_label} schedule · Quantum {current_quantum}")
        if not timeline:
            st.info("No schedule events were generated for this workload.")
        else:
            st.caption(
                f"The {'AI selected' if active_scheduler == 'AI-Adaptive Scheduler' else 'fixed'} quantum of {current_quantum} time units. "
                "Each block's width shows how long that process used the CPU in this turn."
            )
            # Use a unique key for the slider that includes the schedule type
            focus_index = st.slider(
                "Step through execution",
                min_value=1,
                max_value=len(timeline),
                value=1,
                format="Event %d",
                help="Select an execution, context-switch, or idle interval to highlight it on the Gantt chart.",
                key=f"focus_slider_{active_scheduler.lower().replace(' ', '_')}_tab2"
            )
            selected_event = timeline[focus_index - 1]
            st.caption(
                f"Event {focus_index}/{len(timeline)} · {selected_event['label']} · "
                f"time {selected_event['start']}–{selected_event['end']}"
            )

            # Ready Queue visualization for the selected event
            ready_queue_pids = selected_event.get("ready_queue", [])
            running_pid = selected_event["label"] if selected_event["kind"] == "execution" else None
            card_bg = "#ffffff" if light_mode else "#111d2d"
            card_border = "#cbd5e1" if light_mode else "#26364a"
            running_bg = "#31c48d"
            waiting_bg = "#1e293b" if light_mode else "#1a2540"
            waiting_border = "#94a3b8" if light_mode else "#334155"
            waiting_color = "#172033" if light_mode else "#e2e8f0"
            no_q_color = "#64748b" if light_mode else "#94a3b8"

            st.markdown(
                f'<div style="background:{card_bg};border:1px solid {card_border};'
                f'border-radius:12px;padding:14px 18px;margin:6px 0 14px 0;">'
                f'<div style="font-size:13px;font-weight:700;color:{running_bg};letter-spacing:0.5px;'
                f'text-transform:uppercase;margin-bottom:10px;">Ready Queue</div>',
                unsafe_allow_html=True,
            )
            if running_pid:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
                    f'<span style="background:{running_bg};color:#07111f;padding:6px 16px;'
                    f'border-radius:8px;font-weight:700;font-size:14px;">▶ {running_pid}</span>'
                    f'<span style="font-size:12px;color:{no_q_color};font-style:italic;">currently executing</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
                    f'<span style="background:#64748b;color:#fff;padding:6px 16px;'
                    f'border-radius:8px;font-weight:700;font-size:14px;">⏸ Idle</span>'
                    f'<span style="font-size:12px;color:{no_q_color};font-style:italic;">no process running</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if ready_queue_pids:
                chips = "".join(
                    f'<span style="background:{waiting_bg};border:1px solid {waiting_border};'
                    f'color:{waiting_color};padding:6px 14px;border-radius:8px;'
                    f'font-weight:600;font-size:13px;">{pid}</span>'
                    for pid in ready_queue_pids
                )
                st.markdown(
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{chips}</div>'
                    f'<div style="font-size:11px;color:{no_q_color};margin-top:8px;">'
                    f'{len(ready_queue_pids)} process(es) waiting in queue</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="font-size:13px;color:{no_q_color};font-style:italic;">'
                    f'Ready queue is empty</span></div>',
                    unsafe_allow_html=True,
                )

            timeline_fig = draw_timeline(timeline, focus_index - 1, light_mode)
            st.pyplot(timeline_fig, width="stretch")
            plt.close(timeline_fig)
            st.caption("Blocks run left-to-right in CPU order. Green = process execution · amber = context-switch overhead · slate = idle CPU")

        st.subheader("Per-process metrics")
        st.dataframe(results_df, width="stretch", hide_index=True)

        if feature_data:
            queue_df = pd.DataFrame(feature_data)
            queue_df["step"] = np.arange(1, len(queue_df) + 1)
            st.subheader("Queue state during scheduling")
            st.line_chart(queue_df.set_index("step")[["ready_queue_length", "average_cpu_burst"]])

        st.download_button(
            "Download process results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_prefix}_results.csv",
            mime="text/csv",
        )

        timeline_df = pd.DataFrame(timeline)
        st.download_button(
            "Download execution timeline (CSV)",
            data=timeline_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{file_prefix}_timeline.csv",
            mime="text/csv",
        )

    with tab3:
        # ===== 4-Step AI Pipeline Visualization =====
        st.subheader("AI Prediction Pipeline: How the Model Selects the Optimal Quantum")

        # Prepare live feature values for this workload
        q_len = len(process_list)
        avg_burst = sum(p.burst_time for p in process_list) / len(process_list) if process_list else 0.0
        io_ratio = sum(1 for p in process_list if p.burst_time <= 5) / len(process_list) if process_list else 0.0

        # Step cards container
        step_bg = "#ffffff" if light_mode else "#111d2d"
        step_border = "#cbd5e1" if light_mode else "#26364a"
        step_accent = "#31c48d"
        text_main = "#172033" if light_mode else "#e2e8f0"
        text_muted = "#64748b" if light_mode else "#94a3b8"

        st.markdown(
            f'<div style="background:{step_bg};border:1px solid {step_border};'
            f'border-radius:14px;padding:20px 24px;margin-bottom:24px;">',
            unsafe_allow_html=True,
        )

        # Step 1: Feature Extraction
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:24px;">'
            f'<div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;background:{step_accent};'
            f'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;color:#07111f;">1</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:16px;font-weight:700;color:{text_main};margin-bottom:6px;">Step 1: Feature Extraction</div>'
            f'<div style="font-size:13px;color:{text_muted};line-height:1.6;">'
            f'The scheduler captures three key workload characteristics from the current process set:<br/>'
            f'<span style="font-family:monospace;background:{step_bg};border:1px solid {step_border};padding:2px 6px;border-radius:4px;">ready_queue_length</span> = <b>{q_len}</b> &nbsp;|&nbsp; '
            f'<span style="font-family:monospace;background:{step_bg};border:1px solid {step_border};padding:2px 6px;border-radius:4px;">average_cpu_burst</span> = <b>{avg_burst:.2f}</b> &nbsp;|&nbsp; '
            f'<span style="font-family:monospace;background:{step_bg};border:1px solid {step_border};padding:2px 6px;border-radius:4px;">io_ratio</span> = <b>{io_ratio:.0%}</b>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Step 2: Vectorization
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:24px;">'
            f'<div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;background:{step_accent};'
            f'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;color:#07111f;">2</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:16px;font-weight:700;color:{text_main};margin-bottom:6px;">Step 2: Vectorization</div>'
            f'<div style="font-size:13px;color:{text_muted};line-height:1.6;">'
            f'Features are arranged into a 1 x 3 feature vector matching the model\'s training schema:<br/>'
            f'<div style="font-family:monospace;background:#0b1220;color:#67e8bd;padding:10px 14px;border-radius:8px;'
            f'margin:8px 0;font-size:12px;overflow-x:auto;">'
            f'X = [[{q_len}, {avg_burst:.4f}, {io_ratio:.4f}]]  &nbsp;<span style="color:#94a3b8;"># shape (1, 3)</span>'
            f'</div>'
            f'<div style="font-size:12px;color:{text_muted};">No scaling/normalization needed — Random Forest handles raw feature values directly.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Step 3: Ensemble Tree Voting
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:24px;">'
            f'<div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;background:{step_accent};'
            f'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;color:#07111f;">3</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:16px;font-weight:700;color:{text_main};margin-bottom:6px;">Step 3: Ensemble Tree Voting</div>'
            f'<div style="font-size:13px;color:{text_muted};line-height:1.6;">'
            f'The trained Random Forest (typically 100-500 trees) independently evaluates the feature vector.<br/>'
            f'Each tree traverses its decision path by comparing features against learned thresholds:<br/>'
            f'<div style="font-family:monospace;background:#0b1220;color:#67e8bd;padding:10px 14px;border-radius:8px;'
            f'margin:8px 0;font-size:12px;overflow-x:auto;">'
            f'<span style="color:#94a3b8;"># e.g., Tree decision logic</span><br/>'
            f'if ready_queue_length <= 5.5:<br/>'
            f'&nbsp;&nbsp;if average_cpu_burst <= 8.2:<br/>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;predict 3<br/>'
            f'&nbsp;&nbsp;else:<br/>'
            f'&nbsp;&nbsp;&nbsp;&nbsp;predict 7<br/>'
            f'else:<br/>'
            f'&nbsp;&nbsp;predict 12'
            f'</div>'
            f'Final prediction = majority vote across all trees (class with highest vote count).'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Step 4: Optimal Quantum Output
        pred_q = int(max(1, min(20, round(float(predicted_quantum)))))
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:16px;">'
            f'<div style="flex-shrink:0;width:44px;height:44px;border-radius:12px;background:{step_accent};'
            f'display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;color:#07111f;">4</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:16px;font-weight:700;color:{text_main};margin-bottom:6px;">Step 4: Optimal Quantum Output</div>'
            f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
            f'<div style="background:{step_accent};color:#07111f;padding:14px 28px;'
            f'border-radius:10px;font-weight:800;font-size:28px;">Q = {pred_q}</div>'
            f'<div style="font-size:14px;color:{text_muted};line-height:1.5;">'
            f'This quantum is applied to a fresh Round Robin simulation of the same workload.<br/>'
            f'Predicted by the trained Random Forest classifier on the extracted features above.'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)
        # ===== End Pipeline Visualization =====

        st.markdown("---")

        training_df = load_training_data()
        label_distribution = training_df["optimal_quantum"].value_counts().sort_index()
        dist_df = pd.DataFrame({"Quantum": label_distribution.index, "Records": label_distribution.values})
        st.bar_chart(dist_df.set_index("Quantum"))

        feature_importance = pd.DataFrame(
            {
                "Feature": ["ready_queue_length", "average_cpu_burst", "io_ratio"],
                "Importance": model.feature_importances_,
            }
        )
        st.bar_chart(feature_importance.set_index("Feature"), color="#31c48d")

        st.subheader("Training dataset")
        st.caption(f"{len(training_df):,} records · {training_df.shape[1]} model and label columns")
        st.dataframe(training_df.head(10), width="stretch", hide_index=True)

        st.download_button(
            "Download training dataset (CSV)",
            data=training_df.to_csv(index=False).encode("utf-8"),
            file_name="training_data.csv",
            mime="text/csv",
        )

st.markdown("---")
with st.container():
    st.subheader("How the demonstration works")
    st.markdown(
        """
        1. Workload features are passed to the trained Random Forest classifier.
        2. The predicted quantum is used to run an independent Round Robin simulation.
        3. Results are compared with the selected fixed quantum using identical fresh copies of the workload.
        4. Simulated elapsed time is split into process execution, context-switch overhead, and idle time.
        5. The timeline and CSV exports expose the schedule and its simulated outcomes.

        **Interpretation:** the AI cannot reduce the workload's required CPU burst time in this simulation.
        It can change the number of context switches and when processes run, which may change elapsed,
        waiting, response, and turnaround times. These are simulation time units—not a benchmark of
        real computer runtime or the cost of running the ML model.
        """
    )
