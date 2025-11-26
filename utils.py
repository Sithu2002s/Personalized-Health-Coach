# utils.py
import matplotlib.pyplot as plt
import pandas as pd                #handling dates and tabular data
import matplotlib.dates as mdates  #formatting dates on the x axis

def plot_history(history):
    if not history:
        return None, None, None

    df = pd.DataFrame(history)
    # Convert timestamp 
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, infer_datetime_format=True)
    df = df.sort_values("timestamp")

    # --------------------------- 1. STEPS ---------------------------
    fig_steps, ax1 = plt.subplots(figsize=(5.5, 2.4))
    ax1.plot(df["timestamp"], df["steps"], linewidth=1.8, label="Step Count")
    avg_steps = df["steps"].mean()
    ax1.axhline(avg_steps, color="red", linestyle="--", linewidth=1, label=f"Avg: {avg_steps:.0f}")
    ax1.set_title("Daily Steps Trend", fontsize=10)
    ax1.set_ylabel("Steps", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.get_xticklabels(), rotation=30, fontsize=8, ha="right")
    ax1.legend(fontsize=8)

    # --------------------------- 2. SLEEP ---------------------------
    fig_sleep, ax2 = plt.subplots(figsize=(5.5, 2.4))
    ax2.plot(df["timestamp"], df["sleep"], linewidth=1.8, label="Sleep Hours")
    avg_sleep = df["sleep"].mean()
    ax2.axhline(avg_sleep, color="red", linestyle="--", linewidth=1, label=f"Avg: {avg_sleep:.1f}h")
    ax2.set_title("Sleep Duration (Hours)", fontsize=10)
    ax2.set_ylabel("Hours", fontsize=9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=8, ha="right")
    ax2.legend(fontsize=8)

    # --------------------------- 3. WATER ---------------------------
    fig_water, ax3 = plt.subplots(figsize=(4.5, 1.9))
    ax3.plot(df["timestamp"], df["water"], linewidth=1.6, label="Water Intake")
    avg_water = df["water"].mean()
    ax3.axhline(avg_water, color="red", linestyle="--", linewidth=1, label=f"Avg: {avg_water:.1f}L")
    ax3.set_title("Daily Water Intake (Liters)", fontsize=9)
    ax3.set_ylabel("Liters", fontsize=8)
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.get_xticklabels(), rotation=30, fontsize=7, ha="right")
    ax3.legend(fontsize=7)

    return fig_steps, fig_sleep, fig_water
