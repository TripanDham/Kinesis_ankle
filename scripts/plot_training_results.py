import re
import os
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_log_file(log_path):
    """
    Parses the training log to extract epoch, average reward, and component rewards.
    Assumes the format:
    [TIMESTAMP][src.agents.agent_humanoid][INFO] - Ep: 1 ... eps_R_avg 11.9875 ... [v1 v2 v3 ...] ... eps_len 22.95
    """
    epochs = []
    total_rewards = []
    component_rewards = []
    eps_lens = []
    
    # Regex to capture the key metrics from the AgentHumanoid log line
    # Groups: 1=Ep, 2=eps_R_avg, 3=bracketed_values, 4=eps_len
    pattern = re.compile(r"Ep:\s+(\d+).*?eps_R_avg\s+([\d.-]+).*?\[(.*?)\]\s+.*?eps_len\s+([\d.-]+)")

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return None

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                total_rewards.append(float(match.group(2)))
                
                # Parse bracketed values (handle 'nan' as 0.0)
                raw_components = match.group(3).split()
                comps = []
                for val in raw_components:
                    try:
                        comps.append(float(val) if val != 'nan' else 0.0)
                    except ValueError:
                        comps.append(0.0)
                component_rewards.append(comps)
                
                eps_lens.append(float(match.group(4)))

    if not epochs:
        print("No training data found in the log file.")
        return None

    return {
        "epoch": np.array(epochs),
        "total_reward": np.array(total_rewards),
        "components": np.array(component_rewards),
        "eps_len": np.array(eps_lens)
    }

def plot_training_results(data, output_path="training_dashboard_fixed.html"):
    """
    Creates a Plotly dashboard with individual graphs for each metric in a single column.
    """
    labels = ["Gail", "Vel", "Upright", "Muscle", "Motor", "Total_Step"]
    num_comps = data["components"].shape[1]
    if len(labels) < num_comps:
        labels += [f"Comp {i}" for i in range(len(labels), num_comps)]
    elif len(labels) > num_comps:
        labels = labels[:num_comps]

    # total_reward + eps_len + all components
    num_plots = 2 + num_comps
    
    fig = make_subplots(
        rows=num_plots, cols=1,
        subplot_titles=["Total Episode Reward (Cumulative)", "Episode Length (Steps)"] + [f"Reward component: {l}" for l in labels],
        vertical_spacing=0.03,
    )

    # 1. Total Reward (Cumulative)
    fig.add_trace(go.Scatter(x=data["epoch"], y=data["total_reward"], name="Cumulative Reward", line=dict(color='blue')), row=1, col=1)
    fig.update_yaxes(title_text="Total Reward", row=1, col=1)

    # 2. Episode Length
    fig.add_trace(go.Scatter(x=data["epoch"], y=data["eps_len"], name="Avg Length", line=dict(color='green')), row=2, col=1)
    fig.update_yaxes(title_text="Steps", row=2, col=1)

    # 3+ Individual Components
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for i in range(num_comps):
        row = 3 + i
        fig.add_trace(go.Scatter(
            x=data["epoch"], 
            y=data["components"][:, i], 
            name=labels[i], 
            line=dict(color=colors[i % len(colors)])
        ), row=row, col=1)
        fig.update_yaxes(title_text=labels[i], row=row, col=1)

    # Update global layout
    fig.update_layout(
        height=300 * num_plots,  # Expand height based on plot count
        width=1000, 
        title_text=f"Detailed Training Analysis Dashboard",
        showlegend=False
    )
    fig.update_xaxes(title_text="Epoch")

    # Save to HTML
    fig.write_html(output_path)
    print(f"\n" + "="*80)
    print(f"ANALYSIS COMPLETE: {num_plots} individual graphs generated.")
    print(f"Dashboard saved to: {os.path.abspath(output_path)}")
    print(f"="*80 + "\n")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GAIL training results from a log file.")
    parser.add_argument("log_file", help="Path to the training log file (e.g. outputs/run_IL.log)")
    parser.add_argument("--output", default=None, help="Path to save the HTML dashboard.")
    
    args = parser.parse_args()
    
    # By default, save the dashboard to the same folder as the log file
    if args.output is None:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        args.output = os.path.join(log_dir, "training_dashboard.html")
    
    results = parse_log_file(args.log_file)
    if results:
        plot_training_results(results, args.output)
