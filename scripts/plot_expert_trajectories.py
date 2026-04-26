import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

JOINT_COLUMNS = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',         # 0, 1, 2
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', # 3, 4, 5
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', # 6, 7, 8, 9, 10
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'  # 11, 12, 13, 14, 15
]

# Labels for the 24 dimensions
DIM_LABELS = [
    # 10 Angles
    'hip_flexion_r_angle', 'hip_adduction_r_angle', 'hip_rotation_r_angle', 'knee_angle_r_angle', 'ankle_angle_r_angle',
    'hip_flexion_l_angle', 'hip_adduction_l_angle', 'hip_rotation_l_angle', 'knee_angle_l_angle', 'ankle_angle_l_angle',
    # 13 Velocities
    'pelvis_tx_vel', 'pelvis_ty_vel', 'pelvis_tz_vel',
    'hip_flexion_r_vel', 'hip_adduction_r_vel', 'hip_rotation_r_vel', 'knee_angle_r_vel', 'ankle_angle_r_vel',
    'hip_flexion_l_vel', 'hip_adduction_l_vel', 'hip_rotation_l_vel', 'knee_angle_l_vel', 'ankle_angle_l_vel',
    # 1 Height
    'root_height (pelvis_ty)'
]

def extract_speed(filename):
    match = re.search(r'_(\d+)p(\d+)_', filename)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.0

def find_col(df_cols, target):
    if target in df_cols:
        return target
    for c in df_cols:
        if target in c:
            return c
    return None

def parse_mot(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_end = 0
    in_degrees = True
    for i, line in enumerate(lines):
        if 'inDegrees' in line:
            in_degrees = 'yes' in line.lower()
        if 'endheader' in line:
            header_end = i + 1
            break
    df = pd.read_csv(filepath, sep='\t', skiprows=header_end)
    if len(df.columns) <= 1:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_end)
    return df, in_degrees

def generate_trajectories_for_plotting(data_dir, target_freq=30.0):
    mot_files = sorted(list(Path(data_dir).glob("*.mot")))
    print(f"Found {len(mot_files)} .mot files.")
    target_dt = 1.0 / target_freq
    trajectories = []
    
    VEL_LIMIT = 20.0
    TRANS_VEL_LIMIT = 5.0
    
    for mot_file in tqdm(mot_files):
        filename = mot_file.name
        if filename.endswith(".joblib"): continue
        
        speed = extract_speed(filename)
        df, in_degrees = parse_mot(str(mot_file))
        if len(df) < 5: continue
        
        times = df['time'].values
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        raw_pos = np.zeros((len(df), 16), dtype=np.float32)
        for i, target in enumerate(JOINT_COLUMNS):
            col_name = find_col(df.columns, target)
            if col_name:
                scale = 1.0 if 'pelvis_t' in target else unit_scale
                val = df[col_name].values * scale
                if 'knee' in target or 'ankle' in target:
                    val = -val
                raw_pos[:, i] = val

        avg_yaw = np.mean(raw_pos[:, 5])
        raw_pos[:, 5] = raw_pos[:, 5] - avg_yaw
        
        new_times = np.arange(times[0], times[-1], target_dt)
        resampled_raw_pos = np.zeros((len(new_times), 16), dtype=np.float32)
        for i in range(16):
            resampled_raw_pos[:, i] = np.interp(new_times, times, raw_pos[:, i])
            
        # 10 Angles
        obs_angles = resampled_raw_pos[:, 6:16]
        
        # 13 Velocities
        resampled_vels = np.zeros((len(new_times), 16), dtype=np.float32)
        diffs = np.diff(resampled_raw_pos, axis=0) / (target_dt + 1e-8)
        
        diffs[:, 0:3] = np.clip(diffs[:, 0:3], -TRANS_VEL_LIMIT, TRANS_VEL_LIMIT)
        diffs[:, 0] += speed
        diffs[:, 3:16] = np.clip(diffs[:, 3:16], -VEL_LIMIT, VEL_LIMIT)
        
        resampled_vels[:-1] = diffs
        resampled_vels[-1] = resampled_vels[-2]

        vel_indices = [0, 1, 2] + list(range(6, 16))
        obs_vels = resampled_vels[:, vel_indices]
        
        # 1 Height (added back for plotting, making it exactly 24D)
        height_col = resampled_raw_pos[:, 1:2]
        
        obs_traj = np.concatenate([obs_angles, obs_vels, height_col], axis=1)
        
        trajectories.append({
            'filename': filename,
            'speed': speed,
            'time': new_times,
            'data': obs_traj
        })

    return trajectories

def plot_trajectories(trajectories, output_file):
    print(f"Plotting {len(trajectories)} trajectories...")
    # 24 dimensions -> 6 rows, 4 cols
    rows, cols = 6, 4
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=DIM_LABELS)
    
    # We want each file to have the same color across all subplots
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    for idx, traj in enumerate(trajectories):
        color = colors[idx % len(colors)]
        name = traj['filename']
        
        for dim in range(24):
            r = (dim // cols) + 1
            c = (dim % cols) + 1
            
            # Only show legend once per file (on the first subplot)
            show_legend = True if dim == 0 else False
            
            fig.add_trace(go.Scatter(
                x=traj['time'],
                y=traj['data'][:, dim],
                mode='lines',
                name=name,
                line=dict(color=color),
                showlegend=show_legend
            ), row=r, col=c)

    fig.update_layout(
        height=1500, 
        width=1800, 
        title_text="Expert Trajectories: 24D Extracted Features",
        template='plotly_dark'
    )
    
    fig.write_html(output_file)
    print(f"Plot successfully saved to {output_file}")

if __name__ == "__main__":
    data_dir = "/media/tripan/Data/DDP/amputee_data/training_data"
    output_html = "data/expert_trajectories_plot.html"
    
    trajectories = generate_trajectories_for_plotting(data_dir)
    plot_trajectories(trajectories, output_html)
