"""
Visualize the pelvis trajectory from the expert .mot data to verify
that the Y-up (OpenSim) -> Z-up (MuJoCo) frame rotation is correct.

Plots:
1. 3D pelvis trajectory (before and after rotation)
2. Per-axis translation over time (tx, ty, tz)
3. Pelvis euler angles over time

Usage:
    python visualize_frame_rotation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as sRot

pio.renderers.default = "browser"

MOT_FILE = "/media/tripan/Data/DDP/amputee_data/training_data/tf01_1p0_03_rotated_ik.mot"

def load_raw_mot(filepath):
    """Load raw .mot file and return pelvis translation + euler angles."""
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
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_end)
    unit_scale = np.pi / 180.0 if in_degrees else 1.0
    
    time = df['time'].values
    trans_raw = df[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].values
    euler_raw = df[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].values * unit_scale
    
    return time, trans_raw, euler_raw

def apply_inverse_rotation(trans_raw, euler_raw):
    """Apply the inverse rotation: Ry(-π/2) then Rx(π/2)."""
    # R_y_neg = sRot.from_euler('y', -np.pi/2)
    R_x_pos = sRot.from_euler('x', np.pi/2)
    R_frame = R_x_pos
    R_mat = R_frame.as_matrix()
    
    # Rotate translation
    trans_rotated = (R_mat @ trans_raw.T).T
    
    # Rotate orientation
    rot_opensim = sRot.from_euler('xyz', euler_raw)
    rot_mujoco = R_frame * rot_opensim
    euler_rotated = rot_mujoco.as_euler('xyz')
    
    return trans_rotated, euler_rotated

def main():
    time, trans_raw, euler_raw = load_raw_mot(MOT_FILE)
    trans_rot, euler_rot = apply_inverse_rotation(trans_raw, euler_raw)
    
    # ============================================================
    # PLOT 1: 3D Trajectory comparison
    # ============================================================
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter3d(
        x=trans_raw[:, 0], y=trans_raw[:, 1], z=trans_raw[:, 2],
        mode='lines', name='Raw (OpenSim Y-up)',
        line=dict(color='red', width=3)
    ))
    fig1.add_trace(go.Scatter3d(
        x=trans_rot[:, 0], y=trans_rot[:, 1], z=trans_rot[:, 2],
        mode='lines', name='Rotated (MuJoCo Z-up)',
        line=dict(color='blue', width=3)
    ))
    fig1.update_layout(
        title="Pelvis 3D Trajectory: Raw vs Rotated",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode='data'
        ),
        height=700, width=900
    )
    fig1.show(renderer="browser")
    
    # ============================================================
    # PLOT 2: Per-axis translation over time 
    # ============================================================
    fig2 = make_subplots(rows=3, cols=2,
                         subplot_titles=(
                             "Raw TX (OpenSim)", "Rotated X (MuJoCo)",
                             "Raw TY (OpenSim) - should be HEIGHT", "Rotated Y (MuJoCo)",
                             "Raw TZ (OpenSim)", "Rotated Z (MuJoCo) - should be HEIGHT"
                         ))
    
    axis_labels = ['TX', 'TY', 'TZ']
    colors_raw = ['#e74c3c', '#e67e22', '#f1c40f']
    colors_rot = ['#3498db', '#2ecc71', '#9b59b6']
    
    for i in range(3):
        fig2.add_trace(go.Scatter(x=time, y=trans_raw[:, i], name=f"Raw {axis_labels[i]}",
                                  line=dict(color=colors_raw[i])), row=i+1, col=1)
        fig2.add_trace(go.Scatter(x=time, y=trans_rot[:, i], name=f"Rotated {axis_labels[i]}",
                                  line=dict(color=colors_rot[i])), row=i+1, col=2)
        fig2.update_yaxes(title_text="meters", row=i+1, col=1)
        fig2.update_yaxes(title_text="meters", row=i+1, col=2)
    
    fig2.update_layout(height=800, width=1100, title_text="Pelvis Translation Over Time")
    fig2.show(renderer="browser")
    
    # ============================================================
    # PLOT 3: Euler angles comparison
    # ============================================================
    fig3 = make_subplots(rows=3, cols=2,
                         subplot_titles=(
                             "Raw Tilt (OpenSim)", "Rotated Tilt (MuJoCo)",
                             "Raw List (OpenSim)", "Rotated List (MuJoCo)",
                             "Raw Rotation (OpenSim)", "Rotated Rotation (MuJoCo)"
                         ))
    
    euler_labels = ['Tilt', 'List', 'Rotation']
    for i in range(3):
        fig3.add_trace(go.Scatter(x=time, y=np.degrees(euler_raw[:, i]), name=f"Raw {euler_labels[i]}",
                                  line=dict(color=colors_raw[i])), row=i+1, col=1)
        fig3.add_trace(go.Scatter(x=time, y=np.degrees(euler_rot[:, i]), name=f"Rotated {euler_labels[i]}",
                                  line=dict(color=colors_rot[i])), row=i+1, col=2)
        fig3.update_yaxes(title_text="degrees", row=i+1, col=1)
        fig3.update_yaxes(title_text="degrees", row=i+1, col=2)
    
    fig3.update_layout(height=800, width=1100, title_text="Pelvis Euler Angles Over Time")
    fig3.show(renderer="browser")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Raw pelvis translation ranges:")
    print(f"  TX: [{trans_raw[:, 0].min():.3f}, {trans_raw[:, 0].max():.3f}]")
    print(f"  TY: [{trans_raw[:, 1].min():.3f}, {trans_raw[:, 1].max():.3f}]  <-- HEIGHT in OpenSim Y-up")
    print(f"  TZ: [{trans_raw[:, 2].min():.3f}, {trans_raw[:, 2].max():.3f}]")
    print(f"\nRotated pelvis translation ranges:")
    print(f"  X:  [{trans_rot[:, 0].min():.3f}, {trans_rot[:, 0].max():.3f}]")
    print(f"  Y:  [{trans_rot[:, 1].min():.3f}, {trans_rot[:, 1].max():.3f}]")
    print(f"  Z:  [{trans_rot[:, 2].min():.3f}, {trans_rot[:, 2].max():.3f}]  <-- should be HEIGHT (~0.95)")

if __name__ == "__main__":
    main()
