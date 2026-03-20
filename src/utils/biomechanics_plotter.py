import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import logging

# Force browser renderer to avoid nbformat dependency errors in terminal
pio.renderers.default = "browser"

logger = logging.getLogger(__name__)

def plot_biomechanics(all_biomechanics, env):
    """
    Processes and plots biomechanics data collected over multiple evaluation runs.
    
    Args:
        all_biomechanics (list of lists): [episode][timestep][dict of metrics]
        env: The environment instance, used to extract joint and actuator indices.
    """
    logger.info(f"Processing biomechanics data for {len(all_biomechanics)} episodes...")
    
    # 1. Truncate to min episode length to compute strict means, or pad.
    # For simplicity, we truncate to the shortest episode length among the runs
    min_len = min(len(ep) for ep in all_biomechanics)
    if min_len == 0:
        logger.warning("No biomechanics data to plot (empty episodes).")
        return
        
    logger.info(f"Analyzing up to {min_len} timesteps across {len(all_biomechanics)} runs.")
    
    # Extract actuator names and joint indices
    actuator_names = env.actuator_names
    
    # Find specific actuator indices
    def get_act_idx(name_substring):
        for i, name in enumerate(actuator_names):
            if name_substring in name:
                return i
        return -1
        
    soleus_l_idx = get_act_idx("soleus_l")
    prosthetic_knee_idx = get_act_idx("osl_knee_torque_actuator")
    prosthetic_ankle_idx = get_act_idx("osl_ankle_torque_actuator")
    
    # QPOS Joint indices based on myolegs structure:
    # Hip Flexion L/R, Knee L/R, Ankle L/R
    # (Based on standard myolegs mapping, but extracted directly from environment state layout)
    # The environment has angles[9:11] from qpos[[14, 15]] -> osl knee, ankle
    # angles[14:16] from qpos[[21, 24]] -> left knee, ankle
    # hip flexion R = qpos[7], hip flexion L = qpos[16]
    joint_indices = {
        "Hip Flexion R": 7,
        "Knee R (Prosthetic)": 14,
        "Ankle R (Prosthetic)": 15,
        "Hip Flexion L": 16,
        "Knee L": 21,
        "Ankle L": 24
    }
    
    # Pre-allocate arrays [num_episodes, min_len]
    num_eps = len(all_biomechanics)
    soleus_act = np.zeros((num_eps, min_len))
    knee_torque_rt = np.zeros((num_eps, min_len))
    ankle_torque_rt = np.zeros((num_eps, min_len))
    
    joint_angles = {name: np.zeros((num_eps, min_len)) for name in joint_indices.keys()}
    
    # Compile Data
    for ep_idx in range(num_eps):
        for t in range(min_len):
            step_data = all_biomechanics[ep_idx][t]
            
            # Muscle Activation (ctrl or actuator_activation)
            if soleus_l_idx != -1:
                # ctrl is shaped [nu], same as actuators
                soleus_act[ep_idx, t] = step_data["ctrl"][soleus_l_idx]
                
            # Prosthetic Actuator Torques
            if prosthetic_knee_idx != -1:
                knee_torque_rt[ep_idx, t] = step_data["actuator_force"][prosthetic_knee_idx]
            if prosthetic_ankle_idx != -1:
                ankle_torque_rt[ep_idx, t] = step_data["actuator_force"][prosthetic_ankle_idx]
                
            # Joint Angles
            for name, qpos_idx in joint_indices.items():
                joint_angles[name][ep_idx, t] = step_data["qpos"][qpos_idx]


    # Helper to add trace with std dev
    def add_shaded_trace(fig, data_matrix, name, row, col, color="blue", y_title=""):
        mean_val = np.mean(data_matrix, axis=0)
        std_val = np.std(data_matrix, axis=0)
        x_vals = np.arange(min_len)
        
        # Upper/Lower Bounds
        upper_bound = mean_val + std_val
        lower_bound = mean_val - std_val
        
        fig.add_trace(go.Scatter(
            x=list(x_vals) + list(x_vals)[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor=f'rgba({color}, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f"{name} StdDev"
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=mean_val,
            line=dict(color=f'rgb({color})'),
            mode='lines',
            name=f"{name} Mean"
        ), row=row, col=col)
        
        fig.update_yaxes(title_text=y_title, row=row, col=col)
        fig.update_xaxes(title_text="Timestep", row=row, col=col)

    # ---------------------------------------------------------
    # PLOT 1: Activations & Torques (3 subplots)
    # ---------------------------------------------------------
    fig1 = make_subplots(rows=3, cols=1, 
                         subplot_titles=("Left Soleus Activation", 
                                         "Right Knee Actuator Torque (Prosthetic)", 
                                         "Right Ankle Actuator Torque (Prosthetic)"))
    
    add_shaded_trace(fig1, soleus_act, "Soleus L", row=1, col=1, color="255, 0, 0", y_title="Activation")
    add_shaded_trace(fig1, knee_torque_rt, "Knee Torque R", row=2, col=1, color="0, 100, 255", y_title="Torque (Nm)")
    add_shaded_trace(fig1, ankle_torque_rt, "Ankle Torque R", row=3, col=1, color="0, 200, 100", y_title="Torque (Nm)")
    
    fig1.update_layout(height=900, width=800, title_text="Muscle & Prosthetic Actuator Analysis")
    fig1.show(renderer="browser")

    # ---------------------------------------------------------
    # PLOT 2: Kinematics (6 subplots)
    # ---------------------------------------------------------
    fig2 = make_subplots(rows=3, cols=2, 
                         subplot_titles=("Hip Flexion L", "Hip Flexion R", 
                                         "Knee L", "Knee R (Prosthetic)", 
                                         "Ankle L", "Ankle R (Prosthetic)"))
    
    # Left Leg (Col 1)
    add_shaded_trace(fig2, joint_angles["Hip Flexion L"], "Hip L", row=1, col=1, color="100, 0, 200", y_title="Angle (rad)")
    add_shaded_trace(fig2, joint_angles["Knee L"], "Knee L", row=2, col=1, color="100, 0, 200", y_title="Angle (rad)")
    add_shaded_trace(fig2, joint_angles["Ankle L"], "Ankle L", row=3, col=1, color="100, 0, 200", y_title="Angle (rad)")
    
    # Right Leg (Col 2)
    add_shaded_trace(fig2, joint_angles["Hip Flexion R"], "Hip R", row=1, col=2, color="255, 100, 0", y_title="Angle (rad)")
    add_shaded_trace(fig2, joint_angles["Knee R (Prosthetic)"], "Knee R", row=2, col=2, color="255, 100, 0", y_title="Angle (rad)")
    add_shaded_trace(fig2, joint_angles["Ankle R (Prosthetic)"], "Ankle R", row=3, col=2, color="255, 100, 0", y_title="Angle (rad)")
    
    fig2.update_layout(height=1000, width=1000, title_text="Leg Kinematics Analysis")
    fig2.show(renderer="browser")
