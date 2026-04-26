import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import logging
import os

logger = logging.getLogger(__name__)

def plot_biomechanics(all_biomechanics, env):
    """
    Processes and plots biomechanics data collected over multiple evaluation runs into a single dashboard.
    
    Args:
        all_biomechanics (list of lists): [episode][timestep][dict of metrics]
        env: The environment instance, used to extract joint and actuator indices.
    """
    logger.info(f"Processing biomechanics data for {len(all_biomechanics)} episodes...")
    
    # 1. Truncate to min episode length
    min_len = min(len(ep) for ep in all_biomechanics)
    if min_len == 0:
        logger.warning("No biomechanics data to plot (empty episodes).")
        return
        
    num_eps = len(all_biomechanics)
    logger.info(f"Analyzing up to {min_len} timesteps across {num_eps} runs.")

    # 2. Extract Indices (Dynamic Discovery)
    def get_jnt_id(name):
        try:
            return env.mj_model.joint(name).id
        except (ValueError, KeyError):
            return -1

    def get_act_id(name):
        try:
            return env.mj_model.actuator(name).id
        except (ValueError, KeyError):
            return -1

    # Define potential joint names (Prosthetic vs Biological)
    # The right knee might be "osl_knee_angle_r" (prosthetic) or just "knee_angle_r" (O-model)
    knee_r_name = "osl_knee_angle_r" if get_jnt_id("osl_knee_angle_r") != -1 else "knee_angle_r"
    ankle_r_name = "osl_ankle_angle_r" if get_jnt_id("osl_ankle_angle_r") != -1 else "ankle_angle_r"

    joint_names = {
        "Hip Flexion R": "hip_flexion_r",
        "Knee R": knee_r_name,
        "Ankle R": ankle_r_name,
        "Hip Flexion L": "hip_flexion_l",
        "Knee L": "knee_angle_l",
        "Ankle L": "ankle_angle_l"
    }
    
    # Filter only available joints to avoid KeyError in jnt_qposadr
    joint_qpos_indices = {}
    joint_qvel_indices = {}
    for display, name in joint_names.items():
        jid = get_jnt_id(name)
        if jid != -1:
            joint_qpos_indices[display] = env.mj_model.jnt_qposadr[jid]
            joint_qvel_indices[display] = env.mj_model.joint(name).dofadr[0]
    
    # Actuator Indices
    soleus_l_idx = get_act_id("soleus_l")
    glutmax_l_idx = get_act_id("glutmax_l")
    glutmax_r_idx = get_act_id("glutmax_r")
    
    # Hip Torques (Net Moments)
    hip_l_dof = env.mj_model.joint("hip_flexion_l").dofadr[0] if get_jnt_id("hip_flexion_l") != -1 else -1
    hip_r_dof = env.mj_model.joint("hip_flexion_r").dofadr[0] if get_jnt_id("hip_flexion_r") != -1 else -1
    
    # Knee act/dof
    knee_act_idx = get_act_id("osl_knee_torque_actuator") 
    knee_l_dof = env.mj_model.joint("knee_angle_l").dofadr[0] if get_jnt_id("knee_angle_l") != -1 else -1
    knee_r_dof = env.mj_model.joint(knee_r_name).dofadr[0] if get_jnt_id(knee_r_name) != -1 else -1
    
    # Ankle act/dof
    ankle_act_idx = get_act_id("osl_ankle_torque_actuator")
    ankle_l_dof = env.mj_model.joint("ankle_angle_l").dofadr[0] if get_jnt_id("ankle_angle_l") != -1 else -1
    ankle_r_dof = env.mj_model.joint(ankle_r_name).dofadr[0] if get_jnt_id(ankle_r_name) != -1 else -1
    
    # Gears
    gear_ankle = env.mj_model.actuator_gear[ankle_act_idx, 0] if ankle_act_idx != -1 else 1.0

    # 3. Pre-allocate Data
    soleus_l_act = np.zeros((num_eps, min_len))
    glutmax_l_act = np.zeros((num_eps, min_len))
    glutmax_r_act = np.zeros((num_eps, min_len))
    
    hip_l_moment = np.zeros((num_eps, min_len))
    hip_r_moment = np.zeros((num_eps, min_len))
    knee_l_moment = np.zeros((num_eps, min_len))
    knee_r_moment = np.zeros((num_eps, min_len))
    ankle_l_moment = np.zeros((num_eps, min_len))
    ankle_r_moment = np.zeros((num_eps, min_len))
    
    com_vel = np.zeros((num_eps, min_len, 3)) # X, Y, Z
    
    joint_angles = {name: np.zeros((num_eps, min_len)) for name in joint_qpos_indices.keys()}
    joint_vels = {name: np.zeros((num_eps, min_len)) for name in joint_qvel_indices.keys()}
    
    imp_keys = ["knee_K", "knee_B", "knee_target", "ankle_K", "ankle_B", "ankle_target"]
    impedance_data = {k: np.zeros((num_eps, min_len)) for k in imp_keys}
    has_impedance = False

    # 4. Extract Steps
    for ep_idx in range(num_eps):
        for t in range(min_len):
            step_data = all_biomechanics[ep_idx][t]
            
            if soleus_l_idx != -1:
                soleus_l_act[ep_idx, t] = step_data["ctrl"][soleus_l_idx]
            if glutmax_l_idx != -1:
                glutmax_l_act[ep_idx, t] = step_data["ctrl"][glutmax_l_idx]
            if glutmax_r_idx != -1:
                glutmax_r_act[ep_idx, t] = step_data["ctrl"][glutmax_r_idx]
                
            if "qfrc_actuator" in step_data:
                if hip_l_dof != -1: hip_l_moment[ep_idx, t] = step_data["qfrc_actuator"][hip_l_dof]
                if hip_r_dof != -1: hip_r_moment[ep_idx, t] = step_data["qfrc_actuator"][hip_r_dof]
                if knee_l_dof != -1: knee_l_moment[ep_idx, t] = step_data["qfrc_actuator"][knee_l_dof]
                if knee_r_dof != -1: knee_r_moment[ep_idx, t] = step_data["qfrc_actuator"][knee_r_dof]
                if ankle_l_dof != -1: ankle_l_moment[ep_idx, t] = step_data["qfrc_actuator"][ankle_l_dof]
                if ankle_r_dof != -1: ankle_r_moment[ep_idx, t] = step_data["qfrc_actuator"][ankle_r_dof]
            
            # COM Velocity (Base root velocity)
            com_vel[ep_idx, t, :] = step_data["qvel"][:3]
                
            for name, qidx in joint_qpos_indices.items():
                joint_angles[name][ep_idx, t] = step_data["qpos"][qidx]
                
            for name, vidx in joint_qvel_indices.items():
                joint_vels[name][ep_idx, t] = step_data["qvel"][vidx]

            if "impedance" in step_data and step_data["impedance"]:
                has_impedance = True
                for k in imp_keys:
                    impedance_data[k][ep_idx, t] = step_data["impedance"].get(k, 0.0)

    # 5. Build Subplots (Restructured Checklist)
    rows = 7
    # Flat list of 28 titles (7 rows x 4 columns)
    subplot_titles = [
        "Hip Angle - L (Rad)", "Hip Angle - R (Rad)", "Hip Velocity - L (Rad/s)", "Hip Velocity - R (Rad/s)",
        "Hip Torque - L (Nm)", "Hip Torque - R (Nm)", "", "",
        "Knee Angle - L (Rad)", "Knee Angle - R (Rad)", "Knee Velocity - L (Rad/s)", "Knee Velocity - R (Rad/s)",
        "Ankle Angle - L (Rad)", "Ankle Angle - R (Rad)", "Ankle Velocity - L (Rad/s)", "Ankle Velocity - R (Rad/s)",
        "Ankle Stiffness K", "Ankle Damping B", "Ankle Target Angle (Rad)", "",
        "Muscle: Soleus L", "Muscle: Gluteus L", "Muscle: Gluteus R", "",
        "COM Velocity X (Fwd)", "COM Velocity Y (Lat)", "COM Velocity Z (Up)", ""
    ]

    fig = make_subplots(
        rows=rows, cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.05
    )

    def add_shaded_trace(fig, data, name, row, col, color="blue", y_title=""):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        fig.add_trace(go.Scatter(x=list(x)+list(x)[::-1], y=list(mean+std)+list(mean-std)[::-1],
                                 fill='toself', fillcolor=f'rgba({color}, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False, name=f"{name} StdDev"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=mean, line=dict(color=f'rgb({color})'), name=f"{name} Mean"), row=row, col=col)
        if y_title:
            fig.update_yaxes(title_text=y_title, row=row, col=col)

    # Row 1: Hip Kinematics
    add_shaded_trace(fig, joint_angles.get("Hip Flexion L", np.zeros(min_len)), "Hip L Angle", 1, 1, "150,0,250")
    add_shaded_trace(fig, joint_angles.get("Hip Flexion R", np.zeros(min_len)), "Hip R Angle", 1, 2, "255,100,0")
    add_shaded_trace(fig, joint_vels.get("Hip Flexion L", np.zeros(min_len)), "Hip L Vel", 1, 3, "150,0,250")
    add_shaded_trace(fig, joint_vels.get("Hip Flexion R", np.zeros(min_len)), "Hip R Vel", 1, 4, "255,100,0")

    # Row 2: Hip Torques
    add_shaded_trace(fig, hip_l_moment, "Hip L Torque", 2, 1, "100,100,100")
    add_shaded_trace(fig, hip_r_moment, "Hip R Torque", 2, 2, "0,150,255")

    # Row 3: Knee Kinematics
    add_shaded_trace(fig, joint_angles.get("Knee L", np.zeros(min_len)), "Knee L Angle", 3, 1, "150,0,250")
    add_shaded_trace(fig, joint_angles.get("Knee R", np.zeros(min_len)), "Knee R Angle", 3, 2, "255,100,0")
    add_shaded_trace(fig, joint_vels.get("Knee L", np.zeros(min_len)), "Knee L Vel", 3, 3, "150,0,250")
    add_shaded_trace(fig, joint_vels.get("Knee R", np.zeros(min_len)), "Knee R Vel", 3, 4, "255,100,0")

    # Row 4: Ankle Kinematics
    add_shaded_trace(fig, joint_angles.get("Ankle L", np.zeros(min_len)), "Ankle L Angle", 4, 1, "150,0,250")
    add_shaded_trace(fig, joint_angles.get("Ankle R", np.zeros(min_len)), "Ankle R Angle", 4, 2, "255,100,0")
    add_shaded_trace(fig, joint_vels.get("Ankle L", np.zeros(min_len)), "Ankle L Vel", 4, 3, "150,0,250")
    add_shaded_trace(fig, joint_vels.get("Ankle R", np.zeros(min_len)), "Ankle R Vel", 4, 4, "255,100,0")

    # Row 5: Impedance Parameters
    if has_impedance:
        add_shaded_trace(fig, impedance_data["ankle_K"], "Ankle Stiffness K", 5, 1, "0,150,150")
        add_shaded_trace(fig, impedance_data["ankle_B"], "Ankle Damping B", 5, 2, "150,150,0")
        add_shaded_trace(fig, impedance_data["ankle_target"], "Ankle Target Angle", 5, 3, "255,0,0")

    # Row 6: Muscle Activations
    add_shaded_trace(fig, soleus_l_act, "Soleus L", 6, 1, "255,0,0")
    add_shaded_trace(fig, glutmax_l_act, "GlutMax L", 6, 2, "200,0,50")
    add_shaded_trace(fig, glutmax_r_act, "GlutMax R", 6, 3, "200,50,0")

    # Row 7: COM Velocity
    add_shaded_trace(fig, com_vel[:, :, 0], "COM X (Fwd)", 7, 1, "0,0,0")
    add_shaded_trace(fig, com_vel[:, :, 1], "COM Y (Lat)", 7, 2, "150,150,150")
    add_shaded_trace(fig, com_vel[:, :, 2], "COM Z (Up)", 7, 3, "200,0,0")

    fig.update_layout(height=350 * rows, width=1400, title_text="MuJoCo Biomechanics Evaluation Dashboard (7-Row)", showlegend=True)
    
    output_path = os.path.abspath("biomechanics_dashboard.html")
    fig.write_html(output_path)
    
    print(f"\n" + "="*80)
    print(f"BIOMECHANICS ANALYSIS COMPLETE")
    print(f"Dashboard saved to: {output_path}")
    print(f"Indices: Ankle R Act={ankle_act_idx} (Gear={gear_ankle:.1f}), Knee L DOF={knee_l_dof}")
    print(f"="*80 + "\n")

    # Also plot the exact GAIL observations dashboard
    plot_gail_obs_dashboard(all_biomechanics, env)

def plot_gail_obs_dashboard(all_biomechanics, env):
    """Plots the exact 24D GAIL observation state for the agent's test runs."""
    logger.info("Generating 24D GAIL Observation Dashboard...")
    
    min_len = min(len(ep) for ep in all_biomechanics)
    if min_len == 0: return
    num_eps = len(all_biomechanics)

    def get_jnt(name):
        try: return env.mj_model.joint(name)
        except: return None

    # Determine prosthetic ankle/knee names
    knee_r_name = "osl_knee_angle_r" if get_jnt("osl_knee_angle_r") else "knee_angle_r"
    ankle_r_name = "osl_ankle_angle_r" if get_jnt("osl_ankle_angle_r") else "ankle_angle_r"

    angle_names = [
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", knee_r_name, ankle_r_name,
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l"
    ]
    
    vel_names = ["pelvis_tx", "pelvis_ty", "pelvis_tz"] + angle_names

    DIM_LABELS = [
        'hip_flexion_r_angle', 'hip_adduction_r_angle', 'hip_rotation_r_angle', 'knee_angle_r_angle', 'ankle_angle_r_angle',
        'hip_flexion_l_angle', 'hip_adduction_l_angle', 'hip_rotation_l_angle', 'knee_angle_l_angle', 'ankle_angle_l_angle',
        'pelvis_tx_vel', 'pelvis_ty_vel', 'pelvis_tz_vel',
        'hip_flexion_r_vel', 'hip_adduction_r_vel', 'hip_rotation_r_vel', 'knee_angle_r_vel', 'ankle_angle_r_vel',
        'hip_flexion_l_vel', 'hip_adduction_l_vel', 'hip_rotation_l_vel', 'knee_angle_l_vel', 'ankle_angle_l_vel',
        'root_height (pelvis_ty)'
    ]

    # Pre-allocate 24D data: shape (num_eps, min_len, 24)
    data_24d = np.zeros((num_eps, min_len, 24))

    # Get addresses safely
    qpos_addrs = [get_jnt(n).qposadr[0] if get_jnt(n) else 0 for n in angle_names]
    qvel_addrs = [get_jnt(n).dofadr[0] if get_jnt(n) else 0 for n in vel_names]
    pelvis_ty_qpos = get_jnt("pelvis_ty").qposadr[0] if get_jnt("pelvis_ty") else 1

    for ep_idx in range(num_eps):
        for t in range(min_len):
            step_data = all_biomechanics[ep_idx][t]
            qpos = step_data["qpos"]
            qvel = step_data["qvel"]
            
            # 10 Angles
            for i, addr in enumerate(qpos_addrs):
                data_24d[ep_idx, t, i] = qpos[addr]
                
            # 13 Velocities
            for i, addr in enumerate(qvel_addrs):
                data_24d[ep_idx, t, 10 + i] = qvel[addr]
                
            # 1 Root Height
            data_24d[ep_idx, t, 23] = qpos[pelvis_ty_qpos]

    rows, cols = 6, 4
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=DIM_LABELS, vertical_spacing=0.05)

    def add_shaded_trace(fig, data, name, row, col, color="blue"):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        fig.add_trace(go.Scatter(x=list(x)+list(x)[::-1], y=list(mean+std)+list(mean-std)[::-1],
                                 fill='toself', fillcolor=f'rgba({color}, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(x=x, y=mean, line=dict(color=f'rgb({color})'), name=f"{name} Mean", showlegend=(row==1 and col==1)), row=row, col=col)

    for dim in range(24):
        r = (dim // cols) + 1
        c = (dim % cols) + 1
        add_shaded_trace(fig, data_24d[:, :, dim], "Agent Run", r, c, "31, 119, 180")

    fig.update_layout(height=1500, width=1800, title_text="Agent Evaluation: 24D GAIL Observations", template='plotly_dark')
    
    output_path = os.path.abspath(os.path.join(os.path.dirname("biomechanics_dashboard.html"), "agent_gail_obs_dashboard.html"))
    fig.write_html(output_path)
    logger.info(f"Agent GAIL Dashboard saved to: {output_path}")
