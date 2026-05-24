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

def plot_discriminator_saliency(saliency_data, feature_names, output_path="discriminator_saliency.html"):
    """
    Plots the discriminator saliency as a heatmap.
    
    Args:
        saliency_data (np.ndarray): Shape (num_steps, obs_dim)
        feature_names (list): List of feature names
        output_path (str): Save path
    """
    import plotly.express as px
    
    # Transpose so states are on Y axis, time on X axis
    fig = px.imshow(
        saliency_data.T,
        labels=dict(x="Timestep", y="State Feature", color="Gradient Magnitude"),
        y=feature_names,
        aspect="auto",
        title="Discriminator Saliency Heatmap (Gradients w.r.t Input State)",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=1200, width=1500, template='plotly_dark')
    fig.write_html(output_path)
    logger.info(f"Discriminator saliency dashboard saved to: {output_path}")


def plot_reference_trajectory(env, motion_key: str, output_path: str = "reference_trajectory.html"):
    """
    Queries ProstWalkCore.get_reference_state across the full motion duration and plots
    all 7 tracked body-segment positions (body_xpos_ref) and velocities (body_vel_ref)
    in an interactive Plotly dashboard.

    Args:
        env:         A MyoLegsGAIL instance with motion_lib and tracking metadata loaded.
        motion_key:  Motion to visualise (e.g. 'tf01_0p6_01_rotated_ik').
        output_path: Where to write the HTML file.
    """
    from src.KinesisCore.prostwalk_core import TRACKED_BODY_NAMES
    axis_labels = ["X", "Y", "Z"]
    axis_colors = ["#00BFFF", "#FF6B6B", "#90EE90"]

    # Determine subject from motion key
    subject_id = "tf01" if "tf01" in motion_key else \
                 "tf11" if "tf11" in motion_key else \
                 "tf08" if "tf08" in motion_key else None
    subj_settings = env._subject_correction_map.get(subject_id) if subject_id else None

    # Determine motion duration
    motion_data = env.motion_lib.motion_data.get(motion_key)
    if motion_data is None:
        logger.warning(f"plot_reference_trajectory: motion '{motion_key}' not found in motion_lib.")
        return
    fps = float(motion_data.get("fps", 200))
    n_frames = len(motion_data.get("qpos", []))
    duration = n_frames / fps
    dt = 1.0 / fps
    times = np.arange(0.0, duration, dt)

    # Collect body position and velocity reference trajectories
    xpos_traj = []   # (T, 7, 3)
    vel_traj  = []   # (T, 7, 3)

    for t in times:
        ref = env.motion_lib.get_reference_state(
            motion_key=motion_key,
            t_start=0.0,
            elapsed=t,
            joint_qpos_idx=env.obs_qpos_idx,
            joint_qvel_idx=env.obs_qvel_idx,
            subject_settings=subj_settings,
        )
        xpos_traj.append(ref["body_xpos_ref"].reshape(-1, 3))
        vel_traj.append(ref["body_vel_ref"].reshape(-1, 3))

    xpos_traj = np.array(xpos_traj)   # (T, 7, 3)
    vel_traj  = np.array(vel_traj)    # (T, 7, 3)

    n_bodies = len(TRACKED_BODY_NAMES)

    # Layout: 7 bodies × 2 cols (position | velocity), 3 lines per subplot (X/Y/Z)
    subplot_titles = []
    for body in TRACKED_BODY_NAMES:
        subplot_titles += [f"{body} — position (m)", f"{body} — velocity (m/s)"]

    fig = make_subplots(
        rows=n_bodies, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.025,
        horizontal_spacing=0.08,
    )

    for i, body in enumerate(TRACKED_BODY_NAMES):
        row = i + 1
        for ax_i, (ax_label, ax_color) in enumerate(zip(axis_labels, axis_colors)):
            show_legend = (i == 0)
            # Position
            fig.add_trace(
                go.Scatter(x=times, y=xpos_traj[:, i, ax_i], mode="lines",
                           line=dict(color=ax_color, width=1.5),
                           name=ax_label, showlegend=show_legend,
                           legendgroup=ax_label),
                row=row, col=1,
            )
            # Velocity
            fig.add_trace(
                go.Scatter(x=times, y=vel_traj[:, i, ax_i], mode="lines",
                           line=dict(color=ax_color, width=1.5, dash="dot"),
                           name=ax_label, showlegend=False,
                           legendgroup=ax_label),
                row=row, col=2,
            )

    fig.update_layout(
        height=260 * n_bodies,
        width=1600,
        title_text=f"Reference Body Trajectory — {motion_key}  (subject: {subject_id or 'unknown'})",
        template="plotly_dark",
        legend=dict(orientation="h", y=1.01, x=0),
        font=dict(size=11),
    )

    for col in (1, 2):
        fig.update_xaxes(title_text="Time (s)", row=n_bodies, col=col)

    fig.write_html(output_path)
    logger.info(f"Reference trajectory plot saved to: {output_path}")

def plot_gail_obs_dashboard_episode(episode_biomech, env, episode_idx, output_dir=None):
    """Plot the 24‑D GAIL observation data for a single evaluation episode.

    Args:
        episode_biomech (list): List of biomechanics dicts for each timestep of the episode.
        env: Environment instance for joint mapping.
        episode_idx (int): Index of the episode (for filename labeling).
        output_dir (str, optional): Directory to write the HTML file. If None, defaults to the same folder as the aggregated dashboard.
    """
    if not episode_biomech:
        logger.warning(f"Episode {episode_idx} contains no biomechanics data; skipping plot.")
        return

    # Helper to safely get joint objects.
    def get_jnt(name):
        try:
            return env.mj_model.joint(name)
        except Exception:
            return None

    # Determine prosthetic joint names.
    knee_r_name = "osl_knee_angle_r" if get_jnt("osl_knee_angle_r") else "knee_angle_r"
    ankle_r_name = "osl_ankle_angle_r" if get_jnt("osl_ankle_angle_r") else "ankle_angle_r"

    angle_names = [
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", knee_r_name, ankle_r_name,
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    ]
    vel_names = ["pelvis_tx", "pelvis_ty", "pelvis_tz"] + angle_names

    # Resolve addresses, fall back to 0 if missing.
    qpos_addrs = [get_jnt(n).qposadr[0] if get_jnt(n) else 0 for n in angle_names]
    qvel_addrs = [get_jnt(n).dofadr[0] if get_jnt(n) else 0 for n in vel_names]
    pelvis_ty_qpos = get_jnt("pelvis_ty").qposadr[0] if get_jnt("pelvis_ty") else 0

    min_len = len(episode_biomech)
    data_24d = np.zeros((min_len, 24))
    for t, step in enumerate(episode_biomech):
        qpos = step["qpos"]
        qvel = step["qvel"]
        for i, adr in enumerate(qpos_addrs):
            data_24d[t, i] = qpos[adr]
        for i, adr in enumerate(qvel_addrs):
            data_24d[t, 10 + i] = qvel[adr]
        data_24d[t, 23] = qpos[pelvis_ty_qpos]

    # Plot each dimension in a 6x4 grid (same layout as aggregated dashboard).
    rows, cols = 6, 4
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[
        'hip_flexion_r_angle', 'hip_adduction_r_angle', 'hip_rotation_r_angle', 'knee_angle_r_angle',
        'ankle_angle_r_angle', 'hip_flexion_l_angle', 'hip_adduction_l_angle', 'hip_rotation_l_angle',
        'knee_angle_l_angle', 'ankle_angle_l_angle', 'pelvis_tx_vel', 'pelvis_ty_vel',
        'pelvis_tz_vel', 'hip_flexion_r_vel', 'hip_adduction_r_vel', 'hip_rotation_r_vel',
        'knee_angle_r_vel', 'ankle_angle_r_vel', 'hip_flexion_l_vel', 'hip_adduction_l_vel',
        'hip_rotation_l_vel', 'knee_angle_l_vel', 'ankle_angle_l_vel', 'root_height (pelvis_ty)'
    ], vertical_spacing=0.05)

    def add_shaded(fig, data, name, r, c, color="31, 119, 180"):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(mean + std) + list(mean - std),
                                 fill='toself', fillcolor=f'rgba({color}, 0.2)',
                                 line=dict(color='rgba(255,255,255,0)'), showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=mean, line=dict(color=f'rgb({color})'), name=name, showlegend=False), row=r, col=c)

    for dim in range(24):
        r = dim // cols + 1
        c = dim % cols + 1
        add_shaded(fig, data_24d[:, dim:dim+1], f"Dim {dim}", r, c)

    fig.update_layout(height=1500, width=1800,
                      title_text=f"Episode {episode_idx} – 24D GAIL Observations", template='plotly_dark')
    out_dir = output_dir or os.path.abspath(os.path.dirname("biomechanics_dashboard.html"))
    out_path = os.path.join(out_dir, f"gail_obs_ep{episode_idx:03d}.html")
    fig.write_html(out_path)
    logger.info(f"Saved per‑episode GAIL observation plot to {out_path}")


def plot_body_and_reference_episode(episode_body, env, motion_key, t_start, episode_idx, output_dir=None):
    """Plot simulated body positions/velocities together with the reference trajectory for a single episode.

    Args:
        episode_body (list): List of dicts with keys ``pos`` and ``vel`` (both (7,3) arrays).
        env: Environment (provides TRACKED_BODY_NAMES and reference helper).
        motion_key (str): Identifier for the reference motion.
        t_start (float): The actual motion start time for this episode.
        episode_idx (int): Index for naming the output file.
        output_dir (str, optional): Where to write the HTML file.
    """
    if not episode_body:
        logger.warning(f"Episode {episode_idx} has no body tracking data; skipping plot.")
        return

    from src.KinesisCore.prostwalk_core import TRACKED_BODY_NAMES
    
    # ── Time Alignment ──
    # The simulation steps at 30 Hz (control step dt) starting from the RSI t_start phase.
    dt = env._sim_dt
    times_sim = np.arange(len(episode_body)) * dt
    times_ref = times_sim  # Keep times_ref name so Scatter lines don't need signature edits

    # Subject correction settings.
    subj_id = None
    if "tf01" in motion_key:
        subj_id = "tf01"
    elif "tf11" in motion_key:
        subj_id = "tf11"
    elif "tf08" in motion_key:
        subj_id = "tf08"
    subj_settings = env._subject_correction_map.get(subj_id) if subj_id else None

    # Query reference at elapsed=0 to compute the static horizontal alignment offset.
    # This mirrors what myolegs_IL.py does at the first step of each episode so that
    # both curves share the same absolute coordinate origin.
    ref_t0 = env.motion_lib.get_reference_state(
        motion_key=motion_key,
        t_start=t_start,
        elapsed=0.0,
        joint_qpos_idx=env.obs_qpos_idx,
        joint_qvel_idx=env.obs_qvel_idx,
        subject_settings=subj_settings,
    )
    ref_pelvis_xy_t0 = ref_t0["body_xpos_ref"].reshape(-1, 3)[0, :2].copy()

    xpos_ref, vel_ref = [], []
    for t in times_sim:
        ref = env.motion_lib.get_reference_state(
            motion_key=motion_key,
            t_start=t_start,
            elapsed=t,
            joint_qpos_idx=env.obs_qpos_idx,
            joint_qvel_idx=env.obs_qvel_idx,
            subject_settings=subj_settings,
        )
        bxpos = ref["body_xpos_ref"].reshape(-1, 3).copy()
        xpos_ref.append(bxpos)
        vel_ref.append(ref["body_vel_ref"].reshape(-1, 3))
    xpos_ref = np.array(xpos_ref)
    vel_ref = np.array(vel_ref)

    # Convert episode_body to arrays.
    pos_sim = np.stack([b["pos"] for b in episode_body])
    vel_sim = np.stack([b["vel"] for b in episode_body])

    # Apply horizontal alignment: align reference origin to simulation origin at t=0.
    # sim origin at t=0 is pos_sim[0, 0, :2] (pelvis XY at first recorded step).
    sim_pelvis_xy_t0 = pos_sim[0, 0, :2].copy()
    horizontal_offset = ref_pelvis_xy_t0 - sim_pelvis_xy_t0  # (2,)
    xpos_ref[:, :, :2] -= horizontal_offset  # broadcast over T and all 7 bodies


    axis_labels = ["X", "Y", "Z"]
    # Curated premium colors: X = Neon Red, Y = Neon Blue, Z = Neon Green
    axis_colors = ["#FF2D55", "#00F0FF", "#39FF14"]

    subplot_titles = []
    for body in TRACKED_BODY_NAMES:
        subplot_titles += [
            f"{body} Pos X", f"{body} Pos Y", f"{body} Pos Z",
            f"{body} Vel X", f"{body} Vel Y", f"{body} Vel Z"
        ]

    fig = make_subplots(rows=len(TRACKED_BODY_NAMES), cols=6, subplot_titles=subplot_titles,
                        shared_xaxes=True, vertical_spacing=0.04, horizontal_spacing=0.04)

    for i, body in enumerate(TRACKED_BODY_NAMES):
        row = i + 1
        show_legend = (i == 0)
        
        for ax_i, (label, color) in enumerate(zip(axis_labels, axis_colors)):
            # Position columns: 1, 2, 3
            col_pos = ax_i + 1
            
            # Position – Simulation (Solid Line)
            fig.add_trace(go.Scatter(
                x=times_ref, y=pos_sim[:, i, ax_i], mode="lines",
                line=dict(color=color, width=2.0),
                name=f"Sim {label}",
                showlegend=show_legend,
                legendgroup=f"sim_{label}"
            ), row=row, col=col_pos)
            
            # Position – Reference (Dashed Line)
            fig.add_trace(go.Scatter(
                x=times_ref, y=xpos_ref[:, i, ax_i], mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                name=f"Ref {label}",
                showlegend=show_legend,
                legendgroup=f"ref_{label}"
            ), row=row, col=col_pos)

            # Velocity columns: 4, 5, 6
            col_vel = ax_i + 4
            
            # Velocity – Simulation (Solid Line)
            fig.add_trace(go.Scatter(
                x=times_ref, y=vel_sim[:, i, ax_i], mode="lines",
                line=dict(color=color, width=2.0),
                name=f"Sim {label}",
                showlegend=False,
                legendgroup=f"sim_{label}"
            ), row=row, col=col_vel)
            
            # Velocity – Reference (Dashed Line)
            fig.add_trace(go.Scatter(
                x=times_ref, y=vel_ref[:, i, ax_i], mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                name=f"Ref {label}",
                showlegend=False,
                legendgroup=f"ref_{label}"
            ), row=row, col=col_vel)

    fig.update_layout(height=280 * len(TRACKED_BODY_NAMES), width=1800,
                      title_text=f"Episode {episode_idx} – Body Position & Velocity vs Reference ({motion_key})",
                      template="plotly_dark", legend=dict(orientation="h", y=1.02, x=0))
    out_dir = output_dir or os.path.abspath(os.path.dirname("biomechanics_dashboard.html"))
    out_path = os.path.join(out_dir, f"body_vs_ref_ep{episode_idx:03d}.html")
    fig.write_html(out_path)
    logger.info(f"Saved per‑episode body vs reference plot to {out_path}")
