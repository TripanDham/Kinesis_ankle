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
