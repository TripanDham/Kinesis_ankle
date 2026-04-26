import os
import sys
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.myolegs_IL import MyoLegsGAIL
from scripts.plot_expert_trajectories import generate_trajectories_for_plotting

def main():
    # 1. Get expert data for tf01_0p6_01_rotated_ik.mot
    data_dir = "/media/tripan/Data/DDP/amputee_data/training_data"
    trajectories = generate_trajectories_for_plotting(data_dir)
    
    target_traj = None
    for t in trajectories:
        if '01_rotated_ik' in t['filename'] and '0p6' in t['filename']:
            target_traj = t
            break
            
    if target_traj is None:
        print("Could not find tf01_0p6_01_rotated_ik.mot")
        return
        
    expert_24d = target_traj['data'][0] # First frame
    
    # We need the 23D version (remove root height which is index 23)
    expert_23d = expert_24d[:23]
    
    # 2. Get environment initial state (walk_right keyframe)
    with hydra.initialize(config_path="../cfg", version_base="1.1"):
        cfg = hydra.compose(config_name="config", overrides=["env=myolegs_gail", "run=myolegs_gail", "learning=gail_mlp"])
    
    # Mock wandb/logger config if needed
    
    env = MyoLegsGAIL(cfg)
    env.reset()
    
    # Get the raw 23D observation directly from the environment mapping
    # 10 Angles
    angles = env.mj_data.qpos[env.obs_qpos_idx].astype(np.float32)
    # 13 Velocities 
    vels = env.mj_data.qvel[env.obs_qvel_idx].astype(np.float32)
    
    env_23d = np.concatenate([angles, vels])
    
    DIM_LABELS_23D = [
        'hip_flexion_r_ang', 'hip_adduction_r_ang', 'hip_rotation_r_ang', 'knee_angle_r_ang', 'ankle_angle_r_ang',
        'hip_flexion_l_ang', 'hip_adduction_l_ang', 'hip_rotation_l_ang', 'knee_angle_l_ang', 'ankle_angle_l_ang',
        'pelvis_tx_vel', 'pelvis_ty_vel', 'pelvis_tz_vel',
        'hip_flexion_r_vel', 'hip_adduction_r_vel', 'hip_rotation_r_vel', 'knee_angle_r_vel', 'ankle_angle_r_vel',
        'hip_flexion_l_vel', 'hip_adduction_l_vel', 'hip_rotation_l_vel', 'knee_angle_l_vel', 'ankle_angle_l_vel'
    ]
    
    print("\n" + "="*80)
    print(f"{'Dimension':<25} | {'Expert (First Frame)':<20} | {'Env (walk_right key)':<20} | {'Diff'}")
    print("-" * 80)
    
    for i in range(23):
        exp_val = expert_23d[i]
        env_val = env_23d[i]
        diff = abs(exp_val - env_val)
        print(f"{DIM_LABELS_23D[i]:<25} | {exp_val:>20.4f} | {env_val:>20.4f} | {diff:>10.4f}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
