import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from src.env.myolegs_IL import MyoLegsGAIL
import logging

# Disable hydra file logging for this tool
logging.getLogger('hydra').setLevel(logging.WARNING)

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "="*85)
    print(" OBSERVATION MATCHING DIAGNOSTIC TOOL (FULL REPORT)")
    print("="*85)

    # 1. Load Expert Buffer
    expert_path = cfg.run.get("expert_buffer_path", "data/expert_trajectories.pth")
    if not os.path.exists(expert_path):
        print(f"Error: Expert buffer not found at {expert_path}")
        return

    print(f"Loading Expert Buffer: {expert_path}")
    expert_data = torch.load(expert_path, map_location='cpu')
    
    expert_obs = []
    for traj in expert_data:
        obs = traj['observation']
        if torch.is_tensor(obs):
            obs = obs.numpy()
        if obs.shape[1] > 30:
            obs = obs[:, -30:] # Assume [Batch, History*30] -> grab last frame
        expert_obs.append(obs)
    expert_obs = np.concatenate(expert_obs, axis=0)
    
    # 2. Initialize Environment
    print("Initializing MyoLegsGAIL Environment...")
    cfg.run.headless = True
    cfg.run.test = True
    env = MyoLegsGAIL(cfg)
    
    # 3. Collect Agent Observations
    print("Collecting Agent Observations (1000 steps)...")
    agent_obs = []
    obs, _ = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        raw_frame = env.get_disc_obs()
        agent_obs.append(raw_frame)
        if done or truncated:
            obs, _ = env.reset()
            
    agent_obs = np.array(agent_obs)
    env.close()

    # 4. Compare Statistics
    print("\nExpert vs Agent Distribution Details:")
    header = f"{'Idx':<3} | {'Feature':<18} | {'Exp (Mean±Std)':<18} | {'Agent (Mean±Std)':<18} | {'Exp Range':<15} | {'Stat'}"
    print(header)
    print("-" * 105)

    root_angles = ["p_tilt", "p_list", "p_rot"]
    r_leg = ["hip_f_r", "hip_a_r", "hip_r_r", "knee_r", "ank_r"]
    l_leg = ["hip_f_l", "hip_a_l", "hip_r_l", "knee_l", "ank_l"]
    angle_names = root_angles + r_leg + l_leg
    
    root_vels = ["v_tx", "v_ty", "v_tz", "v_tp", "v_tl", "v_tr"]
    vel_names = root_vels + r_leg + l_leg 
    
    all_names = angle_names + vel_names + ["target_speed"]
    total_dims = min(expert_obs.shape[1], agent_obs.shape[1], len(all_names))

    for i in range(total_dims):
        e_m, e_s = np.mean(expert_obs[:, i]), np.std(expert_obs[:, i])
        a_m, a_s = np.mean(agent_obs[:, i]), np.std(agent_obs[:, i])
        e_min, e_max = np.min(expert_obs[:, i]), np.max(expert_obs[:, i])
        
        name = all_names[i]
        diff_scaled = abs(e_m - a_m) / (e_s + 1e-6)
        
        status = "OK"
        if diff_scaled > 10 and abs(e_m - a_m) > 0.5:
            status = "!!! CHEAT !!!"
        elif diff_scaled > 5:
            status = "Warning"

        exp_str = f"{e_m:5.2f}±{e_s:4.2f}"
        agt_str = f"{a_m:5.2f}±{a_s:4.2f}"
        range_str = f"[{e_min:4.1f}, {e_max:4.1f}]"

        print(f"{i:<3} | {name:<18} | {exp_str:<18} | {agt_str:<18} | {range_str:<15} | {status}")

    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    main()
