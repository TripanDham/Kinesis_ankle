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
    print(" OBSERVATION MATCHING DIAGNOSTIC TOOL (NORMALIZED REPORT)")
    print("="*85)

    # 1. Load Expert Buffer
    expert_path = cfg.run.get("expert_buffer_path", "data/expert_trajectories.pth")
    if not os.path.exists(expert_path):
        print(f"Error: Expert buffer not found at {expert_path}")
        return

    print(f"Loading Expert Buffer: {expert_path}")
    expert_data = torch.load(expert_path, map_location='cpu')
    
    expert_obs_frames = []
    for traj in expert_data:
        obs = traj['observation']
        if torch.is_tensor(obs):
            obs = obs.numpy()
        # The buffer has 30D frames possibly repeated by history.
        # We want the base 30D features for statistics.
        # Assuming [T, 30*History]
        num_frames = obs.shape[0]
        # Just grab the last 30D from each concatenated frame
        raw_frames = obs.reshape(num_frames, -1, 30)[:, -1, :]
        expert_obs_frames.append(raw_frames)
    expert_obs_raw = np.concatenate(expert_obs_frames, axis=0)
    
    # 2. Initialize Environment
    print("Initializing MyoLegsGAIL Environment...")
    cfg.run.headless = True
    cfg.run.test = True
    env = MyoLegsGAIL(cfg)
    
    # 3. Collect Agent Observations
    print("Collecting Agent Observations (1000 steps)...")
    agent_obs_raw = []
    obs, _ = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        raw_frame = env.get_disc_obs()
        agent_obs_raw.append(raw_frame)
        if done or truncated:
            obs, _ = env.reset()
            
    agent_obs_raw = np.array(agent_obs_raw)
    
    # 4. Update Normalizer with both sets
    print("Updating Shared Normalizer (env.gail_norm)...")
    env.gail_norm.train()
    # History normalization actually happens on the CONCATENATED obs.
    # For diagnosis, let's normalize the 30D raw frames assuming 1 frame history for simplicity
    # or just use the first 30D of the normalizer.
    
    exp_tensor = torch.from_numpy(expert_obs_raw).float()
    agt_tensor = torch.from_numpy(agent_obs_raw).float()
    
    # We need to broadcast the 30D to the full history size if gail_norm is larger
    total_norm_dim = env.gail_norm.dim
    if total_norm_dim > 30:
        # Repeat the 30D to fit history
        history_repeat = total_norm_dim // 30
        exp_full = exp_tensor.repeat(1, history_repeat)
        agt_full = agt_tensor.repeat(1, history_repeat)
    else:
        exp_full = exp_tensor
        agt_full = agt_tensor

    env.gail_norm(exp_full)
    env.gail_norm(agt_full)
    
    # 5. Apply Normalization
    env.gail_norm.eval()
    expert_obs_norm = env.gail_norm(exp_full).numpy()[:, :30]
    agent_obs_norm = env.gail_norm(agt_full).numpy()[:, :30]

    # 6. Compare Normalized Statistics
    print("\nNormalized Distributions (Should have Mean~0, Std~1 and overlap):")
    header = f"{'Idx':<3} | {'Feature':<18} | {'Exp (Mean±Std)':<18} | {'Agent (Mean±Std)':<18} | {'Status'}"
    print(header)
    print("-" * 105)

    root_angles = ["p_tilt", "p_list", "p_rot"]
    r_leg = ["hip_f_r", "hip_a_r", "hip_r_r", "knee_r", "ank_r"]
    l_leg = ["hip_f_l", "hip_a_l", "hip_r_l", "knee_l", "ank_l"]
    angle_names = root_angles + r_leg + l_leg
    
    root_vels = ["v_tx", "v_ty", "v_tz", "v_tp", "v_tl", "v_tr"]
    vel_names = root_vels + r_leg + l_leg 
    
    all_names = angle_names + vel_names + ["target_speed"]

    for i in range(30):
        e_m, e_s = np.mean(expert_obs_norm[:, i]), np.std(expert_obs_norm[:, i])
        a_m, a_s = np.mean(agent_obs_norm[:, i]), np.std(agent_obs_norm[:, i])
        
        name = all_names[i]
        diff = abs(e_m - a_m)
        
        status = "OK"
        if diff > 1.0:
            status = "Mismatch"

        exp_str = f"{e_m:5.2f}±{e_s:4.2f}"
        agt_str = f"{a_m:5.2f}±{a_s:4.2f}"

        print(f"{i:<3} | {name:<18} | {exp_str:<18} | {agt_str:<18} | {status}")

    print("\nDiagnostic Complete.")
    env.close()

if __name__ == "__main__":
    main()
