import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.myolegs_IL import MyoLegsGAIL
from src.agents.agent_gail import AgentGAIL

@hydra.main(config_path="../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configurations loaded via Hydra.")
    
    # Overrides for verification
    cfg.run.num_threads = 1
    cfg.run.im_eval = False
    cfg.run.fast_forward = False

    device = "cpu"
    print("Initializing environment and agent...")
    agent = AgentGAIL(
        cfg=cfg,
        dtype=torch.float32,
        device=device,
        training=True
    )
    
    agent.setup_env()
    env = agent.env
    
    output_file = "verify_step_output.txt"
    print(f"Running 20 steps and saving to {output_file}...")
    
    # Calculate sizes for formatting
    task_obs_size = env.get_task_obs_size()
    hist_len = env.history_len
    frame_size = 24
    
    # Temporarily reset to discover proprio mapping safely
    env.reset()
    prop_sizes = {k: v.size for k, v in env.proprioception.items()}
    
    # Define labels for the 24D GAIL slice
    angle_labels = ["hip_flex_r", "hip_add_r", "hip_rot_r", "knee_r", "ankle_r", "hip_flex_l", "hip_add_l", "hip_rot_l", "knee_l", "ankle_l"]
    vel_labels = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "hip_flex_r", "hip_add_r", "hip_rot_r", "knee_r", "ankle_r", "hip_flex_l", "hip_add_l", "hip_rot_l", "knee_l", "ankle_l"]
    gail_labels = angle_labels + vel_labels + ["root_height"]
    
    with open(output_file, "w") as f:
        f.write("=== GAIL Pipeline Verification (20 Steps) ===\n\n")
        
        # --- Pre-seeded Normalizer Stats ---
        f.write("=== [0] PRE-SEEDED NORMALIZER STATS (30D) ===\n")
        f.write("These values were calculated from the expert dataset and are FROZEN for training.\n\n")
        norm_mean = agent.policy_net.norm.mean[:30].numpy()
        norm_std = agent.policy_net.norm.std[:30].numpy()
        
        f.write(f"{'Dimension':<25} | {'Mean':<10} | {'Std':<10}\n")
        f.write("-" * 50 + "\n")
        for i, label in enumerate(gail_labels):
            f.write(f"{i:02d}: {label:<21} | {norm_mean[i]:>10.4f} | {norm_std[i]:>10.4f}\n")
        f.write("\n")

        obs, _ = env.reset()
        for step in range(20):
            action = env.action_space.sample()
            
            f.write(f"--- STEP {step+1} ---\n")
            f.write(f"[1] Observations (Policy full state, {len(obs)}D):\n")
            
            # --- Format Task Obs (GAIL History) ---
            gail_obs = obs[:task_obs_size]
            proprio_obs = obs[task_obs_size:]
            
            f.write(f"  -> GAIL History Vector ({task_obs_size}D):\n")
            frames = gail_obs.reshape(hist_len, frame_size)
            for i, frame in enumerate(frames):
                f.write(f"     Frame t-{hist_len - 1 - i}: {np.array2string(frame[:10], precision=2, suppress_small=True)} (Angles)\n")
                f.write(f"                 {np.array2string(frame[10:23], precision=2, suppress_small=True)} (Vels)\n")
                f.write(f"                 Root Height: {frame[23]:.3f}\n")
            
            # --- Format Proprioceptive Obs ---
            f.write(f"\n  -> Proprioceptive Inputs ({len(proprio_obs)}D):\n")
            idx = 0
            for name, size in prop_sizes.items():
                chunk = proprio_obs[idx:idx+size]
                f.write(f"     {name} ({size}D): {np.array2string(chunk, precision=3, suppress_small=True, max_line_width=120)}\n")
                idx += size
            f.write("\n")
            
            # 2. PPO Normalised Observations
            f.write(f"[2] Normalised Observations (PPO Policy Internal RunningNorm):\n")
            
            # The policy network has an internal RunningNorm that normalizes the full 943D state
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                norm_obs_tensor = agent.policy_net.norm(obs_tensor)
                norm_obs = norm_obs_tensor.numpy()[0]
                
            f.write(f"  -> {len(norm_obs)}D Vector Statistics:\n")
            f.write(f"     Min: {np.min(norm_obs):.4f} | Max: {np.max(norm_obs):.4f} | Mean: {np.mean(norm_obs):.4f}\n")
            f.write(f"     First 20 vals: {np.array2string(norm_obs[:20], precision=3, suppress_small=True)}\n")
            
            f.write(f"\n  -> (Note: The first {task_obs_size} dimensions of this normalizer are automatically applied to the Discriminator inputs. The expert data seeding procedure locked these values.)\n\n")
            
            # 3. Expert data fed to discriminator
            target_speed = env.target_speed
            expert_tensor = agent.loader_exp.dataset.sample_by_speed(np.array([target_speed]), 1)
            
            with torch.no_grad():
                agent.policy_net.norm.eval()
                # Dummy pad for the normalizer
                padded_exp = torch.zeros(1, agent.policy_net.norm.dim)
                padded_exp[:, :task_obs_size] = expert_tensor
                exp_norm = agent.policy_net.norm(padded_exp)[0, :task_obs_size].numpy()
                
            f.write(f"[3] Expert Data Fed to Discriminator (Matching V={target_speed:.2f}, {len(exp_norm)}D):\n")
            f.write(f"    (This is normalized by the PRE-SEEDED PPO Normalizer over the first {task_obs_size} indices)\n")
            exp_frames = exp_norm.reshape(hist_len, frame_size)
            for i, frame in enumerate(exp_frames):
                f.write(f"     Expert Frame t-{hist_len - 1 - i}: {np.array2string(frame[:10], precision=2, suppress_small=True)} (Angles)\n")
                f.write(f"                        {np.array2string(frame[10:23], precision=2, suppress_small=True)} (Vels)\n")
                f.write(f"                        Root Height: {frame[23]:.3f}\n")
            f.write("\n")
            
            # 4. Rewards
            next_obs, reward, term, trunc, info = env.step(action)
            f.write(f"[4] Reward Breakdown:\n")
            f.write(f"Total Reward: {reward:.4f}\n")
            if hasattr(env, 'reward_info'):
                for key, val in env.reward_info.items():
                    f.write(f"  - {key}: {val:.4f}\n")
            f.write("\n=============================================\n\n")
            
            obs = next_obs
            
    print(f"Done! Results formatted and written to {output_file}")

if __name__ == "__main__":
    main()
