import torch
import matplotlib.pyplot as plt
import numpy as np

def verify_expert_obs(buffer_path="data/expert_trajectories.pth"):
    print(f"Loading buffer from {buffer_path}...")
    data = torch.load(buffer_path)
    
    # Handle list of dictionaries
    if isinstance(data, list):
        print(f"Found {len(data)} trajectories. Verifying the first one...")
        traj = data[0]
        curr_obs = traj['observation'].numpy() # (T, 30)
        target_speed = traj.get('speed', 0.0)
    else:
        # Fallback for old format
        states = data['state'].numpy()
        curr_obs = states[:, -30:]
        target_speed = curr_obs[0, -1]
    
    # Key indices from parse_expert_trajectories.py:
    # 0: pelvis_tilt
    # 6: knee_angle_r
    # 7: osl_ankle_angle_r
    # 11: knee_angle_l
    # 12: ankle_angle_l
    # 29: speed
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Pelvis Tilt
    axs[0].plot(curr_obs[:, 0], label="Pelvis Tilt")
    axs[0].set_title(f"Root Orientation (Euler) - Target Speed: {target_speed}")
    axs[0].legend()
    
    # 2. Right Leg (Knee & Ankle)
    axs[1].plot(curr_obs[:, 6], label="Bio Knee (R)", color='blue')
    axs[1].plot(curr_obs[:, 7], label="OSL Ankle (R)", color='red', linestyle='--')
    axs[1].set_title("Right Leg Joints")
    axs[1].legend()
    
    # 3. Left Leg (Knee & Ankle)
    axs[2].plot(curr_obs[:, 11], label="Bio Knee (L)", color='green')
    axs[2].plot(curr_obs[:, 12], label="Bio Ankle (L)", color='orange', linestyle='--')
    axs[2].set_title("Left Leg Joints")
    axs[2].legend()
    
    plt.tight_layout()
    plot_path = "expert_obs_verification.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Summary Statistics
    print(f"\nObservation Statistics (Total Frames: {curr_obs.shape[0]}):")
    print(f"Pelvis Tilt: min={curr_obs[:,0].min():.3f}, max={curr_obs[:,0].max():.3f}")
    print(f"R-Knee (Bio): min={curr_obs[:,6].min():.3f}, max={curr_obs[:,6].max():.3f}")
    print(f"R-Ankle (OSL): min={curr_obs[:,7].min():.3f}, max={curr_obs[:,7].max():.3f}")
    print(f"L-Knee (Bio): min={curr_obs[:,11].min():.3f}, max={curr_obs[:,11].max():.3f}")
    print(f"L-Ankle (Bio): min={curr_obs[:,12].min():.3f}, max={curr_obs[:,12].max():.3f}")
    print(f"Speed Column (Index 29): min={curr_obs[:,29].min():.3f}, max={curr_obs[:,29].max():.3f}")

if __name__ == "__main__":
    import os
    # Ensure buffer exists
    buffer_file = "data/expert_trajectories.pth"
    if not os.path.exists(buffer_file):
        print("Expert buffer not found. Please run scripts/parse_expert_trajectories.py first.")
    else:
        verify_expert_obs(buffer_file)
