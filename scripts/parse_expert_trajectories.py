import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import re

# Observation Layout (30D) expected by the agent:
# 0-2: Root orientation (tilt, list, rot)
# 3-5: Right Hip (flexion, adduction, rotation)
# 6: Right Knee (knee_angle_r)
# 7: Right Ankle (osl_ankle_angle_r)
# 8-10: Left Hip (flexion, adduction, rotation)
# 11: Left Knee (knee_angle_l)
# 12: Left Ankle (ankle_angle_l)
# 13-18: Root Velocity (tx, ty, tz, tp, tl, tr)
# 19-28: Joint Velocities (same order as angles)
# 29: Target Speed

JOINT_COLUMNS = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',         # 0, 1, 2
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', # 3, 4, 5
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', # 6, 7, 8, 9, 10
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'  # 11, 12, 13, 14, 15
]

def extract_speed(filename):
    match = re.search(r'_(\d+)p(\d+)_', filename)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.0

def find_col(df_cols, target):
    if target in df_cols:
        return target
    for c in df_cols:
        if target in c:
            return c
    return None

def parse_mot(filepath):
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
    df = pd.read_csv(filepath, sep='\t', skiprows=header_end)
    if len(df.columns) <= 1:
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_end)
    return df, in_degrees

def generate_trajectories(data_dir, output_path, target_freq=30.0):
    mot_files = sorted(list(Path(data_dir).glob("*.mot")))
    print(f"Found {len(mot_files)} .mot files.")
    target_dt = 1.0 / target_freq
    trajectories = []
    
    # Range constraints to prevent discriminator "cheating" on numerical noise
    VEL_LIMIT = 20.0 # rad/s. Physiologically reasonable for joints.
    TRANS_VEL_LIMIT = 5.0 # m/s.
    
    for mot_file in tqdm(mot_files):
        filename = mot_file.name
        if filename.endswith(".joblib"): continue
        
        speed = extract_speed(filename)
        df, in_degrees = parse_mot(str(mot_file))
        if len(df) < 5: continue
        
        times = df['time'].values
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        # 1. Map columns and scale
        raw_pos = np.zeros((len(df), 16), dtype=np.float32)
        for i, target in enumerate(JOINT_COLUMNS):
            col_name = find_col(df.columns, target)
            if col_name:
                scale = 1.0 if 'pelvis_t' in target else unit_scale
                val = df[col_name].values * scale
                
                # Biomechanical Flip: OpenSim (+) -> MuJoCo (-) Flexion
                if 'knee' in target or 'ankle' in target:
                    val = -val
                
                raw_pos[:, i] = val

        # 2. 180-Degree Y-Rotation (Turn character around if walking in -X)
        if raw_pos[-1, 0] < raw_pos[0, 0]:
            raw_pos[:, 0] = -raw_pos[:, 0]  # tx
            raw_pos[:, 2] = -raw_pos[:, 2]  # tz (if applicable)
            raw_pos[:, 3] = -raw_pos[:, 3]  # pelvis_tilt
            raw_pos[:, 4] = -raw_pos[:, 4]  # pelvis_list
            raw_pos[:, 5] = raw_pos[:, 5] + np.pi # pelvis_rotation
        
        # 3. Heading Normalization (Center Yaw around 0)
        avg_yaw = np.mean(raw_pos[:, 5])
        raw_pos[:, 5] = raw_pos[:, 5] - avg_yaw
        
        # 4. Resample to 30Hz
        new_times = np.arange(times[0], times[-1], target_dt)
        resampled_raw_pos = np.zeros((len(new_times), 16), dtype=np.float32)
        for i in range(16):
            resampled_raw_pos[:, i] = np.interp(new_times, times, raw_pos[:, i])
            
        # 5. Construct 24D Observation Frame
        # 10 Angles: skips 0,1,2 (Pos) and 3,4,5 (Rot)
        obs_angles = resampled_raw_pos[:, 6:16]
        
        # Velocities: Finite difference (16D raw -> 13D pruned)
        resampled_vels = np.zeros((len(new_times), 16), dtype=np.float32)
        diffs = np.diff(resampled_raw_pos, axis=0) / (target_dt + 1e-8)
        
        # CLEANING: Clip velocity outliers that cause discriminator collapse
        TRANS_VEL_LIMIT, VEL_LIMIT = 5.0, 40.0
        diffs[:, 0:3] = np.clip(diffs[:, 0:3], -TRANS_VEL_LIMIT, TRANS_VEL_LIMIT)
        diffs[:, 3:16] = np.clip(diffs[:, 3:16], -VEL_LIMIT, VEL_LIMIT)
        
        resampled_vels[:-1] = diffs
        resampled_vels[-1] = resampled_vels[-2]

        # Prune to 13D: We keep 0,1,2 (Translational Vels) and 6:16 (Joint Vels)
        vel_indices = [0, 1, 2] + list(range(6, 16))
        obs_vels = resampled_vels[:, vel_indices]
        
        # Sequence: [Angles 10] + [Vels 13] + [Pelvis Height 1] = 24D
        height_col = resampled_raw_pos[:, 1:2]
        obs_traj = np.concatenate([obs_angles, obs_vels, height_col], axis=1)
        
        trajectories.append({
            'speed': speed,
            'observation': torch.from_numpy(obs_traj)
        })

    print(f"Saving {len(trajectories)} trajectories to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(trajectories, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/tripan/Data/DDP/amputee_data/training_data")
    parser.add_argument("--output_path", type=str, default="data/expert_trajectories.pth")
    parser.add_argument("--target_freq", type=float, default=30.0)
    args = parser.parse_args()
    generate_trajectories(args.data_dir, args.output_path, args.target_freq)
