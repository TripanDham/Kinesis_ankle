import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import re

# Observation Layout (30D):
# 0-2: Root orientation (tilt, list, rot)
# 3-5: Right Hip (flexion, adduction, rotation)
# 6: Right Knee (knee_angle_r)
# 7: Right Ankle (osl_ankle_angle_r)
# 8-10: Left Hip (flexion, adduction, rotation)
# 11: Left Knee (knee_angle_l)
# 12: Left Ankle (ankle_angle_l)
# 13-18: Root Velocity (Linear 3, Angular 3)
# 19-28: Joint Velocities (Right 5, Left 5)
# 29: Target Speed

JOINT_COLUMNS = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'
]

# Mapping specifically for OSL model names
MOT_TO_OBS = {
    'osl_knee_angle_r': 'knee_angle_r',
    'osl_ankle_angle_r': 'ankle_angle_r',
}

def extract_speed(filename):
    """Extracts speed from filename like tf01_0p6_01_rotated_ik.mot"""
    match = re.search(r'_(\d+)p(\d+)_', filename)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.0

def parse_mot(filepath):
    """Parses .mot file and returns a dataframe + metadata."""
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
        # Retry with space separator
        df = pd.read_csv(filepath, sep='\s+', skiprows=header_end)
        
    return df, in_degrees

def generate_trajectories(data_dir, output_path):
    mot_files = sorted(list(Path(data_dir).glob("*.mot")))
    print(f"Found {len(mot_files)} .mot files.")
    
    trajectories = []
    
    for mot_file in tqdm(mot_files):
        filename = mot_file.name
        # Skip joblib if mistakenly found
        if filename.endswith(".joblib"): continue
        
        speed = extract_speed(filename)
        df, in_degrees = parse_mot(str(mot_file))
        
        num_frames = len(df)
        if num_frames < 2: continue
        
        times = df['time'].values
        dt = times[1] - times[0]
        
        unit_scale = np.pi / 180.0 if in_degrees else 1.0
        
        # 1. Extract raw positions for all 16 columns (6 root + 10 joints)
        raw_pos = np.zeros((num_frames, len(JOINT_COLUMNS)), dtype=np.float32)
        translation_cols = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        for i, col in enumerate(JOINT_COLUMNS):
            if col in df.columns:
                # Scale by degrees-to-rad UNLESS it's a translation column
                scale = 1.0 if col in translation_cols else unit_scale
                raw_pos[:, i] = df[col].values * scale

        # 2. Extract angles for observation (13D: Exclude pelvis_tx/ty/tz at 0,1,2)
        angles = raw_pos[:, 3:16] # (num_frames, 13)
        
        # 3. Compute Velocities (16D: Finite difference of all 16 pos columns)
        velocities = np.zeros((num_frames, 16), dtype=np.float32)
        velocities[:-1] = np.diff(raw_pos, axis=0) / dt
        velocities[-1] = velocities[-2] # Pad last frame
        
        # 4. Target Speed (1D)
        speed_col = np.full((num_frames, 1), speed, dtype=np.float32)
        
        # 5. Concatenate into Final Observation (30D)
        # Sequence: Angles (13) + Velocities (16) + Speed (1)
        obs_traj = np.concatenate([angles, velocities, speed_col], axis=1) # (T, 30)
            
        trajectories.append({
            'speed': speed,
            'observation': torch.from_numpy(obs_traj)
        })

    # Save all trajectories
    print(f"Saving {len(trajectories)} trajectories to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(trajectories, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/tripan/Data/DDP/amputee_data/training_data")
    parser.add_argument("--output_path", type=str, default="data/expert_trajectories.pth")
    args = parser.parse_args()
    
    generate_trajectories(args.data_dir, args.output_path)
