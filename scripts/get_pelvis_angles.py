import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.parse_expert_trajectories import parse_mot

def main():
    data_dir = "/media/tripan/Data/DDP/amputee_data/training_data"
    mot_file = os.path.join(data_dir, "tf01_0p6_01_rotated_ik.mot")
    
    # Parse mot file
    data, column_names = parse_mot(mot_file)
    
    # Extract first frame
    first_frame = data.iloc[0]
    
    pelvis_cols = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']
    
    print("\nExpert Pelvis Angles (Degrees) - First Frame:")
    for col in pelvis_cols:
        if col in first_frame:
            print(f"{col}: {first_frame[col]:.4f}")
        else:
            print(f"{col}: NOT FOUND")
            
    print("\nExpert Pelvis Angles (Radians) - First Frame:")
    for col in pelvis_cols:
        if col in first_frame:
            print(f"{col}: {np.deg2rad(first_frame[col]):.4f}")

if __name__ == "__main__":
    main()
