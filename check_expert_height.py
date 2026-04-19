import torch
import numpy as np
import os

expert_path = "/media/tripan/Data/DDP/Kinesis_ankle/data/expert_trajectories.pth"

def check_height():
    if not os.path.exists(expert_path):
        print(f"Error: Expert file not found at {expert_path}")
        return

    try:
        # Load with weights_only=False if it's a legacy pickling
        data = torch.load(expert_path, map_location='cpu')
        print(f"Loaded expert data from {expert_path}")
        
        # Handle list of trajectories or flattened buffer
        if isinstance(data, list):
            print(f"Detected trajectory list with {len(data)} trajectories.")
            obs = data[0]['observation']
        elif isinstance(data, dict):
            obs = data.get('states') or data.get('state') or data.get('observations')
        else:
            obs = data

        if obs is None:
            print("Could not find observation data in the file.")
            return

        print(f"Observation shape: {obs.shape}")
        
        # If the dimension is large (e.g. 30+), check if index 1 is height
        # Standard MyoLegs format often uses index 1 for Y-height
        if obs.shape[-1] >= 30:
            # We check the first few frames
            heights = obs[..., 1]
            if torch.is_tensor(heights):
                heights = heights.numpy()
            
            # Filter out zeros if it's a padded buffer
            valid_heights = heights[heights != 0]
            if len(valid_heights) == 0:
                valid_heights = heights

            avg_h = np.mean(valid_heights)
            print(f"\nPelvis Height Analysis (Column Index 1):")
            print(f"  Mean Height: {avg_h:.4f}")
            print(f"  Min Height:  {np.min(valid_heights):.4f}")
            print(f"  Max Height:  {np.max(valid_heights):.4f}")
            
            if avg_h > 0.95:
                print("\n[!] The expert data height is high (>0.95m).")
                print("The model's standing height is ~0.91m, so this will appear 'floating'.")
            elif avg_h < 0.85:
                print("\n[!] The expert data height is low (<0.85m).")
        else:
            print("\nObservation dimension is too small for raw height index.")

    except Exception as e:
        print(f"Error analyzing expert data: {e}")

if __name__ == "__main__":
    check_height()
