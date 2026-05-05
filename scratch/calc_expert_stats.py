import torch
import numpy as np
from src.KinesisCore.expert_dataset import get_expert_loader

def main():
    # Paths from config
    expert_buffer_path = "/media/tripan/Data/DDP/Kinesis_ankle/data/expert_trajectories.pth"
    history_len = 1 # We want the per-frame 23D stats
    
    print(f"Loading expert buffer from {expert_buffer_path}...")
    
    loader_exp = get_expert_loader(
        path=expert_buffer_path,
        batch_size=4096,
        history_len=history_len,
        shuffle=False
    )
    
    # Sample a large batch to get stable stats
    # We'll just take the whole dataset if possible, or a large sample
    with torch.no_grad():
        # Get all samples
        all_data = []
        for i in range(len(loader_exp.dataset.all_trajectories)):
            all_data.append(loader_exp.dataset.all_trajectories[i]['observation'])
        
        # Concatenate all frames (N, 23)
        states = torch.cat(all_data, dim=0).to(torch.float32)
        
        mean = states.mean(dim=0).numpy()
        std = states.std(dim=0, unbiased=False).numpy()
        
        print("\n=== Expert Statistics (23D) ===")
        print("MEAN = np.array([")
        print(", ".join([f"{x:.6f}" for x in mean]))
        print("])")
        
        print("\nSTD = np.array([")
        print(", ".join([f"{x:.6f}" for x in std]))
        print("])")

if __name__ == "__main__":
    main()
