import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ExpertDataset(Dataset):
    """
    Dataset for loading expert trajectories and providing temporal history windows.
    Each trajectory is a dictionary with 'speed' and 'observation' (torch.Tensor of shape (T, ObsDim)).
    """
    def __init__(self, path, history_len=3):
        data = torch.load(path)
        if isinstance(data, dict):
            if 'state' in data:
                obs_data = data['state']
            elif 'states' in data:
                obs_data = data['states']
            else:
                raise ValueError(f"Unknown expert buffer format in {path}. Keys: {list(data.keys())}")
            
            # Flattened buffer format
            print(f"Loaded flattened expert buffer. Shape: {obs_data.shape}")
            self.all_trajectories = [{'observation': obs_data, 'speed': 0.0}]
        elif isinstance(data, list):
            self.all_trajectories = data
            if len(data) > 0:
                print(f"Loaded trajectory-list expert buffer. {len(data)} trajectories.")
                print(f"First trajectory observation shape: {data[0]['observation'].shape}")
        else:
            raise ValueError(f"Unknown expert buffer format in {path}")

        self.history_len = history_len
        
        # Build index of valid windows and group by speed
        self.valid_indices = []
        self.speed_groups = {} # {speed: [valid_indices]}
        
        for traj_idx, traj in enumerate(self.all_trajectories):
            obs = traj['observation']
            speed = traj.get('speed', 0.0)
            num_frames = obs.size(0)
            
            # If the observation is already a flattened history window (rank 2, dim > 33)
            # then we don't need to slice windows, every row is a sample.
            if obs.dim() == 2 and obs.size(1) > 33:
                is_preprocessed = True
            else:
                is_preprocessed = False

            if speed not in self.speed_groups:
                self.speed_groups[speed] = []

            if is_preprocessed:
                # Every frame is a valid sample
                for frame_idx in range(num_frames):
                    global_idx = len(self.valid_indices)
                    self.valid_indices.append((traj_idx, frame_idx, True)) # Mark as preprocessed
                    self.speed_groups[speed].append(global_idx)
            else:
                # Need to build windows
                if num_frames >= history_len:
                    for frame_idx in range(history_len - 1, num_frames):
                        global_idx = len(self.valid_indices)
                        self.valid_indices.append((traj_idx, frame_idx, False))
                        self.speed_groups[speed].append(global_idx)
        
        self.available_speeds = sorted(list(self.speed_groups.keys()))
        print(f"ExpertDataset initialized with {len(self.valid_indices)} samples.")
        print(f"Available speeds: {self.available_speeds}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, frame_idx, is_preprocessed = self.valid_indices[idx]
        trait = self.all_trajectories[traj_idx]
        obs = trait['observation']
        speed = trait.get('speed', 0.0)
        
        if is_preprocessed:
            # frame_idx points to a single row that is already (HistoryLen * ObsDim)
            return obs[frame_idx]
        
        # Original logic for building windows from raw frames
        # Extract window [frame_idx - history_len + 1, ..., frame_idx]
        window = obs[frame_idx - self.history_len + 1 : frame_idx + 1]
        
        # Check dimensionality to determine if stripping is needed.
        # If per-frame dimension is 32 or 33, we strip tx,ty,tz (indices 0,1,2)
        # to match the environment's new 30D frame.
        if window.size(1) >= 32:
            keep_idx = list(range(3, window.size(1)))
            window = window[:, keep_idx]  # (history_len, 29 or 30)
        
        # No longer appending speed here, as the discriminator should only see biomechanical states.
        # Speed is still available in the trajectory metadata for sampling.
        
        # Concatenate temporal history into a single vector
        return window.reshape(-1)

    def sample_by_speed(self, target_speeds, batch_size):
        """
        Samples a batch of expert states matching the provided target speeds.
        target_speeds: tensor or array of speeds of length batch_size.
        """
        batch_samples = []
        for speed in target_speeds:
            # Find the closest available speed
            speed_val = float(speed)
            closest_speed = min(self.available_speeds, key=lambda x: abs(x - speed_val))
            
            # Sample a random index from that speed group
            idx = np.random.choice(self.speed_groups[closest_speed])
            batch_samples.append(self[idx])
            
        return torch.stack(batch_samples)

def get_expert_loader(path, batch_size, history_len=3, shuffle=True):
    dataset = ExpertDataset(path, history_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
