# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.

import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from src.agents.agent_humanoid import AgentHumanoid
from src.env.myolegs_IL import MyoLegsGAIL
from src.KinesisCore.expert_dataset import get_expert_loader
from src.learning.learning_utils import to_train, to_test, to_device

logger = logging.getLogger(__name__)

class AgentGAIL(AgentHumanoid):
    """
    AgentGAIL integrates GAIL discriminator training into the AgentHumanoid framework.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)
        
        # Expert buffer for discriminator training
        if training:
            self.history_len = cfg.run.get("history_len", 6)
            self.batch_size_disc = cfg.learning.get("batch_size_disc", 64)
            self.loader_exp = get_expert_loader(
                path=cfg.run.expert_buffer_path,
                batch_size=self.batch_size_disc,
                history_len=self.history_len,
                shuffle=True
            )
            self.epoch_disc = cfg.learning.get("epoch_disc", 10)
            
            # PRE-SEED NORMALIZER: Establish biomechanical scale from expert data
            print("Pre-seeding GAIL normalizer with expert data...")
            self.env.gail_norm.train()
            with torch.no_grad():
                # Sample a large representative batch
                init_speeds = np.random.uniform(0.5, 1.5, 4096)
                states_init = self.loader_exp.dataset.sample_by_speed(init_speeds, 4096).to(self.device).to(self.dtype)
                self.env.gail_norm(states_init)
            print(f"Normalizer initialized: Mean shape {self.env.gail_norm.mean.shape}")
    def setup_env(self):
        """
        Initializes the MyoLegsGAIL environment based on the configuration.
        """
        self.env = MyoLegsGAIL(self.cfg)
        logger.info("MyoLegsGAIL environment initialized.")

    def update_params(self, batch) -> float:
        """
        Extends parameter updates with GAIL discriminator training.
        """
        t0 = time.time()
        
        # 1. Update Discriminator
        disc_metrics = {}
        if self.training:
            disc_metrics = self.train_discriminator(batch)
        
        # 2. Update Policy and Value (Standard PPO)
        # Note: In AgentIM/PPO, update_params handles the conversion to tensors and calls update_policy.
        super().update_params(batch)
        
        return time.time() - t0, disc_metrics

    def train_discriminator(self, batch) -> dict:
        """
        Trains the discriminator using agent rollouts and expert demonstrations.
        """
        to_train(self.env.gail_disc)
        metrics = {"loss_disc": [], "loss_pi": [], "loss_exp": []}
        
        # We need to extract (s, s') or just s from the batch for the discriminator.
        # However, our environment's GAILDiscrim expects concatenated history observations.
        # Agent observations in the batch might not be the history-concatenated ones 
        # depending on how compute_task_obs is implemented.
        
        # In our MyoLegsIm.compute_reward, we maintain self.history_buffer.
        # But the batch contains what was returned by step().
        
        # Move discriminator to GPU for training, then back to CPU for sampling workers
        self.env.gail_disc.to(self.device)
        
        for _ in range(self.epoch_disc):
            # Sample from agent's batch
            # Assuming batch.states contains the history-concatenated observations
            # if that's what compute_task_obs returns.
            # However, compute_task_obs returns a concatenated vector of length obs_size.
            # The GAIL rewards in compute_reward use self.get_obs() which is often full state.
            
            # Let's assume the agent batch contains the same "state" used by the discriminator.
            # Allow replacement sampling if batch is smaller than discriminator batch size
            replace = (len(batch.states) < self.batch_size_disc)
            indices = np.random.choice(len(batch.states), self.batch_size_disc, replace=replace)
            states_pi = torch.from_numpy(batch.states[indices]).to(self.dtype).to(self.device)
            # State-only GAIL uses a dummy action
            actions_pi = torch.zeros((self.batch_size_disc, 0), device=self.device)
            
            # Extract target speeds from agent rollouts
            # States are (batch, 30 * history_len). Reshape to (batch, history_len, 30)
            # The speed is the last element of each 30D frame.
            states_pi_frames = states_pi.view(self.batch_size_disc, self.history_len, 30)
            target_speeds = states_pi_frames[:, -1, -1]
            
            # Sample expert data matching these speeds
            states_exp = self.loader_exp.dataset.sample_by_speed(target_speeds, self.batch_size_disc)
            states_exp = states_exp.to(self.dtype).to(self.device)
            # Combine and normalize states
            states_combined = torch.cat([states_pi, states_exp], dim=0)
            
            # Update shared normalizer with both distributions
            self.env.gail_norm.train()
            states_norm = self.env.gail_norm(states_combined)
            
            states_pi_norm = states_norm[:self.batch_size_disc]
            states_exp_norm = states_norm[self.batch_size_disc:]

            actions_pi = torch.zeros((self.batch_size_disc, 0), device=self.device)
            
            # Discriminator predictions
            logits_pi = self.env.gail_disc(states_pi_norm, actions_pi)
            logits_exp = self.env.gail_disc(states_exp_norm, actions_pi) # state-only GAIL
            
            loss_pi = -F.logsigmoid(-logits_pi).mean()
            loss_exp = -F.logsigmoid(logits_exp).mean()
            loss_disc = loss_pi + loss_exp
            
            self.env.optim_disc.zero_grad()
            loss_disc.backward()
            self.env.optim_disc.step()
            
            # Record losses
            metrics["loss_disc"].append(loss_disc.item())
            metrics["loss_pi"].append(loss_pi.item())
            metrics["loss_exp"].append(loss_exp.item())
            
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logger.debug(f"Discriminator loss: {avg_metrics['loss_disc']:.4f}")
        
        # Move discriminator back to CPU for forked sampling workers
        self.env.gail_disc.to("cpu")
        
        return avg_metrics
