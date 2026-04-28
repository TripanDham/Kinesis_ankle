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
from src.learning.learning_utils import to_train, to_test, to_device, to_cpu

logger = logging.getLogger(__name__)

class AgentGAIL(AgentHumanoid):
    """
    AgentGAIL integrates GAIL discriminator training into the AgentHumanoid framework.
    """
    
    def __init__(self, cfg, dtype, device, training: bool = True, checkpoint_epoch: int = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)
        
        # Force sync history_len with env regardless of training/test mode
        self.history_len = self.env.history_len 
        
        # Expert buffer for discriminator training
        if training:
            self.batch_size_disc = cfg.learning.get("batch_size_disc", 64)
            self.loader_exp = get_expert_loader(
                path=cfg.run.expert_buffer_path,
                batch_size=self.batch_size_disc,
                history_len=self.history_len,
                shuffle=True
            )
            self.epoch_disc = cfg.learning.get("epoch_disc", 10)
            
            # SYNCHRONIZED NORMALIZATION: 
            # 1. Sample expert dataset to get the mean/var for the GAIL History subspace.
            # 2. Inject it into the PPO policy's internal normalizer and FREEZE it. 
            gail_obs_size = self.env.get_task_obs_size()
            print(f"Pre-seeding PPO internal normalizer with expert data for the {gail_obs_size}D history slice...")
            with torch.no_grad():
                init_speeds = np.random.uniform(0.5, 1.5, 4096)
                states_init = self.loader_exp.dataset.sample_by_speed(init_speeds, 4096).to("cpu").to(self.dtype)
                
                mean_gail = states_init.mean(dim=0)
                var_gail = states_init.var(dim=0, unbiased=False)
                
                # Freeze the first N indices of PPO's overall normalizer
                self.policy_net.norm.frozen_slice = gail_obs_size
                self.policy_net.norm.mean[:gail_obs_size] = mean_gail.to(self.policy_net.norm.mean.device)
                self.policy_net.norm.var[:gail_obs_size] = var_gail.to(self.policy_net.norm.var.device)
                self.policy_net.norm.std[:gail_obs_size] = torch.sqrt(self.policy_net.norm.var[:gail_obs_size])
                
                # Advance tracking `n` gently so PPO doesnt overwrite blindly on step 1
                self.policy_net.norm.n += 4096
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

    def sample_trajectories(self):
        # Inject the current epoch into the environment so it can scale the im_reward
        self.env.current_epoch = getattr(self, 'epoch', 0)
        return super().sample_trajectories()

    def train_discriminator(self, batch) -> dict:
        """
        Trains the discriminator using agent rollouts and expert demonstrations.
        """
        epoch = getattr(self, 'epoch', 0)
        
        # DELAYED START: Skip GAIL entirely for the first 3000 epochs
        if epoch < 3000:
            return {"loss_disc": 0.0, "loss_pi": 0.0, "loss_exp": 0.0}
            
        current_lr = 2e-5
        for param_group in self.env.optim_disc.param_groups:
            param_group['lr'] = current_lr

        to_train(self.env.gail_disc)
        metrics = {"loss_disc": [], "loss_pi": [], "loss_exp": []}
        
        # We need to extract (s, s') or just s from the batch for the discriminator.
        # However, our environment's GAILDiscrim expects concatenated history observations.
        # Agent observations in the batch might not be the history-concatenated ones 
        # depending on how compute_task_obs is implemented.
        
        # In our MyoLegsIm.compute_reward, we maintain self.history_buffer.
        # But the batch contains what was returned by step().
        
        # Move discriminator to GPU for training
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
            states_pi_full = torch.from_numpy(batch.states[indices]).to(self.dtype).to(self.device)
            
            # The GAIL state is always the first subset of the full RL observation vector
            gail_obs_size = self.env.get_task_obs_size()
            states_pi = states_pi_full[:, :gail_obs_size]
            
            # Extract target speeds from agent rollouts
            # Now that height is in history, target_speed is the FIRST element 
            # of the proprioceptive slice [Speed, Muscles]
            target_speeds = states_pi_full[:, gail_obs_size]
            
            # Sample expert data matching these speeds
            states_exp = self.loader_exp.dataset.sample_by_speed(target_speeds, self.batch_size_disc)
            states_exp = states_exp.to(self.dtype).to(self.device)
            
            # NORMALIZATION PARITY:
            # We map BOTH the expert state and the agent state through PPO's normalizer so the discriminator matches
            self.policy_net.norm.eval()
            with torch.no_grad():
                # For expert states, we temporarily pad to full dimension to use the normalizer, then extract just the GAIL slice
                padded_exp = torch.zeros(self.batch_size_disc, self.policy_net.norm.dim, device=self.device, dtype=self.dtype)
                padded_exp[:, :gail_obs_size] = states_exp
                states_exp_norm = self.policy_net.norm(padded_exp)[:, :gail_obs_size]
                
                # For agent states, just take the first N dimensions of their fully normalized variant
                states_pi_norm = self.policy_net.norm(states_pi_full)[:, :gail_obs_size]

            # INSTANCE NOISE: Prevents the discriminator from instantly solving the problem by 
            # memorizing clipped extremes (e.g., [5.0, 5.0]). Also adds continuous gradients.
            noise_std = self.cfg.learning.get("gail_noise_std", 0.1)
            states_pi_noisy = states_pi_norm + torch.randn_like(states_pi_norm) * noise_std
            states_exp_noisy = states_exp_norm + torch.randn_like(states_exp_norm) * noise_std

            actions_pi = torch.zeros((self.batch_size_disc, 0), device=self.device)
            
            # Discriminator predictions
            logits_pi = self.env.gail_disc(states_pi_noisy, actions_pi)
            logits_exp = self.env.gail_disc(states_exp_noisy, actions_pi) # state-only GAIL
            
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
            
        # Move back to CPU for sampling workers
        self.env.gail_disc.to("cpu")
        
        return {k: np.mean(v) for k, v in metrics.items()}
    def eval_policy(self, runs: int = 10, dump: bool = False) -> dict:
        """
        Extended evaluation that computes and visualizes discriminator saliency.
        """
        logger.info(f"Running evaluation with Discriminator Saliency for {runs} episodes...")
        
        # 1. Prepare Feature Names for the Heatmap
        base_names = self.env.get_gail_feature_names()
        feature_names = []
        # History in MyoLegsGAIL is [current, previous, ...]
        for h in range(self.env.history_len):
            suffix = f" (t-{h})" if h > 0 else " (t)"
            feature_names.extend([n + suffix for n in base_names])

        all_saliency = []
        
        # 2. Run Evaluation with Gradients Enabled for Saliency
        with to_test(*self.sample_modules):
            with to_cpu(*self.sample_modules):
                # Keep Discriminator on CPU to match policy/normalizer device in this block
                self.env.gail_disc.to("cpu")
                self.env.gail_disc.eval()
                
                # Get device from policy (which should be 'cpu' now)
                device = next(self.policy_net.parameters()).device
                
                for i in range(runs):
                    obs_dict, info = self.env.reset()
                    state = self.preprocess_obs(obs_dict)
                    
                    episode_saliency = []
                    
                    # We only collect saliency for the first ~500 steps to keep the heatmap readable
                    for t in range(500): 
                        # A. Select Action (No Grad)
                        with torch.no_grad():
                            actions = self.policy_net.select_action(
                                torch.from_numpy(state).to(self.dtype).to(device), True
                            )[0].numpy()

                        # B. Compute Discriminator Saliency (With Grad)
                        gail_obs_size = self.env.get_task_obs_size()
                        gail_state = torch.from_numpy(state[:, :gail_obs_size]).to(self.dtype).to(device)
                        gail_state.requires_grad = True
                        
                        # Normalize state using policy's normalizer (same as training)
                        self.policy_net.norm.eval()
                        padded = torch.zeros(1, self.policy_net.norm.dim, device=device, dtype=self.dtype)
                        padded[:, :gail_obs_size] = gail_state
                        norm_state = self.policy_net.norm(padded)[:, :gail_obs_size]
                        
                        # Forward pass through discriminator
                        logits = self.env.gail_disc(norm_state)
                        
                        # Backward pass to get gradients w.r.t input state
                        self.env.gail_disc.zero_grad()
                        logits.backward()
                        
                        # Saliency = Magnitude of gradients
                        saliency = gail_state.grad.abs().squeeze().cpu().numpy()
                        episode_saliency.append(saliency)

                        # C. Step Environment
                        next_obs, reward, terminated, truncated, info = self.env.step(
                            self.preprocess_actions(actions)
                        )
                        state = self.preprocess_obs(next_obs)
                        
                        if not self.headless:
                            self.env.render()
                            
                        if terminated or truncated:
                            break
                    
                    if episode_saliency:
                        all_saliency.append(np.array(episode_saliency))
                        logger.info(f"Episode {i} saliency collected ({len(episode_saliency)} steps)")

                # 3. Visualization
                if all_saliency:
                    from src.utils.biomechanics_plotter import plot_discriminator_saliency
                    # Plot the first episode's saliency map
                    plot_discriminator_saliency(all_saliency[0], feature_names)
        
        # 4. Cleanup: Move discriminator back to CPU for sampling worker compatibility
        self.env.gail_disc.to("cpu")
        
        # Run standard evaluation for biomechanics plots
        return super().eval_policy(runs=runs, dump=dump)
