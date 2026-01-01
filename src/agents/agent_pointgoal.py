# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.
#
# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.

import os
import torch
import numpy as np
import logging
import wandb
from typing import Tuple

os.environ["OMP_NUM_THREADS"] = "1"

from src.agents.agent_im import AgentIM
from src.learning.learning_utils import to_test, to_cpu
from src.env.myolegs_pointgoal import MyoLegsPointGoal

logger = logging.getLogger(__name__)

class AgentPointGoal(AgentIM):
    """
    AgentPointGoal is a specialized reinforcement learning agent for humanoid environments,
    adapting AgentIM for a point goal task, in accordance with the MyoLegsPointGoal environment.
    """

    def __init__(self, cfg, dtype, device, training = True, checkpoint_epoch = 0):
        super().__init__(cfg, dtype, device, training, checkpoint_epoch)

    def setup_env(self):
        self.env = MyoLegsPointGoal(self.cfg)
        logger.info("PointGoal environment initialized.")

    def eval_policy(self, epoch = 0, dump = False, runs = 100):
        logger.info("Starting policy evaluation on target goal reaching.")
        self.env.start_eval(im_eval=True)

        to_test(*self.sample_modules)

        success_list = []
        last_reward_list = []

        with to_cpu(*self.sample_modules), torch.no_grad():
            for i in range(runs):
                result, last_reward, frames, joint_angles = self.eval_single_thread(capture=True)
                print(f"Episode {i} result: {result}")
                success_list.append(result)
                last_reward_list.append(last_reward)

                if not self.cfg.no_log:
                    if len(frames) > 0:
                        video_data = np.array(frames).transpose(0, 3, 1, 2)
                        wandb.log({f"eval/video_{i}": wandb.Video(video_data, fps=20, format="mp4")}, step=self.epoch)
                    
                    if len(joint_angles) > 0:
                        angles_data = np.array(joint_angles)
                        table = wandb.Table(data=joint_angles, columns=[f"joint_{j}" for j in range(angles_data.shape[1])])
                        wandb.log({f"eval/joint_angles_{i}": table}, step=self.epoch)

        success_rate = np.mean(success_list)
        logger.info(f"Policy evaluation success rate: {success_rate}")

        res_dict = {
            "eval/success_rate": success_rate,
            "eval/avg_last_reward": np.mean(last_reward_list),
        }

        return res_dict
    
    def eval_single_thread(self, capture: bool = False) -> tuple:
        """
        Evaluates the policy in a single thread by running an episode.

        Returns:
            tuple: (success, last_reward, frames, joint_angles)
        """
        frames = []
        joint_angles = []
        with to_cpu(*self.sample_modules), torch.no_grad():
            obs_dict, info = self.env.reset()
            state = self.preprocess_obs(obs_dict)
            for t in range(10000):
                actions = self.policy_net.select_action(
                    torch.from_numpy(state).to(self.dtype), True
                )[0].numpy()

                if capture:
                    frame = self.env.render()
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)
                    if hasattr(self.env, 'data') and hasattr(self.env.data, 'qpos'):
                        joint_angles.append(self.env.data.qpos[7:].copy().tolist())

                next_obs, reward, terminated, truncated, info = self.env.step(
                    self.preprocess_actions(actions)
                )
                next_state = self.preprocess_obs(next_obs)
                done = terminated or truncated

                if done:                      
                    return not terminated, reward, frames, joint_angles
                state = next_state

        # If the loop exits without termination, consider it a failure
        return False, reward, frames, joint_angles