# This file contains code adapted from:
#
# 1. SMPLSim (https://github.com/ZhengyiLuo/SMPLSim)
#    Copyright (c) 2024 Zhengyi Luo
#    Licensed under the BSD 3-Clause License.
#
# 2. PyTorch-RL (https://github.com/Khrylx/PyTorch-RL)
#   Copyright (c) 2020 Ye Yuan

import math
from collections import defaultdict
import numpy as np

class LoggerRL:

    def __init__(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.avg_episode_len = 0
        self.total_reward = 0
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.min_reward = math.inf
        self.max_reward = -math.inf
        self.episode_reward = 0
        self.episode_reward_list = []
        self.avg_episode_reward = 0
        self.avg_episode_reward_std = 0
        self.sample_time = 0
        self.info_dict = defaultdict(list)
        

    def start_episode(self, env):
        self.episode_reward = 0

    def step(self, env, reward, info):
        self.episode_reward += reward
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.num_steps += 1
        {self.info_dict[k].append(v) for k, v in info.items()}

    def end_episode(self, env):
        self.num_episodes += 1
        self.total_reward += self.episode_reward
        self.episode_reward_list.append(self.episode_reward)
        self.min_episode_reward = min(self.min_episode_reward, self.episode_reward)
        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)

    def end_sampling(self):
        self.avg_episode_len = self.num_steps / self.num_episodes if self.num_episodes > 0 else 0.0
        self.avg_episode_reward = self.total_reward / self.num_episodes if self.num_episodes > 0 else 0.0

    @classmethod
    def merge(cls, logger_list):
        logger = cls()
        logger.total_reward = sum([x.total_reward for x in logger_list])
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        
        # Episode reward stats
        all_episode_rewards = []
        for l in logger_list:
            all_episode_rewards.extend(l.episode_reward_list)
        
        if len(all_episode_rewards) > 0:
            logger.avg_episode_reward = np.mean(all_episode_rewards)
            logger.avg_episode_reward_std = np.std(all_episode_rewards)
            logger.max_episode_reward = np.max(all_episode_rewards)
            logger.min_episode_reward = np.min(all_episode_rewards)
        
        logger.avg_episode_len = logger.num_steps / logger.num_episodes if logger.num_episodes > 0 else 0
        logger.avg_reward = logger.total_reward / logger.num_steps if logger.num_steps > 0 else 0
        
        # Merge info_dict with variance
        logger.info_dict = {}
        all_step_rewards = []
        for k in logger_list[0].info_dict.keys():
            try:
                # Concatenate all steps for this metric
                data = np.concatenate([np.array(x.info_dict[k], dtype=np.float32) for x in logger_list if len(x.info_dict[k]) > 0])
                if data.size > 0:
                    logger.info_dict[k] = np.mean(data)
                    # If this is the total reward component, use it for step reward variance
                    if k == "total_reward":
                        all_step_rewards = data
            except (ValueError, TypeError):
                continue
        
        logger.avg_reward_std = np.std(all_step_rewards) if len(all_step_rewards) > 0 else 0
        
        # Step reward stats (min/max over all steps)
        logger.max_reward = max([x.max_reward for x in logger_list])
        logger.min_reward = min([x.min_reward for x in logger_list])
        
        return logger