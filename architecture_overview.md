# Kinesis GAIL Architecture Overview

This document summarizes the architecture for the imitation learning system used to train an amputee musculoskeletal model for walking.

## 1. System Overview
The system uses **Generative Adversarial Imitation Learning (GAIL)** combined with **Proximal Policy Optimization (PPO)** to train a musculoskeletal agent to mimic human walking data. The agent controls a MuJoCo model featuring biological muscles and a powered prosthetic ankle.

## 2. Core Components

### A. Environment: `MyoLegsGAIL`
*   **Model**: `myoLeg26_OSL_A.xml` (MyoLegs 2.0 with OSL Ankle).
*   **Physics Engine**: MuJoCo.
*   **Reference State Initialization (RSI)**: Instead of a fixed pose, the agent is initialized using a pool of 2,700 expert-sampled states (300 frames from each of 9 .mot files).
    *   **Subject Scaling**: Poses are scaled to subject-specific heights (TF01: 0.903, TF11: 0.985) and include kinematic offsets (-6°/+2° ankle, -1° pelvis list).
    *   **Dynamic Reset**: Episodes start with expert finite-difference velocities (including treadmill offsets). The forward position (`pelvis_tx`) is zeroed at each reset.
*   **Observation Space (46D per frame)**:
    * Task Observations (Shared with discriminator):
        *   **Joint Angles (10D)**: Hip flexion/adduction/rotation, Knee flexion, Ankle (Biological & Prosthetic).
        *   **Joint Velocities (13D)**: Pelvis linear velocities (3D) + joint velocities (10D).
        *   **Temporal History**: The policy sees a history of these states ($s_t, s_{t-1}, ...$) to capture gait dynamics. This history is currently set at just 1 step. 
    * Proprioceptive observations: 
        * Target speed (1D): 0.6 m/s in this case
        * Muscle activations (22D): Activation levels [0,1] of all 22 biological muscles
*   **Action Space**:
    *   **Muscles**: Control signals for 22 biological muscles.
    *   **Prosthetic Ankle**: PD control parameters (Stiffness $K$, Damping $B$, and Target Angle $\theta_{target}$).

### B. Expert Data Pipeline: `ProstWalkCore`
*   **Source**: 14 OpenSim `.mot` files (IK results) from 3 different users walking at 0.6 m/s. The users use a knee and ankle prosthetic. 
*   **Preprocessing**:
    *   Coordinate system mapping (OpenSim Y-up to MuJoCo Z-up).
    *   Joint sign flipping (e.g., knee flexion inversion).
    *   Velocity computation via finite difference ($\Delta q / \Delta t$).
*   **Data Buffer**: Processed trajectories are stored in `expert_trajectories.pth` for discriminator training.

### C. Agent: `AgentGAIL`
*   **Actor-Critic**: MLP networks trained via PPO for policy optimization.
*   **Discriminator (WGAN-GP)**: An MLP that learns a Wasserstein distance metric to distinguish between expert and agent observations. 
    *   **Gradient Penalty**: Enforces 1-Lipschitz continuity for training stability.
    *   **EMA Reward**: The raw critic output is normalized via an Exponential Moving Average (EMA) to provide a stable imitation reward $R_{gail}$ in the range $[0, 1]$.
*   **Normalization**: A shared running normalizer ensures expert and agent states are statistically consistent ($z = (x - \mu)/\sigma$). For the observations that are common between the discriminator and the PPO agent, the normalisation mean and std are fixed to the range that is seen in the expert data. 

## 3. Reward Structure
The total reward $R_{total}$ is a weighted sum of imitation signals and physical constraints:

$$R_{total} = w_{im} \cdot R_{gail} + 0.2 \cdot R_{dist} + 0.3 \cdot R_{upright} - w_{mus} \cdot P_{mus} - w_{mot} \cdot P_{mot} - w_{oob} \cdot P_{oob} + 0.1$$

### Components:
1.  **GAIL Imitation ($R_{gail}$)**: 
    $$R_{gail} = \text{EMA\_Norm}(\text{Critic}(s, a))$$
    *Where Critic is the WGAN output. Mapped to approximately $[0, 1]$.*

2.  **Distance Progress ($R_{dist}$)**: 
    $$R_{dist} = x_{pelvis} - x_{start}$$
    *Encourages forward movement along the X-axis.*

3.  **Upright Posture ($R_{upright}$)**: 
    $$R_{upright} = \exp(-3 \cdot (\phi^2 + \theta^2))$$
    *Where $\phi$ and $\theta$ are pelvis tilt and list angles.*

4.  **Muscle Effort ($P_{mus}$)**: 
    $$P_{mus} = \sum a_{muscle}^2$$
    *Penalizes metabolic cost (activations squared).*

5.  **Motor Effort ($P_{mot}$)**: 
    $$P_{mot} = \sum a_{motor}^2$$
    *Penalizes high prosthetic parameter usage.*

6.  **State Out-of-Bounds ($P_{oob}$)**: 
    $$P_{oob} = \sum_{i=1}^{23} \mathbb{1} \left( \left| \frac{s_i - \mu_i^{exp}}{\sigma_i^{exp}} \right| > 5.0 \right)$$
    *Counts features exceeding 5 standard deviations of the expert distribution.*

## 4. Training Hyperparameters:
*   **Schedule** GAIL Start Epoch: 3000.
*   **Learning Rates** Policy LR: $5.0 \times 10^{-5}$, Value LR: $3.0 \times 10^{-4}$, GAIL LR: $2.0 \times 10^{-5}$ 
*   **Optimization** Policy Updates/Epoch: 10, Discrim Updates/Epoch: 10
*   **Batching** Rollout Batch Size: 8192, Discrim Batch Size: 4096

## 5. Network Architectures

### Policy & Value (PPO)
*   **Hidden Layers**: `[512, 256, 128]`
*   **Activation**: `SiLU`

### Discriminator (GAIL)
*   **Hidden Layers**: `[512, 256, 256]`
*   **Activation**: `LeakyReLU`

## 6. Configuration Management
The system uses **Hydra** for modularity:
*   `cfg/env/`: Reward weights, observation mapping, and MuJoCo model paths.
*   `cfg/learning/`: Optimizer settings, batch sizes, and epoch limits.
*   `cfg/run/`: Checkpoint management, device selection (`cuda`), and visualization toggles.