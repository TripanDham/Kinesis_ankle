# Kinesis GAIL Architecture Overview

This document summarizes the architecture for the imitation learning system used to train an amputee musculoskeletal model for walking.  It incorporates all design details relevant to the project report, including reference tracking, RSI, data transformation, velocity computation methods, and reward formulation.

---

## 1. System Overview
The system uses **Generative Adversarial Imitation Learning (GAIL)** combined with **Proximal Policy Optimization (PPO)** to train a musculoskeletal agent to mimic human walking data.  The agent controls a MuJoCo model featuring biological muscles and a powered prosthetic ankle.  Key extensions introduced relative to the original Kinesis framework are:

- **Reference State Initialization (RSI)** — sampling from a pool of expert-derived poses rather than a fixed keyframe.
- **Body-segment Forward-Kinematics (FK) tracking** — Cartesian reward signal instead of pure joint-space tracking.
- **Separate pelvis vs. limb error weighting** in the exponential reward exponent.
- **World-frame `framelinvel` sensors** — direct simulation velocity reads instead of finite-difference estimates.

---

## 2. Core Components

### A. Environment: `MyoLegsGAIL`
*   **Model**: `myoLeg26_OSL_A.xml` (MyoLegs 2.0 with OSL Ankle).
*   **Physics Engine**: MuJoCo at 200 Hz simulation / 40 Hz control (5 physics steps per control step).

#### Observation Space

| Observation group | Dimension | Description |
|---|---|---|
| **Task observations (discriminator)** | 23 D | Joint angles (10 D) + velocities (13 D) – shared with the GAIL discriminator. |
| **Proprioception** | 77 D | Target speed (1 D) + muscle activations (22 D) + foot-contact forces (12 D) + FK body positions (7 × 3 = 21 D) + FK body velocities (7 × 3 = 21 D). |
| **Total** | **100 D** (`history_len = 1`) | Concatenation of task observations and proprioception. |

*   **Joint Angles (10 D)**: Hip flexion/adduction/rotation, Knee flexion, Ankle — both biological left and prosthetic right sides.
*   **Joint Velocities (13 D)**: Pelvis linear velocities (3 D) + the same 10 joint angular velocities.
*   **Temporal History**: The policy sees $h$ consecutive frames $(s_t, s_{t-1}, \dots)$; currently $h = 1$.

#### Action Space
*   **Muscles**: Control signals for 22 biological muscles.
*   **Prosthetic Ankle**: PD control parameters — Stiffness $K$, Damping $B$, and Target Angle $\theta_{\text{target}}$.

---

### B. Expert Data Pipeline: `ProstWalkCore`

#### Source Data
*   14 OpenSim `.mot` files (inverse-kinematics results) from 3 amputee subjects walking at 0.6 m/s.
*   The subjects use a powered knee-and-ankle prosthetic.  
*   Raw data are sampled at **200 Hz** — the same as the MuJoCo simulation — so no temporal up/downsampling is required.

#### Data Transformation Pipeline
1.  **File parsing** — `.mot` header is scanned for `inDegrees`; angles are converted to radians (`unit_scale = π/180`).
2.  **Coordinate-frame handling** — The `myoLeg26_OSL_A` model uses a body-level quaternion on the pelvis to absorb the OpenSim Y-up → MuJoCo Z-up rotation, so joint `qpos` values are used *directly* without an additional frame rotation.
3.  **Pelvis translation** — Raw `pelvis_tx/ty/tz` columns are used unchanged.
4.  **Pelvis orientation** — Raw OpenSim Euler angles (`tilt`, `list`, `rotation`) are concatenated into the hinge-based root representation used by `myoLeg26_OSL_A`.
5.  **Joint mapping** — Each MuJoCo joint name is looked up in the `.mot` columns.  Prosthetic joints (`osl_ankle_angle_r` → `ankle_angle_r`) are handled via a name-map.  Joints absent from the `.mot` file (e.g. coupled translations, wrapping points) default to the `stand` keyframe value.
6.  **Sign conventions** — Knee flexion is negated (`knee_angle_r`, `knee_angle_l`, `osl_knee_angle_r`) to match MuJoCo's joint-angle polarity.
7.  **Treadmill offset** — The forward walking speed is extracted from the filename (e.g. `_0p6_` → 0.6 m/s) and added back to the pelvis X-velocity, because the original treadmill IK data references a belt-relative pelvis that moves at near-zero velocity in the data.
8.  **FK body positions** — For the 7 tracked body segments, MuJoCo forward kinematics (`mj_kinematics`) are run frame-by-frame on the reconstructed `qpos` to obtain world-frame body positions at full 200 Hz resolution.

#### Velocity Computation Methods

Two distinct velocity sources are used in the pipeline:

| Context | Source | Method | Why |
|---|---|---|---|
| **Joint-space reference velocities** (for discriminator `task_obs`) | Offline `.mot` data | Central finite difference: $\dot{q} = \Delta q / \Delta t$ at 200 Hz | Simple and accurate for smooth IK trajectories. |
| **Body-segment reference velocities** (for FK tracking reward) | Offline FK positions (`xpos`) | Finite difference **then Gaussian smoothing**: $\dot{x}_{FK} = \text{GaussianFilter}\!\bigl(\Delta x_{FK} / \Delta t,\; \sigma=2\text{ frames}\bigr)$ | Finite-difference of Cartesian positions amplifies high-frequency noise from joint-angle quantisation.  Gaussian smoothing (σ = 2 frames, causal at 200 Hz ≈ 10 ms) acts as a low-pass derivative filter, eliminating noise while introducing negligible phase lag. |
| **Body-segment simulation velocities** (for FK tracking reward, live) | Online, MuJoCo `sensordata` | `<framelinvel>` world-frame sensor — direct read with no delay | Avoids the one-step finite-difference lag present at episode start; physically consistent with MuJoCo's integrator; zero computational overhead. |

The combination of Gaussian-smoothed reference velocities (offline) and raw sensor velocities (online) ensures that the reward signal is both accurate and free of causal artifacts.

*   **Data Buffer**: Processed trajectories are cached in `processed_motions.joblib` and the discriminator expert buffer is stored in `expert_trajectories.pth`.

---

### C. Reference State Initialization (RSI)

RSI replaces a fixed keyframe reset with a randomized draw from a large pool of expert poses, ensuring the agent trains across the full gait cycle rather than only from the standing posture.

#### Pose Pool Generation (`generate_rsi_poses.py`)
*   **300 frames** are sampled uniformly from each of the 9 training `.mot` files → **2,700 total poses**.
*   Each pose stores:
    *   `rsi_qpos` — full MuJoCo `qpos` vector (geometry-matched to `myoLeg26_OSL_A`).
    *   `rsi_qvel` — full MuJoCo `qvel` vector computed by finite difference of the `.mot` data.
    *   `rsi_frame_idx`, `rsi_fps`, `rsi_motion_key`, `rsi_subject_id` — metadata used to synchronise the reference trajectory at reset.

#### Subject-Specific Corrections
Because the MuJoCo model has a fixed morphology, poses are scaled and offset per subject:

| Setting | TF01 | TF11 |
|---|---|---|
| Height scale (`pelvis_ty`) | 0.903 | 0.985 |
| Pelvis list offset | −1° | −1° |
| Prosthetic ankle offset (right) | −6° | +2° |

These corrections are applied identically both during RSI pose generation and at every environment reset, ensuring consistency between the stored poses and the live simulation.

#### Reset Procedure
1.  A random index is drawn from the 2,700-pose pool.
2.  `mj_data.qpos` ← stored RSI `qpos`.
3.  `mj_data.qvel` ← stored RSI `qvel` (finite-difference expert velocity, including treadmill offset) — when `rsi_init_vel = True`.
4.  `pelvis_tx` is zeroed so the episode always starts at X = 0 regardless of where in the motion cycle the sample falls.
5.  The corresponding `motion_key` and `frame_idx` are used to synchronise `_body_pos_hat` / `_body_vel_hat` for the tracking reward from the very first step.

---

### D. Agent: `AgentGAIL`
*   **Actor-Critic**: MLP networks trained via PPO for policy optimization.
*   **Discriminator (WGAN-GP)**: An MLP that learns a Wasserstein distance metric to distinguish expert from agent observations.
    *   **Gradient Penalty**: Enforces 1-Lipschitz continuity for training stability.
    *   **EMA Reward**: The raw critic output is normalized via an Exponential Moving Average (EMA) to provide a stable imitation reward $R_{\text{gail}} \in [0, 1]$.
*   **Normalization**: A shared running normalizer ensures expert and agent states are statistically consistent ($z = (x - \mu)/\sigma$).  For task observations shared with the discriminator, $\mu$ and $\sigma$ are **fixed** to the expert distribution range.

---

## 3. Reference Tracking

### 3.1 Tracked Degrees of Freedom

The system tracks **seven body segments** (defined in `TRACKED_BODY_NAMES`):

| # | Body | Position (X, Y, Z) | Velocity (Vx, Vy, Vz) |
|---|---|---|---|
| 1 | `pelvis` | ✓ | ✓ (sensor) |
| 2 | `femur_r` | ✓ | ✓ (sensor) |
| 3 | `femur_l` | ✓ | ✓ (sensor) |
| 4 | `tibia_r` | ✓ | ✓ (sensor) |
| 5 | `tibia_l` | ✓ | ✓ (sensor) |
| 6 | `calcn_l` | ✓ | ✓ (sensor) |
| 7 | `toes_l` | ✓ | ✓ (sensor) |

Total: **7 segments × 6 DOFs = 42 FK DOFs**.  The discriminator additionally uses **23 joint-space DOFs** (10 angles + 13 velocities).

### 3.2 Global vs. Local (Pelvis-Relative) Coordinates

*   **Global (world-frame) coordinates** — raw MuJoCo `mj_data.xpos` values expressed in the world frame, including the absolute pelvis translation along X, Y, Z.
*   **Local (pelvis-relative) coordinates** — used in the tracking reward.  The pelvis X and Y positions are subtracted from all tracked bodies before computing the error.

This makes the reward **translation-invariant**: the agent is free to drift forward on the treadmill as long as the *relative* body configuration matches the expert.  The pelvis Z (height) is kept absolute, as vertical pelvis excursion is biomechanically critical.

### 3.3 Joint-Space vs. Body-Segment (FK) Tracking — Comparison

| Aspect | Joint-Space Tracking | Body-Segment (FK) Tracking |
|---|---|---|
| **Data source** | Directly from `.mot` angles/velocities | FK positions run on the same joint configuration |
| **Pros** | Simple 1-to-1 mapping with expert data; no extra sensor setup | Cartesian; aligned with GRF measurement points; insensitive to joint-sign ambiguities |
| **Cons** | Chain-multiplication amplifies angle errors; harder to compare with force-plate data | Requires FK computation and careful coordinate-frame bookkeeping; velocity from sensor or FD |
| **Robustness** | Sensitive to sign flips / joint-coupling errors | More tolerant; reward is on absolute body-segment positions and velocities |
| **Role in this project** | Discriminator `task_obs` only | Tracking reward (`compute_body_tracking_reward`) + RSI pool |

---

## 4. Reward Structure

The total reward is a weighted sum of imitation signals and physical constraints:

$$R_{\text{total}} = w_{\text{im}} \cdot R_{\text{gail}} + 0.2 \cdot R_{\text{dist}} + 0.3 \cdot R_{\text{upright}} + w_{\text{pos}} \cdot R_{\text{pos}} + w_{\text{vel}} \cdot R_{\text{vel}} - w_{\text{mus}} \cdot P_{\text{mus}} - w_{\text{mot}} \cdot P_{\text{mot}} - w_{\text{oob}} \cdot P_{\text{oob}} + 0.1$$

### Components

1.  **GAIL Imitation ($R_{\text{gail}}$)**:
    $$R_{\text{gail}} = \text{EMA\_Norm}(\text{Critic}(s))$$
    *WGAN critic output normalized to approximately $[0, 1]$.*

2.  **Distance Progress ($R_{\text{dist}}$)**:
    $$R_{\text{dist}} = x_{\text{pelvis}} - x_{\text{start}}$$
    *Encourages forward movement along X.*

3.  **Upright Posture ($R_{\text{upright}}$)**:
    $$R_{\text{upright}} = \exp\!\bigl(-3 \cdot (\phi^2 + \theta^2)\bigr)$$
    *$\phi$, $\theta$ are pelvis tilt and list angles.*

4.  **Body-Segment Position Tracking ($R_{\text{pos}}$)**:
    $$R_{\text{pos}} = \frac{1}{2}\Bigl[\exp\!\bigl(-w_{\text{track\_pelvis}}\,\|x_{\text{pelvis}}^{\text{sim}} - x_{\text{pelvis}}^{\text{ref}}\|^2\bigr) + \exp\!\bigl(-w_{\text{track\_body}}\,\|X_{\text{body}}^{\text{sim}} - X_{\text{body}}^{\text{ref}}\|^2\bigr)\Bigr]$$

5.  **Body-Segment Velocity Tracking ($R_{\text{vel}}$)**:
    $$R_{\text{vel}} = \frac{1}{2}\Bigl[\exp\!\bigl(-w_{\text{track\_pelvis}}\,\|v_{\text{pelvis}}^{\text{sim}} - v_{\text{pelvis}}^{\text{ref}}\|^2\bigr) + \exp\!\bigl(-w_{\text{track\_body}}\,\|V_{\text{body}}^{\text{sim}} - V_{\text{body}}^{\text{ref}}\|^2\bigr)\Bigr]$$

6.  **Muscle Effort ($P_{\text{mus}}$)**:
    $$P_{\text{mus}} = \sum a_{\text{muscle}}^2$$

7.  **Motor Effort ($P_{\text{mot}}$)**:
    $$P_{\text{mot}} = \sum a_{\text{motor}}^2$$

8.  **State Out-of-Bounds ($P_{\text{oob}}$)**:
    $$P_{\text{oob}} = \sum_{i=1}^{23} \mathbb{1}\!\left(\left|\frac{s_i - \mu_i^{\text{exp}}}{\sigma_i^{\text{exp}}}\right| > 5.0\right)$$

### 4.1 Why Separate Pelvis vs. Body Weights in the Exponent?

The position and velocity tracking rewards use **two separate exponential weights**:

*   **`w_track_pelvis`** — applied only to the pelvis error.  After the pelvis-relative transformation, the pelvis X/Y error is zero by definition, so this weight penalises solely the **vertical height error (Z)**.  A higher weight here directly penalises unrealistic pelvis bobbing, which is critical for stable treadmill gait.
*   **`w_track_body`** — applied to the remaining 6 body segments.  A somewhat lower value lets the agent tolerate small Cartesian deviations in the distal limbs while still following the overall trajectory.

If both bodies and pelvis shared a single weight, the pelvis height error (typically very small, ~0.01 m) would be drowned out by the larger limb errors or vice versa.  Splitting them allows independent calibration.

### 4.2 Weight Selection

Final values: `w_track_pelvis = 2.0`, `w_track_body = 1.0` (from `cfg/env/myolegs_gail.yaml`).  Selected by:

1.  **Reward magnitude check** — exponential terms should stay in $[0.1, 0.9]$ for most steps, avoiding gradient vanishing.
2.  **Visualization** — higher pelvis weights eliminated unrealistic bouncing in rollouts.
3.  **Training curve** — overly large weights caused reward collapse within the first 1 000 epochs; the chosen values gave stable monotone improvement.

Both weights are YAML-exposed for per-subject fine-tuning.

---

## 5. Training Hyperparameters

| Parameter | Value |
|---|---|
| GAIL start epoch | 3 000 |
| Policy LR | $5.0 \times 10^{-5}$ |
| Value LR | $3.0 \times 10^{-4}$ |
| GAIL (discriminator) LR | $2.0 \times 10^{-5}$ |
| Policy updates / epoch | 10 |
| Discriminator updates / epoch | 10 |
| Rollout batch size | 8 192 |
| Discriminator batch size | 4 096 |

---

## 6. Network Architectures

### Policy & Value (PPO)
*   **Hidden Layers**: `[512, 256, 128]`
*   **Activation**: `SiLU`

### Discriminator (GAIL)
*   **Hidden Layers**: `[512, 256, 256]`
*   **Activation**: `LeakyReLU`

---

## 7. Configuration Management
The system uses **Hydra** for modularity:
*   `cfg/env/` — reward weights, observation mapping, and MuJoCo model paths.
*   `cfg/learning/` — optimizer settings, batch sizes, epoch limits, and `save_frequency`.
*   `cfg/run/` — checkpoint management, device selection (`cuda`), visualization toggles (`plot_saliency`, `plot_reference`, `plot_joint`, `plot_body`).