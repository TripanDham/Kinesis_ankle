"""
Reference State Initialization (RSI) Visualizer — With Velocities

Uses ProstWalkCore to load expert .mot data (with correct sign flips,
deg→rad, and velocity computation), then plays it back on the MuJoCo
model with pelvis height scaling and velocity initialization.

Usage:
    python scripts/visualize_rsi_with_vel.py --mot_file /path/to/expert.mot --user_height 0.95
    python scripts/visualize_rsi_with_vel.py --sim_steps 100  # Run physics after init
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.KinesisCore.prostwalk_core import ProstWalkCore


def main():
    parser = argparse.ArgumentParser(description="RSI Visualizer (With Velocities)")
    parser.add_argument("--mot_file", type=str,
                        default="/media/tripan/Data/DDP/amputee_data/training_data_combined/tf11_0p6_01_rotated_ik.mot")
    parser.add_argument("--xml_path", type=str,
                        default="/media/tripan/Data/DDP/Kinesis_ankle/data/xml/myoLeg26_OSL_A.xml")
    parser.add_argument("--user_height", type=float, default=None)
    parser.add_argument("--hold_time", type=float, default=0.0,
                        help="Additional sleep time after the simulation steps")
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--pelvis_list_offset", type=float, default=1.0,
                        help="Constant offset added to pelvis_list (degrees) to adjust foot clearance")
    parser.add_argument("--ankle_r_offset", type=float, default=-1.0,
                        help="Constant offset added to right ankle (degrees)")
    args = parser.parse_args()

    # ── 1. Load Model ──
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    sim_dt = model.opt.timestep

    # ── 2. Get "stand" keyframe ──
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    stand_qpos = model.key_qpos[stand_id].copy()
    stand_qvel = model.key_qvel[stand_id].copy()
    
    ty_qpos_adr = model.joint("pelvis_ty").qposadr[0]
    list_qpos_adr = model.joint("pelvis_list").qposadr[0]
    ankle_r_adr = model.joint("osl_ankle_angle_r").qposadr[0]
    
    ty_dof_adr = model.joint("pelvis_ty").dofadr[0]
    tx_dof_adr = model.joint("pelvis_tx").dofadr[0]
    
    model_stand_height = stand_qpos[ty_qpos_adr]
    print(f"Model stand height: {model_stand_height:.4f} m")
    print(f"Sim timestep: {sim_dt:.4f} s")

    # ── 3. Parse expert data via ProstWalkCore ──
    dummy = ProstWalkCore.__new__(ProstWalkCore)
    dummy._mj_model = model
    parsed = dummy._parse_mot(args.mot_file, joint_names)
    
    expert_qpos = parsed['qpos']  # (N, nq)
    expert_qvel = parsed['qvel']  # (N, nv)
    fps = parsed['fps']
    num_frames = len(expert_qpos)
    print(f"Loaded {num_frames} frames @ {fps:.1f}Hz")
    print(f"Expert qpos: {expert_qpos.shape}, qvel: {expert_qvel.shape}")
    print(f"Model nq={model.nq}, nv={model.nv}")

    # ── 4. Height scaling ──
    expert_ty_frame0 = expert_qpos[0, ty_qpos_adr]
    if args.user_height is not None:
        user_stand_height = args.user_height
    else:
        user_stand_height = expert_ty_frame0
        print(f"  (Auto-detected user stand height: {user_stand_height:.4f} m)")

    height_scale = model_stand_height / user_stand_height if abs(user_stand_height) > 0.01 else 1.0
    print(f"Height scale: {height_scale:.4f}")

    # ── 5. Playback with velocities ──
    frame_indices = list(range(0, num_frames, args.skip_frames))
    list_offset_rad = np.deg2rad(args.pelvis_list_offset)
    
    with viewer.launch_passive(model, data) as v:
        for frame_num, fi in enumerate(frame_indices):
            if not v.is_running():
                break

            # Start from stand keyframe for unmapped DOFs
            data.qpos[:] = stand_qpos.copy()
            data.qvel[:] = stand_qvel.copy()

            # Overwrite positions
            nq = min(expert_qpos.shape[1], model.nq)
            data.qpos[:nq] = expert_qpos[fi, :nq]
            data.qpos[ty_qpos_adr] *= height_scale
            
            # Apply offsets
            data.qpos[list_qpos_adr] += list_offset_rad
            data.qpos[ankle_r_adr] += np.deg2rad(args.ankle_r_offset)

            # Overwrite velocities
            nv = min(expert_qvel.shape[1], model.nv)
            data.qvel[:nv] = expert_qvel[fi, :nv]
            data.qvel[ty_dof_adr] *= height_scale  # Scale pelvis_ty velocity too

            # Run physics steps and synchronize viewer
            for _ in range(max(1, args.sim_steps)):
                mujoco.mj_step(model, data)
                v.sync()
            
            if frame_num % 10 == 0:
                pelvis_ty = data.qpos[ty_qpos_adr]
                vx = data.qvel[tx_dof_adr]
                vy = data.qvel[ty_dof_adr]
                print(f"Frame {fi:5d}/{num_frames}  |  "
                      f"ty={pelvis_ty:.4f}m  vx={vx:.3f}m/s  vy={vy:.3f}m/s")

            if args.hold_time > 0:
                time.sleep(args.hold_time)

    print("\nDone.")


if __name__ == "__main__":
    main()
