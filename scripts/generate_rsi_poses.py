"""
Generate RSI (Reference State Initialization) Poses

Creates a pool of pre-computed (qpos, qvel) pairs from expert .mot files,
applying subject-specific height scaling, pelvis tilt, and ankle offsets.

These poses are used during training to initialize the agent in diverse,
biomechanically-valid starting states.

Usage:
    python scripts/generate_rsi_poses.py
    python scripts/generate_rsi_poses.py --samples_per_file 500
"""

import os
import sys
import glob
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mujoco
from src.KinesisCore.prostwalk_core import ProstWalkCore


# ── Subject-Specific Settings ──
# Validated via visualize_rsi.py interactive tuning
SUBJECT_SETTINGS = {
    'tf01': {
        'height_scale': 0.903,
        'ankle_r_offset_deg': -6.0,
        'pelvis_list_offset_deg': -1.0,
    },
    'tf11': {
        'height_scale': 0.985,
        'ankle_r_offset_deg': 2.0,
        'pelvis_list_offset_deg': -1.0,
    },
}


def get_subject_from_filename(filename: str) -> str:
    """Extracts subject ID (e.g. 'tf01', 'tf11') from a .mot filename."""
    basename = os.path.basename(filename).lower()
    for subject in SUBJECT_SETTINGS:
        if basename.startswith(subject):
            return subject
    raise ValueError(f"Cannot determine subject from filename: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate RSI pose pool for training")
    parser.add_argument("--data_dir", type=str,
                        default="/media/tripan/Data/DDP/amputee_data/training_data_combined")
    parser.add_argument("--xml_path", type=str,
                        default="/media/tripan/Data/DDP/Kinesis_ankle/data/xml/myoLeg26_OSL_A.xml")
    parser.add_argument("--output_path", type=str,
                        default="/media/tripan/Data/DDP/Kinesis_ankle/data/rsi_poses.npz")
    parser.add_argument("--samples_per_file", type=int, default=300,
                        help="Number of frames to randomly sample from each .mot file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── 1. Load Model ──
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    joint_names = [model.joint(i).name for i in range(model.njnt)]
    nq = model.nq
    nv = model.nv

    # Joint address lookups
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    stand_qpos = model.key_qpos[stand_id].copy()

    ty_qpos_adr = model.joint("pelvis_ty").qposadr[0]
    tx_qpos_adr = model.joint("pelvis_tx").qposadr[0]
    list_qpos_adr = model.joint("pelvis_list").qposadr[0]
    ankle_r_qpos_adr = model.joint("osl_ankle_angle_r").qposadr[0]
    ty_dof_adr = model.joint("pelvis_ty").dofadr[0]

    model_stand_height = stand_qpos[ty_qpos_adr]
    print(f"Model stand height (pelvis_ty): {model_stand_height:.4f} m")
    print(f"Model nq={nq}, nv={nv}")

    # ── 2. Create ProstWalkCore dummy for parsing ──
    dummy = ProstWalkCore.__new__(ProstWalkCore)
    dummy._mj_model = model

    # ── 3. Find all .mot files ──
    mot_files = sorted(glob.glob(os.path.join(args.data_dir, "*.mot")))
    print(f"\nFound {len(mot_files)} .mot files in {args.data_dir}")
    for f in mot_files:
        print(f"  {os.path.basename(f)}")

    # ── 4. Process each file ──
    all_qpos = []
    all_qvel = []
    file_sources = []  # Track which file each pose came from

    for mot_file in mot_files:
        basename = os.path.basename(mot_file)
        subject = get_subject_from_filename(mot_file)
        settings = SUBJECT_SETTINGS[subject]

        print(f"\nProcessing {basename} (subject={subject})")
        print(f"  height_scale={settings['height_scale']}, "
              f"ankle_r={settings['ankle_r_offset_deg']}°, "
              f"pelvis_list={settings['pelvis_list_offset_deg']}°")

        # Parse expert data
        parsed = dummy._parse_mot(mot_file, joint_names)
        expert_qpos = parsed['qpos']  # (N, nq)
        expert_qvel = parsed['qvel']  # (N, nv)
        num_frames = len(expert_qpos)
        print(f"  {num_frames} frames @ {parsed['fps']:.0f}Hz")

        # Random sample indices
        sample_indices = np.random.choice(num_frames, size=args.samples_per_file, replace=(num_frames < args.samples_per_file))

        # Pre-compute offsets
        list_offset_rad = np.deg2rad(settings['pelvis_list_offset_deg'])
        ankle_r_offset_rad = np.deg2rad(settings['ankle_r_offset_deg'])
        height_scale = settings['height_scale']

        for fi in sample_indices:
            # Start from stand keyframe (fills unmapped DOFs)
            qpos = stand_qpos.copy()
            qvel = np.zeros(nv, dtype=np.float32)

            # Overwrite with expert positions
            n = min(expert_qpos.shape[1], nq)
            qpos[:n] = expert_qpos[fi, :n]

            # Apply subject-specific corrections
            qpos[ty_qpos_adr] *= height_scale
            qpos[list_qpos_adr] += list_offset_rad
            qpos[ankle_r_qpos_adr] += ankle_r_offset_rad

            # Zero pelvis_tx (agent always starts at x=0)
            qpos[tx_qpos_adr] = 0.0

            # Overwrite with expert velocities
            nv_expert = min(expert_qvel.shape[1], nv)
            qvel[:nv_expert] = expert_qvel[fi, :nv_expert]

            # Scale pelvis_ty velocity to match height scaling
            qvel[ty_dof_adr] *= height_scale

            all_qpos.append(qpos)
            all_qvel.append(qvel)
            file_sources.append(basename)

    # ── 5. Stack and save ──
    rsi_qpos = np.stack(all_qpos, axis=0)  # (2700, nq)
    rsi_qvel = np.stack(all_qvel, axis=0)  # (2700, nv)

    print(f"\n{'='*60}")
    print(f"RSI pose pool generated:")
    print(f"  qpos shape: {rsi_qpos.shape}")
    print(f"  qvel shape: {rsi_qvel.shape}")
    print(f"  Total poses: {len(rsi_qpos)}")

    # Quick sanity checks
    print(f"\n  pelvis_ty range: [{rsi_qpos[:, ty_qpos_adr].min():.4f}, {rsi_qpos[:, ty_qpos_adr].max():.4f}] m")
    print(f"  pelvis_tx range: [{rsi_qpos[:, tx_qpos_adr].min():.4f}, {rsi_qpos[:, tx_qpos_adr].max():.4f}] m (should be 0)")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(
        args.output_path,
        rsi_qpos=rsi_qpos,
        rsi_qvel=rsi_qvel,
        subject_settings={str(k): str(v) for k, v in SUBJECT_SETTINGS.items()},
        source_files=np.array(file_sources),
    )
    print(f"\nSaved to {args.output_path}")
    print(f"File size: {os.path.getsize(args.output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
