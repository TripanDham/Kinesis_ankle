import mujoco
import mujoco.viewer
import pickle
import numpy as np
import time
import argparse
from scipy.spatial.transform import Rotation as sRot


def load_motion_data(motion_file):
    """Loads motion data from a pickle file."""
    print(f"Loading motion data from: {motion_file}")
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f, encoding='latin1')
    print(f"Loaded {len(motion_data)} motions.")
    return motion_data


def main(args):
    """Main function to load and visualize a motion."""
    # Load the motion dictionary
    motion_dict = load_motion_data(args.motion_file)
    motion_keys = list(motion_dict.keys())

    if args.list_motions:
        print("\nAvailable motions:")
        for key in motion_keys:
            print(f"- {key}")
        return

    if args.motion_key not in motion_dict:
        print(f"Error: Motion key '{args.motion_key}' not found in {args.motion_file}.")
        print("Use --list-motions to see available keys.")
        return

    # Load the MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_path(args.xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model from {args.xml_path}: {e}")
        return

    # Extract the selected motion
    motion = motion_dict[args.motion_key]
    qpos_data = motion["qpos"]
    num_frames = qpos_data.shape[0]
    print(f"Visualizing motion '{args.motion_key}' with {num_frames} frames.")

    # The qpos from the motion file is for an SMPL model and needs to be
    # adapted for the MyoLegs model. We will only use the root position
    # and rotation, and the joint angles.

    # Initial rotation to align SMPL with MuJoCo's world frame
    initial_rot = sRot.from_euler("XYZ", [-np.pi / 2, 0, -np.pi / 2])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Get qpos for the current frame
            qpos_frame = qpos_data[frame_idx]

            # Set root position
            data.qpos[:3] = qpos_frame[:3]

            # Set root rotation (and apply initial rotation)
            smpl_quat = qpos_frame[[4, 5, 6, 3]]  # (x, y, z, w)
            rotated_quat = (sRot.from_quat(smpl_quat) * initial_rot).as_quat()
            data.qpos[3:7] = np.roll(rotated_quat, 1)  # (w, x, y, z)

            # Set joint angles
            data.qpos[7:] = qpos_frame[7:]

            # Step the simulation to update kinematics
            mujoco.mj_kinematics(model, data)
            mujoco.mj_step(model, data)

            # Sync the viewer
            viewer.sync()

            # Move to the next frame
            frame_idx = (frame_idx + 1) % num_frames

            # Rudimentary framerate control
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motions from a pickle file.")
    parser.add_argument(
        "--motion_file",
        type=str,
        default="data/kit_test_motion_dict.pkl",
        help="Path to the motion dictionary pickle file.",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="data/xml/myolegs.xml",
        help="Path to the MuJoCo XML model file.",
    )
    parser.add_argument("--motion_key", type=str, help="The key of the motion to visualize.")
    parser.add_argument(
        "--list-motions", action="store_true", help="List all available motion keys and exit."
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)