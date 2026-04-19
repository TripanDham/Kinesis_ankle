import mujoco
import mujoco.viewer
import time
import os
import numpy as np

# Path to the model
model_path = "data/xml/myoLeg26_OSL_A.xml"

def main():
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load the model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Find the 'stand' keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        print(f"Loading 'stand' keyframe (id: {key_id})")
        data.qpos[:] = model.key_qpos[key_id]
        data.qvel[:] = model.key_qvel[key_id]
        
        # Ensure kinematics are updated after loading qpos
        mujoco.mj_kinematics(model, data)
        print(f"Initial Pelvis Height (qpos[1]): {data.qpos[1]:.4f}")
    else:
        print("Warning: 'stand' keyframe not found. Using default pose.")

    # Launch the viewer (Removed 'with' to stay compatible with different API versions)
    print("Launching viewer. If it doesn't open in a window, check your X11/Display settings.")
    viewer = mujoco.viewer.launch(model, data)
    
    # Note: viewer.launch() is blocking in older versions, returning only when closed.
    # In newer versions it might be different, but this is the safest call.

if __name__ == "__main__":
    main()
