import mujoco
import mujoco.viewer

xml_path = "data/xml/myolegs_OSL_KA.xml"
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    print(f"Total qpos elements (nq): {model.nq}")
    print(f"Total qvel elements (nv): {model.nv}")
    print(f"Total actuators (nu): {model.nu}")
except Exception as e:
    print(f"Error: {e}")

data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
