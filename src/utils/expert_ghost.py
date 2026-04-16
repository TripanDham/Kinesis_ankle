"""
Expert Ghost Visualization Utilities.

Renders a transparent "ghost" of the expert motion reference alongside the
agent in the MuJoCo viewer by rendering the full model with a second MjData
and copying the geoms (with reduced alpha) into the viewer's user scene.
"""

import numpy as np
import mujoco


class ExpertGhost:
    """
    Maintains a ghost MjData and renders the expert reference pose
    as a transparent copy of the full model in the MuJoCo viewer.
    """

    def __init__(self, mj_model, lateral_offset=-1.5):
        """
        Args:
            mj_model: The MuJoCo model (shared with the agent).
            lateral_offset: How far to offset the ghost laterally (in local Z,
                           which maps to world -Y due to body rotation).
        """
        self.mj_model = mj_model
        self.ghost_data = mujoco.MjData(mj_model)
        self.lateral_offset = lateral_offset
        self.enabled = True
        self.alpha = 0.3  # Ghost transparency
        
        # Create a temporary scene for rendering the ghost
        self.ghost_scene = mujoco.MjvScene(mj_model, maxgeom=5000)
        self.ghost_opt = mujoco.MjvOption()
        self.ghost_pert = mujoco.MjvPerturb()
        self.ghost_cam = mujoco.MjvCamera()
        
    def update_pose(self, expert_qpos):
        """
        Sets the ghost's qpos to the expert reference and runs forward kinematics.
        
        Args:
            expert_qpos: The expert qpos array (same size as model nq).
        """
        nq = min(len(expert_qpos), self.mj_model.nq)
        self.ghost_data.qpos[:nq] = expert_qpos[:nq]
        self.ghost_data.qvel[:] = 0
        mujoco.mj_forward(self.mj_model, self.ghost_data)
    
    def apply_offset(self, agent_root_qpos):
        """
        Offsets the ghost's root position to be beside the agent.
        
        Args:
            agent_root_qpos: The agent's qpos[0:3] (pelvis_tx, pelvis_ty, pelvis_tz).
        """
        # Match agent's forward position (X), keep ghost height (Y), offset lateral (Z)
        self.ghost_data.qpos[0] = agent_root_qpos[0]   # Match forward (X)
        # Keep ghost's own height (Y) from expert data
        self.ghost_data.qpos[2] += self.lateral_offset   # Offset lateral (Z in local = -Y in world)
        
        # Re-run kinematics after offset
        mujoco.mj_forward(self.mj_model, self.ghost_data)
    
    def draw(self, viewer):
        """
        Renders the ghost model into the viewer's user scene with transparency.
        
        Args:
            viewer: The MuJoCo passive viewer instance.
        """
        if not self.enabled:
            return
        
        # Reset user scene geom count to prevent accumulation
        viewer._user_scn.ngeom = 0
            
        # Render ghost model into temporary scene
        mujoco.mjv_updateScene(
            self.mj_model,
            self.ghost_data,
            self.ghost_opt,
            self.ghost_pert,
            self.ghost_cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.ghost_scene
        )
        
        # Copy geoms from ghost scene to viewer's user scene with transparency
        for i in range(self.ghost_scene.ngeom):
            if viewer._user_scn.ngeom >= viewer._user_scn.maxgeom:
                break
                
            src_geom = self.ghost_scene.geoms[i]
            
            # Skip non-visual geoms (e.g. collision geoms, sites)
            # Category 0 = dynamic bodies, 2 = decorative
            # Skip ground plane and other static geoms
            if src_geom.category == 4:  # static/world
                continue
            
            dst_idx = viewer._user_scn.ngeom
            viewer._user_scn.ngeom += 1
            dst_geom = viewer._user_scn.geoms[dst_idx]
            
            # Copy all fields from source geom 
            dst_geom.type = src_geom.type
            dst_geom.dataid = src_geom.dataid
            dst_geom.objtype = src_geom.objtype
            dst_geom.objid = src_geom.objid
            dst_geom.category = src_geom.category
            dst_geom.texid = src_geom.texid
            dst_geom.texuniform = src_geom.texuniform
            dst_geom.texrepeat[:] = src_geom.texrepeat
            dst_geom.size[:] = src_geom.size
            dst_geom.pos[:] = src_geom.pos
            dst_geom.mat[:] = src_geom.mat
            
            # Set ghost color: semi-transparent blue tint
            dst_geom.rgba[0] = src_geom.rgba[0] * 0.3 + 0.15  # Tint blue
            dst_geom.rgba[1] = src_geom.rgba[1] * 0.3 + 0.25
            dst_geom.rgba[2] = src_geom.rgba[2] * 0.3 + 0.6
            dst_geom.rgba[3] = self.alpha  # Transparency
