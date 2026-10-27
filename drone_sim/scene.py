"""Procedural scene; no downloaded meshes or textures are required."""
import math
import cv2
from .config import Config


def scene_xml(c: Config):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, c.marker_id, 6)
    cell = c.marker_size / 6
    cells = []
    # White quiet zone. Pixel boxes are seen from BELOW by the upward camera.
    for row in range(-1, 7):
        for col in range(-1, 7):
            white = not (0 <= row < 6 and 0 <= col < 6) or marker[row, col] > 128
            color = '0.96 0.96 0.96 1' if white else '0.015 0.015 0.015 1'
            x, y = (col - 2.5) * cell, (row - 2.5) * cell
            cells.append(f'<geom type="box" pos="{x} {y} {c.marker_z}" size="{cell/2} {cell/2} 0.0005" '
                         f'rgba="{color}" material="marker" contype="0" conaffinity="0" mass="0"/>')
    parts = []
    for x in (-0.065, 0.065):
        for y in (-0.065, 0.065):
            parts.append(f'<geom type="cylinder" pos="{x} {y} 0.017" size="0.042 0.002" '
                         'rgba="0.12 0.23 0.28 0.65" contype="0" conaffinity="0" mass="0"/>')
            parts.append(f'<geom type="capsule" fromto="0 0 0 {x} {y} 0.008" size="0.008" '
                         'rgba="0.15 0.2 0.25 1" contype="0" conaffinity="0" mass="0"/>')
            parts.append(f'<geom name="foot_{len(parts)}" type="sphere" pos="{x} {y} {c.feet_z+0.006}" '
                         'size="0.006" rgba="0.16 0.18 0.20 1" mass="0"/>')
    return f'''<mujoco model="Indoor visual landing laboratory">
      <compiler angle="radian"/>
      <option timestep="{c.physics_dt}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/>
      <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/></visual>
      <asset>
        <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"
          rgb1="0.10 0.14 0.19" rgb2="0.13 0.18 0.23"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="12 12" reflectance="0.1"/>
        <material name="marker" emission="0.6" specular="0"/>
      </asset>
      <worldbody>
        <light pos="0 -1 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="0 0 -0.2" dir="0 0 1" diffuse="0.7 0.7 0.7" castshadow="false"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
        <geom name="pad" type="cylinder" pos="0 0 {c.pad_top/2}" size="0.24 {c.pad_top/2}"
          rgba="0.09 0.28 0.31 1" friction="0.9 0.02 0.001"/>
        <geom name="target" type="cylinder" pos="0 0 {c.pad_top+0.0006}" size="0.075 0.0005"
          rgba="0.15 0.85 0.66 1" contype="0" conaffinity="0"/>
        <geom name="camera_aperture" type="cylinder" pos="0 0 {c.pad_top+0.002}" size="0.012 0.001"
          rgba="0.02 0.04 0.06 1" contype="0" conaffinity="0"/>
        {''.join(f'<geom type="cylinder" pos="{x} {y} {c.pad_top+0.001}" size="{c.electrode_radius} 0.0005" rgba="0.8 0.43 0.15 1" contype="0" conaffinity="0"/>' for x in (-0.065,0.065) for y in (-0.065,0.065))}
        <camera name="ground_camera" pos="0 0 {c.camera_z}" xyaxes="1 0 0 0 -1 0" fovy="{c.camera_fovy}"/>
        <camera name="overview" pos="2.0 -2.5 1.9" xyaxes="0.781 0.625 0 -0.285 0.356 0.89" fovy="48"/>
        <body name="drone" pos="{c.initial_x} {c.initial_y} {c.initial_z}" quat="{math.cos(c.initial_yaw/2)} 0 0 {math.sin(c.initial_yaw/2)}">
          <freejoint/>
          <inertial pos="0 0 0" mass="{c.mass}" diaginertia="0.00014 0.00014 0.00024"/>
          <geom name="hull" type="box" size="0.055 0.04 0.019" rgba="0.85 0.90 0.94 1" mass="0"/>
          <geom type="box" pos="0 0.039 0.014" size="0.022 0.003 0.004" rgba="0.03 0.8 0.95 1"
            contype="0" conaffinity="0" mass="0"/>
          {''.join(parts)}
          {''.join(cells)}
        </body>
      </worldbody>
    </mujoco>'''
