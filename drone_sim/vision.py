"""Image-derived ArUco pose. Does not receive or inspect true drone state."""
import math
import cv2
import numpy as np
from .config import Config
from .control import Pose


class MarkerVision:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        focal = cfg.camera_height / (2 * math.tan(math.radians(cfg.camera_fovy) / 2))
        self.K = np.array([[focal, 0, cfg.camera_width/2],
                           [0, focal, cfg.camera_height/2], [0, 0, 1]], dtype=float)
        self.dist = np.zeros(5)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        h = cfg.marker_size / 2
        self.points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
        self.corners = None
        self.error = None

    def estimate(self, rgb: np.ndarray, captured_at: float):
        self.corners = None
        self.error = None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.cfg.marker_id not in ids.flatten():
            return None
        image_points = corners[list(ids.flatten()).index(self.cfg.marker_id)].reshape(4, 2)
        self.corners = image_points
        ok, rvec, tvec = cv2.solvePnP(self.points, image_points, self.K, self.dist,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            return None
        projected, _ = cv2.projectPoints(self.points, rvec, tvec, self.K, self.dist)
        error = float(np.sqrt(np.mean((projected.reshape(4, 2) - image_points) ** 2)))
        self.error = error
        if error > 2.0:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        # Camera optical coordinates are aligned with world X/Y/Z in this scene.
        # Marker +X is drone +X. Translation is marker centre, not drone COM.
        center = tvec.ravel() + np.array([0, 0, self.cfg.camera_z])
        center += rotation[:, 2] * self.cfg.marker_z
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return Pose(*map(float, center), yaw, captured_at, error)
