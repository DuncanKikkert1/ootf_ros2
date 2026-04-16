# =============================================================================
# Name        : mech_eye_camera.py
# Author      : Duncan Kikkert
# Created     : 16/4/2026
# Description : MechEye 3D camera interface. Connects to the camera via the
#               Mech-Eye Python SDK, captures the 2D colour image, and returns
#               it as an OpenCV-compatible numpy array (HxWx3, RGB).
#
# Install SDK : pip install MechEyeAPI
#               https://github.com/MechMindRobotics/mecheye_python_samples
#
# Usage:
#   with MechEyeCamera('192.168.137.100', 50005) as cam:
#       rgb = cam.capture_rgb()   # numpy uint8 HxWx3
# =============================================================================

import cv2
import numpy as np

from mecheye.area_scan_3d_camera import Camera, CameraInfo, Frame2D


class MechEyeCamera:
    """2D colour image capture from a MechEye 3D camera (SDK v2.x)."""

    def __init__(self, ip: str, port: int = 50005):
        self.ip   = ip
        self.port = port
        self._camera = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        info            = CameraInfo()
        info.ip_address = self.ip
        info.port       = self.port

        self._camera = Camera()
        status = self._camera.connect(info)

        if not status.is_ok():
            raise RuntimeError(
                f"Failed to connect to MechEye camera at {self.ip}:{self.port}. "
                f"Error: {status.error_description}"
            )

        print(f"[CAM] Connected to MechEye at {self.ip}:{self.port}")

    def close(self):
        if self._camera is not None:
            self._camera.disconnect()
            self._camera = None
            print("[CAM] MechEye disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_rgb(self) -> np.ndarray:
        """Capture the 2D colour image and return as HxWx3 RGB uint8 array."""
        if self._camera is None:
            raise RuntimeError("Camera not connected. Call connect() first.")

        frame  = Frame2D()
        status = self._camera.capture_2d(frame)

        if not status.is_ok():
            raise RuntimeError(f"capture_2d failed: {status.error_description}")

        color_image = frame.get_color_image()
        bgr = np.array(color_image.data(), dtype=np.uint8).reshape(
            color_image.height(), color_image.width(), 3
        )

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
