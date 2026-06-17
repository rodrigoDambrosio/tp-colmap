"""Read COLMAP binary model files (cameras.bin, images.bin, points3D.bin)."""
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple
import struct
import numpy as np


class Camera(NamedTuple):
    id: int
    model: int
    width: int
    height: int
    params: Tuple


class Image(NamedTuple):
    id: int
    qvec: Tuple          # qw qx qy qz  (world-to-camera rotation)
    tvec: Tuple          # tx ty tz      (world-to-camera translation)
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: Tuple


class Point3D(NamedTuple):
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: Tuple
    point2D_idxs: Tuple


# Number of intrinsic parameters per camera model
_MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 9, 6: 12, 7: 6, 8: 3, 9: 4, 10: 5, 11: 5}


def qvec2rotmat(qvec) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qw*qz,   2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,      2*qy*qz + 2*qw*qx,   1 - 2*qx**2 - 2*qy**2],
    ])


def image_camera_center(img: Image) -> np.ndarray:
    """Compute camera center in world coords: c = -R^T @ tvec."""
    R = qvec2rotmat(img.qvec)
    return -R.T @ np.array(img.tvec)


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model  = struct.unpack("<I", f.read(4))[0]
            width  = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n = _MODEL_PARAMS.get(model, 4)
            params = struct.unpack(f"<{n}d", f.read(8 * n))
            cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras


def read_images_binary(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id  = struct.unpack("<I", f.read(4))[0]
            qvec    = struct.unpack("<4d", f.read(32))
            tvec    = struct.unpack("<3d", f.read(24))
            cam_id  = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            xys_flat = struct.unpack(f"<{2 * n_pts}d", f.read(16 * n_pts))
            pt3d_ids = struct.unpack(f"<{n_pts}q", f.read(8 * n_pts))
            images[img_id] = Image(
                img_id, qvec, tvec, cam_id, name,
                np.array(xys_flat).reshape(-1, 2), pt3d_ids,
            )
    return images


def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
    points: Dict[int, Point3D] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pt_id  = struct.unpack("<Q", f.read(8))[0]
            xyz    = np.array(struct.unpack("<3d", f.read(24)))
            rgb    = np.array(struct.unpack("<3B", f.read(3)))
            error  = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = struct.unpack(f"<{2 * track_len}I", f.read(8 * track_len))
            points[pt_id] = Point3D(
                pt_id, xyz, rgb, error, track[0::2], track[1::2]
            )
    return points


def read_model(sfm_dir: Path):
    """Returns (cameras, images, points3D) dicts from a COLMAP binary model directory."""
    cameras  = read_cameras_binary(sfm_dir / "cameras.bin")
    images   = read_images_binary(sfm_dir / "images.bin")
    points3D = read_points3D_binary(sfm_dir / "points3D.bin")
    return cameras, images, points3D
