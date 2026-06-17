from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import zipfile
import requests
import numpy as np
from .base import Dataset, DATA_DIR

# Scene -> (download URL, ~size hint)
CAMBRIDGE_SCENES: Dict[str, str] = {
    "KingsCollege":  "https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip",
    "OldHospital":   "https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip",
    "ShopFacade":    "https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip",
    "StMarysChurch": "https://www.repository.cam.ac.uk/bitstream/handle/1810/251339/StMarysChurch.zip",
}

MANUAL_DOWNLOAD_URL = "https://www.repository.cam.ac.uk/handle/1810/251342"


class CambridgeDataset(Dataset):
    """Cambridge Landmarks visual localization dataset (Kendall et al., ICCV 2015)."""

    def __init__(self, scene: str):
        if scene not in CAMBRIDGE_SCENES:
            raise ValueError(f"Scene '{scene}' desconocida. Opciones: {list(CAMBRIDGE_SCENES)}")
        self._scene = scene
        self._root = DATA_DIR / "cambridge" / scene

    @property
    def name(self) -> str:
        return f"cambridge_{self._scene}"

    def download(self, log_fn: Callable[[str], None] = print) -> None:
        marker = self._root / "dataset_train.txt"
        if marker.exists():
            log_fn(f"[Cambridge/{self._scene}] Dataset ya existe, saltando descarga.")
            return

        self._root.mkdir(parents=True, exist_ok=True)
        url = CAMBRIDGE_SCENES[self._scene]
        zip_path = self._root.parent / f"{self._scene}.zip"

        log_fn(f"[Cambridge/{self._scene}] Descargando desde Cambridge repository...")
        log_fn(f"  URL: {url}")
        try:
            self._download_file(url, zip_path, log_fn)
        except Exception as e:
            log_fn(f"  ERROR al descargar: {e}")
            log_fn(f"  Descargá el dataset manualmente desde: {MANUAL_DOWNLOAD_URL}")
            log_fn(f"  y extraelo en: {self._root}")
            raise

        log_fn("  Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._root.parent)
        zip_path.unlink(missing_ok=True)
        log_fn(f"[Cambridge/{self._scene}] Listo.")

    def _download_file(self, url: str, dest: Path, log_fn: Callable) -> None:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct != last_pct and pct % 10 == 0:
                        log_fn(f"  Descargando... {pct}%  ({downloaded // 1_048_576} MB)")
                        last_pct = pct

    def get_images_dir(self) -> Path:
        return self._root

    def _parse_pose_file(self, path: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Parse Cambridge .txt: name qw qx qy qz x y z (camera center in world)."""
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    q = np.array([float(v) for v in parts[1:5]])   # qw qx qy qz
                    t = np.array([float(v) for v in parts[5:8]])   # camera center (world)
                except ValueError:
                    continue  # salta líneas de cabecera como "ImageFile, Camera Position..."
                entries.append((parts[0], q, t))
        return entries

    def get_db_images(self) -> List[str]:
        return [name for name, _, _ in self._parse_pose_file(self._root / "dataset_train.txt")]

    def get_query_images(self) -> List[str]:
        return [name for name, _, _ in self._parse_pose_file(self._root / "dataset_test.txt")]

    def get_queries_txt(self) -> Optional[Path]:
        return self._root / "dataset_test.txt"

    def get_ground_truth(self) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        gt = {}
        for name, q, t in self._parse_pose_file(self._root / "dataset_test.txt"):
            gt[name] = (t, q)  # (camera_center_world, quaternion_w2c)
        return gt
