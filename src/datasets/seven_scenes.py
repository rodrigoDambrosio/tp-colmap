from pathlib import Path
from typing import Callable, List, Optional
import zipfile
import requests
from .base import Dataset, DATA_DIR

# Microsoft Research 7-Scenes dataset
_BASE = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"
SEVEN_SCENES_URLS: dict = {
    "chess":      f"{_BASE}/chess.zip",
    "fire":       f"{_BASE}/fire.zip",
    "heads":      f"{_BASE}/heads.zip",
    "office":     f"{_BASE}/office.zip",
    "pumpkin":    f"{_BASE}/pumpkin.zip",
    "redkitchen": f"{_BASE}/redkitchen.zip",
    "stairs":     f"{_BASE}/stairs.zip",
}

MANUAL_URL = "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/"


class SevenScenesDataset(Dataset):
    """Microsoft 7-Scenes indoor RGB-D dataset (uses only color images)."""

    def __init__(self, scene: str):
        if scene not in SEVEN_SCENES_URLS:
            raise ValueError(f"Scene '{scene}' desconocida. Opciones: {list(SEVEN_SCENES_URLS)}")
        self._scene = scene
        self._root = DATA_DIR / "7scenes" / scene

    @property
    def name(self) -> str:
        return f"7scenes_{self._scene}"

    def download(self, log_fn: Callable[[str], None] = print) -> None:
        if self._root.exists() and any(self._root.rglob("*.color.png")):
            log_fn(f"[7-Scenes/{self._scene}] Dataset ya existe.")
            return

        self._root.mkdir(parents=True, exist_ok=True)
        url = SEVEN_SCENES_URLS[self._scene]
        zip_path = self._root.parent / f"{self._scene}.zip"

        log_fn(f"[7-Scenes/{self._scene}] Descargando... (~0.3-3 GB según escena)")
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded, last_pct = 0, -1
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        if pct != last_pct and pct % 10 == 0:
                            log_fn(f"  {pct}%  ({downloaded // 1_048_576} MB)")
                            last_pct = pct
        except Exception as e:
            log_fn(f"  ERROR: {e}")
            log_fn(f"  Descargá manualmente desde: {MANUAL_URL}")
            raise

        log_fn("  Extrayendo archivo principal...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self._root.parent)
        zip_path.unlink(missing_ok=True)

        # 7-Scenes tiene sub-zips por secuencia dentro del zip principal
        for sub_zip in self._root.glob("*.zip"):
            log_fn(f"  Extrayendo secuencia: {sub_zip.name}")
            with zipfile.ZipFile(sub_zip, "r") as zf:
                zf.extractall(self._root)
            sub_zip.unlink(missing_ok=True)

        log_fn(f"[7-Scenes/{self._scene}] Listo.")

    def get_images_dir(self) -> Path:
        return self._root

    def _all_color_images(self) -> List[str]:
        return sorted(
            str(p.relative_to(self._root)).replace("\\", "/")
            for p in self._root.rglob("*.color.png")
        )

    def get_db_images(self) -> List[str]:
        imgs = self._all_color_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[:split]

    def get_query_images(self) -> List[str]:
        imgs = self._all_color_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[split:]

    def get_queries_txt(self) -> Optional[Path]:
        return None  # 7-Scenes poses are per-frame .pose.txt files, not in COLMAP format
