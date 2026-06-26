"""
VideoDataset — extrae frames de un video y los usa como dataset propio.
Los frames se guardan en <video_dir>/<video_stem>_frames/ y se reusan en ejecuciones
sucesivas (no se reextraen si la carpeta ya existe y tiene contenido).
"""
from pathlib import Path
from typing import Callable, List, Optional

from .base import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class VideoDataset(Dataset):
    """Dataset generado desde un archivo de video."""

    def __init__(self, video_path: Path, fps: float = 2.0):
        self._video      = Path(video_path)
        self._fps        = max(0.1, fps)
        self._frames_dir = self._video.parent / f"{self._video.stem}_frames"

    @property
    def name(self) -> str:
        return f"video_{self._video.stem}"

    # ------------------------------------------------------------------
    # Download / extraction
    # ------------------------------------------------------------------

    def download(self, log_fn: Callable[[str], None] = print) -> None:
        if self._frames_dir.exists() and any(self._frames_dir.iterdir()):
            count = sum(1 for f in self._frames_dir.iterdir() if f.suffix == ".jpg")
            log_fn(f"[Video] Frames ya extraídos ({count} imágenes), saltando extracción.")
            return
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._extract(log_fn)

    def _extract(self, log_fn: Callable) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV no está instalado. Ejecutá: pip install opencv-python"
            )

        cap = cv2.VideoCapture(str(self._video))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self._video}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step      = max(1, round(video_fps / self._fps))
        expected  = total // step

        log_fn(f"[Video] {self._video.name}")
        log_fn(f"  {video_fps:.1f} fps nativos  |  {total} frames totales")
        log_fn(f"  Extrayendo 1 cada {step} frames → ~{expected} imágenes a {self._fps} fps")

        frame_idx = 0
        saved     = 0
        last_pct  = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                out_path = self._frames_dir / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                if total:
                    pct = frame_idx * 100 // total
                    if pct != last_pct and pct % 10 == 0:
                        log_fn(f"  {pct}%  ({saved} frames guardados)")
                        last_pct = pct
            frame_idx += 1

        cap.release()
        log_fn(f"[Video] Extracción lista: {saved} frames en {self._frames_dir}")

        if saved < 10:
            log_fn("  Advertencia: muy pocos frames para una buena reconstrucción. "
                   "Bajá el FPS objetivo o usá un video más largo.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def get_images_dir(self) -> Path:
        return self._frames_dir

    def _all_images(self) -> List[str]:
        if not self._frames_dir.exists():
            return []
        return sorted(
            f.name for f in self._frames_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )

    def get_db_images(self) -> List[str]:
        imgs  = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[:split]

    def get_query_images(self) -> List[str]:
        imgs  = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[split:]

    def get_ground_truth(self) -> Optional[dict]:
        return None
