"""
VideoDataset — extrae frames de un video y los usa como dataset propio.
Los frames se guardan en <video_dir>/<video_stem>_frames/ y se reusan en ejecuciones
sucesivas (no se reextraen si la carpeta ya existe y tiene contenido).
"""
from pathlib import Path
from typing import Callable, List, Optional

from .base import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def ffprobe_available() -> bool:
    """Return True if ffprobe is on PATH."""
    import subprocess
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, timeout=5,
        )
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def read_calibration_from_video(
    video_path: Path,
    log_fn: Optional[callable] = None,
) -> Optional[dict]:
    """
    Try to extract focal length from video container metadata via ffprobe.
    Returns a calibration dict (SIMPLE_RADIAL model) or None if not found.

    Conversion: f_px = f_35mm_equiv * frame_width / 36.0
    (36 mm is the horizontal width of a 35 mm full-frame sensor)
    """
    import subprocess, json

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError:
        _log("ffprobe no encontrado en PATH")
        return None
    except Exception as e:
        _log(f"Error al correr ffprobe: {e}")
        return None

    if result.returncode != 0:
        _log(f"ffprobe retornó código {result.returncode}")
        if result.stderr:
            _log(f"stderr: {result.stderr.strip()}")
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        _log(f"JSON inválido de ffprobe: {e}")
        return None

    # Collect all tags and the video stream dimensions
    w = h = 0
    tags: dict = {}
    tags.update(data.get("format", {}).get("tags", {}))
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            w = stream.get("width", w)
            h = stream.get("height", h)
            tags.update(stream.get("tags", {}))

    if not w or not h:
        _log(f"No se pudo obtener dimensiones del video")
        return None

    _log(f"Dimensiones: {w}×{h}")

    # Normalize keys to lowercase for case-insensitive lookup
    tl = {k.lower(): v for k, v in tags.items()}

    if tl:
        _log(f"Tags encontradas ({len(tl)}):")
        for k, v in sorted(tl.items()):
            _log(f"  {k} = {v}")
    else:
        _log("No se encontraron tags de metadatos")

    def _float(key: str) -> Optional[float]:
        val = tl.get(key)
        if val is None:
            return None
        try:
            return float(str(val).split()[0])
        except (ValueError, IndexError):
            return None

    # Priority 1: 35 mm equivalent focal length (most cameras write this)
    f35 = (
        _float("focal_length_in_35mm_format")
        or _float("focal_length_35mm")
        or _float("com.apple.quicktime.camera.focal_length35mm")
    )
    if f35:
        f_px = f35 * w / 36.0
        _log(f"Focal length (35mm equiv): {f35:.1f} mm → {f_px:.1f} px")
        return {
            "model":  "SIMPLE_RADIAL",
            "width":  w,
            "height": h,
            "params": [f_px, w / 2.0, h / 2.0, 0.0],
            "source": f"video metadata (f35={f35:.1f} mm)",
        }

    # Priority 2: real focal length in mm + sensor crop factor heuristic.
    fl_mm = (
        _float("focal_length")
        or _float("com.apple.quicktime.camera.focal_length")
    )
    f35_apple = _float("com.apple.quicktime.camera.focal_length35mm_equiv")
    if f35_apple:
        f_px = f35_apple * w / 36.0
        _log(f"Focal length (Apple 35mm equiv): {f35_apple:.1f} mm → {f_px:.1f} px")
        return {
            "model":  "SIMPLE_RADIAL",
            "width":  w,
            "height": h,
            "params": [f_px, w / 2.0, h / 2.0, 0.0],
            "source": f"video metadata (Apple f35={f35_apple:.1f} mm)",
        }
    if fl_mm:
        # Rough heuristic: assume typical smartphone sensor ~5.1 mm wide
        SENSOR_W_MM = 5.1
        f_px = fl_mm * w / SENSOR_W_MM
        _log(f"Focal length (real): {fl_mm:.2f} mm, heurística sensor {SENSOR_W_MM} mm → {f_px:.1f} px")
        return {
            "model":  "SIMPLE_RADIAL",
            "width":  w,
            "height": h,
            "params": [f_px, w / 2.0, h / 2.0, 0.0],
            "source": f"video metadata (fl={fl_mm:.2f} mm, heuristic sensor)",
        }

    _log("No se encontró focal length en ninguna tag conocida")
    return None


class VideoDataset(Dataset):
    """Dataset generado desde un archivo de video."""

    def __init__(self, video_path: Path, fps: float = 2.0, run_name: str = ""):
        self._video      = Path(video_path)
        self._fps        = max(0.1, fps)
        self._run_name   = run_name.strip()
        self._frames_dir = self._video.parent / f"{self._video.stem}_frames"

    @property
    def name(self) -> str:
        return self._run_name or f"video_{self._video.stem}"

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
