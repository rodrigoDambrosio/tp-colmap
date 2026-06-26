"""
Pipeline runner — executes the full hloc localization pipeline in a QThread.
Emits Qt signals for progress updates so the GUI stays responsive.

hloc 1.5 API notes:
  - extract_features.main(conf, image_dir, export_dir, image_list) -> Path (h5)
  - pairs_from_retrieval.main(descriptors_h5, output, num_matched, query_list, db_list)
  - match_features.main(conf, pairs, features, export_dir, matches) -> Path (h5)
  - reconstruction.main(sfm_dir, image_dir, pairs, features, matches) -> Reconstruction
  - localize_sfm.main(reconstruction, queries_txt, retrieval_pairs, features, matches, results)
"""
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from src.datasets.base import Dataset

OUTPUT_DIR = Path("outputs")

# Retrieval method name (UI) -> hloc extract_features conf key
RETRIEVAL_CONFS = {
    "netvlad":               "netvlad",
    "dir":                   "dir",
    "openibl":               "openibl",
    "exhaustive (dataset pequeño)": None,   # None = skip retrieval step
}


class PipelineRunner(QThread):
    log_message           = Signal(str)
    progress_updated      = Signal(int)
    stage_changed         = Signal(str)
    reconstruction_ready  = Signal(str, str)         # sfm_dir, images_dir
    localization_started  = Signal(str, str)        # sfm_dir, images_dir
    query_localized       = Signal(str, float)      # results_line, elapsed_ms
    localization_ready    = Signal(str, str, str)   # sfm_dir, images_dir, results_path
    finished              = Signal(bool)

    def __init__(self, dataset: Dataset, config: dict):
        super().__init__()
        self.dataset = dataset
        self.config  = config

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_message.emit(f"[{ts}] {msg}")

    def _stage(self, name: str, pct: int) -> None:
        self.stage_changed.emit(name)
        self.progress_updated.emit(pct)
        self._log(f"--- {name} ---")

    def run(self) -> None:
        try:
            import hloc  # noqa: F401
            import logging
            _hl = logging.getLogger("hloc")
            while len(_hl.handlers) > 1:
                _hl.handlers.pop()
            self._execute()
        except ImportError as e:
            self._log(f"Dependencia faltante: {e}")
            self._log("Instalá hloc: pip install git+https://github.com/cvg/Hierarchical-Localization.git")
            self.finished.emit(False)
        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}")
            self._log(traceback.format_exc())
            self.finished.emit(False)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _execute(self) -> None:
        from hloc import extract_features, match_features, reconstruction
        from hloc import pairs_from_retrieval, pairs_from_exhaustive, localize_sfm

        # ── 0. Dataset ────────────────────────────────────────────────
        self._stage("Preparando dataset", 2)
        self.dataset.download(log_fn=self._log)

        images_dir   = self.dataset.get_images_dir()
        db_images    = self.dataset.get_db_images()
        query_images = self.dataset.get_query_images()
        self._log(f"DB: {len(db_images)} imágenes | Queries: {len(query_images)} imágenes")

        out_dir = OUTPUT_DIR / self.dataset.name
        out_dir.mkdir(parents=True, exist_ok=True)

        sfm_pairs    = out_dir / "pairs_sfm.txt"
        query_pairs  = out_dir / "pairs_query.txt"
        sfm_dir      = out_dir / "sparse"
        results_path = out_dir / "results.txt"

        ext_key  = self.config["extractor"]   # e.g. "disk"
        mat_key  = self.config["matcher"]     # e.g. "disk+lightglue"
        ret_name = self.config["retrieval"]

        ret_conf_key   = RETRIEVAL_CONFS.get(ret_name)
        use_exhaustive = ret_conf_key is None or len(db_images) < 150

        feature_conf  = extract_features.confs[ext_key]
        matcher_conf  = match_features.confs[mat_key]
        features_name = feature_conf["output"]   # string name, e.g. "feats-disk"

        # ── 1. Extract local features (DB) ───────────────────────────
        self._stage("Extrayendo features (DB)", 10)
        feature_path = extract_features.main(
            feature_conf, images_dir,
            export_dir=out_dir, image_list=db_images, overwrite=False,
        )
        self.progress_updated.emit(25)

        # ── 2. Pairs for SfM ─────────────────────────────────────────
        self._stage("Generando pares para SfM", 25)
        if use_exhaustive:
            self._log("Pares exhaustivos")
            pairs_from_exhaustive.main(sfm_pairs, image_list=db_images)
            retrieval_path = None
        else:
            self._log(f"Retrieval: {ret_conf_key}")
            retrieval_conf = extract_features.confs[ret_conf_key]
            retrieval_path = extract_features.main(
                retrieval_conf, images_dir,
                export_dir=out_dir, image_list=db_images, overwrite=False,
            )
            pairs_from_retrieval.main(
                retrieval_path, sfm_pairs, num_matched=20,
                query_list=db_images, db_list=db_images,
            )
        self.progress_updated.emit(38)

        # ── 3. Match features (DB) ────────────────────────────────────
        self._stage("Matching features (SfM)", 38)
        matches_path = match_features.main(
            matcher_conf, sfm_pairs, features=features_name,
            export_dir=out_dir, overwrite=False,
        )
        self.progress_updated.emit(52)

        # ── 4. SfM reconstruction ─────────────────────────────────────
        calibration = self.config.get("calibration")
        sfm_extra   = {}
        if calibration:
            import pycolmap
            from src.utils.calibration import calibration_to_image_options
            sfm_extra["camera_mode"]   = pycolmap.CameraMode.SINGLE
            sfm_extra["image_options"] = calibration_to_image_options(calibration)
            self._log(
                f"Calibración ({calibration['source']}): "
                f"f={calibration['params'][0]:.1f}px  "
                f"{calibration['width']}×{calibration['height']}"
            )

        sfm_model = None
        if not self.config.get("skip_reconstruction"):
            # Try loading an existing reconstruction before re-running COLMAP
            if sfm_dir.exists() and any(sfm_dir.iterdir()):
                try:
                    import pycolmap
                    sfm_model = pycolmap.Reconstruction(str(sfm_dir))
                    self._log(
                        f"Reconstrucción cargada desde disco: "
                        f"{len(sfm_model.images)} imgs, {len(sfm_model.points3D)} pts 3D"
                    )
                    self.reconstruction_ready.emit(str(sfm_dir), str(images_dir))
                except Exception as e:
                    self._log(f"No se pudo cargar reconstrucción existente ({e}), reconstruyendo…")
                    sfm_model = None

            if sfm_model is None:
                self._stage("Reconstrucción SfM (COLMAP)", 52)
                sfm_model = reconstruction.main(
                    sfm_dir, images_dir, sfm_pairs,
                    feature_path, matches_path,
                    image_list=db_images,
                    **sfm_extra,
                )
                self._log(f"Reconstrucción: {len(sfm_model.images)} imágenes, {len(sfm_model.points3D)} puntos 3D")
                self.reconstruction_ready.emit(str(sfm_dir), str(images_dir))
        else:
            self._log("Reconstrucción salteada")
        self.progress_updated.emit(70)

        # ── 5. Query localization ─────────────────────────────────────
        if query_images and sfm_model is not None:
            self._stage("Extrayendo features (queries)", 70)
            extract_features.main(
                feature_conf, images_dir,
                export_dir=out_dir, image_list=query_images, overwrite=False,
            )
            self.progress_updated.emit(78)

            self._stage("Retrieval para queries", 78)
            if use_exhaustive:
                pairs_from_exhaustive.main(
                    query_pairs, image_list=query_images, ref_list=db_images,
                )
            else:
                # Add query descriptors to the same retrieval h5
                retrieval_path = extract_features.main(
                    extract_features.confs[ret_conf_key], images_dir,
                    export_dir=out_dir, image_list=query_images, overwrite=False,
                )
                pairs_from_retrieval.main(
                    retrieval_path, query_pairs, num_matched=10,
                    query_list=query_images, db_list=db_images,
                )
            self.progress_updated.emit(84)

            self._stage("Matching features (queries)", 84)
            query_matches_path = match_features.main(
                matcher_conf, query_pairs, features=features_name,
                export_dir=out_dir, overwrite=False,
            )
            self.progress_updated.emit(90)

            self._stage("Localizando queries", 90)
            queries_txt = self._write_query_list(
                query_images, sfm_model, out_dir, calibration=calibration
            )
            self.localization_started.emit(str(sfm_dir), str(images_dir))
            self._localize_realtime(
                sfm_model, queries_txt, query_pairs,
                feature_path, query_matches_path, results_path,
            )
            self._log(f"Resultados en: {results_path}")
            self.localization_ready.emit(
                str(sfm_dir), str(images_dir), str(results_path)
            )
            self.progress_updated.emit(96)
            self._report_metrics(results_path)

        self.progress_updated.emit(100)
        self._stage("Completado", 100)
        self._log("Pipeline finalizado exitosamente.")
        self.finished.emit(True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _localize_realtime(
        self,
        sfm_model,
        queries_txt: Path,
        query_pairs: Path,
        feature_path: Path,
        matches_path: Path,
        results_path: Path,
    ) -> None:
        """Localize queries one by one, emitting query_localized per query."""
        import time
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
        from hloc.utils.parsers import parse_image_lists, parse_retrieval
        from hloc.utils.io import write_poses

        queries        = parse_image_lists(queries_txt, with_intrinsics=True)
        retrieval_dict = parse_retrieval(query_pairs)
        db_name_to_id  = {img.name: i for i, img in sfm_model.images.items()}

        config    = {"estimation": {"ransac": {"max_error": 12}}}
        localizer = QueryLocalizer(sfm_model, config)

        cam_from_world = {}
        total   = len(queries)
        times   = []

        for i, (qname, qcam) in enumerate(queries):
            if qname not in retrieval_dict:
                self._log(f"  [{i+1}/{total}] {qname} — sin retrieval, saltando")
                continue

            db_ids = [
                db_name_to_id[n]
                for n in retrieval_dict[qname]
                if n in db_name_to_id
            ]
            if not db_ids:
                self._log(f"  [{i+1}/{total}] {qname} — sin DB matches")
                continue

            t0 = time.perf_counter()
            ret, _ = pose_from_cluster(
                localizer, qname, qcam, db_ids, feature_path, matches_path
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

            if ret is not None:
                cam_from_world[qname] = ret["cam_from_world"]
            else:
                cam_from_world[qname] = sfm_model.images[db_ids[0]].cam_from_world()

            # Build results line in hloc format (basename only, as write_poses does)
            t   = cam_from_world[qname]
            q   = t.rotation.quat          # [qx, qy, qz, qw]
            qw, qx, qy, qz = q[3], q[0], q[1], q[2]
            tx, ty, tz     = t.translation
            basename       = qname.split("/")[-1]
            line = f"{basename} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

            self._log(f"  [{i+1}/{total}] {basename}  {elapsed_ms:.1f} ms")
            self.query_localized.emit(line, elapsed_ms)

        if times:
            import numpy as np
            self._log(
                f"Localización: {len(times)}/{total} queries  |  "
                f"mediana {np.median(times):.1f} ms  |  "
                f"total {sum(times)/1000:.2f} s"
            )

        write_poses(cam_from_world, results_path, prepend_camera_name=False)

    def _write_query_list(
        self, query_images: list, sfm_model, out_dir: Path,
        calibration: dict | None = None,
    ) -> Path:
        """Generate hloc-format query list.

        If calibration is provided (EXIF or manual), those intrinsics are used.
        Otherwise falls back to the camera estimated by COLMAP during SfM.
        """
        if calibration:
            model_name = calibration["model"]
            w, h       = calibration["width"], calibration["height"]
            params_str = " ".join(f"{p:.6f}" for p in calibration["params"])
            self._log(f"Query list (cal {calibration['source']}): {model_name} {w}×{h}")
        else:
            cam = next(iter(sfm_model.cameras.values()))
            try:
                model_name = cam.model_name
            except AttributeError:
                model_name = str(cam.model).split(".")[-1]
            w, h       = cam.width, cam.height
            params_str = " ".join(f"{p:.6f}" for p in cam.params)
            self._log(f"Query list (SfM): {model_name} {w}×{h}")

        queries_txt = out_dir / "queries_hloc.txt"
        with open(queries_txt, "w") as fh:
            for name in query_images:
                fh.write(f"{name} {model_name} {w} {h} {params_str}\n")
        return queries_txt

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _report_metrics(self, results_path: Path) -> None:
        gt = self.dataset.get_ground_truth()
        if gt is None:
            return
        from src.utils.metrics import compute_localization_errors, recall_at_thresholds
        import numpy as np

        errors = compute_localization_errors(results_path, gt)
        if not errors:
            self._log("Sin métricas (no hay GT para las queries localizadas).")
            return

        t_errs = [e[0] for e in errors.values()]
        r_errs = [e[1] for e in errors.values()]
        self._log(f"Localización: {len(errors)}/{len(gt)} queries")
        self._log(f"  Mediana error translación : {np.median(t_errs):.3f} m")
        self._log(f"  Mediana error rotación     : {np.median(r_errs):.2f} °")
        for thr, pct in recall_at_thresholds(errors).items():
            self._log(f"  Recall {thr}: {pct:.1f}%")
