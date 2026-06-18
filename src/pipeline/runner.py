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
    log_message          = Signal(str)
    progress_updated     = Signal(int)
    stage_changed        = Signal(str)
    reconstruction_ready = Signal(str)
    localization_ready   = Signal(str, str, str)   # sfm_dir, images_dir, results_path
    finished             = Signal(bool)

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
        sfm_model = None
        if not self.config.get("skip_reconstruction"):
            self._stage("Reconstrucción SfM (COLMAP)", 52)
            sfm_model = reconstruction.main(
                sfm_dir, images_dir, sfm_pairs,
                feature_path, matches_path,
                image_list=db_images,
            )
            self._log(f"Reconstrucção: {len(sfm_model.images)} imágenes, {len(sfm_model.points3D)} puntos 3D")
            self.reconstruction_ready.emit(str(sfm_dir))
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
            queries_txt = self._write_query_list(query_images, sfm_model, out_dir)
            localize_sfm.main(
                sfm_model, queries_txt, query_pairs,
                feature_path, query_matches_path, results_path,
                covisibility_clustering=False,
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

    def _write_query_list(self, query_images: list, sfm_model, out_dir: Path) -> Path:
        """Generate hloc-format query list reusing camera intrinsics from SfM model."""
        cam = next(iter(sfm_model.cameras.values()))
        try:
            model_name = cam.model_name
        except AttributeError:
            model_name = str(cam.model).split(".")[-1]
        params_str = " ".join(f"{p:.6f}" for p in cam.params)
        queries_txt = out_dir / "queries_hloc.txt"
        with open(queries_txt, "w") as fh:
            for name in query_images:
                fh.write(f"{name} {model_name} {cam.width} {cam.height} {params_str}\n")
        self._log(f"Query list: {model_name} {cam.width}×{cam.height}")
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
