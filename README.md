# TP COLMAP — Localización Visual

Pipeline de localización visual con Structure from Motion. Construye un mapa 3D de un lugar a partir de fotos y localiza imágenes nuevas dentro de ese mapa.

## Instalación

```bash
pip install -r requirements.txt
pip install git+https://github.com/cvg/Hierarchical-Localization.git
pip install git+https://github.com/cvg/LightGlue.git
```

> Requiere CUDA y GPU

## Uso

```bash
python main.py
```

1. Seleccioná un dataset (Cambridge Landmarks, COLMAP demo, o carpeta/video propio)
2. Configurá el extractor de features y el método de retrieval
3. Ejecutá el pipeline
4. Explorá los resultados en el visor 3D

## Datasets

| Dataset | Descripción |
|---|---|
| Cambridge Landmarks | Outdoor, se descarga automáticamente, tiene ground truth de poses |
| COLMAP demos | Datasets de ejemplo de COLMAP, se descargan automáticamente |
| Dataset propio | Carpeta de imágenes o video — split 80% DB / 20% query automático |

## Pipeline

```
Extracción de features (DB)
    → Retrieval (NetVLAD / MegaLoc / exhaustivo)
        → Matching (LightGlue)
            → Reconstrucción SfM (COLMAP)
                → Localización de queries (hloc + PnP/RANSAC)
```

## Stack

- **hloc** — pipeline de localización jerárquica
- **pycolmap** — reconstrucción SfM y estimación de pose
- **LightGlue** — feature matching
- **PySide6** — interfaz gráfica
