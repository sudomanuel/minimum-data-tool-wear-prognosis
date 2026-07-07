"""
config.py — configuracion central del proyecto.

Mantener este archivo simple. Si hay que agregar opciones, prefiere
pasarlas por argparse en los scripts en vez de inflar este modulo.
"""
from pathlib import Path

# =============================================================================
# RUTAS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
SEGMENTS_DIR  = RAW_DIR / "segments"
TARGETS_DIR   = RAW_DIR / "targets"
METADATA_DIR  = RAW_DIR / "metadata"
TARGET_FILE   = TARGETS_DIR  / "vb_targets.csv"
METADATA_FILE = METADATA_DIR / "experiment_metadata.csv"

INTERIM_DIR        = DATA_DIR / "interim"
INTERIM_AUG_DIR    = INTERIM_DIR / "augmentation"
PROCESSED_DIR      = DATA_DIR / "processed"
PROCESSED_DATASET  = PROCESSED_DIR / "experiment_features.csv"

OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
MODELS_DIR      = OUTPUTS_DIR / "models"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
SPLITS_DIR      = OUTPUTS_DIR / "splits"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
ARCHIVE_DIR     = OUTPUTS_DIR / "archive"
LOGS_DIR        = OUTPUTS_DIR / "logs"

# Subcarpetas de figuras (una por etapa)
FIG_DATA_QUALITY  = FIGURES_DIR / "data_quality"
FIG_SIGNALS       = FIGURES_DIR / "signals"
FIG_FEATURES      = FIGURES_DIR / "features"
FIG_HOLDOUT       = FIGURES_DIR / "holdout"
FIG_LOEO          = FIGURES_DIR / "loeo"
FIG_TUNING        = FIGURES_DIR / "tuning"
FIG_AUGMENTATION  = FIGURES_DIR / "augmentation"
FIG_SHAP          = FIGURES_DIR / "shap"

# Subcarpeta de metricas SHAP
METRICS_SHAP = METRICS_DIR / "shap"

# Archivos clave
SPLIT_FILE          = SPLITS_DIR / "train_test_split.csv"
LOEO_FOLDS_FILE     = SPLITS_DIR / "loeo_folds.csv"
MODELING_DATASET    = PROCESSED_DIR / "modeling_dataset.csv"
CONTACT_FEATURES    = INTERIM_DIR / "contact_features.csv"
FEATURE_COLUMNS_CSV = METRICS_DIR / "feature_columns.csv"
DATA_INVENTORY_CSV  = METRICS_DIR / "data_inventory.csv"
MISSING_SEGMENTS    = METRICS_DIR / "missing_segments.csv"
LEAKAGE_CHECKS_CSV  = METRICS_DIR / "leakage_checks.csv"

REPORTS_DIR  = PROJECT_ROOT / "reports"

# =============================================================================
# TARGET Y COLUMNAS NO-FEATURE
# =============================================================================
TARGET_COLUMN     = "VB_um"
EXPERIMENT_ID_COL = "experiment_id"
TOOL_ID_COL       = "tool_id"
EXP_ORDER_COL     = "experiment_order"

# Columnas que NUNCA se usan como features predictoras.
NON_FEATURE_COLS = {
    EXPERIMENT_ID_COL,
    TOOL_ID_COL,
    EXP_ORDER_COL,
    TARGET_COLUMN,
    "end_of_life",
    "is_augmented",
}

# =============================================================================
# ESTRUCTURA DE SENAL
# =============================================================================
N_CONTACTS = 6            # legacy T01 (6 partes fijas). El builder multi-cuchilla
                          # NO lo usa: agrega sobre las partes que existan.
AXIAL_PREFIX = "A"
ROT_PREFIX   = "R"

# --- Ingesta multi-cuchilla -------------------------------------------------
# Canal por token en el nombre. 'AE'=acustica (NO 'A', que es axial).
# Confirmar el token real de acustica antes de ingerirla.
CHANNEL_TOKENS = {"A": "axial", "R": "rotacional", "AE": "acustica"}
DEFAULT_TOOL_ID = "T01"   # nombres legacy sin cuchilla: A{exp}_p{n}

# Driver fisico de la ley de tasa de la PINN: energia de vibracion ROTACIONAL.
# Validado en scripts/validate_physics_assumptions.py (Spearman vs tasa = 0.76).
# Candidatos por esquema (legacy per-contacto vs agregado multi-cuchilla); la
# PINN resuelve el primero presente y AVISA fuerte si ninguno esta (C2).
PHYSICS_DRIVER_COL = "R_energy_total_6_contacts"
PHYSICS_DRIVER_CANDIDATES = ["R_energy_total_6_contacts", "R_energy_total"]

# Columnas de proceso/metadata que NO entran como features de senal de forma
# automatica (C3). Si se quieren usar como features, hacerlo explicitamente.
PROCESS_METADATA_COLS = {"VC", "F"}

# Provenance del dataset: sidecar JSON junto al CSV (C1).
DATASET_META_SUFFIX = ".meta.json"
DATASET_SCHEMA_LEGACY = "per_contact_v1"
DATASET_SCHEMA_MULTITOOL = "aggregate_over_parts_v1"

# Agregacion sobre PARTES (ancho fijo, robusto a nº variable de partes).
AGG_OVER_PARTS = ["mean", "std", "slope"]
FEATURE_CACHE_DIR = INTERIM_DIR / "feature_cache"   # cache por archivo (hash)
N_INGEST_WORKERS = -1                               # joblib (-1 = todos los cores)
# exp 1..183 globalmente unicos. Si fueran por-cuchilla, usar uid compuesto.
EXPERIMENT_ID_GLOBALLY_UNIQUE = True

# =============================================================================
# FEATURES
# =============================================================================
ENABLE_FREQUENCY_FEATURES = True
MIN_SAMPLES_FOR_FFT       = 64

# =============================================================================
# SPLIT / VALIDACION
# =============================================================================
RANDOM_SEED = 42
TEST_SIZE   = 0.2   # → 8/2 con 10 experimentos

# =============================================================================
# AUGMENTATION
# =============================================================================
N_AUGMENTED_PER_EXPERIMENT = 3
AUGMENTATION_NOISE_SIGMA   = 0.01     # 1% std relativo
AUGMENTATION_SCALING_RANGE = (0.98, 1.02)

# Columnas que JAMAS se perturban en augmentation.
AUGMENTATION_PROTECTED_COLS = {
    EXPERIMENT_ID_COL,
    TOOL_ID_COL,
    EXP_ORDER_COL,
    TARGET_COLUMN,
    "end_of_life",
    "is_augmented",
}

# Sufijos que deben permanecer >= 0 (clipping fisico).
PHYSICAL_NONNEGATIVE_SUFFIXES = (
    "_rms", "_energy", "_duration_s", "_n_samples",
    "_peak_to_peak", "_std", "_absolute_mean",
    "_spectral_energy", "_dominant_freq_hz", "_spectral_centroid_hz",
)

# =============================================================================
# FIGURAS
# =============================================================================
FIGURE_DPI    = 130
FIGURE_FORMAT = "png"


def ensure_output_dirs():
    """Crea todas las carpetas de salida si no existen."""
    for d in [
        PROCESSED_DIR, INTERIM_DIR, INTERIM_AUG_DIR, FEATURE_CACHE_DIR,
        MODELS_DIR, METRICS_DIR, FIGURES_DIR, SPLITS_DIR,
        PREDICTIONS_DIR, ARCHIVE_DIR, LOGS_DIR,
        METRICS_SHAP,
        FIG_DATA_QUALITY, FIG_SIGNALS, FIG_FEATURES,
        FIG_HOLDOUT, FIG_LOEO, FIG_TUNING, FIG_AUGMENTATION, FIG_SHAP,
        REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
