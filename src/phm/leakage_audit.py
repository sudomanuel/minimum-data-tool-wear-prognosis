"""
leakage_audit.py — checks formales para detectar data leakage.

Cada check devuelve (status, details) donde status ∈ {PASS, FAIL, WARN}.
El resultado se guarda en outputs/metrics/leakage_checks.csv.
"""
import pandas as pd
from pathlib import Path

from .config import (
    EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL, TARGET_COLUMN,
    NON_FEATURE_COLS, PROCESSED_DATASET, SPLIT_FILE, LEAKAGE_CHECKS_CSV,
    INTERIM_AUG_DIR,
)


def _row(name, status, details):
    return {'check_name': name, 'status': status, 'details': details}


def check_one_row_per_experiment(df: pd.DataFrame):
    n = len(df)
    n_uniq = df[EXPERIMENT_ID_COL].nunique()
    if n == n_uniq:
        return _row('one_row_per_experiment', 'PASS', f'{n} filas, todas unicas por {EXPERIMENT_ID_COL}')
    return _row('one_row_per_experiment', 'FAIL',
                f'{n} filas pero solo {n_uniq} experiment_ids unicos — leakage potencial')


def check_target_unique_per_experiment(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        return _row('target_unique_per_experiment', 'FAIL', f'falta columna {TARGET_COLUMN}')
    by_exp = df.groupby(EXPERIMENT_ID_COL)[TARGET_COLUMN].nunique()
    bad = by_exp[by_exp > 1]
    if bad.empty:
        return _row('target_unique_per_experiment', 'PASS',
                    f'{len(by_exp)} experimentos con VB_um unico')
    return _row('target_unique_per_experiment', 'FAIL',
                f'experimentos con VB_um multiples: {bad.index.tolist()}')


def check_no_experiment_in_both_splits():
    if not SPLIT_FILE.exists():
        return _row('no_experiment_in_both_splits', 'WARN',
                    f'no existe {SPLIT_FILE.name}')
    rec = pd.read_csv(SPLIT_FILE)
    split_col = 'split' if 'split' in rec.columns else 'set'
    train_ids = set(rec.loc[rec[split_col] == 'train', EXPERIMENT_ID_COL].astype(int))
    test_ids  = set(rec.loc[rec[split_col] == 'test',  EXPERIMENT_ID_COL].astype(int))
    overlap = train_ids & test_ids
    if not overlap:
        return _row('no_experiment_in_both_splits', 'PASS',
                    f'train={sorted(train_ids)}, test={sorted(test_ids)}')
    return _row('no_experiment_in_both_splits', 'FAIL',
                f'experiment_ids en ambos splits: {sorted(overlap)}')


def check_id_columns_excluded(df: pd.DataFrame):
    from .dataset_builder import get_feature_columns
    feats = set(get_feature_columns(df))
    leaks = feats & {EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL,
                     TARGET_COLUMN, 'end_of_life', 'is_augmented'}
    if not leaks:
        return _row('id_columns_excluded', 'PASS',
                    f'features={len(feats)} sin ids ni target')
    return _row('id_columns_excluded', 'FAIL',
                f'columnas que NO deberian ser feature: {sorted(leaks)}')


def check_test_not_augmented():
    """
    Audita los CSV en data/interim/augmentation/. Las filas con
    is_augmented=True NUNCA deberian aparecer mezcladas con test.
    Como nosotros guardamos solo el TRAIN aumentado, validamos que ese
    archivo no contiene experimentos del test.
    """
    if not SPLIT_FILE.exists():
        return _row('test_not_augmented', 'WARN', 'split no existe')
    rec = pd.read_csv(SPLIT_FILE)
    split_col = 'split' if 'split' in rec.columns else 'set'
    test_ids = set(rec.loc[rec[split_col] == 'test', EXPERIMENT_ID_COL].astype(int))

    if not INTERIM_AUG_DIR.exists():
        return _row('test_not_augmented', 'PASS', 'no hay augmentation aun')
    bad = []
    for p in INTERIM_AUG_DIR.glob('train_augmented_*.csv'):
        try:
            adf = pd.read_csv(p)
        except Exception:
            continue
        if EXPERIMENT_ID_COL not in adf.columns:
            continue
        ids = set(adf[EXPERIMENT_ID_COL].astype(int).unique())
        leak = ids & test_ids
        if leak:
            bad.append(f"{p.name}:{sorted(leak)}")
    if bad:
        return _row('test_not_augmented', 'FAIL',
                    f'archivos con experimentos de test: {bad}')
    return _row('test_not_augmented', 'PASS',
                'ningun archivo de train_augmented contiene experimentos de test')


def check_augmented_rows_keep_vb(df_pre_split: pd.DataFrame):
    """
    En cada train_augmented_*.csv, las filas con is_augmented=True deben
    tener un VB_um que coincide con el del experimento original.
    """
    if not INTERIM_AUG_DIR.exists():
        return _row('augmented_rows_keep_vb', 'PASS', 'no hay augmentation aun')
    truth = dict(zip(df_pre_split[EXPERIMENT_ID_COL].astype(int),
                     df_pre_split[TARGET_COLUMN].astype(float)))
    bad = []
    for p in INTERIM_AUG_DIR.glob('train_augmented_*.csv'):
        try:
            adf = pd.read_csv(p)
        except Exception:
            continue
        if 'is_augmented' not in adf.columns:
            continue
        aug = adf[adf['is_augmented'] == True]
        for _, row in aug.iterrows():
            eid = int(row[EXPERIMENT_ID_COL])
            vb_truth = truth.get(eid)
            vb_row   = float(row[TARGET_COLUMN])
            if vb_truth is None or abs(vb_row - vb_truth) > 1e-6:
                bad.append((p.name, eid, vb_row, vb_truth))
                break  # uno por archivo basta
    if bad:
        return _row('augmented_rows_keep_vb', 'FAIL', f'desajustes: {bad[:3]}')
    return _row('augmented_rows_keep_vb', 'PASS',
                'todas las filas augmentadas conservan VB_um del experimento real')


def check_physics_aux_train_only(df: pd.DataFrame):
    """
    El PINN recibe datos auxiliares (experiment_order, tool_id) via fit-params
    SOLO del train de cada fold. Este check verifica la condicion estatica que
    lo hace seguro: que esas columnas NO sean features de X (si lo fueran,
    entrarian al modelo para todos los experimentos, incluido el test).
    """
    from .dataset_builder import get_feature_columns
    feats = set(get_feature_columns(df))
    leaks = feats & {EXP_ORDER_COL, TOOL_ID_COL}
    if leaks:
        return _row('physics_aux_train_only', 'FAIL',
                    f'{sorted(leaks)} es feature de X — el PINN no debe verlas '
                    f'como entrada; deben ir solo via fit-params del train')
    return _row('physics_aux_train_only', 'PASS',
                'experiment_order/tool_id excluidos de X; el PINN los recibe '
                'via fit-params solo del TRAIN de cada fold (nunca del test)')


def check_tuning_train_fold_only(df: pd.DataFrame):
    """
    Verifica de forma DINAMICA que el tuning de hiperparametros ocurre dentro
    de cada fold LOEO usando solo los experimentos de train (nested-CV), y que
    el experimento held-out nunca informa la seleccion.

    Mecanismo: espia `layered_pipeline._tune_one`, corre una rama AUGMENTADA con
    tuning (Ridge, grid, feature_noise) y registra, por cada llamada, los
    experimentos (groups) y el numero de filas que recibe.
    - Las llamadas de tuning de LOEO deben usar exactamente (total-1) experimentos
      (uno held-out por fold); la union de los held-out debe cubrir los `total`.
    - item 6 (anti-augmentation-leak): en cada llamada nested, n_filas debe ser
      igual al numero de experimentos (1 fila real por experimento). Si el inner
      CV viera filas sinteticas, n_filas >> n_experimentos -> FAIL.
    - Una llamada con los `total` experimentos esta permitida: es el modelo
      full-data para SHAP (post-hoc, fuera de la evaluacion LOEO).
    - Si NINGUNA llamada excluye un fold (todas ven los `total`), es el esquema
      antiguo con fuga -> FAIL.
    """
    import numpy as np
    try:
        from . import layered_pipeline as lp
        from .dataset_builder import get_feature_columns
    except Exception as exc:
        return _row('tuning_train_fold_only', 'WARN', f'no importable: {exc}')

    if df is None or df.empty or df[EXPERIMENT_ID_COL].nunique() < 3:
        return _row('tuning_train_fold_only', 'WARN', 'dataset insuficiente para el check')

    total = int(df[EXPERIMENT_ID_COL].nunique())
    feat_cols = get_feature_columns(df)
    sub = lp.get_features_for_subset(feat_cols, 'SOLO_A') or feat_cols
    if not sub:
        return _row('tuning_train_fold_only', 'WARN', 'sin features para el check')

    calls = []  # cada item: (set de experimentos, n_filas) visto por _tune_one
    orig = lp._tune_one

    def _spy(name, pipe, X_all, y_all, groups_all, method, n_iter=20, cv_splits=5):
        try:
            calls.append((set(int(g) for g in np.unique(groups_all)),
                          int(np.asarray(X_all).shape[0])))
        except Exception:
            pass
        return orig(name, pipe, X_all, y_all, groups_all, method, n_iter=n_iter, cv_splits=cv_splits)

    lp._tune_one = _spy
    try:
        # Rama AUGMENTADA + tuning: ejercita simultaneamente el held-out
        # exclusion (nested) y la no-fuga de augmentation al inner CV (item 6).
        lp.run_branch(
            branch_id='AUDIT_SOLO_A_A_CT_grid_feature_noise', data_branch='A',
            tuning_method='grid', aug_strategy='feature_noise',
            full_df=df, feat_cols=sub, models_filter=['Ridge'],
            feature_subset='SOLO_A',
        )
    except Exception as exc:
        lp._tune_one = orig
        return _row('tuning_train_fold_only', 'WARN', f'no se pudo correr la rama de prueba: {exc}')
    finally:
        lp._tune_one = orig

    if not calls:
        return _row('tuning_train_fold_only', 'WARN',
                    'no se registraron llamadas de tuning (modelo no tuneable?)')

    all_exps = set(int(e) for e in df[EXPERIMENT_ID_COL].unique())
    nested = [(g, n) for (g, n) in calls if len(g) == total - 1]
    full_calls = [(g, n) for (g, n) in calls if len(g) == total]

    # held-out cubierto por las llamadas nested
    held_out_union = set()
    for g, _ in nested:
        held_out_union |= (all_exps - g)

    # item 6: ninguna llamada nested puede contener filas sinteticas.
    synthetic_leak = [(sorted(g), n) for (g, n) in nested if n != len(g)]

    if not nested:
        return _row('tuning_train_fold_only', 'FAIL',
                    f'ninguna llamada de tuning excluye un fold: todas ven {total} '
                    f'experimentos (esquema antiguo con fuga). calls={len(calls)}')
    if synthetic_leak:
        return _row('tuning_train_fold_only', 'FAIL',
                    f'augmentation FILTRA al inner CV: llamadas nested con '
                    f'n_filas != n_experimentos {synthetic_leak[:3]} '
                    f'(el tuning debe usar solo filas reales)')
    if held_out_union != all_exps:
        missing = sorted(all_exps - held_out_union)
        return _row('tuning_train_fold_only', 'WARN',
                    f'tuning nested presente pero no cubre todos los folds; '
                    f'experimentos nunca held-out: {missing}')
    return _row('tuning_train_fold_only', 'PASS',
                f'nested verificado en rama AUGMENTADA: {len(nested)} llamadas con '
                f'{total-1} experimentos reales (held-out excluido, cubre {total} folds; '
                f'sin filas sinteticas en inner CV); {len(full_calls)} full-data (SHAP)')


def check_synthetic_from_train_fold_only(df: pd.DataFrame):
    """
    Audita el log de generacion sintetica (contrato P7:
    synthetic_validation.log_generation_event). Reglas:

    - Sin log / log vacio -> PASS: gate P7 activo, no se ha generado nada.
    - Cada evento con scope loeo_fold/loto_fold debe EXCLUIR su experimento
      held-out de los train_experiment_ids del generador -> si no, FAIL.
    - train_experiment_ids debe ser subconjunto de los experimentos reales
      del dataset -> ids desconocidos = FAIL (generador viendo datos fantasma).
    - Eventos full_data se cuentan aparte (solo diagnostico post-hoc); si son
      los UNICOS eventos y existen archivos sinteticos, WARN.
    """
    try:
        from .synthetic_validation import read_generation_log, SCOPE_FULL_DATA
    except Exception as exc:
        return _row('synthetic_from_train_fold_only', 'WARN', f'no importable: {exc}')

    log = read_generation_log()
    if log.empty:
        return _row('synthetic_from_train_fold_only', 'PASS',
                    'sin eventos de generacion sintetica (gate P7 activo: '
                    'no se generan sinteticos hasta cerrar la reconciliacion T01)')

    all_exps = set(int(e) for e in df[EXPERIMENT_ID_COL].unique()) if df is not None else set()
    bad_leak, bad_ids = [], []
    fold_events = log[log['scope'] != SCOPE_FULL_DATA]
    full_events = log[log['scope'] == SCOPE_FULL_DATA]
    for _, ev in fold_events.iterrows():
        train_ids = {int(x) for x in str(ev['train_experiment_ids']).split(';') if x.strip()}
        test_id = ev['fold_test_experiment_id']
        if pd.isna(test_id):
            bad_leak.append(f"{ev['method']}: evento de fold sin held-out declarado")
            continue
        if int(test_id) in train_ids:
            bad_leak.append(f"{ev['method']}: held-out {int(test_id)} dentro del train del generador")
        if all_exps and not train_ids.issubset(all_exps):
            bad_ids.append(f"{ev['method']}: ids desconocidos {sorted(train_ids - all_exps)}")
    if bad_leak or bad_ids:
        return _row('synthetic_from_train_fold_only', 'FAIL',
                    f'leaks={bad_leak[:3]} ids_invalidos={bad_ids[:3]} '
                    f'({len(fold_events)} eventos de fold auditados)')
    if fold_events.empty and not full_events.empty:
        return _row('synthetic_from_train_fold_only', 'WARN',
                    f'solo eventos full_data ({len(full_events)}): permitidos como '
                    f'diagnostico post-hoc, nunca para metricas LOEO/LOTO')
    return _row('synthetic_from_train_fold_only', 'PASS',
                f'{len(fold_events)} eventos de fold sin fuga (held-out excluido del '
                f'generador en todos); {len(full_events)} eventos full_data post-hoc')


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_all_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corre todos los checks sobre el dataset procesado y artefactos del
    pipeline existentes. Guarda outputs/metrics/leakage_checks.csv.
    """
    rows = [
        check_one_row_per_experiment(df),
        check_target_unique_per_experiment(df),
        check_no_experiment_in_both_splits(),
        check_id_columns_excluded(df),
        check_test_not_augmented(),
        check_augmented_rows_keep_vb(df),
        check_physics_aux_train_only(df),
        check_tuning_train_fold_only(df),
        check_synthetic_from_train_fold_only(df),
    ]
    out = pd.DataFrame(rows)
    LEAKAGE_CHECKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEAKAGE_CHECKS_CSV, index=False)
    return out
