"""Distance Matrix - Matriz de distancias vectorizada con numpy."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .camelot import CamelotKey, camelot_compatibility

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "bpm": 0.35,
    "key": 0.25,
    "energy": 0.15,
    "valence": 0.10,
    "dance": 0.10,
    "acoustic": 0.05,
}


def prepare_feature_vectors(df: pd.DataFrame) -> dict:
    n = len(df)

    bpm = df["bpm"].fillna(120.0).to_numpy(dtype=np.float64)
    bpm = np.where(bpm <= 0, 120.0, bpm)

    camelot_keys: list[CamelotKey | None] = []
    for _, row in df.iterrows():
        ck = None
        camelot_str = row.get("camelot")
        if pd.notna(camelot_str) and str(camelot_str).strip():
            try:
                ck = CamelotKey.from_string(str(camelot_str))
            except (ValueError, TypeError):
                ck = None

        if ck is None and "key" in row and "mode" in row:
            key_val = row.get("key")
            mode_val = row.get("mode")
            if pd.notna(key_val) and pd.notna(mode_val):
                try:
                    ck = CamelotKey.from_spotify(int(key_val), int(mode_val))
                except (ValueError, TypeError):
                    ck = None

        camelot_keys.append(ck)

    features_01 = {}
    for col in ["energy", "valence", "dance", "acoustic"]:
        if col in df.columns:
            arr = df[col].fillna(0.5).to_numpy(dtype=np.float64)
            arr = np.clip(arr, 0.0, 1.0)
        else:
            arr = np.full(n, 0.5, dtype=np.float64)
            logger.warning("Columna '%s' no encontrada, usando 0.5", col)
        features_01[col] = arr

    return {"bpm": bpm, "camelot": camelot_keys, "features_01": features_01, "n": n}


def _bpm_distance_matrix(bpm: np.ndarray) -> np.ndarray:
    log_bpm = np.log2(bpm)
    log_bpm_frac = log_bpm - np.floor(log_bpm)
    diff = np.abs(log_bpm_frac[:, np.newaxis] - log_bpm_frac[np.newaxis, :])
    dist = np.minimum(diff, 1.0 - diff)
    max_dist = dist.max() if dist.max() > 0 else 1.0
    return dist / max_dist


def _key_compatibility_matrix(camelot: list) -> np.ndarray:
    n = len(camelot)
    matrix = np.ones((n, n), dtype=np.float64)

    for i in range(n):
        ki = camelot[i]
        if ki is None:
            continue
        for j in range(n):
            kj = camelot[j]
            if kj is None:
                continue
            compat = camelot_compatibility(ki, kj)
            matrix[i, j] = 1.0 - compat

    np.fill_diagonal(matrix, 0.0)
    return matrix


def _feature_distance_matrix(arr: np.ndarray) -> np.ndarray:
    diff = np.abs(arr[:, np.newaxis] - arr[np.newaxis, :])
    return diff


def build_distance_matrix(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict]:
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.05:
        logger.warning("Pesos no suman 1.0 (suman %.2f). Normalizando.", total_weight)
        weights = {k: v / total_weight for k, v in weights.items()}

    vectors = prepare_feature_vectors(df)
    n = vectors["n"]

    d_bpm = _bpm_distance_matrix(vectors["bpm"])
    d_key = _key_compatibility_matrix(vectors["camelot"])
    d_energy = _feature_distance_matrix(vectors["features_01"]["energy"])
    d_valence = _feature_distance_matrix(vectors["features_01"]["valence"])
    d_dance = _feature_distance_matrix(vectors["features_01"]["dance"])
    d_acoustic = _feature_distance_matrix(vectors["features_01"]["acoustic"])

    D = (
        weights["bpm"] * d_bpm
        + weights["key"] * d_key
        + weights["energy"] * d_energy
        + weights["valence"] * d_valence
        + weights["dance"] * d_dance
        + weights["acoustic"] * d_acoustic
    )

    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 1.0)

    info = {
        "n": n,
        "weights": weights,
        "bpm_range": (float(vectors["bpm"].min()), float(vectors["bpm"].max())),
        "valid_camelot": sum(1 for k in vectors["camelot"] if k is not None),
        "invalid_camelot": sum(1 for k in vectors["camelot"] if k is None),
    }

    logger.info("Matriz de distancias: %dx%d, pesos=%s", n, n, weights)
    return D, info
