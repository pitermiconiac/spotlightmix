"""
Deduplicación Inteligente
=========================
Detecta y elimina canciones duplicadas en una playlist usando:
  1. ISRC (mismo master) → mantener la de mayor popularity
  2. Fuzzy matching name+artist (rapidfuzz) → detectar versiones alternativas
  3. Criterios de selección de mejor versión:
       duration > 4:30 (extended mix > radio edit)
       popularity (Spotify 0-100)
       explicit flag (preferir explicit = master sin censura)
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Normalización de texto
# ────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """
    Normaliza texto para fuzzy matching:
      - lowercase
      - sin acentos
      - sin puntuación
      - sin espacios múltiples
    """
    if pd.isna(text):
        return ""

    s = str(text).lower().strip()
    # Quitar acentos (unicode normalize → ASCII)
    s = re.sub(r"[àáâãäå]", "a", s)
    s = re.sub(r"[èéêë]", "e", s)
    s = re.sub(r"[ìíîï]", "i", s)
    s = re.sub(r"[òóôõö]", "o", s)
    s = re.sub(r"[ùúûü]", "u", s)
    s = re.sub(r"[ñ]", "n", s)
    s = re.sub(r"[ç]", "c", s)
    # Quitar puntuación y caracteres especiales
    s = re.sub(r"[^\w\s]", " ", s)
    # Espacios múltiples → uno
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_artist(text: str) -> str:
    """
    Normaliza nombres de artistas además de la normalización de texto:
      - Quitar sufijos comunes: "Official", "Music", "Topic"
      - Quitar "feat." / "ft." / "vs." → no afecta el matching principal
    """
    s = _normalize_text(text)
    # Quitar palabras comunes en nombres de canales de YouTube/Spotify
    s = re.sub(r"\b(official|music|topic|vevo|remastered|remaster)\b", "", s)
    # Quitar "feat.", "ft.", "vs."
    s = re.sub(r"\b(feat|ft|vs)\b", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ────────────────────────────────────────────────────────────────
# Clasificación de versiones
# ────────────────────────────────────────────────────────────────

@dataclass
class VersionInfo:
    """Información extraída del nombre para clasificar versiones."""
    is_remaster: bool
    is_remix: bool
    is_live: bool
    is_acoustic: bool
    is_radio_edit: bool
    is_extended: bool
    is_explicit: bool
    year: int | None  # Año extraído del nombre o album_date


def detect_version_type(name: str, album: str = "", album_date: str = "") -> VersionInfo:
    """
    Detecta el tipo de versión de una pista basándose en su nombre y álbum.
    Busca patrones como "(Remaster)", "(Live)", "(Radio Edit)", "(Extended Mix)".
    """
    text = f"{name} {album}".lower()

    return VersionInfo(
        is_remaster="remaster" in text or "re-master" in text,
        is_remix="remix" in text or "re-mix" in text,
        is_live="live" in text or "concert" in text or "tour" in text,
        is_acoustic="acoustic" in text or "unplugged" in text or "strip" in text,
        is_radio_edit="radio edit" in text or "radio" in text,
        is_extended="extended" in text or "original mix" in text or "full length" in text,
        is_explicit="explicit" in text,
        year=_extract_year(album_date) or _extract_year(name) or _extract_year(album),
    )


def _extract_year(text: str) -> int | None:
    """Extrae un año (4 dígitos, 1950-2025+) de un texto."""
    if pd.isna(text):
        return None
    s = str(text)
    match = re.search(r"(19[5-9]\d|20[0-5]\d)", s)
    if match:
        return int(match.group(1))
    return None


# ────────────────────────────────────────────────────────────────
# Puntuación de calidad de versión
# ────────────────────────────────────────────────────────────────

def version_quality_score(row: pd.Series) -> float:
    """
    Puntúa la calidad de una versión (0.0 - 1.0) para seleccionar la mejor
    entre duplicados. Criterios (en orden de prioridad):
      1. duration > 4:30 (270s) → extended mix preferido
      2. popularity (Spotify 0-100) → versión más conocida
      3. explicit flag → master sin censura
      4. remaster más reciente → mejor calidad de audio
    """
    score = 0.0

    # Duration: extended (>270s) = mejor
    duration = row.get("duration")
    try:
        duration = float(duration)
        if duration > 270:
            score += 0.30
        elif duration > 180:
            score += 0.15
        else:
            score += 0.05
    except (TypeError, ValueError):
        pass  # NaN o string no parseable

    # Popularity: 0-100 → normalizar a 0-0.30
    popularity = row.get("popularity")
    try:
        popularity = float(popularity)
        score += (popularity / 100.0) * 0.30
    except (TypeError, ValueError):
        pass

    # Explicit: master sin censura = mejor
    explicit = row.get("explicit")
    if explicit is True or str(explicit).lower() in ("true", "1", "yes"):
        score += 0.15

    # Remaster reciente
    version = detect_version_type(
        str(row.get("name", "")),
        str(row.get("album", "")),
        str(row.get("album_date", "")),
    )
    if version.is_remaster and version.year and version.year >= 2010:
        score += 0.15
    elif version.is_remaster:
        score += 0.10
    elif version.is_extended:
        score += 0.10  # extended mix sin ser remaster

    # Penalizar versiones live/acústicas si el set es electrónico
    # (el caller puede filtrarlas antes)
    if version.is_live:
        score -= 0.05
    if version.is_acoustic:
        score -= 0.05

    return max(0.0, min(1.0, score))


# ────────────────────────────────────────────────────────────────
# Deduplicación: pipeline completo
# ────────────────────────────────────────────────────────────────

def deduplicate_playlist(
    df: pd.DataFrame,
    fuzzy_threshold: int = 90,
    drop_isrc_duplicates: bool = True,
    prefer_extended: bool = True,
    remove_live: bool = False,
    remove_acoustic: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Deduplica una playlist de forma inteligente.

    Args:
        df: DataFrame con la playlist (debe tener columnas name, artist).
        fuzzy_threshold: Umbral de similitud (0-100) para fuzzy matching.
        drop_isrc_duplicates: Si True, elimina duplicados por ISRC.
        prefer_extended: Si True, prefiere versiones extended/longer.
        remove_live: Si True, elimina versiones live.
        remove_acoustic: Si True, elimina versiones acústicas.

    Returns:
        (df_deduplicated, duplicates_removed) donde duplicates_removed
        es una lista de dicts con info sobre cada duplicado eliminado.
    """
    if df.empty:
        return df, []

    df = df.copy()
    removed: list[dict] = []

    # ─────────────────────────────────────────────────────────────
    # 0. Filtrar versiones live/acústicas si se solicita
    # ─────────────────────────────────────────────────────────────
    if remove_live or remove_acoustic:
        for idx, row in df.iterrows():
            version = detect_version_type(
                str(row.get("name", "")),
                str(row.get("album", "")),
            )
            if remove_live and version.is_live:
                removed.append({
                    "index": idx,
                    "reason": "live_version",
                    "name": row.get("name"),
                    "artist": row.get("artist"),
                })
            elif remove_acoustic and version.is_acoustic:
                removed.append({
                    "index": idx,
                    "reason": "acoustic_version",
                    "name": row.get("name"),
                    "artist": row.get("artist"),
                })
        # Filtrar
        live_acoustic_indices = [r["index"] for r in removed if r["reason"] in ("live_version", "acoustic_version")]
        df = df.drop(index=live_acoustic_indices, errors="ignore").reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────
    # 1. Duplicados exactos por Spotify ID
    # ─────────────────────────────────────────────────────────────
    if "spotify_id" in df.columns:
        exact_dupes = df.duplicated(subset=["spotify_id"], keep=False)
        if exact_dupes.any():
            # Quedarse con la de mayor quality_score
            df["_quality"] = df.apply(version_quality_score, axis=1)
            df = df.sort_values(["spotify_id", "_quality"], ascending=[True, False])
            df = df.drop_duplicates(subset=["spotify_id"], keep="first")
            df = df.drop(columns=["_quality"])
            logger.info("Duplicados exactos por Spotify ID eliminados")

    # ─────────────────────────────────────────────────────────────
    # 2. Duplicados por ISRC (mismo master)
    # ─────────────────────────────────────────────────────────────
    if drop_isrc_duplicates and "isrc" in df.columns:
        isrc_dupes = df.duplicated(subset=["isrc"], keep=False) & df["isrc"].notna() & (df["isrc"] != "")
        if isrc_dupes.any():
            df["_quality"] = df.apply(version_quality_score, axis=1)
            df = df.sort_values(["isrc", "_quality"], ascending=[True, False])
            df_isrc = df.drop_duplicates(subset=["isrc"], keep="first")
            removed_isrc = df[~df.index.isin(df_isrc.index)]
            for _, row in removed_isrc.iterrows():
                removed.append({
                    "index": row.name,
                    "reason": "isrc_duplicate",
                    "name": row.get("name"),
                    "artist": row.get("artist"),
                    "isrc": row.get("isrc"),
                })
            df = df_isrc.drop(columns=["_quality"])
            logger.info("Duplicados por ISRC eliminados")

    # ─────────────────────────────────────────────────────────────
    # 3. Fuzzy matching: nombre+artista para detectar versiones distintas
    #    (remaster, remix, radio edit, etc. del mismo tema)
    # ─────────────────────────────────────────────────────────────
    if "name" in df.columns and "artist" in df.columns:
        df["_norm_name"] = df["name"].apply(_normalize_text)
        df["_norm_artist"] = df["artist"].apply(_normalize_artist)
        df["_match_key"] = df["_norm_name"] + " :: " + df["_norm_artist"]

        # Comparar todos contra todos (O(n²) — para 2.000 tracks son 4M comparaciones,
        # rapidfuzz procesa ~100K/s, total ~40s. Aceptable.)
        # Optimización: primero agrupar por nombre exacto normalizado,
        # luego fuzzy solo dentro de cada grupo.
        groups: dict[str, list[int]] = {}
        for idx, row in df.iterrows():
            # Usar una clave de agrupación menos estricta (primeras 10 chars del nombre)
            grouping_key = row["_norm_name"][:10]
            groups.setdefault(grouping_key, []).append(idx)

        for key, indices in groups.items():
            if len(indices) <= 1:
                continue

            # Comparar fuzzy dentro del grupo
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]

                    name_i = df.at[idx_i, "_norm_name"]
                    name_j = df.at[idx_j, "_norm_name"]
                    artist_i = df.at[idx_i, "_norm_artist"]
                    artist_j = df.at[idx_j, "_norm_artist"]

                    # Score de similitud: media de name y artist
                    name_ratio = fuzz.ratio(name_i, name_j)
                    artist_ratio = fuzz.ratio(artist_i, artist_j)
                    combined = (name_ratio + artist_ratio) / 2

                    if combined >= fuzzy_threshold:
                        # Son duplicados (versiones distintas del mismo tema)
                        # Quedarse con la de mayor quality_score
                        quality_i = version_quality_score(df.loc[idx_i])
                        quality_j = version_quality_score(df.loc[idx_j])

                        if quality_i >= quality_j:
                            loser_idx = idx_j
                        else:
                            loser_idx = idx_i

                        removed.append({
                            "index": loser_idx,
                            "reason": "fuzzy_duplicate",
                            "name": df.at[loser_idx, "name"],
                            "artist": df.at[loser_idx, "artist"],
                            "similarity": combined,
                        })

        # Eliminar los perdedores de fuzzy matching
        fuzzy_loser_indices = [r["index"] for r in removed if r["reason"] == "fuzzy_duplicate"]
        df = df.drop(index=fuzzy_loser_indices, errors="ignore")

        # Limpiar columnas temporales
        df = df.drop(columns=["_norm_name", "_norm_artist", "_match_key"], errors="ignore")
        df = df.reset_index(drop=True)

    logger.info(
        "Deduplicación completa: %d → %d pistas (%d eliminadas)",
        len(df) + len(removed), len(df), len(removed),
    )

    return df, removed


# ────────────────────────────────────────────────────────────────
# Funciones auxiliares para tests y compatibilidad con CLI
# ────────────────────────────────────────────────────────────────

def _is_duplicate_pair(row1: pd.Series, row2: pd.Series) -> bool:
    """
    Determina si dos filas son duplicadas combinando:
      1. ISRC identico (si ambas tienen ISRC valido)
      2. Fuzzy matching de name+artist (ratio >= 90)
    """
    # ISRC
    isrc1 = str(row1.get("isrc", "")).strip().upper()
    isrc2 = str(row2.get("isrc", "")).strip().upper()
    if isrc1 and isrc1 != "NAN" and len(isrc1) >= 12 and isrc2 and isrc2 != "NAN" and len(isrc2) >= 12:
        if isrc1 == isrc2:
            return True

    # Fuzzy matching
    name1 = _normalize_text(row1.get("name", ""))
    artist1 = _normalize_artist(row1.get("artist", ""))
    name2 = _normalize_text(row2.get("name", ""))
    artist2 = _normalize_artist(row2.get("artist", ""))

    if not name1 or not name2:
        return False

    name_ratio = fuzz.ratio(name1, name2)
    artist_ratio = fuzz.ratio(artist1, artist2)
    combined = (name_ratio + artist_ratio) / 2

    return combined >= 90


def _select_best_version(rows: pd.DataFrame) -> pd.Series:
    """Selecciona la mejor version de un grupo de duplicados."""
    if len(rows) == 1:
        return rows.iloc[0]
    scores = rows.apply(version_quality_score, axis=1)
    best_idx = scores.idxmax()
    return rows.loc[best_idx]


def deduplicate(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, list[dict]]:
    """Alias de deduplicate_playlist para compatibilidad con CLI."""
    return deduplicate_playlist(df, **kwargs)


def flag_live_acoustic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca pistas que son probablemente live o acusticas para revision manual.
    Anade columna 'flag_review' con etiquetas separadas por coma.
    No elimina las pistas, solo las marca.
    """
    df = df.copy()
    df["flag_review"] = ""

    for idx, row in df.iterrows():
        flags = []

        # Liveness alta (>0.8)
        live_val = row.get("live")
        if live_val is not None and not pd.isna(live_val):
            try:
                if float(live_val) > 0.8:
                    flags.append("live")
            except (ValueError, TypeError):
                pass

        # Acousticness alta (>0.5)
        acoustic_val = row.get("acoustic")
        if acoustic_val is not None and not pd.isna(acoustic_val):
            try:
                if float(acoustic_val) > 0.5:
                    flags.append("acoustic")
            except (ValueError, TypeError):
                pass

        # Detectar "live" en el nombre
        name = str(row.get("name", "")).lower()
        if any(kw in name for kw in ["live", "en vivo", "concert", "tour"]):
            flags.append("live_in_name")

        # Detectar "acoustic" o "acustico" en el nombre
        if any(kw in name for kw in ["acoustic", "acustico", "unplugged", "stripped"]):
            flags.append("acoustic_in_name")

        if flags:
            df.at[idx, "flag_review"] = ",".join(flags)

    n_flagged = (df["flag_review"] != "").sum()
    if n_flagged:
        logger.info("%d pistas marcadas para revision manual (live/acusticas)", n_flagged)

    return df


# ────────────────────────────────────────────────────────────────
# Flag de pistas live/acústicas para revisión manual
# ────────────────────────────────────────────────────────────────

def flag_live_acoustic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca pistas que son probablemente live o acústicas para revisión manual.
    No las elimina, solo añade una columna 'flag_review'.
    """
    df = df.copy()
    df["flag_review"] = ""

    for idx, row in df.iterrows():
        flags = []

        live = row.get("live")
        if pd.notna(live) and live and float(live) > 0.8:
            flags.append("live")

        acoustic = row.get("acoustic")
        if pd.notna(acoustic) and acoustic and float(acoustic) > 0.5:
            flags.append("acoustic")

        name = str(row.get("name", "")).lower()
        if any(kw in name for kw in ["live", "en vivo", "concert", "tour"]):
            flags.append("live_in_name")

        if any(kw in name for kw in ["acoustic", "acústico", "unplugged", "stripped"]):
            flags.append("acoustic_in_name")

        if flags:
            df.at[idx, "flag_review"] = ",".join(flags)

    n_flagged = (df["flag_review"] != "").sum()
    if n_flagged:
        logger.info("%d pistas marcadas para revisión manual", n_flagged)

    return df
