"""
CSV Loader — Validador y cargador de CSV flexible
=================================================
Acepta headers con diferentes nombres de columnas y los normaliza.
"""
import re
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Mapping de sinónimos de columnas → nombre canónico
# ────────────────────────────────────────────────────────────────
# El CSV del usuario puede tener headers con diferentes nombres
# (ej: "BPM" o "Tempo", "Song" o "Track Name", "Spotify Track Id" o "ID").
# Esta tabla normaliza todo a un conjunto canónico.

COLUMN_ALIASES: dict[str, list[str]] = {
    # Identificadores
    "spotify_id": [
        "spotify track id", "spotify id", "track id", "id",
        "spotify uri", "uri", "spotify_track_id",
    ],
    "isrc": ["isrc", "isrc code", "international standard recording code"],
    # Metadatos básicos
    "name": ["song", "track name", "title", "name", "track", "song name"],
    "artist": ["artist", "artists", "artist name", "performer"],
    "album": ["album", "album name", "release"],
    "album_date": ["album date", "release date", "release_year", "year", "date"],
    "duration": ["duration", "duration (s)", "duration (ms)", "length", "duration_ms"],
    "popularity": ["popularity", "pop", "popularity score"],
    "explicit": ["explicit", "explicit content", "is explicit"],
    "genres": ["genres", "genre", "tags", "musical style"],
    # Tags de audio (ordena)
    "bpm": ["bpm", "tempo", "beats per minute", "tempo (bpm)"],
    "camelot": ["camelot", "camelot key", "key (camelot)", "harmonic key"],
    "key": ["key", "musical key", "key signature", "key (spotify)"],
    "mode": ["mode", "tonal mode", "major minor", "maj min"],
    "energy": ["energy", "energy score", "energy level"],
    "dance": ["dance", "danceability", "dance score"],
    "acoustic": ["acoustic", "acousticness", "acoustic score"],
    "instrumental": ["instrumental", "instrumentalness"],
    "valence": ["valence", "mood", "happiness", "valence score"],
    "speech": ["speech", "speechiness", "speech score"],
    "live": ["live", "liveness", "live score"],
    "loud": ["loud", "loud (db)", "loudness", "loudness (db)"],
    "time_signature": ["time signature", "time sig", "time_signature", "meter"],
    # Metadata de playlist
    "added_at": ["added at", "added", "date added", "added date"],
    "position": ["#", "position", "index", "track number"],
}


def _normalize_header(h: str) -> str:
    """Normaliza un header: lowercase, sin espacios extra, sin puntuación."""
    h = h.strip().lower()
    h = re.sub(r"[^\w\s]", " ", h)  # puntuación → espacio
    h = re.sub(r"\s+", " ", h)      # espacios múltiples → uno
    return h.strip()


# Construye el mapping inverso: alias (normalizado) → canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in COLUMN_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        norm = _normalize_header(alias)
        if norm not in _ALIAS_TO_CANONICAL:
            _ALIAS_TO_CANONICAL[norm] = canonical


# Columnas canónicas esperadas (orden del header del usuario)
EXPECTED_COLUMNS = [
    "position", "name", "artist", "bpm", "camelot", "energy",
    "added_at", "duration", "popularity", "genres", "album", "album_date",
    "dance", "acoustic", "instrumental", "valence", "speech", "live",
    "loud", "key", "mode", "time_signature",
    "spotify_id", "isrc", "explicit",
]

# Columnas críticas (sin las cuales no se puede ordenar)
CRITICAL_COLUMNS = ["bpm", "camelot", "energy"]


def load_playlist_csv(
    csv_path: str,
    encoding: str = "utf-8",
    separator: str | None = None,
) -> pd.DataFrame:
    """
    Carga un CSV de playlist y normaliza los nombres de columnas.

    Args:
        csv_path: Ruta al archivo CSV.
        encoding: Encoding del archivo (default utf-8).
        separator: Separador de columnas. Si None, se detecta automáticamente.

    Returns:
        DataFrame con columnas normalizadas.

    Raises:
        ValueError: Si faltan columnas críticas o el CSV está vacío.
    """
    # Detectar separador automáticamente si no se especifica
    if separator is None:
        try:
            df = pd.read_csv(csv_path, encoding=encoding, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(csv_path, encoding=encoding)
    else:
        df = pd.read_csv(csv_path, encoding=encoding, sep=separator)

    if df.empty:
        raise ValueError(f"CSV vacío: {csv_path}")

    # Normalizar headers
    original_columns = list(df.columns)
    normalized_columns = []

    for col in original_columns:
        norm = _normalize_header(col)
        canonical = _ALIAS_TO_CANONICAL.get(norm, norm)
        normalized_columns.append(canonical)

    df.columns = normalized_columns
    logger.info(
        "Headers normalizados: %s → %s",
        original_columns, normalized_columns,
    )

    # Validar columnas críticas
    missing_critical = [c for c in CRITICAL_COLUMNS if c not in df.columns]
    if missing_critical:
        raise ValueError(
            f"Faltan columnas críticas: {missing_critical}. "
            f"El CSV debe contener al menos: {CRITICAL_COLUMNS}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Reportar columnas faltantes (no críticas)
    missing_non_critical = [
        c for c in EXPECTED_COLUMNS if c not in df.columns
    ]
    if missing_non_critical:
        logger.warning(
            "Columnas faltantes (no críticas): %s. "
            "El algoritmo funcionará con valores por defecto.",
            missing_non_critical,
        )

    # ────────────────────────────────────────────────────────────────
    # Limpieza y tipos
    # ────────────────────────────────────────────────────────────────

    # BPM: entero positivo (o float). Si hay strings, convertir.
    if "bpm" in df.columns:
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
        invalid_bpm = df["bpm"].isna().sum()
        if invalid_bpm > 0:
            logger.warning("%d filas con BPM inválido (NaN)", invalid_bpm)

    # Energy, Dance, Acoustic, Valence, etc.: float 0-1
    for col in ["energy", "dance", "acoustic", "instrumental", "valence", "speech", "live"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Clamp a 0-1 (algunos CSVs traen 0-100 o porcentajes)
            if df[col].max() > 1.0 and df[col].max() <= 100.0:
                logger.info("%s en escala 0-100 → normalizando a 0-1", col)
                df[col] = df[col] / 100.0
            df[col] = df[col].clip(0.0, 1.0)

    # Loudness: float (dB, generalmente -60 a 0)
    if "loud" in df.columns:
        df["loud"] = pd.to_numeric(df["loud"], errors="coerce")

    # Key: int 0-11 (Pitch Class) o string (C, C#, D, ...)
    if "key" in df.columns:
        df["key"] = pd.to_numeric(df["key"], errors="coerce")

    # Mode: int 0 o 1 (mayor o menor)
    if "mode" in df.columns:
        df["mode"] = pd.to_numeric(df["mode"], errors="coerce")

    # Time signature: int 3-7
    if "time_signature" in df.columns:
        df["time_signature"] = pd.to_numeric(df["time_signature"], errors="coerce")

    # Duration: si está en formato mm:ss, convertir a segundos
    if "duration" in df.columns:
        if df["duration"].dtype == object:
            # Intentar parsear formatos como "3:45" o "3:45.123"
            df["duration"] = df["duration"].apply(_parse_duration)
        # Asegurar que sea numérico (float) incluso si _parse_duration devolvió None
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

    # Popularity: int 0-100
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")

    # Explicit: bool
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])

    # Spotify ID: limpiar (quitar prefijo spotify:track: si viene como URI)
    if "spotify_id" in df.columns:
        df["spotify_id"] = df["spotify_id"].astype(str).str.replace(
            r"^spotify:track:", "", regex=True
        ).str.strip()

    # Camelot: normalizar a formato "NA" o "NB" (ej: "8A", "12B")
    if "camelot" in df.columns:
        df["camelot"] = df["camelot"].astype(str).str.upper().str.strip()
        # Validar formato: 1-12 + A/B
        invalid_camelot = ~df["camelot"].str.match(r"^(1[0-2]|[1-9])[AB]$")
        n_invalid = invalid_camelot.sum()
        if n_invalid > 0:
            logger.warning(
                "%d filas con Camelot inválido. Ejemplos: %s",
                n_invalid, df.loc[invalid_camelot, "camelot"].unique()[:5],
            )

    logger.info(
        "CSV cargado: %d filas, %d columnas → %s",
        len(df), len(df.columns), list(df.columns),
    )

    return df


def _parse_duration(value: Any) -> float | None:
    """
    Convierte un valor de duración a segundos.
    Acepta: "3:45", "3:45.123", "225" (s), "225000" (ms), "3:45:30" (h:m:s).
    """
    if pd.isna(value):
        return None


def parse_weights(weights_str: str | None) -> dict[str, float] | None:
    """
    Parsea un string de pesos del CLI: 'bpm=0.40 key=0.20 energy=0.15'
    Devuelve None si weights_str es None o vacío.
    """
    if not weights_str:
        return None

    weights: dict[str, float] = {}
    for pair in weights_str.split():
        if "=" not in pair:
            logger.warning("Par de peso inválido (ignorado): %s", pair)
            continue
        key, val = pair.split("=", 1)
        key = key.strip().lower()
        try:
            weights[key] = float(val)
        except ValueError:
            logger.warning("Valor de peso inválido para %s: %s", key, val)

    return weights if weights else None

    s = str(value).strip()

    # Si es solo dígitos, asumir segundos o ms
    if s.replace(".", "", 1).isdigit():
        v = float(s)
        # Si > 1000, asumir ms → segundos
        if v > 1000:
            return v / 1000.0
        return v

    # Si contiene ":" intentar parsear como m:ss o h:m:s
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None

    if len(parts) == 2:        # m:ss
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:     # h:m:ss
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 4:     # d:h:m:s (improbable)
        return parts[0] * 86400 + parts[1] * 3600 + parts[2] * 60 + parts[3]

    return None
