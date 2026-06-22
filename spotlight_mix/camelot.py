"""
Camelot Wheel — Mapeo y compatibilidad armónica
===============================================
Sistema de 24 llaves (1A-12B) usado por DJs para harmonic mixing.
Basado en el circle of fifths: llaves adyacentes son armónicamente compatibles.

Tabla completa y reglas de compatibilidad:
  - Mismo código (8A→8A)         → 1.00 (perfecto)
  - ±1, misma letra (8A→9A)    → 0.85
  - A↔B, mismo número (8A→8B)  → 0.80 (relativo menor/mayor)
  - ±1 + A↔B (8A→9B)          → 0.60
  - +2, misma letra (8A→10A)    → 0.50 (energy boost)
  - Otros saltos                → 0.05-0.35
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ────────────────────────────────────────────────────────────────
# Camelot Wheel: tabla completa 24 llaves
# ────────────────────────────────────────────────────────────────
# Estructura: (numero, letra) → (notación musical, nombre común)
#
#  Mayores (B): 8B=C, 3B=C#/Db, 10B=D, 5B=D#/Eb, 12B=E, 7B=F,
#                2B=F#/Gb, 9B=G, 4B=G#/Ab, 11B=A, 6B=A#/Bb, 1B=B
#
#  Menores (A): 8A=A, 3A=A#/Bb, 10A=B, 5A=C, 12A=C#/Db, 7A=D,
#                2A=D#/Eb, 9A=E, 4A=F, 11A=F#/Gb, 6A=G, 1A=G#/Ab

CAMELOT_NOTATION: dict[str, str] = {
    # Menores (A)
    "1A": "G#/Ab minor",  "2A": "D#/Eb minor", "3A": "A#/Bb minor",
    "4A": "F minor",       "5A": "C minor",     "6A": "G minor",
    "7A": "D minor",       "8A": "A minor",     "9A": "E minor",
    "10A": "B minor",     "11A": "F#/Gb minor", "12A": "C#/Db minor",
    # Mayores (B)
    "1B": "B major",       "2B": "F#/Gb major",  "3B": "C#/Db major",
    "4B": "G#/Ab major",   "5B": "D#/Eb major",  "6B": "A#/Bb major",
    "7B": "F major",        "8B": "C major",      "9B": "G major",
    "10B": "D major",      "11B": "A major",     "12B": "E major",
}

# Lista ordenada de las 24 llaves (1A, 1B, 2A, 2B, ..., 12A, 12B)
ALL_CAMELOT_KEYS: list[str] = [
    f"{n}{letter}" for n in range(1, 13) for letter in ("A", "B")
]


# ────────────────────────────────────────────────────────────────
# Mapeo Spotify (key, mode) → Camelot
# ────────────────────────────────────────────────────────────────
# Spotify devuelve:
#   key: 0-11 (Pitch Class: 0=C, 1=C#/Db, 2=D, ..., 11=B)
#   mode: 0=menor, 1=mayor
#
# Mapeo a Camelot:
#   Mayor (mode=1) → llaves B
#   Menor (mode=0) → llaves A

# Tabla: (key, mode) → Camelot
# Mejor (B): 0→8B, 1→3B, 2→10B, 3→5B, 4→12B, 5→7B, 6→2B, 7→9B, 8→4B, 9→11B, 10→6B, 11→1B
# Menor (A): 0→5A, 1→12A, 2→7A, 3→2A, 4→9A, 5→4A, 6→11A, 7→6A, 8→1A, 9→8A, 10→3A, 11→10A

SPOTIFY_KEY_MODE_TO_CAMELOT: dict[tuple[int, int], str] = {
    # Mayor (mode=1 → B)
    (0, 1): "8B",   # C major
    (1, 1): "3B",   # C#/Db major
    (2, 1): "10B",  # D major
    (3, 1): "5B",   # D#/Eb major
    (4, 1): "12B",  # E major
    (5, 1): "7B",   # F major
    (6, 1): "2B",   # F#/Gb major
    (7, 1): "9B",   # G major
    (8, 1): "4B",   # G#/Ab major
    (9, 1): "11B",  # A major
    (10, 1): "6B",  # A#/Bb major
    (11, 1): "1B",  # B major
    # Menor (mode=0 → A)
    (0, 0): "5A",   # C minor
    (1, 0): "12A",  # C#/Db minor
    (2, 0): "7A",   # D minor
    (3, 0): "2A",   # D#/Eb minor
    (4, 0): "9A",   # E minor
    (5, 0): "4A",   # F minor
    (6, 0): "11A",  # F#/Gb minor
    (7, 0): "6A",   # G minor
    (8, 0): "1A",   # G#/Ab minor
    (9, 0): "8A",   # A minor
    (10, 0): "3A",  # A#/Bb minor
    (11, 0): "10A", # B minor
}

# Mapping inverso: Camelot → (key, mode)
CAMELOT_TO_SPOTIFY_KEY_MODE: dict[str, tuple[int, int]] = {
    v: k for k, v in SPOTIFY_KEY_MODE_TO_CAMELOT.items()
}


# ────────────────────────────────────────────────────────────────
# CamelotKey: dataclass para operaciones type-safe
# ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CamelotKey:
    """Representa una llave del Camelot Wheel (ej: 8A, 12B)."""
    number: int  # 1-12
    letter: str  # "A" (menor) o "B" (mayor)

    @classmethod
    def from_string(cls, s: str) -> CamelotKey:
        """Crea desde string '8A', '12B', etc."""
        s = s.strip().upper()
        if len(s) < 2 or len(s) > 3:
            raise ValueError(f"Camelot inválido: {s!r}")

        # Formato "8A" o "12B"
        letter = s[-1]
        if letter not in ("A", "B"):
            raise ValueError(f"Letra Camelot inválida: {letter!r} en {s!r}")

        number_str = s[:-1]
        try:
            number = int(number_str)
        except ValueError:
            raise ValueError(f"Número Camelot inválido: {number_str!r} en {s!r}")

        if not 1 <= number <= 12:
            raise ValueError(f"Número Camelot fuera de rango: {number} en {s!r}")

        return cls(number=number, letter=letter)

    @classmethod
    def from_spotify(cls, key: int, mode: int) -> CamelotKey | None:
        """Crea desde Spotify (key 0-11, mode 0/1). None si key=-1."""
        if key == -1 or key not in range(12) or mode not in (0, 1):
            return None
        camelot_str = SPOTIFY_KEY_MODE_TO_CAMELOT.get((key, mode))
        if not camelot_str:
            return None
        return cls.from_string(camelot_str)

    def __str__(self) -> str:
        return f"{self.number}{self.letter}"

    def to_spotify(self) -> tuple[int, int]:
        """Convierte a (key, mode) de Spotify."""
        return CAMELOT_TO_SPOTIFY_KEY_MODE[str(self)]


# ────────────────────────────────────────────────────────────────
# Compatibilidad armónica: scoring 0.0 - 1.0
# ────────────────────────────────────────────────────────────────

def _circular_distance(n1: int, n2: int) -> int:
    """Distancia circular entre dos números Camelot (1-12)."""
    diff = abs(n1 - n2)
    return min(diff, 12 - diff)


def camelot_compatibility(k1: CamelotKey, k2: CamelotKey) -> float:
    """
    Calcula la compatibilidad armónica entre dos llaves Camelot.

    Devuelve un score 0.0 - 1.0 basado en las reglas:
      - Mismo código (8A→8A)           → 1.00 (perfecto)
      - ±1, misma letra (8A→9A)       → 0.85
      - A↔B, mismo número (8A→8B)     → 0.80 (relativo menor/mayor)
      - ±1 + A↔B (8A→9B)              → 0.60
      - +2, misma letra (8A→10A)       → 0.50 (energy boost)
      - Otros saltos (dist=3-4)       → 0.05-0.35
      - Distancia >=5                 → 0.00 (incompatible)
    """
    # Mismo código
    if k1 == k2:
        return 1.0

    dist = _circular_distance(k1.number, k2.number)
    same_letter = k1.letter == k2.letter
    same_number = k1.number == k2.number

    # A↔B mismo número (relativo menor/mayor)
    if same_number and not same_letter:
        return 0.80

    # ±1 misma letra (movimiento adyacente en circle of fifths)
    if dist == 1 and same_letter:
        return 0.85

    # ±1 + A↔B (saltar una posición y cambiar modo)
    if dist == 1 and not same_letter:
        return 0.60

    # +2 misma letra (energy boost dos posiciones)
    if dist == 2 and same_letter:
        return 0.50

    # +2 + A↔B
    if dist == 2 and not same_letter:
        return 0.35

    # Distancia 3-4 (saltos largos, raramente compatibles)
    if dist == 3:
        return 0.20
    if dist == 4:
        return 0.10

    # Distancia >=5: incompatible
    return 0.0


# ────────────────────────────────────────────────────────────────
# Utilidades para iterar todas las 24 llaves
# ────────────────────────────────────────────────────────────────

def all_camelot_keys() -> Iterable[CamelotKey]:
    """Itera las 24 llaves en orden (1A, 1B, 2A, 2B, ..., 12A, 12B)."""
    for s in ALL_CAMELOT_KEYS:
        yield CamelotKey.from_string(s)


def key_to_camelot(key: int | str, mode: int | None = None) -> CamelotKey:
    """
    Convierte a CamelotKey desde:
      - Spotify (key: int 0-11, mode: int 0/1)
      - String Camelot directo ("8A")
    """
    if isinstance(key, str):
        return CamelotKey.from_string(key)

    if mode is None:
        raise ValueError("Si key es int (Spotify), se requiere mode (0 o 1)")

    return CamelotKey.from_spotify(key, mode)


def build_compatibility_matrix() -> dict[str, dict[str, float]]:
    """
    Construye la matriz completa de compatibilidad 24x24.
    Útil para debugging, visualización y tests.
    """
    matrix: dict[str, dict[str, float]] = {}
    for k1 in all_camelot_keys():
        matrix[str(k1)] = {}
        for k2 in all_camelot_keys():
            matrix[str(k1)][str(k2)] = camelot_compatibility(k1, k2)
    return matrix
