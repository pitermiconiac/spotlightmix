"""
Tests unitarios del módulo Camelot
===================================
Cubre las 6 reglas de compatibilidad armónica + casos edge.
"""
import pytest

from spotlight_mix.camelot import (
    CamelotKey,
    camelot_compatibility,
    _circular_distance,
    SPOTIFY_KEY_MODE_TO_CAMELOT,
    CAMELOT_TO_SPOTIFY_KEY_MODE,
    ALL_CAMELOT_KEYS,
    all_camelot_keys,
    build_compatibility_matrix,
)


class TestCamelotKey:
    def test_from_string_valid(self):
        assert CamelotKey.from_string("8A").number == 8
        assert CamelotKey.from_string("8A").letter == "A"
        assert CamelotKey.from_string("12B").number == 12
        assert CamelotKey.from_string("12B").letter == "B"

    def test_from_string_lowercase(self):
        assert CamelotKey.from_string("8a").number == 8
        assert CamelotKey.from_string("8a").letter == "A"

    def test_from_string_whitespace(self):
        assert str(CamelotKey.from_string("  8A  ")) == "8A"

    def test_from_string_invalid_letter(self):
        with pytest.raises(ValueError):
            CamelotKey.from_string("8C")

    def test_from_string_invalid_number(self):
        with pytest.raises(ValueError):
            CamelotKey.from_string("0A")
        with pytest.raises(ValueError):
            CamelotKey.from_string("13A")

    def test_from_string_empty(self):
        with pytest.raises(ValueError):
            CamelotKey.from_string("")

    def test_from_spotify_valid(self):
        # C major = key 0, mode 1 → 8B
        ck = CamelotKey.from_spotify(0, 1)
        assert ck is not None
        assert str(ck) == "8B"

        # A minor = key 9, mode 0 → 8A
        ck = CamelotKey.from_spotify(9, 0)
        assert ck is not None
        assert str(ck) == "8A"

    def test_from_spotify_invalid_key(self):
        assert CamelotKey.from_spotify(-1, 1) is None
        assert CamelotKey.from_spotify(12, 0) is None

    def test_from_spotify_invalid_mode(self):
        assert CamelotKey.from_spotify(0, 2) is None
        assert CamelotKey.from_spotify(0, -1) is None

    def test_to_spotify_roundtrip(self):
        for s in ALL_CAMELOT_KEYS:
            ck = CamelotKey.from_string(s)
            key, mode = ck.to_spotify()
            assert CAMELOT_TO_SPOTIFY_KEY_MODE[s] == (key, mode)
            # Reverse roundtrip
            ck2 = CamelotKey.from_spotify(key, mode)
            assert str(ck2) == s

    def test_all_24_keys_present(self):
        keys = list(all_camelot_keys())
        assert len(keys) == 24
        # Verificar que todas las 24 llaves están
        for s in ALL_CAMELOT_KEYS:
            assert any(str(k) == s for k in keys)


class TestCircularDistance:
    @pytest.mark.parametrize("n1,n2,expected", [
        (1, 1, 0),
        (1, 2, 1),
        (1, 12, 1),   # Circular: 1 y 12 son adyacentes
        (1, 6, 5),    # Distancia 5
        (1, 7, 6),    # Distancia 6 (maxima posible, = 12-6)
        (8, 8, 0),
        (8, 9, 1),
        (8, 7, 1),
        (8, 10, 2),
        (8, 6, 2),    # Circular: min(2, 10) = 2
        (8, 12, 4),   # Circular: min(4, 8) = 4
        (8, 4, 4),    # Circular: min(4, 8) = 4
    ])
    def test_circular_distance(self, n1, n2, expected):
        assert _circular_distance(n1, n2) == expected


class TestCamelotCompatibility:
    """Test de las 6 reglas de compatibilidad armónica."""

    def test_mismo_codigo(self):
        """Mismo código → 1.0"""
        for s in ["8A", "8B", "1A", "12B", "5A"]:
            k = CamelotKey.from_string(s)
            assert camelot_compatibility(k, k) == 1.0

    def test_plus_minus_1_misma_letra(self):
        """±1 misma letra → 0.85"""
        k8a = CamelotKey.from_string("8A")
        k9a = CamelotKey.from_string("9A")
        k7a = CamelotKey.from_string("7A")
        assert camelot_compatibility(k8a, k9a) == 0.85
        assert camelot_compatibility(k8a, k7a) == 0.85
        assert camelot_compatibility(k9a, k8a) == 0.85  # Simétrica

    def test_a_b_mismo_numero(self):
        """A↔B mismo número → 0.80"""
        k8a = CamelotKey.from_string("8A")
        k8b = CamelotKey.from_string("8B")
        assert camelot_compatibility(k8a, k8b) == 0.80
        assert camelot_compatibility(k8b, k8a) == 0.80

    def test_plus_minus_1_a_b(self):
        """±1 + A↔B → 0.60"""
        k8a = CamelotKey.from_string("8A")
        k9b = CamelotKey.from_string("9B")
        k7b = CamelotKey.from_string("7B")
        assert camelot_compatibility(k8a, k9b) == 0.60
        assert camelot_compatibility(k8a, k7b) == 0.60
        assert camelot_compatibility(k9b, k8a) == 0.60

    def test_plus_2_misma_letra(self):
        """+2 misma letra → 0.50"""
        k8a = CamelotKey.from_string("8A")
        k10a = CamelotKey.from_string("10A")
        k6a = CamelotKey.from_string("6A")
        assert camelot_compatibility(k8a, k10a) == 0.50
        assert camelot_compatibility(k8a, k6a) == 0.50  # Circular: min(2, 10) = 2

    def test_plus_2_a_b(self):
        """+2 + A↔B → 0.35"""
        k8a = CamelotKey.from_string("8A")
        k10b = CamelotKey.from_string("10B")
        k6b = CamelotKey.from_string("6B")
        assert camelot_compatibility(k8a, k10b) == 0.35
        assert camelot_compatibility(k8a, k6b) == 0.35

    def test_dist_3(self):
        """Distancia 3 → 0.20"""
        k8a = CamelotKey.from_string("8A")
        k11a = CamelotKey.from_string("11A")
        k5a = CamelotKey.from_string("5A")
        assert camelot_compatibility(k8a, k11a) == 0.20
        assert camelot_compatibility(k8a, k5a) == 0.20

    def test_dist_4(self):
        """Distancia 4 → 0.10"""
        k8a = CamelotKey.from_string("8A")
        k12a = CamelotKey.from_string("12A")
        k4a = CamelotKey.from_string("4A")
        assert camelot_compatibility(k8a, k12a) == 0.10
        assert camelot_compatibility(k8a, k4a) == 0.10

    def test_dist_5_plus_incompatible(self):
        """Distancia >=5 → 0.0"""
        k8a = CamelotKey.from_string("8A")
        k1a = CamelotKey.from_string("1A")  # Dist 5 (circular: min(7, 5) = 5)
        k2a = CamelotKey.from_string("2A")  # Dist 6 (maximo)
        assert camelot_compatibility(k8a, k1a) == 0.0
        assert camelot_compatibility(k8a, k2a) == 0.0

    def test_simetria_total(self):
        """La compatibilidad es simétrica para todas las 24x24 combinaciones."""
        matrix = build_compatibility_matrix()
        for k1 in matrix:
            for k2 in matrix:
                assert matrix[k1][k2] == matrix[k2][k1], f"{k1}↔{k2}"

    def test_compatibilidad_alta_cerca(self):
        """Llaves cercanas en el wheel tienen alta compatibilidad (>0.5)."""
        # Las llaves adyacentes en el circle of fifths son compatibles
        k8a = CamelotKey.from_string("8A")
        for s in ["7A", "8A", "9A", "8B", "7B", "9B"]:
            k = CamelotKey.from_string(s)
            assert camelot_compatibility(k8a, k) > 0.5

    def test_compatibilidad_baja_lejos(self):
        """Llaves lejanas en el wheel tienen baja compatibilidad (<0.1)."""
        k8a = CamelotKey.from_string("8A")
        for s in ["1A", "2A", "3A", "1B", "2B", "3B"]:
            k = CamelotKey.from_string(s)
            assert camelot_compatibility(k8a, k) < 0.1

    def test_diagonal_uno(self):
        """La diagonal de la matriz de compatibilidad es 1.0."""
        matrix = build_compatibility_matrix()
        for k in matrix:
            assert matrix[k][k] == 1.0
