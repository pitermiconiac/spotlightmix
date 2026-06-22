"""
Tests unitarios del módulo de Deduplicación
===========================================
Cubre: duplicado exacto, remaster vs original, radio edit vs extended mix,
ISRC matching, selección de mejor versión.
"""
import pytest
import pandas as pd

from spotlight_mix.dedup import (
    deduplicate_playlist,
    _is_duplicate_pair,
    _select_best_version,
    _normalize_text,
    flag_live_acoustic,
)


def make_row(**kwargs):
    """Crea una fila (pd.Series) con valores por defecto."""
    defaults = {
        "name": "Test Song",
        "artist": "Test Artist",
        "bpm": 120,
        "camelot": "8A",
        "energy": 0.5,
        "isrc": "",
        "duration": 200,
        "popularity": 50,
        "explicit": False,
        "album_date": "2020",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text("SONG NAME") == "song name"

    def test_accents(self):
        assert _normalize_text("Canción") == "cancion"
        assert _normalize_text("Nuñez") == "nunez"

    def test_empty_and_none(self):
        assert _normalize_text("") == ""
        assert _normalize_text(None) == ""


class TestIsDuplicatePair:
    def test_exact_duplicate(self):
        r1 = make_row(name="Song A", artist="Artist X", isrc="USRC123")
        r2 = make_row(name="Song A", artist="Artist X", isrc="USRC123")
        assert _is_duplicate_pair(r1, r2)

    def test_same_isrc_different_name(self):
        r1 = make_row(name="Song A", artist="Artist X", isrc="USRC123456789")
        r2 = make_row(name="Song A Remastered", artist="Artist X", isrc="USRC123456789")
        assert _is_duplicate_pair(r1, r2)

    def test_different_isrc_same_song(self):
        r1 = make_row(name="Song A", artist="Artist X", isrc="USRC111111111")
        r2 = make_row(name="Song A", artist="Artist X", isrc="USRC222222222")
        assert _is_duplicate_pair(r1, r2)

    def test_completely_different_songs(self):
        r1 = make_row(name="Song A", artist="Artist X")
        r2 = make_row(name="Song B", artist="Artist Y")
        assert not _is_duplicate_pair(r1, r2)


class TestSelectBestVersion:
    def test_prefer_extended_duration(self):
        rows = pd.DataFrame([
            make_row(name="Song A", duration=200, popularity=50, explicit=False),
            make_row(name="Song A", duration=300, popularity=60, explicit=True),
        ])
        best = _select_best_version(rows)
        assert best["duration"] == 300

    def test_prefer_higher_popularity(self):
        rows = pd.DataFrame([
            make_row(name="Song A", duration=200, popularity=30, explicit=False),
            make_row(name="Song A", duration=210, popularity=80, explicit=False),
        ])
        best = _select_best_version(rows)
        assert best["popularity"] == 80

    def test_single_row(self):
        rows = pd.DataFrame([make_row(name="Song A")])
        best = _select_best_version(rows)
        assert best["name"] == "Song A"


class TestDeduplicatePlaylist:
    def test_no_duplicates(self):
        df = pd.DataFrame([
            make_row(name="Song A", artist="Artist X"),
            make_row(name="Song B", artist="Artist Y"),
            make_row(name="Song C", artist="Artist Z"),
        ])
        df_dedup, removed = deduplicate_playlist(df)
        assert len(df_dedup) == 3
        assert len(removed) == 0

    def test_exact_duplicate_removed(self):
        df = pd.DataFrame([
            make_row(name="Song A", artist="Artist X", isrc="USRC111111111"),
            make_row(name="Song A", artist="Artist X", isrc="USRC111111111"),
        ])
        df_dedup, removed = deduplicate_playlist(df)
        assert len(df_dedup) == 1
        assert len(removed) == 1

    def test_empty_df(self):
        df = pd.DataFrame(columns=["name", "artist"])
        df_dedup, removed = deduplicate_playlist(df)
        assert len(df_dedup) == 0

    def test_single_row(self):
        df = pd.DataFrame([make_row(name="Song A")])
        df_dedup, removed = deduplicate_playlist(df)
        assert len(df_dedup) == 1
        assert len(removed) == 0


class TestFlagLiveAcoustic:
    def test_flag_live(self):
        df = pd.DataFrame([make_row(name="Song A", live=0.9)])
        df_flagged = flag_live_acoustic(df)
        assert "live" in df_flagged.iloc[0]["flag_review"]

    def test_flag_acoustic(self):
        df = pd.DataFrame([make_row(name="Song A", acoustic=0.7)])
        df_flagged = flag_live_acoustic(df)
        assert "acoustic" in df_flagged.iloc[0]["flag_review"]

    def test_no_flag_for_normal_track(self):
        df = pd.DataFrame([make_row(name="Song A", live=0.1, acoustic=0.1)])
        df_flagged = flag_live_acoustic(df)
        assert df_flagged.iloc[0]["flag_review"] == ""
