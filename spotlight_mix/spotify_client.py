"""
Spotify Client — Wrapper de spotipy con OAuth, throttle y reintentos
====================================================================
"""
import time
import logging
import functools
from typing import Any, Callable

import spotipy
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger(__name__)

# Scopes requeridos para leer y modificar playlists del usuario
SCOPE = (
    "playlist-read-private "
    "playlist-read-collaborative "
    "playlist-modify-public "
    "playlist-modify-private"
)


def _throttle_and_retry(
    min_interval: float = 0.5,
    max_retries: int = 5,
) -> Callable:
    """
    Decorador que aplica throttle (pausa entre llamadas) y reintentos
    exponenciales con jitter en caso de HTTP 429 Too Many Requests.
    """
    def decorator(func: Callable) -> Callable:
        last_call_times: dict = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            now = time.time()

            # Throttle: respetar intervalo mínimo entre llamadas del mismo método
            last = last_call_times.get(func_name, 0)
            elapsed = now - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    last_call_times[func_name] = time.time()
                    return result
                except spotipy.SpotifyException as e:
                    # 429 Too Many Requests
                    if e.http_status == 429:
                        retry_after = e.headers.get("Retry-After", 2)
                        try:
                            retry_after = int(retry_after)
                        except (ValueError, TypeError):
                            retry_after = 2

                        # Fallback exponencial si Retry-After es 0 o anómalo
                        wait = max(retry_after, 2 ** attempt)

                        # Jitter aleatorio (±25%)
                        jitter = wait * 0.25
                        actual_wait = wait + (time.time() % 1) * jitter - jitter / 2

                        logger.warning(
                            "429 en %s (intento %d/%d). "
                            "Esperando %.1fs (Retry-After=%s)",
                            func_name, attempt + 1, max_retries,
                            actual_wait, retry_after,
                        )
                        time.sleep(max(actual_wait, 1.0))
                        continue

                    # Otras excepciones de Spotify (401, 403, 404, 412...)
                    raise

            raise RuntimeError(
                f"Máximo de reintentos ({max_retries}) excedido en {func_name}"
            )

        return wrapper
    return decorator


class SpotifyClient:
    """
    Cliente wrapper de spotipy con:
      - OAuth Authorization Code (con refresh automático)
      - Throttle de 0.5s entre llamadas
      - Reintentos en HTTP 429 con Retry-After + backoff exponencial + jitter
      - Métodos de alto nivel para el flujo de reordenamiento
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://127.0.0.1:8080",
        cache_path: str = ".cache-spotlight-mix",
    ):
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPE,
            cache_path=cache_path,
            open_browser=True,
        )
        self.sp = spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=30,
        )
        self._user_id = None

    # ────────────────────────────────────────────────────────────────
    # Identidad del usuario
    # ────────────────────────────────────────────────────────────────

    def get_user_id(self) -> str:
        """Obtiene el user_id del usuario autenticado (cacheado en instancia)."""
        if self._user_id is None:
            user = self._current_user()
            self._user_id = user["id"]
        return self._user_id

    @_throttle_and_retry()
    def _current_user(self) -> dict:
        return self.sp.current_user()

    # ────────────────────────────────────────────────────────────────
    # Lectura de playlist (paginada)
    # ────────────────────────────────────────────────────────────────

    def get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """
        Lee todas las pistas de una playlist con paginación (100 por request).
        Para 2.000 pistas → 20 requests.
        """
        all_items = []
        offset = 0

        while True:
            response = self._playlist_items(playlist_id, limit=100, offset=offset)
            items = response.get("items", [])

            all_items.extend(items)
            logger.info(
                "Leídos %d/%d pistas (offset=%d)",
                len(all_items), response.get("total", len(all_items)), offset,
            )

            # Spotify devuelve <= limit cuando no hay más
            if len(items) < 100:
                break

            offset += 100

            # Safety check: evitar loop infinito si el total es <100
            if offset >= response.get("total", 0):
                break

        return all_items

    @_throttle_and_retry()
    def _playlist_items(self, playlist_id: str, limit: int = 100, offset: int = 0) -> dict:
        """Wrapper de sp.playlist_items() con throttle y retry."""
        return self.sp.playlist_items(playlist_id, limit=limit, offset=offset)

    # ────────────────────────────────────────────────────────────────
    # Metadata de pistas (batch)
    # ────────────────────────────────────────────────────────────────

    def get_tracks_metadata(self, track_ids: list[str]) -> list[dict]:
        """
        Obtiene metadata de pistas en batches de 100 (límite de Spotify).
        Devuelve: name, artist, album, duration_ms, popularity, explicit, ISRC.
        """
        all_tracks = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i : i + 100]
            # Filtrar None IDs (pistas no encontradas)
            batch = [t for t in batch if t]
            if not batch:
                continue

            result = self._tracks(batch)
            all_tracks.extend(result.get("tracks", []))
            logger.info("Metadata obtenida: %d/%d pistas", len(all_tracks), len(track_ids))

        return all_tracks

    @_throttle_and_retry()
    def _tracks(self, track_ids: list[str]) -> dict:
        """Wrapper de sp.tracks() (batch de hasta 100 IDs)."""
        return self.sp.tracks(track_ids)

    # ────────────────────────────────────────────────────────────────
    # Audio Features (batch) — solo disponible si la app tiene acceso
    # pre-noviembre-2024. Si el endpoint está deprecated, usar el CSV.
    # ────────────────────────────────────────────────────────────────

    def get_audio_features(self, track_ids: list[str]) -> list[dict] | None:
        """
        Intenta obtener audio features en batch (100 por request).
        Devuelve None si el endpoint no está disponible (deprecated nov 2024).
        """
        try:
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i : i + 100]
                batch = [t for t in batch if t]
                if not batch:
                    continue

                result = self._audio_features(batch)
                all_features.extend(result)
                logger.info("Audio features: %d/%d pistas", len(all_features), len(track_ids))

            return all_features
        except spotipy.SpotifyException as e:
            if e.http_status in (403, 404):
                logger.warning(
                    "Audio Features no disponible (deprecated nov 2024). "
                    "Usar el CSV como fuente de tags."
                )
                return None
            raise

    @_throttle_and_retry()
    def _audio_features(self, track_ids: list[str]) -> list[dict]:
        """Wrapper de sp.audio_features() (batch de hasta 100 IDs)."""
        return self.sp.audio_features(track_ids)

    # ────────────────────────────────────────────────────────────────
    # Crear playlist y añadir pistas
    # ────────────────────────────────────────────────────────────────

    def create_playlist(
        self,
        name: str,
        description: str = "",
        public: bool = False,
    ) -> dict:
        """Crea una nueva playlist en la cuenta del usuario."""
        user_id = self.get_user_id()
        return self._user_playlist_create(
            user_id, name, public=public, description=description
        )

    @_throttle_and_retry()
    def _user_playlist_create(
        self, user_id: str, name: str, public: bool, description: str
    ) -> dict:
        return self.sp.user_playlist_create(
            user_id, name, public=public, description=description
        )

    def add_tracks_to_playlist(
        self,
        playlist_id: str,
        track_uris: list[str],
    ) -> str:
        """
        Añade pistas a una playlist en batches de 100 URIs (límite de Spotify).
        Para 2.000 pistas → 20 requests POST.
        Devuelve el snapshot_id final.
        """
        from tqdm import tqdm

        snapshot_id = ""

        # Procesar en batches de 100
        batches = [
            track_uris[i : i + 100]
            for i in range(0, len(track_uris), 100)
        ]

        logger.info("Añadiendo %d pistas en %d batches de 100", len(track_uris), len(batches))

        for batch in tqdm(batches, desc="Añadiendo pistas", unit="batch"):
            if not batch:
                continue
            result = self._playlist_add_items(playlist_id, batch)
            snapshot_id = result.get("snapshot_id", snapshot_id)

        return snapshot_id

    @_throttle_and_retry()
    def _playlist_add_items(self, playlist_id: str, uris: list[str]) -> dict:
        """Wrapper de sp.playlist_add_items() con throttle y retry."""
        return self.sp.playlist_add_items(playlist_id, uris)

    # ────────────────────────────────────────────────────────────────
    # Reemplazar contenido de playlist (estrategia alternativa)
    # ────────────────────────────────────────────────────────────────

    def replace_playlist_tracks(self, playlist_id: str, track_uris: list[str]) -> None:
        """
        Reemplaza completamente el contenido de una playlist.
        Útil cuando se quiere reordenar in-situ en lugar de crear una nueva.
        Limitación: el body puede ser muy grande para 2.000 URIs.
        """
        # Spotify acepta máximo 100 URIs en un replace
        # Estrategia: vaciar + añadir en batches
        self._playlist_replace(playlist_id, [])
        self.add_tracks_to_playlist(playlist_id, track_uris)

    @_throttle_and_retry()
    def _playlist_replace(self, playlist_id: str, uris: list[str]) -> None:
        """Wrapper de sp.playlist_replace_items()."""
        self.sp.playlist_replace_items(playlist_id, uris)
