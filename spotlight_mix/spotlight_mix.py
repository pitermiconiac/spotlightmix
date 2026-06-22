#!/usr/bin/env python3
"""
Spotlight Mix — CLI principal
============================
Reordena playlists de Spotify basándose en tags de un CSV para maximizar
la calidad de transiciones Automix.

Uso:
  python spotlight_mix.py \
    --csv mi_playlist.csv \
    --playlist-id 37i9dQZF1... \
    --client-id TU_ID \
    --client-secret TU_SECRET

  # O sin Spotify (solo ordenar y volcar CSV):
  python spotlight_mix.py --csv mi_playlist.csv --output-csv ordenada.csv
"""
from __future__ import annotations

import sys
import time
import logging
import argparse
from pathlib import Path

# Añadir el directorio padre al path para importar spotlight_mix
sys.path.insert(0, str(Path(__file__).parent.parent))

from spotlight_mix.csv_loader import load_playlist_csv, parse_weights
from spotlight_mix.dedup import deduplicate_playlist as deduplicate, flag_live_acoustic
from spotlight_mix.distance_matrix import build_distance_matrix
from spotlight_mix.sequencer import sequence, route_cost


logger = logging.getLogger("spotlight_mix")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def find_seed_track(df) -> int:
    """
    Elige la pista inicial para el greedy: la de menor BPM.
    Una pista de BPM bajo es buena apertura (warm-up del set).
    """
    if "bpm" not in df.columns or df.empty:
        return 0

    bpm = df["bpm"].dropna()
    if bpm.empty:
        return 0

    return int(bpm.idxmin())


def build_ordered_uris(df) -> list[str]:
    """Construye la lista de URIs de Spotify desde el DataFrame ordenado."""
    uris = []
    for _, row in df.iterrows():
        sid = row.get("spotify_id")
        if sid and str(sid).strip() and str(sid) != "nan":
            uris.append(f"spotify:track:{str(sid).strip()}")
    return uris


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spotlight_mix",
        description="Optimizador de Playlists para Spotify Automix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Argumentos obligatorios ──
    parser.add_argument(
        "--csv", required=True,
        help="Ruta al CSV con los tags de la playlist",
    )

    # ── Argumentos de Spotify (opcionales si --output-csv) ──
    parser.add_argument("--client-id", default=None, help="Spotify Client ID")
    parser.add_argument("--client-secret", default=None, help="Spotify Client Secret")
    parser.add_argument(
        "--redirect-uri", default="http://127.0.0.1:8080",
        help="Redirect URI de OAuth (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--playlist-id", default=None,
        help="ID de la playlist a reordenar (para leer metadatos de Spotify)",
    )
    parser.add_argument(
        "--create-new", action="store_true",
        help="Crear una nueva playlist en lugar de reemplazar la original",
    )
    parser.add_argument(
        "--new-playlist-name", default=None,
        help="Nombre de la nueva playlist (default: '{original} [Spotlight Mix]')",
    )

    # ── Argumentos del algoritmo ──
    parser.add_argument(
        "--weights", default=None,
        help="Pesos de features: 'bpm=0.40 key=0.20 energy=0.15 ...'",
    )
    parser.add_argument(
        "--bpm-tolerance", type=float, default=4.0,
        help="Tolerancia BPM ±X% (default: 4.0, Automix no pitch-shiftea)",
    )
    parser.add_argument(
        "--seed-track", default=None,
        help="Nombre de la pista semilla para empezar el orden",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Saltar la deduplicación",
    )
    parser.add_argument(
        "--no-2opt", action="store_true",
        help="Saltar el 2-opt (solo greedy)",
    )
    parser.add_argument(
        "--held-karp", action="store_true",
        help="Usar Held-Karp exacto (solo ≤20 tracks, para validación)",
    )

    # ── Output ──
    parser.add_argument(
        "--output-csv", default=None,
        help="Volcar el orden resultante a un CSV (no necesita Spotify)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Mostrar reporte detallado del ordenamiento",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Logging detallado (DEBUG)",
    )

    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("Spotlight Mix — Optimizador de Playlists para Automix")
    logger.info("=" * 60)

    # ────────────────────────────────────────────────────────────────
    # 1. Cargar CSV
    # ────────────────────────────────────────────────────────────────
    logger.info("Cargando CSV: %s", args.csv)

    try:
        df = load_playlist_csv(args.csv)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Error cargando CSV: %s", e)
        return 1

    logger.info("CSV cargado: %d pistas", len(df))

    # ────────────────────────────────────────────────────────────────
    # 2. Deduplicación
    # ────────────────────────────────────────────────────────────────
    removed = []
    if not args.no_dedup:
        logger.info("Deduplicando...")
        df, removed = deduplicate(df)
        if removed:
            logger.info("Pistas eliminadas (duplicados):")
            for r in removed:
                logger.info("  - %s - %s (reason: %s)", r.get("name", "?"), r.get("artist", "?"), r.get("reason", "?"))

    # Marcar live/acústicas para revisión
    df = flag_live_acoustic(df)

    # ────────────────────────────────────────────────────────────────
    # 3. Construir matriz de distancias
    # ────────────────────────────────────────────────────────────────
    weights = parse_weights(args.weights)

    t0 = time.time()
    dist_matrix, matrix_info = build_distance_matrix(df, weights=weights)
    t_matrix = time.time() - t0
    logger.info("Matriz de distancias construida en %.2fs", t_matrix)

    # ────────────────────────────────────────────────────────────────
    # 4. Secuenciar (greedy + 2-opt)
    # ────────────────────────────────────────────────────────────────
    # Elegir pista semilla
    start_idx = find_seed_track(df)

    # Si se especificó --seed-track por nombre, buscarla
    if args.seed_track:
        seed_lower = args.seed_track.lower().strip()
        matches = df[df["name"].str.lower().str.strip() == seed_lower]
        if not matches.empty:
            start_idx = int(matches.index[0])
            logger.info("Pista semilla: %s (idx=%d)", args.seed_track, start_idx)
        else:
            logger.warning("Pista semilla '%s' no encontrada, usando default", args.seed_track)

    logger.info("Secuenciando %d pistas...", len(df))

    t0 = time.time()
    ordered_indices = sequence(
        dist_matrix,
        start_idx=start_idx,
        use_2opt=not args.no_2opt,
        use_held_karp=args.held_karp,
    )
    t_seq = time.time() - t0
    logger.info("Secuenciación completada en %.2fs", t_seq)

    # Reordenar el DataFrame
    df_ordered = df.iloc[ordered_indices].reset_index(drop=True)

    # ────────────────────────────────────────────────────────────────
    # 5. Reporte (opcional)
    # ────────────────────────────────────────────────────────────────
    if args.report:
        print_report(df_ordered, dist_matrix, ordered_indices, t_matrix, t_seq, removed)

    # ────────────────────────────────────────────────────────────────
    # 6. Volcar CSV ordenado (opcional, no necesita Spotify)
    # ────────────────────────────────────────────────────────────────
    if args.output_csv:
        logger.info("Volcando CSV ordenado: %s", args.output_csv)
        df_ordered.to_csv(args.output_csv, index=False)
        logger.info("CSV guardado. Proceso completado sin Spotify.")
        return 0

    # ────────────────────────────────────────────────────────────────
    # 7. Integración con Spotify
    # ────────────────────────────────────────────────────────────────
    if not args.client_id or not args.client_secret:
        logger.error(
            "Se requiere --client-id y --client-secret para crear la playlist en Spotify. "
            "Usa --output-csv si solo quieres el CSV ordenado."
        )
        return 1

    from spotlight_mix.spotify_client import SpotifyClient

    logger.info("Autenticando con Spotify...")
    client = SpotifyClient(
        client_id=args.client_id,
        client_secret=args.client_secret,
        redirect_uri=args.redirect_uri,
    )

    user_id = client.get_user_id()
    logger.info("Autenticado como: %s", user_id)

    # Construir URIs ordenadas
    ordered_uris = build_ordered_uris(df_ordered)
    logger.info("URIs a añadir: %d / %d", len(ordered_uris), len(df_ordered))

    if not ordered_uris:
        logger.error("No se encontraron Spotify IDs en el CSV. "
                     "Asegúrate de que la columna 'Spotify Track Id' esté presente.")
        return 1

    # Nombre de la nueva playlist
    playlist_name = args.new_playlist_name or "Spotlight Mix"
    if args.playlist_id and not args.new_playlist_name:
        try:
            original = client._playlist_items(args.playlist_id, limit=1)
            orig_name = original.get("name", "Playlist")
            playlist_name = f"{orig_name} [Spotlight Mix]"
        except Exception:
            playlist_name = "Spotlight Mix"

    description = (
        f"Playlist reordenada por Spotlight Mix (BPM/Camelot/Energy). "
        f"{len(ordered_uris)} pistas. Generada el {time.strftime('%Y-%m-%d')}."
    )

    logger.info("Creando playlist: %s", playlist_name)
    new_playlist = client.create_playlist(
        name=playlist_name,
        description=description,
        public=False,
    )
    playlist_id = new_playlist["id"]
    logger.info("Playlist creada: ID=%s", playlist_id)

    # Añadir pistas en batches de 100
    logger.info("Añadiendo %d pistas...", len(ordered_uris))
    snapshot = client.add_tracks_to_playlist(playlist_id, ordered_uris)
    logger.info("Playlist completada. Snapshot: %s", snapshot)

    print()
    print("=" * 60)
    print("✅ Playlist creada exitosamente:")
    print(f"   Nombre: {playlist_name}")
    print(f"   Pistas: {len(ordered_uris)}")
    print(f"   ID: {playlist_id}")
    print(f"   URL: https://open.spotify.com/playlist/{playlist_id}")
    print()
    print("   Activa Automix en Spotify → reproduce → disfruta!")
    print("=" * 60)

    return 0


def print_report(df, dist_matrix, ordered_indices, t_matrix, t_seq, removed):
    """Imprime un reporte detallado del ordenamiento."""
    total_cost = route_cost(dist_matrix, ordered_indices)

    print()
    print("=" * 70)
    print("REPORTE DE SPOTLIGHT MIX")
    print("=" * 70)
    print(f"Pistas totales: {len(df)}")
    print(f"Duplicados eliminados: {len(removed)}")
    print(f"Tiempo matriz de distancias: {t_matrix:.2f}s")
    print(f"Tiempo secuenciación: {t_seq:.2f}s")
    print(f"Coste total de la ruta: {total_cost:.4f}")
    print(f"Coste medio por transición: {total_cost / max(len(ordered_indices)-1, 1):.4f}")
    print()
    print("Primeras 10 pistas del orden:")
    print("-" * 70)
    for i in range(min(10, len(df))):
        row = df.iloc[ordered_indices[i]]
        name = row.get("name", "?")
        artist = row.get("artist", "?")
        bpm = row.get("bpm", "?")
        camelot = row.get("camelot", "?")
        energy = row.get("energy", "?")
        print(f"  {i+1:3d}. {bpm:5} BPM | {camelot:4s} | E={energy:.2f} | {name[:30]:30s} | {artist[:20]:20s}")
    print("-" * 70)
    print()


if __name__ == "__main__":
    sys.exit(main())
