#!/usr/bin/env python3
"""
Spotlight Mix — CLI principal
=============================
Reordena playlists de Spotify basándose en tags de un CSV (BPM, Camelot,
Energy, etc.) para maximizar la calidad de transiciones Automix.

Uso:
  python spotlight_mix.py \\
    --csv mi_playlist.csv \\
    --playlist-id 37i9dQZF1... \\
    --client-id TU_ID \\
    --client-secret TU_SECRET

  # Solo ordenar (sin escribir a Spotify, volcar CSV):
  python spotlight_mix.py --csv mi_playlist.csv --output-csv ordenada.csv

  # Con pesos personalizados:
  python spotlight_mix.py --csv mi.csv --playlist-id X \\
    --client-id A --client-secret B \\
    --weights bpm=0.40 key=0.20 energy=0.15
"""
import sys
import logging
import argparse

import pandas as pd

from spotlight_mix.csv_loader import load_playlist_csv
from spotlight_mix.dedup import deduplicate, flag_live_acoustic
from spotlight_mix.distance_matrix import build_distance_matrix, parse_weights
from spotlight_mix.sequencer import sequence, route_cost

# Si se usa --playlist-id o --create-new, importar SpotifyClient
# (lazy import para que --output-csv funcione sin spotipy instalado)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spotlight Mix — Optimizador de Playlists para Spotify Automix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Solo ordenar y volcar CSV (sin Spotify):
  python spotlight_mix.py --csv playlist.csv --output-csv ordenada.csv

  # Ordenar y crear playlist en Spotify:
  python spotlight_mix.py --csv playlist.csv --playlist-id 37i9dQZF1... \\
    --client-id ID --client-secret SECRET

  # Con pesos personalizados:
  python spotlight_mix.py --csv playlist.csv --output-csv out.csv \\
    --weights bpm=0.40 key=0.20 energy=0.15 valence=0.10 dance=0.10 acoustic=0.05
        """,
    )

    # Entrada
    parser.add_argument(
        "--csv", required=True,
        help="Ruta al CSV con los tags de la playlist",
    )

    # Salida
    parser.add_argument(
        "--output-csv",
        help="Volcar el orden resultante a un CSV (no requiere Spotify)",
    )

    # Spotify (opcional — si no se pasa, solo ordena)
    parser.add_argument(
        "--playlist-id",
        help="ID de la playlist original de Spotify (para leer metadata)",
    )
    parser.add_argument(
        "--create-new",
        action="store_true",
        help="Crear una nueva playlist en Spotify en lugar de reemplazar",
    )
    parser.add_argument(
        "--new-playlist-name",
        default="{original} [Spotlight Mix]",
        help="Nombre de la nueva playlist (default: '{original} [Spotlight Mix]')",
    )
    parser.add_argument(
        "--client-id",
        help="Spotify Client ID (from developer.spotify.com/dashboard)",
    )
    parser.add_argument(
        "--client-secret",
        help="Spotify Client Secret",
    )
    parser.add_argument(
        "--redirect-uri",
        default="http://127.0.0.1:8080",
        help="Redirect URI OAuth (default: http://127.0.0.1:8080)",
    )

    # Algoritmo
    parser.add_argument(
        "--weights",
        help="Pesos personalizados: 'bpm=0.40 key=0.20 energy=0.15 ...'",
    )
    parser.add_argument(
        "--bpm-tolerance",
        type=float, default=4.0,
        help="Tolerancia BPM ±X%% (default: 4.0, Automix no pitch-shiftea)",
    )
    parser.add_argument(
        "--seed-track",
        type=int, default=None,
        help="Índice de la pista inicial (default: menor BPM)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Desactivar deduplicación",
    )
    parser.add_argument(
        "--no-2opt",
        action="store_true",
        help="Desactivar 2-opt (solo greedy nearest-neighbor)",
    )
    parser.add_argument(
        "--use-held-karp",
        action="store_true",
        help="Usar Held-Karp (solo ≤20 tracks, TSP exacto)",
    )
    parser.add_argument(
        "--flag-review",
        action="store_true",
        help="Marcar pistas live/acústicas para revisión manual",
    )

    # Output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Logging detallado (DEBUG)",
    )

    return parser.parse_args()


def find_seed_track(df: pd.DataFrame) -> int:
    """
    Encuentra la pista inicial óptima: la de menor BPM.
    Esto crea una apertura suave que sube gradualmente.
    """
    if "bpm" not in df.columns or df["bpm"].isna().all():
        return 0

    bpm = df["bpm"].fillna(999999)
    return int(bpm.idxmin())


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("spotlight_mix")

    # ────────────────────────────────────────────────────────────────
    # 1. Cargar CSV
    # ────────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Spotlight Mix — Optimizador de Playlists")
    logger.info("═" * 60)

    df = load_playlist_csv(args.csv)
    logger.info("CSV cargado: %d pistas", len(df))

    # ────────────────────────────────────────────────────────────────
    # 2. Deduplicación
    # ────────────────────────────────────────────────────────────────
    if not args.no_dedup:
        logger.info("─" * 40)
        logger.info("Fase 2: Deduplicación")
        df, removed = deduplicate(df)
        if removed:
            logger.info("Pistas eliminadas:")
            for r in removed[:10]:  # Mostrar máximo 10
                logger.info("  ❌ %s — %s (%s)", r["name"], r["artist"], r["reason"])
            if len(removed) > 10:
                logger.info("  ... y %d más", len(removed) - 10)
    else:
        logger.info("Deduplicación desactivada (--no-dedup)")

    # ────────────────────────────────────────────────────────────────
    # 3. Flag live/acústicas
    # ────────────────────────────────────────────────────────────────
    if args.flag_review:
        df = flag_live_acoustic(df)
        flagged = df[df["flag_review"] != ""]
        if not flagged.empty:
            logger.info("Pistas marcadas para revisión:")
            for _, row in flagged.iterrows():
                logger.info("  ⚠️  %s — %s [%s]", row.get("name", "?"), row.get("artist", "?"), row["flag_review"])

    # ────────────────────────────────────────────────────────────────
    # 4. Construir matriz de distancias
    # ────────────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Fase 4: Matriz de distancias")

    weights = parse_weights(args.weights)
    dist_matrix = build_distance_matrix(
        df,
        weights=weights,
        bpm_tolerance_pct=args.bpm_tolerance,
    )

    # ────────────────────────────────────────────────────────────────
    # 5. Secuenciación (greedy + 2-opt o Held-Karp)
    # ────────────────────────────────────────────────────────────────
    logger.info("─" * 40)
    logger.info("Fase 5: Secuenciación")

    # Elegir pista inicial
    if args.seed_track is not None:
        start_idx = args.seed_track
        logger.info("Pista inicial: índice %d (manual)", start_idx)
    else:
        start_idx = find_seed_track(df)
        bpm_val = df.iloc[start_idx].get("bpm", "?")
        name_val = df.iloc[start_idx].get("name", "?")
        logger.info("Pista inicial: '%s' (BPM=%s, índice %d)", name_val, bpm_val, start_idx)

    route = sequence(
        dist_matrix,
        start_idx=start_idx,
        use_2opt=not args.no_2opt,
        use_held_karp=args.use_held_karp,
    )

    cost = route_cost(dist_matrix, route)
    logger.info("Ruta final: %d pistas, coste total=%.4f", len(route), cost)

    # ────────────────────────────────────────────────────────────────
    # 6. Aplicar orden al DataFrame
    # ────────────────────────────────────────────────────────────────
    df_ordered = df.iloc[route].copy().reset_index(drop=True)
    df_ordered["new_position"] = range(1, len(df_ordered) + 1)

    # ────────────────────────────────────────────────────────────────
    # 7. Volcar CSV (opcional)
    # ────────────────────────────────────────────────────────────────
    if args.output_csv:
        df_ordered.to_csv(args.output_csv, index=False, encoding="utf-8")
        logger.info("CSV ordenado guardado: %s", args.output_csv)

    # ────────────────────────────────────────────────────────────────
    # 8. Escribir a Spotify (opcional)
    # ────────────────────────────────────────────────────────────────
    if args.client_id and args.client_secret:
        logger.info("─" * 40)
        logger.info("Fase 8: Escribir a Spotify")

        from spotlight_mix.spotify_client import SpotifyClient

        client = SpotifyClient(
            client_id=args.client_id,
            client_secret=args.client_secret,
            redirect_uri=args.redirect_uri,
        )

        # Obtener URIs de las pistas ordenadas
        if "spotify_id" in df_ordered.columns:
            track_ids = df_ordered["spotify_id"].dropna().tolist()
            track_uris = [f"spotify:track:{tid}" for tid in track_ids if tid and tid != "nan"]
        else:
            logger.error("No hay columna 'spotify_id' en el CSV. No se puede escribir a Spotify.")
            sys.exit(1)

        logger.info("URIs a añadir: %d", len(track_uris))

        if args.create_new or not args.playlist_id:
            # Crear nueva playlist
            original_name = "Playlist"
            if args.playlist_id:
                # Intentar obtener el nombre original
                try:
                    pl = client._playlist_items(args.playlist_id, limit=1)
                    # El nombre no viene en items, necesitamos playlist details
                except Exception:
                    pass

            name = args.new_playlist_name.replace("{original}", original_name)
            logger.info("Creando nueva playlist: '%s'", name)

            description = (
                f"Reordenada por Spotlight Mix | "
                f"Pesos: {weights or 'default'} | "
                f"BPM tolerance: ±{args.bpm_tolerance}% | "
                f"Pistas: {len(track_uris)}"
            )

            new_pl = client.create_playlist(name=name, description=description, public=False)
            new_pl_id = new_pl["id"]
            logger.info("Playlist creada: ID=%s", new_pl_id)

            snapshot = client.add_tracks_to_playlist(new_pl_id, track_uris)
            logger.info("✅ Playlist creada con %d pistas (snapshot=%s)", len(track_uris), snapshot[:12] + "...")
        else:
            # Reemplazar playlist existente
            logger.info("Reemplazando playlist existente: %s", args.playlist_id)
            client.replace_playlist_tracks(args.playlist_id, track_uris)
            logger.info("✅ Playlist actualizada con %d pistas", len(track_uris))

    elif not args.output_csv:
        logger.warning("No se especificó --output-csv ni credenciales de Spotify.")
        logger.warning("El orden se calculó pero no se guardó en ningún sitio.")
        logger.info("Primeras 10 pistas del orden:")
        for i, (_, row) in enumerate(df_ordered.head(10).iterrows()):
            logger.info("  %3d. %s — %s (BPM=%s, Key=%s)",
                        i + 1,
                        row.get("name", "?"),
                        row.get("artist", "?"),
                        row.get("bpm", "?"),
                        row.get("camelot", "?"))

    logger.info("═" * 60)
    logger.info("✅ Spotlight Mix completado en %d pistas", len(df_ordered))
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
