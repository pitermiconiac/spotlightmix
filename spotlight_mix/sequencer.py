"""Sequencer - Algoritmos de secuenciacion de playlist (TSP)."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def greedy_nearest_neighbor(
    distance_matrix: np.ndarray,
    seed_index: int | None = None,
    bpm_array: np.ndarray | None = None,
    start_from_lowest_bpm: bool = True,
) -> list[int]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    if seed_index is None:
        if start_from_lowest_bpm and bpm_array is not None:
            seed_index = int(np.argmin(bpm_array))
            logger.info("Semilla greedy: pista %d (BPM mas bajo)", seed_index)
        else:
            seed_index = 0
    else:
        logger.info("Semilla greedy: pista %d (manual)", seed_index)

    visited = np.zeros(n, dtype=bool)
    order = [seed_index]
    visited[seed_index] = True

    while len(order) < n:
        current = order[-1]
        dists = np.where(visited, np.inf, distance_matrix[current])
        next_idx = int(np.argmin(dists))
        order.append(next_idx)
        visited[next_idx] = True
        logger.debug("Greedy: %d -> %d (dist=%.4f)", current, next_idx, distance_matrix[current, next_idx])

    return order


def two_opt_improve(
    distance_matrix: np.ndarray,
    order: list[int],
    max_iterations: int = 100,
    min_improvement: float = 1e-6,
) -> list[int]:
    """
    2-opt local search optimizado.
    Usa delta de coste (solo los 2 edges que cambian) en lugar de
    recalcular el coste total O(n). Complejidad O(n^2) por iteracion.
    """
    n = len(order)
    if n <= 2:
        return list(order)

    best_order = list(order)
    best_cost = _route_cost(distance_matrix, best_order)
    dm = distance_matrix

    for iteration in range(max_iterations):
        improved = False

        for i in range(1, n - 1):
            a = best_order[i - 1]
            b = best_order[i]

            for j in range(i + 1, n):
                c = best_order[j]
                d = best_order[j + 1] if j + 1 < n else None

                # Coste antes: edge a->b + edge c->d (si d existe)
                cost_before = dm[a, b]
                if d is not None:
                    cost_before += dm[c, d]

                # Coste despues (invertir [i,j]): edge a->c + edge b->d
                cost_after = dm[a, c]
                if d is not None:
                    cost_after += dm[b, d]

                if cost_after < cost_before - min_improvement:
                    # Invertir el subsegmento [i, j] in-place
                    best_order[i:j + 1] = best_order[i:j + 1][::-1]
                    best_cost += (cost_after - cost_before)
                    improved = True
                    b = best_order[i]  # actualizar b tras la inversion

        if not improved:
            break

    initial_cost = _route_cost(distance_matrix, order)
    improvement = (initial_cost - best_cost) / initial_cost * 100 if initial_cost > 0 else 0
    logger.info("2-opt: %d iteraciones, cost=%.6f (%.2f%% mejora)", iteration + 1, best_cost, improvement)

    return best_order


def _route_cost(distance_matrix: np.ndarray, order: list[int]) -> float:
    if len(order) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(order) - 1):
        total += distance_matrix[order[i], order[i + 1]]
    return float(total)


def held_karp_exact(distance_matrix: np.ndarray, seed_index: int = 0) -> list[int]:
    n = distance_matrix.shape[0]
    assert n <= 20, f"Held-Karp solo para <=20 tracks (n={n})"

    INF = float("inf")
    memo = {}

    mask0 = 1 << seed_index
    memo[(mask0, seed_index)] = (0.0, [seed_index])

    for subset_size in range(1, n):
        for mask in range(1 << n):
            if bin(mask).count("1") != subset_size + 1:
                continue
            if not (mask & (1 << seed_index)):
                continue

            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if last == seed_index and mask != mask0:
                    continue

                best = None
                prev_mask = mask ^ (1 << last)

                for prev in range(n):
                    if prev == last or not (prev_mask & (1 << prev)):
                        continue
                    if (prev_mask, prev) not in memo:
                        continue

                    prev_cost, prev_path = memo[(prev_mask, prev)]
                    cost = prev_cost + distance_matrix[prev, last]

                    if best is None or cost < best[0]:
                        best = (cost, prev_path + [last])

                if best is not None:
                    memo[(mask, last)] = best

    full_mask = (1 << n) - 1
    best_final = None

    for last in range(n):
        if (full_mask, last) in memo:
            cost, path = memo[(full_mask, last)]
            if best_final is None or cost < best_final[0]:
                best_final = (cost, path)

    if best_final is None:
        logger.warning("Held-Karp fallo, fallback a greedy")
        return greedy_nearest_neighbor(distance_matrix, seed_index=seed_index)

    logger.info("Held-Karp: cost=%.6f (optimo exacto)", best_final[0])
    return best_final[1]


def sequence_playlist(
    distance_matrix: np.ndarray,
    bpm_array: np.ndarray | None = None,
    seed_index: int | None = None,
    use_two_opt: bool = True,
    two_opt_iterations: int = 100,
    use_held_karp: bool = False,
) -> list[int]:
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    if use_held_karp and n <= 20:
        logger.info("Usando Held-Karp (n=%d, TSP exacto)", n)
        return held_karp_exact(distance_matrix, seed_index=seed_index or 0)
    elif use_held_karp and n > 20:
        logger.info("Held-Karp solicitado pero n=%d > 20, usando greedy+2-opt", n)

    greedy_order = greedy_nearest_neighbor(
        distance_matrix,
        seed_index=seed_index,
        bpm_array=bpm_array,
        start_from_lowest_bpm=(bpm_array is not None),
    )
    initial_cost = _route_cost(distance_matrix, greedy_order)
    logger.info("Greedy: cost=%.6f (%d pistas)", initial_cost, n)

    if use_two_opt:
        refined_order = two_opt_improve(
            distance_matrix,
            greedy_order,
            max_iterations=two_opt_iterations,
        )
        return refined_order
    else:
        return greedy_order


def evaluate_sequence(distance_matrix: np.ndarray, order: list[int]) -> dict:
    cost = _route_cost(distance_matrix, order)
    n = len(order)
    n_consecutive = max(n - 1, 1)

    consecutive_dists = []
    for i in range(len(order) - 1):
        consecutive_dists.append(distance_matrix[order[i], order[i + 1]])

    consecutive_arr = np.array(consecutive_dists) if consecutive_dists else np.array([0.0])

    return {
        "total_cost": float(cost),
        "n_tracks": n,
        "n_transitions": n - 1,
        "mean_consecutive_distance": float(consecutive_arr.mean()),
        "max_consecutive_distance": float(consecutive_arr.max()),
        "min_consecutive_distance": float(consecutive_arr.min()),
        "std_consecutive_distance": float(consecutive_arr.std()),
        "normalized_cost": cost / n_consecutive if n_consecutive > 0 else 0.0,
    }


# ────────────────────────────────────────────────────────────────
# Aliases de compatibilidad (para CLI y tests)
# ────────────────────────────────────────────────────────────────

TWO_OPT_MAX_ITERATIONS = 100
HELD_KARP_MAX_TRACKS = 20


def route_cost(distance_matrix: np.ndarray, order: list[int]) -> float:
    """Alias publico de _route_cost."""
    return _route_cost(distance_matrix, order)


def two_opt(
    distance_matrix: np.ndarray,
    order: list[int],
    max_iterations: int = 100,
) -> list[int]:
    """Alias de two_opt_improve para compatibilidad con CLI y tests."""
    return two_opt_improve(distance_matrix, order, max_iterations=max_iterations)


def held_karp(distance_matrix: np.ndarray) -> tuple[list[int], float]:
    """Alias de held_karp_exact que devuelve (route, cost) para tests."""
    route = held_karp_exact(distance_matrix)
    cost = _route_cost(distance_matrix, route)
    return route, cost


def sequence(
    distance_matrix: np.ndarray,
    start_idx: int | None = None,
    use_2opt: bool = True,
    use_held_karp: bool = False,
) -> list[int]:
    """
    Funcion principal de secuenciacion (alias para CLI).
    Selecciona automaticamente el algoritmo:
      - Si use_held_karp=True y n <= 20: Held-Karp (exacto)
      - Si no: greedy + 2-opt
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    if use_held_karp and n <= HELD_KARP_MAX_TRACKS:
        return held_karp_exact(distance_matrix, seed_index=start_idx or 0)

    greedy_order = greedy_nearest_neighbor(distance_matrix, seed_index=start_idx)
    initial_cost = _route_cost(distance_matrix, greedy_order)
    logger.info("Greedy: cost=%.6f (%d pistas)", initial_cost, n)

    if use_2opt and n > 2:
        refined = two_opt_improve(distance_matrix, greedy_order)
        refined_cost = _route_cost(distance_matrix, refined)
        improvement = (initial_cost - refined_cost) / initial_cost * 100 if initial_cost > 0 else 0
        logger.info("2-opt: cost=%.6f (%.1f%% mejora)", refined_cost, improvement)
        return refined

    return greedy_order
