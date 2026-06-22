"""
Tests unitarios del Sequencer
=============================
Valida: greedy nearest-neighbor, 2-opt improvement, Held-Karp exacto,
y la función sequence() principal.
"""
import pytest
import numpy as np

from spotlight_mix.sequencer import (
    greedy_nearest_neighbor,
    two_opt_improve,
    two_opt,
    held_karp_exact,
    held_karp,
    sequence_playlist,
    sequence,
    route_cost,
    evaluate_sequence,
    HELD_KARP_MAX_TRACKS,
)


def make_dist_matrix(n: int, seed: int = 42) -> np.ndarray:
    """Genera una matriz de distancias n*n simetrica con diagonal=0."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


def make_simple_matrix() -> np.ndarray:
    """Matriz 4x4 con distancias conocidas."""
    return np.array([
        [0, 1, 5, 2],
        [1, 0, 2, 3],
        [5, 2, 0, 1],
        [2, 3, 1, 0],
    ], dtype=float)


class TestGreedyNearestNeighbor:
    def test_single_element(self):
        dm = np.array([[0]])
        assert greedy_nearest_neighbor(dm, seed_index=0) == [0]

    def test_two_elements(self):
        dm = np.array([[0, 0.5], [0.5, 0]])
        assert greedy_nearest_neighbor(dm, seed_index=0) == [0, 1]

    def test_simple_matrix_from_0(self):
        dm = make_simple_matrix()
        route = greedy_nearest_neighbor(dm, seed_index=0)
        assert route == [0, 1, 2, 3]

    def test_simple_matrix_from_2(self):
        dm = make_simple_matrix()
        route = greedy_nearest_neighbor(dm, seed_index=2)
        assert route == [2, 3, 0, 1]

    def test_visits_all_nodes(self):
        dm = make_dist_matrix(20)
        route = greedy_nearest_neighbor(dm, seed_index=0)
        assert len(route) == 20
        assert len(set(route)) == 20

    def test_large_matrix(self):
        import time
        dm = make_dist_matrix(500)
        t0 = time.time()
        route = greedy_nearest_neighbor(dm, seed_index=0)
        elapsed = time.time() - t0
        assert len(route) == 500
        assert elapsed < 5.0


class TestTwoOpt:
    def test_no_improvement_needed(self):
        dm = make_simple_matrix()
        route = [0, 1, 2, 3]
        improved = two_opt(dm, route, max_iterations=10)
        assert len(improved) == 4
        assert route_cost(dm, improved) <= route_cost(dm, route)

    def test_improvement_found(self):
        dm = make_dist_matrix(20)
        bad_route = list(range(20))
        bad_cost = route_cost(dm, bad_route)
        improved = two_opt(dm, bad_route, max_iterations=50)
        assert route_cost(dm, improved) <= bad_cost

    def test_small_route_unchanged(self):
        dm = make_dist_matrix(3)
        route = [0, 1, 2]
        improved = two_opt(dm, route)
        assert improved == route

    def test_does_not_lose_nodes(self):
        dm = make_dist_matrix(50)
        route = greedy_nearest_neighbor(dm, seed_index=0)
        improved = two_opt(dm, route, max_iterations=20)
        assert len(improved) == 50
        assert len(set(improved)) == 50


class TestHeldKarp:
    def test_single_element(self):
        route, cost = held_karp(np.array([[0]]))
        assert route == [0]
        assert cost == 0.0

    def test_two_elements(self):
        dm = np.array([[0, 0.5], [0.5, 0]])
        route, cost = held_karp(dm)
        assert cost == 0.5

    def test_simple_matrix_optimal(self):
        dm = make_simple_matrix()
        route, cost = held_karp(dm)
        assert cost == 4.0

    def test_five_elements(self):
        dm = make_dist_matrix(5, seed=123)
        route, cost = held_karp(dm)
        assert len(route) == 5
        assert len(set(route)) == 5

    def test_max_tracks_limit(self):
        dm = make_dist_matrix(HELD_KARP_MAX_TRACKS + 1)
        with pytest.raises(AssertionError):
            held_karp_exact(dm)


class TestSequence:
    def test_auto_select_greedy(self):
        dm = make_dist_matrix(50)
        route = sequence(dm, use_2opt=True, use_held_karp=False)
        assert len(route) == 50
        assert len(set(route)) == 50

    def test_auto_select_held_karp(self):
        dm = make_dist_matrix(10)
        route = sequence(dm, use_held_karp=True)
        assert len(route) == 10
        assert len(set(route)) == 10

    def test_greedy_better_than_random(self):
        dm = make_dist_matrix(50)
        greedy_route = sequence(dm, use_2opt=False, use_held_karp=False)
        greedy_cost = route_cost(dm, greedy_route)
        random_route = list(range(50))
        random_cost = route_cost(dm, random_route)
        assert greedy_cost < random_cost

    def test_2opt_improves_greedy(self):
        dm = make_dist_matrix(50, seed=999)
        greedy_route = sequence(dm, use_2opt=False, use_held_karp=False)
        greedy_cost = route_cost(dm, greedy_route)
        improved_route = sequence(dm, use_2opt=True, use_held_karp=False)
        improved_cost = route_cost(dm, improved_route)
        assert improved_cost <= greedy_cost

    def test_held_karp_optimal_better_than_greedy(self):
        dm = make_dist_matrix(10, seed=42)
        greedy_route = sequence(dm, use_2opt=True, use_held_karp=False)
        greedy_cost = route_cost(dm, greedy_route)
        hk_route = sequence(dm, use_held_karp=True)
        hk_cost = route_cost(dm, hk_route)
        assert hk_cost <= greedy_cost + 1e-10

    def test_empty_matrix(self):
        route = sequence(np.array([]).reshape(0, 0))
        assert route == []

    def test_single_element(self):
        route = sequence(np.array([[0]]))
        assert route == [0]


class TestRouteCost:
    def test_simple_route(self):
        dm = make_simple_matrix()
        assert route_cost(dm, [0, 1, 2, 3]) == 4.0

    def test_reversed_route(self):
        dm = make_simple_matrix()
        assert route_cost(dm, [3, 2, 1, 0]) == 4.0

    def test_empty_route(self):
        dm = make_dist_matrix(5)
        assert route_cost(dm, []) == 0.0


class TestEvaluateSequence:
    def test_empty(self):
        result = evaluate_sequence(np.array([]).reshape(0, 0), [])
        assert result["n_tracks"] == 0

    def test_single(self):
        dm = np.array([[0.0]])
        result = evaluate_sequence(dm, [0])
        assert result["total_cost"] == 0.0
        assert result["n_tracks"] == 1

    def test_two_tracks(self):
        dm = np.array([[0.0, 0.5], [0.5, 0.0]])
        result = evaluate_sequence(dm, [0, 1])
        assert result["total_cost"] == 0.5
        assert result["n_transitions"] == 1
