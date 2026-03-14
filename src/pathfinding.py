import heapq
from typing import Callable


def _heuristic(r1: int, c1: int, r2: int, c2: int) -> int:
    """Calcola la distanza di Manhattan tra due punti."""
    return abs(r1 - r2) + abs(c1 - c2)


def a_star(
    start: tuple[int, int],
    goal: tuple[int, int],
    is_walkable_fn: Callable[[int, int, int, int], bool]
) -> list[tuple[int, int]] | None:
    """Implementazione dell'algoritmo A* per il pathfinding."""
    open_set = []
    heapq.heappush(open_set, (0 + _heuristic(*start, *goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not is_walkable_fn(neighbor[0], neighbor[1], current[0], current[1]):
                continue

            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + _heuristic(*neighbor, *goal)
                heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                came_from[neighbor] = current

    return None    