import numpy as np
from enum import Enum, auto
from pathfinding import a_star
from environment import EMPTY, WALL, WAREHOUSE, ENTRANCE, EXIT


UNKNOWN = -1


class AgentState(Enum):
    EXPLORE   = auto()
    FETCH     = auto()
    DELIVER   = auto()
    EMERGENCY = auto()
    DEAD      = auto()


class Agent:
    # Target per lo scatter iniziale (indice = quadrant int)
    _SCATTER_TARGETS: dict[int, tuple[int, int]] = {
        0: (0,  0),
        1: (0,  23),
        2: (23,  0),
        3: (23, 23),
    }

    def __init__(self, agent_id: int, vision_radius: int, comm_radius: int, quadrant: int, grid=None):
        self.agent_id = agent_id
        self.r = self.c = 0
        self.battery = 500
        self.vision_radius = vision_radius
        self.comm_radius = comm_radius
        self.quadrant = quadrant
        if grid is not None:
            self.local_map = np.array(grid, dtype=int)
            self._unobserved: set[tuple] = {
                (r, c)
                for r in range(len(grid))
                for c in range(len(grid[0]))
                if grid[r][c] != WALL
            }
        else:
            self.local_map = np.full((25, 25), UNKNOWN, dtype=int)
            self._unobserved: set[tuple] = set()
        self.known_objects = {}
        self.carrying = None
        self.state = AgentState.EXPLORE
        self._path = []
        self.claimed_frontier = None
        self.peer_frontiers: set[tuple] = set()
        self.steps_taken = 0
        self.known_gone_objects: set[int] = set()
        self._scatter_done = False
        self._fetch_target: tuple | None = None
        self._cached_return_cost: int = 0
        self._last_cost_pos: tuple | None = None
        self.can_fetch: bool = True
        self._yielded_objects: set[int] = set()

    @property
    def explore_strategy(self) -> str | None:
        return None

    # ------------------------------------------------------------------
    # Percezione
    # ------------------------------------------------------------------

    def _has_line_of_sight(self, env, tr: int, tc: int) -> bool:
        dr, dc = tr - self.r, tc - self.c
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return True
        for i in range(1, steps):
            if env.grid[round(self.r + dr * i / steps)][round(self.c + dc * i / steps)] == WALL:
                return False
        return True

    def observe(self, env):
        for nr in range(self.r - self.vision_radius, self.r + self.vision_radius + 1):
            for nc in range(self.c - self.vision_radius, self.c + self.vision_radius + 1):
                if not env.in_bound(nr, nc):
                    continue
                if abs(nr - self.r) + abs(nc - self.c) > self.vision_radius:
                    continue
                if not self._has_line_of_sight(env, nr, nc):
                    continue
                self.local_map[nr][nc] = env.grid[nr][nc]
                self._unobserved.discard((nr, nc))
                obj = env.reveal_object_at(nr, nc)
                if obj is not None and obj["id"] not in self.known_objects:
                    self.known_objects[obj["id"]] = obj["pos"]

    # ------------------------------------------------------------------
    # Comunicazione
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pathfinding helpers
    # ------------------------------------------------------------------

    EMERGENCY_MARGIN = 10
    _FETCH_UNKNOWN_THRESHOLD = 0.4  # soglia: >40% celle UNKNOWN → esplorazione guidata

    def _local_is_walkable(self, env, r: int, c: int, from_r: int, from_c: int) -> bool:
        if not env.in_bound(r, c):
            return False
        if self.local_map[r][c] == WALL:
            return False
        # Free Space Assumption: cella sconosciuta → presunta libera
        if self.local_map[r][c] == UNKNOWN:
            return True
        # Cella nota: rispetta i vincoli direzionali (ENTRANCE/EXIT)
        return env.is_walkable(r, c, from_r, from_c)

    def _astar(self, env, goal: tuple) -> list:
        """A* basato esclusivamente sulla mappa locale dell'agente."""
        return a_star(start=(self.r, self.c), goal=goal,
                      is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc)) or []

    def _known_warehouse_entrances(self, env) -> list:
        return [w for w in env.get_warehouse_entrances()
                if self.local_map[w["entrance"][0]][w["entrance"][1]] != UNKNOWN]

    def _cost_to_nearest_entrance(self, env) -> int:
        moved = (
            self._last_cost_pos is None
            or abs(self.r - self._last_cost_pos[0]) + abs(self.c - self._last_cost_pos[1]) >= 5
        )
        near_threshold = self.battery <= self._cached_return_cost + self.EMERGENCY_MARGIN + 15
        if not moved and not near_threshold:
            return self._cached_return_cost
        warehouses = self._known_warehouse_entrances(env) or env.get_warehouse_entrances()
        best = min(abs(self.r - w["entrance"][0]) + abs(self.c - w["entrance"][1])
                   for w in warehouses)
        self._cached_return_cost = int(best * 1.5)
        self._last_cost_pos = (self.r, self.c)
        return self._cached_return_cost

    def _path_to_nearest_entrance(self, env) -> list:
        warehouses = self._known_warehouse_entrances(env) or env.get_warehouse_entrances()
        best_path = None
        for w in warehouses:
            path = a_star(start=(self.r, self.c), goal=tuple(w["entrance"]),
                          is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc))
            if path is not None and (best_path is None or len(path) < len(best_path)):
                best_path = path
        return best_path or []

    @staticmethod
    def _inner_warehouse_cell(w) -> tuple:
        er, ec = w["entrance"]
        offsets = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}
        dr, dc = offsets[w["side"]]
        return (er + dr, ec + dc)

    def _path_into_nearest_warehouse(self, env) -> list:
        best_path = None
        for w in env.get_warehouse_entrances():
            er, ec = w["entrance"]
            if self.local_map[er][ec] == UNKNOWN:
                continue
            inner = self._inner_warehouse_cell(w)
            path = a_star(start=(self.r, self.c), goal=inner,
                          is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc))
            if path is None:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
        return best_path or []

    def _nearest_frontier(self) -> tuple | None:
        """Cella non ancora ispezionata più vicina."""
        if self._unobserved:
            return min(self._unobserved,
                       key=lambda cell: abs(cell[0] - self.r) + abs(cell[1] - self.c))
        # Fallback per mappa non pre-caricata: cerca adiacenti a celle UNKNOWN
        best, best_dist = None, float('inf')
        rows, cols = np.where(self.local_map != UNKNOWN)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if self.local_map[r][c] != EMPTY:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 25 and 0 <= nc < 25 and self.local_map[nr][nc] == UNKNOWN:
                    d = abs(r - self.r) + abs(c - self.c)
                    if d < best_dist:
                        best_dist, best = d, (r, c)
                    break
        return best

    def _repulsion_target(self) -> tuple | None:
        """Cella non ispezionata che massimizza la distanza dai peer (repulsione)."""
        if not self._unobserved:
            return None
        active_peers = {f for f in self.peer_frontiers if f in self._unobserved}
        best, best_score = None, -1.0
        for cell in self._unobserved:
            d_self = abs(cell[0] - self.r) + abs(cell[1] - self.c)
            d_peers = min(
                (abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in active_peers),
                default=1.0,
            )
            score = d_peers / (1.0 + d_self)
            if score > best_score:
                best_score, best = score, cell
        return best

    def _unknown_ratio(self, path: list) -> float:
        """Frazione di celle UNKNOWN nel percorso pianificato."""
        if not path:
            return 0.0
        return sum(1 for r, c in path if self.local_map[r][c] == UNKNOWN) / len(path)

    def _is_path_valid(self) -> bool:
        """Restituisce False se almeno una cella del percorso cached è stata rivelata come muro."""
        return all(self.local_map[r][c] != WALL for r, c in self._path)

    def _frontier_toward(self, target: tuple) -> tuple | None:
        """Frontiera nota più vicina al target (waypoint per esplorazione guidata verso FETCH)."""
        best, best_dist = None, float('inf')
        g = self.local_map.shape[0]
        rows, cols = np.where(self.local_map != UNKNOWN)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if self.local_map[r][c] != EMPTY:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g and 0 <= nc < g and self.local_map[nr][nc] == UNKNOWN:
                    d = abs(r - target[0]) + abs(c - target[1])
                    if d < best_dist:
                        best_dist, best = d, (r, c)
                    break
        return best

    def _explore_target(self) -> tuple | None:
        return self._nearest_frontier()

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------

    def decide(self, env) -> bool:
        if self.battery <= 0:
            self.state = AgentState.DEAD
            return True

        if self._path and not self._is_path_valid():
            self._path = []

        return_cost = self._cost_to_nearest_entrance(env)
        if self.battery <= return_cost + self.EMERGENCY_MARGIN:
            if self.state != AgentState.EMERGENCY:
                self.state = AgentState.EMERGENCY
                self._path = []
            if not self._path:
                self._path = self._path_to_nearest_entrance(env)
            return True

        if self.state == AgentState.DELIVER and self.carrying is not None:
            if not self._path:
                self._path = self._path_into_nearest_warehouse(env)
                if not self._path:
                    # Magazzino non ancora scoperto: esplora per trovarlo
                    frontier = self._nearest_frontier()
                    if frontier:
                        self._path = self._astar(env, frontier)
            return False

        if self.carrying is not None:
            self.state = AgentState.DELIVER
            self._path = self._path_into_nearest_warehouse(env)
            if not self._path:
                # Magazzino non ancora scoperto: esplora per trovarlo
                frontier = self._nearest_frontier()
                if frontier:
                    self._path = self._astar(env, frontier)
            return False

        if self.can_fetch:
            if self.state == AgentState.FETCH and self._path:
                if self._fetch_target is not None and self._fetch_target in self.known_objects.values():
                    # Interrompi se nel frattempo un peer più vicino ha preso il target
                    target_id = next(
                        (k for k, v in self.known_objects.items() if v == self._fetch_target), None
                    )
                    if target_id is not None and target_id in self._yielded_objects:
                        self._path = []
                        self._fetch_target = None
                    else:
                        return False
                else:
                    self._path = []
                    self._fetch_target = None
            if not self._path and self.known_objects:
                best_path = None
                best_pos = None
                candidates = sorted(
                    [(k, v) for k, v in self.known_objects.items() if k not in self._yielded_objects],
                    key=lambda kv: abs(kv[1][0] - self.r) + abs(kv[1][1] - self.c),
                )[:3]
                for _, pos in candidates:
                    path = self._astar(env, pos)
                    if path:
                        if self._unknown_ratio(path) > self._FETCH_UNKNOWN_THRESHOLD:
                            guided = self._frontier_toward(pos)
                            if guided is not None:
                                guided_path = self._astar(env, guided)
                                if guided_path:
                                    path = guided_path
                        if best_path is None or len(path) < len(best_path):
                            best_path = path
                            best_pos = pos
                if best_path is not None:
                    self.state = AgentState.FETCH
                    self._path = best_path
                    self._fetch_target = best_pos
                    return False

        self.state = AgentState.EXPLORE
        return False

    def _move(self, env):
        if not self._path:
            self.decide(env)
            if not self._path:
                return
        next_r, next_c = self._path[0]
        if env.is_walkable(next_r, next_c, self.r, self.c):
            self._path.pop(0)
            self.r, self.c = next_r, next_c
            self.battery -= 1
            self.steps_taken += 1
        else:
            self._path = []
            self.decide(env)

    def act(self, env):
        if self.state == AgentState.DEAD:
            return

        if self.state in (AgentState.EMERGENCY, AgentState.FETCH, AgentState.EXPLORE):
            self._move(env)
            if self.state == AgentState.FETCH:
                for obj_id, pos in list(self.known_objects.items()):
                    if pos == (self.r, self.c):
                        if env.claim_object(obj_id):
                            self.carrying = obj_id
                            del self.known_objects[obj_id]
                        else:
                            del self.known_objects[obj_id]
                            self.known_gone_objects.add(obj_id)
                        self._path = []
                        break

        elif self.state == AgentState.DELIVER:
            self._move(env)
            if env.grid[self.r][self.c] == WAREHOUSE and self.carrying is not None:
                env.deliver_object(self.carrying)
                self.known_gone_objects.add(self.carrying)
                self.carrying = None
                self._path = []




class Scout(Agent):
    def __init__(self, agent_id: int, quadrant: int, grid=None):
        super().__init__(agent_id, vision_radius=3, comm_radius=2, quadrant=quadrant, grid=grid)
        self.role = "scout"
        self.can_fetch = False

    @property
    def explore_strategy(self) -> str:
        return "repulsion"

    def _explore_target(self) -> tuple | None:
        return self._repulsion_target()

    def decide(self, env):
        if super().decide(env):
            return

        if not self._path:
            if not self._scatter_done:
                if self.quadrant is not None:
                    target = self._SCATTER_TARGETS[self.quadrant]
                    if abs(self.r - target[0]) + abs(self.c - target[1]) <= 3:
                        self._scatter_done = True
                    else:
                        self._path = self._astar(env, target)
                        if not self._path:
                            self._scatter_done = True
                else:
                    self._scatter_done = True
            if not self._path:
                best = self._explore_target()
                if best is not None:
                    self.claimed_frontier = best
                    self._path = self._astar(env, best)



class Collector(Agent):
    EMERGENCY_MARGIN = 20

    def __init__(self, agent_id: int, quadrant: int, strategy: str = "nearest", grid=None):
        super().__init__(agent_id, vision_radius=2, comm_radius=1, quadrant=quadrant, grid=grid)
        self.role = "collector"
        self._explore_strategy = strategy
        self._garrison_pos: tuple | None = None
        self._explore_burst: int = 0

    @property
    def explore_strategy(self) -> str:
        return self._explore_strategy

    def _find_garrison_pos(self, env) -> tuple | None:
        """Cella di attesa appena fuori dall'ingresso del magazzino noto più vicino."""
        best, best_dist = None, float('inf')
        side_offsets = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}
        for w in env.get_warehouse_entrances():
            er, ec = w["entrance"]
            if self.local_map[er][ec] == UNKNOWN:
                continue
            dr, dc = side_offsets[w["side"]]
            wait = (er + dr, ec + dc)
            d = abs(wait[0] - self.r) + abs(wait[1] - self.c)
            if d < best_dist:
                best_dist, best = d, wait
        return best

    def decide(self, env):
        # 1. Filtra gli oggetti: consideriamo solo quelli non ceduti ad altri peer
        available_objs = [k for k in self.known_objects if k not in self._yielded_objects]

        # 2. Opportunistic-fetch: se ci sono oggetti validi e stiamo viaggiando,
        # interrompiamo subito la rotta (anche se lo scatter iniziale non è completato)
        if (available_objs and self.can_fetch
                and self.state == AgentState.EXPLORE
                and self._path):
            self._path = []

        if super().decide(env):
            return

        # Fase 1: raggiungi la zona iniziale (Est o Ovest) prima di presidiare
        if not self._scatter_done:
            g = self.local_map.shape[0]
            initial_targets = {
                "east": (g // 2, g - 3),
                "west": (g // 2, 2),
            }
            target = initial_targets.get(self._explore_strategy)
            if target is None:
                self._scatter_done = True
            elif abs(self.r - target[0]) + abs(self.c - target[1]) <= 4:
                self._scatter_done = True
            else:
                if not self._path:
                    self._path = self._astar(env, target)
                    if not self._path:
                        self._scatter_done = True
                if not self._scatter_done:
                    return

        # Fase 2: presidio del magazzino più vicino già scoperto
        if self._path:
            return
        if self._explore_burst > 0:
            self._explore_burst -= 1
            frontier = self._nearest_frontier()
            if frontier:
                self._path = self._astar(env, frontier)
            return
        garrison_target = self._find_garrison_pos(env)
        if garrison_target is not None:
            if self._garrison_pos is None:
                self._garrison_pos = garrison_target
            if abs(self.r - self._garrison_pos[0]) + abs(self.c - self._garrison_pos[1]) <= 1:
                # Arrivato al presidio senza oggetti: esplora in modalità burst
                self._garrison_pos = None
                self._explore_burst = 5
                frontier = self._nearest_frontier()
                if frontier:
                    self._path = self._astar(env, frontier)
                return
            if not self._path:
                self._path = self._astar(env, self._garrison_pos)
            return

        # Fallback: esplora per scoprire magazzini
        if not self._path:
            frontier = self._nearest_frontier()
            if frontier:
                self._path = self._astar(env, frontier)



class Relay(Agent):
    """Agente ponte: si posiziona dinamicamente nel punto medio tra Scout e Collector
    per minimizzare i ritardi di comunicazione, poi si dirige verso i magazzini noti
    per trasferire le info raccolte ai Collector.

    Regola 1 - conosce oggetti da condividere:
        → vai verso l'ingresso del magazzino noto più vicino.
    Regola 2 - nessun oggetto noto:
        → raggiunge il baricentro tra posizione media degli Scout e dei Collector attivi,
          ricalcolando il target solo quando il centroide si sposta di più di 4 celle
          (anti-oscillazione). Fallback al pattugliamento a diamante se agents=None.
    """
    EMERGENCY_MARGIN = 5

    def __init__(self, agent_id: int, grid=None):
        super().__init__(agent_id, vision_radius=2, comm_radius=2, quadrant=None, grid=grid)
        self.role = "relay"
        self._patrol_idx: int = 0
        self.can_fetch = False
        self._relay_frontiers: set[tuple] = set()
        self._bridge_target: tuple | None = None

    @property
    def explore_strategy(self) -> str:
        """Short UI label (full logic in decide / class docstring)."""
        return "Midpoint"

    def _compute_bridge_target(self, agents: list) -> tuple | None:
        """Calcola il punto medio (baricentro) tra la posizione media degli Scout attivi
        e quella media dei Collector attivi. Restituisce None se uno dei due gruppi è vuoto."""
        scouts     = [a for a in agents if getattr(a, 'role', '') == 'scout'     and a.state != AgentState.DEAD]
        collectors = [a for a in agents if getattr(a, 'role', '') == 'collector' and a.state != AgentState.DEAD]
        if not scouts or not collectors:
            return None
        sc_r = sum(a.r for a in scouts)     / len(scouts)
        sc_c = sum(a.c for a in scouts)     / len(scouts)
        co_r = sum(a.r for a in collectors) / len(collectors)
        co_c = sum(a.c for a in collectors) / len(collectors)
        return (int((sc_r + co_r) / 2), int((sc_c + co_c) / 2))

    def decide(self, env, agents: list = None):
        if super().decide(env):
            return

        if self._path:
            return

        # Regola 1: ho oggetti → avvicinati all'ingresso del magazzino più vicino noto
        if self.known_objects:
            warehouses = self._known_warehouse_entrances(env)
            if warehouses:
                target = min(warehouses,
                             key=lambda w: abs(w["entrance"][0] - self.r) + abs(w["entrance"][1] - self.c))
                self._path = a_star(
                    start=(self.r, self.c), goal=tuple(target["entrance"]),
                    is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc)
                ) or []
                if self._path:
                    return

        # Regola 2a: posizionamento dinamico come ponte Scout↔Collector.
        # Il target viene ricalcolato solo quando il baricentro si sposta di più di 4 celle
        # per evitare oscillazioni tick-per-tick.
        if agents:
            bridge = self._compute_bridge_target(agents)
            if bridge is not None:
                if (self._bridge_target is None
                        or abs(bridge[0] - self._bridge_target[0])
                           + abs(bridge[1] - self._bridge_target[1]) > 4):
                    self._bridge_target = bridge
                if abs(self.r - self._bridge_target[0]) + abs(self.c - self._bridge_target[1]) > 2:
                    self._path = a_star(
                        start=(self.r, self.c), goal=self._bridge_target,
                        is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc)
                    ) or []
                # Se A* fallisce verso il bridge, esplora la frontiera più vicina
                if not self._path:
                    frontier = self._nearest_frontier()
                    if frontier:
                        self._path = self._astar(env, frontier)
                return

        # Regola 2b: fallback a pattugliamento a diamante (agents non disponibili)
        g = self.local_map.shape[0]
        m = g // 2
        q = g // 4
        patrol_waypoints = [(q, m), (m, m + q), (m + q, m), (m, q)]
        wp = patrol_waypoints[self._patrol_idx % len(patrol_waypoints)]
        if abs(self.r - wp[0]) + abs(self.c - wp[1]) <= 2:
            self._patrol_idx += 1
            wp = patrol_waypoints[self._patrol_idx % len(patrol_waypoints)]
        self._path = a_star(
            start=(self.r, self.c), goal=wp,
            is_walkable_fn=lambda r, c, fr, fc: self._local_is_walkable(env, r, c, fr, fc)
        ) or []
        if not self._path:
            frontier = self._nearest_frontier()
            if frontier:
                self._path = self._astar(env, frontier)
