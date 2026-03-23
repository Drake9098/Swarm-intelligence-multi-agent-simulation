import numpy as np
from environment import Environment
from agent import Scout, Collector, Relay, AgentState, UNKNOWN


def _merge_component(agents: list) -> None:
    """Merge atomico dello stato epistemico di tutti gli agenti in una componente connessa.

    Opera in sequenza su:
      1. known_gone_objects  - unione
      2. local_map           - unione numpy (celle note sovrascrivono UNKNOWN)
      3. known_objects       - merge dizionario, filtrato per gone
      4. _unobserved         - intersezione (coerente con la local_map unita)
      5. peer_frontiers      - distribuzione di tutte le claimed_frontier del gruppo
      6. _relay_frontiers    - raccolta da Scout + propagazione ai non-Relay
      7. _yielded_objects    - pulizia gone + assegnazione esclusiva per N fetcher
    """
    # 1. known_gone_objects: unione di tutti i set
    all_gone: set = set()
    for a in agents:
        all_gone |= a.known_gone_objects

    # 2. local_map: unione - le celle note di qualsiasi agente sovrascrivono UNKNOWN
    combined_map = agents[0].local_map.copy()
    for a in agents[1:]:
        mask = a.local_map != UNKNOWN
        combined_map[mask] = a.local_map[mask]

    # 3. known_objects: merge dizionario, filtrato per gone
    merged_objects: dict = {}
    for a in agents:
        merged_objects.update(a.known_objects)
    merged_objects = {k: v for k, v in merged_objects.items() if k not in all_gone}

    # 4. _unobserved: intersezione - una cella esce dal set non appena uno del gruppo l'ha osservata
    combined_unobserved: set = agents[0]._unobserved.copy()
    for a in agents[1:]:
        combined_unobserved &= a._unobserved

    # 5. peer_frontiers: raccoglie tutte le claimed_frontier valide del gruppo
    all_claimed: set = set()
    for a in agents:
        if a.claimed_frontier and a.claimed_frontier in combined_unobserved:
            all_claimed.add(a.claimed_frontier)

    # 6. _relay_frontiers: ogni Relay raccoglie le claimed_frontier degli Scout della componente
    #    e accumula le frontiere da ritrasmettere ai non-Relay
    relay_frontiers_combined: set = set()
    for a in agents:
        if getattr(a, 'role', '') == 'relay':
            for other in agents:
                if getattr(other, 'claimed_frontier', None) is not None:
                    a._relay_frontiers.add(other.claimed_frontier)
            relay_frontiers_combined |= a._relay_frontiers

    # 7. Assegnazione dello stato unificato
    for a in agents:
        a.local_map[:]        = combined_map
        a.known_gone_objects  = set(all_gone)
        a.known_objects       = dict(merged_objects)
        a._unobserved         = set(combined_unobserved)
        a._yielded_objects   -= all_gone

        # peer_frontiers: filtra le existenti + aggiungi le nuove (esclusa la propria)
        a.peer_frontiers = {f for f in a.peer_frontiers if f in combined_unobserved}
        for f in all_claimed:
            if f != a.claimed_frontier and f in combined_unobserved:
                a.peer_frontiers.add(f)
        # propaga relay_frontiers ai non-Relay
        if getattr(a, 'role', '') != 'relay':
            for f in relay_frontiers_combined:
                if f in combined_unobserved:
                    a.peer_frontiers.add(f)

    # pulizia _relay_frontiers: rimuove frontiere ormai osservate
    for a in agents:
        if getattr(a, 'role', '') == 'relay':
            a._relay_frontiers = {f for f in a._relay_frontiers if f in a._unobserved}

    # 8. _yielded_objects: assegnazione esclusiva con N fetcher
    #    vince il più vicino all'oggetto, tie-break sull'agent_id (deterministico)
    fetchers = [a for a in agents if a.can_fetch]
    if len(fetchers) >= 2:
        for obj_id, pos in merged_objects.items():
            ordered = sorted(
                fetchers,
                key=lambda a: (abs(a.r - pos[0]) + abs(a.c - pos[1]), a.agent_id),
            )
            ordered[0]._yielded_objects.discard(obj_id)
            for a in ordered[1:]:
                a._yielded_objects.add(obj_id)


def _mesh_communicate(agents: list) -> None:
    """Comunicazione a rete mesh per un tick.

    Sostituisce il doppio ciclo annidato agent.communicate(other):
      Fase 1 - costruisce il grafo di adiacenza con condizione OR sui raggi
               (arco A-B se dist <= max(A.comm_radius, B.comm_radius))
      Fase 2 - trova le componenti connesse tramite BFS
      Fase 3 - per ogni componente con >= 2 agenti esegue _merge_component
    """
    n = len(agents)
    if n < 2:
        return

    # Fase 1: grafo di adiacenza - arco se almeno uno dei due raggi copre la distanza
    adj: dict[int, set] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            dist = abs(a.r - b.r) + abs(a.c - b.c)
            if dist <= max(a.comm_radius, b.comm_radius):
                adj[i].add(j)
                adj[j].add(i)

    # Fase 2: BFS per trovare le componenti connesse
    visited: set = set()
    components: list = []
    for start in range(n):
        if start not in visited:
            component_idx: list = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component_idx.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append([agents[i] for i in component_idx])

    # Fase 3: merge atomico per ogni componente con >= 2 agenti
    for component in components:
        if len(component) >= 2:
            _merge_component(component)


class Simulation:
    def __init__(self, json_path: str, max_ticks: int = 500, config: str = "with_relay"):
        self.env = Environment(json_path)
        g = self.env.grid
        if config == "exploration":
            self.agents = [
                Scout(agent_id=0, quadrant=0, grid=g),
                Scout(agent_id=1, quadrant=3, grid=g),
                Scout(agent_id=2, quadrant=2, grid=g),
                Collector(agent_id=3, quadrant=1, strategy="east", grid=g),
                Collector(agent_id=4, quadrant=2, strategy="west", grid=g),
            ]
        elif config == "collection":
            self.agents = [
                Scout(agent_id=0, quadrant=0, grid=g),
                Scout(agent_id=1, quadrant=3, grid=g),
                Collector(agent_id=2, quadrant=1, strategy="east", grid=g),
                Collector(agent_id=3, quadrant=2, strategy="west", grid=g),
                Collector(agent_id=4, quadrant=0, strategy="east", grid=g),
            ]
        else:  # "with_relay"
            self.agents = [
                Scout(agent_id=0, quadrant=0, grid=g),
                Scout(agent_id=1, quadrant=3, grid=g),
                Relay(agent_id=2, grid=g),
                Collector(agent_id=3, quadrant=1, strategy="east", grid=g),
                Collector(agent_id=4, quadrant=2, strategy="west", grid=g),
            ]
        self.max_ticks = max_ticks
        self.tick = 0
        self.log = []
    

    def _all_delivered(self) -> bool:
        """Controlla se tutti gli oggetti sono stati consegnati."""
        return self.env.all_delivered()
    

    def _build_snapshot(self, tick: int):
        """Costruisce un'istantanea dello stato attuale della simulazione."""
        snapshot = {
            "tick": tick,
            "agents": [
                {
                    "id": a.agent_id,
                    "role": a.role,
                    "pos": [a.r, a.c],
                    "battery": a.battery,
                    "carrying": a.carrying,
                    "state": a.state.name
            }
            for a in self.agents
        ],
            "objects_delivered": 10 - self.env.objects_remaining(),
            "objects_remaining": self.env.objects_remaining()
        }
        return snapshot


    def run(self):
        """Esegue la simulazione fino al completamento o al raggiungimento del limite di tick."""
        while self.tick < self.max_ticks and not self._all_delivered():
            active = [a for a in self.agents if a.state != AgentState.DEAD]
            if not active:
                break
            for a in active:
                a.observe(self.env)
            _mesh_communicate(active)
            for a in active:
                if isinstance(a, Relay):
                    a.decide(self.env, agents=self.agents)
                else:
                    a.decide(self.env)
            for a in active:
                a.act(self.env)
            self.log.append(self._build_snapshot(self.tick))
            self.tick += 1

        return self.log