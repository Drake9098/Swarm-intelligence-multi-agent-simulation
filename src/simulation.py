from environment import Environment
from agent import Scout, Collector, Relay, AgentState

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
            for a in active:
                for other in self.agents:
                    if other is not a:
                        a.communicate(other)
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