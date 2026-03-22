import pygame
import numpy as np
from environment import EMPTY, WALL, ENTRANCE, EXIT, WAREHOUSE
from agent import Scout, Collector, Relay, AgentState
from simulation import Simulation, _mesh_communicate

import os
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# Dimensioni
CELL_SIZE = 24          # pixel per cella
PANEL_WIDTH = 220       # larghezza pannello statistiche a destra
FPS = 10              # tick visualizzati al secondo

# Colori (RGB) — allineati a visualize_environment.py
COLOR_EMPTY      = (255, 255, 255)
COLOR_WALL       = (64,  64,  64)
COLOR_WAREHOUSE  = (74,  144, 217)
COLOR_ENTRANCE   = (46,  204, 113)
COLOR_EXIT       = (231, 76,  60)
COLOR_UNKNOWN    = (180, 180, 180)
COLOR_BACKGROUND = (240, 240, 240)
COLOR_PANEL      = (30,  30,  30)
COLOR_GRID_LINE  = (200, 200, 200)

# Colori agenti per ruolo
COLOR_SCOUT      = (34, 139, 34)
COLOR_COLLECTOR  = (255, 140, 0)
COLOR_RELAY      = (160, 80,  220)
COLOR_DEAD       = (100, 100, 100)

# Colore oggetti — 
COLOR_OBJECT     = (255, 215, 0)


class Visualizer:
    def __init__(self, env, agents: list, max_ticks: int, show_ground_truth: bool = False):
        self.env = env
        self.agents = agents
        self.max_ticks = max_ticks
        pygame.init()
        width = env.size * CELL_SIZE + PANEL_WIDTH
        height = env.size * CELL_SIZE
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 13)
        self.show_ground_truth = show_ground_truth
        self.paused = False
        self.sim = None  # Riferimento alla simulazione, da impostare esternamente
        self.arrow_font = pygame.font.SysFont("segoeuisymbol", CELL_SIZE - 4)

        # Precomputa frecce direzionali per ogni cella ENTRANCE/EXIT
        self._arrow_map: dict[tuple, str] = {}
        for w in env.warehouses:
            side = w["side"]
            er, ec = w["entrance"]
            xr, xc = w["exit"]
            if side == "top":
                self._arrow_map[(er, ec)] = "\u25B2"
                self._arrow_map[(xr, xc)] = "\u25BC"
            elif side == "bottom":
                self._arrow_map[(er, ec)] = "\u25BC"
                self._arrow_map[(xr, xc)] = "\u25B2"
            elif side == "left":
                self._arrow_map[(er, ec)] = "\u25C0"
                self._arrow_map[(xr, xc)] = "\u25B6"
            else:  # right
                self._arrow_map[(er, ec)] = "\u25B6"
                self._arrow_map[(xr, xc)] = "\u25C0"
    

    def run_simulation(self, simulation: Simulation):
        """Gestisce il loop di simulazione tick per tick, aggiornando la visualizzazione."""
        self.sim = simulation
        step = False  # flag per avanzare di un tick con S mentre in pausa
        while self.sim.tick < self.max_ticks and not self.sim._all_delivered():
            step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return self.sim.log
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_g:
                        self.show_ground_truth = not self.show_ground_truth
                    elif event.key == pygame.K_s:
                        step = True
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return self.sim.log

            if not self.paused or step:
                # Esegui un singolo tick (tutti gli agenti in simultanea)
                active = [a for a in self.sim.agents if a.state != AgentState.DEAD]
                if not active:
                    break
                for a in active:
                    a.observe(self.sim.env)
                _mesh_communicate(active)
                for a in active:
                    if isinstance(a, Relay):
                        a.decide(self.sim.env, agents=self.sim.agents)
                    else:
                        a.decide(self.sim.env)
                for a in active:
                    a.act(self.sim.env)
                self.sim.log.append(self.sim._build_snapshot(self.sim.tick))
                self.sim.tick += 1

            self._draw_frame(self.show_ground_truth)
            self.clock.tick(FPS)

        pygame.quit()
        return self.sim.log


    def _draw_frame(self, show_ground_truth: bool):
        """Disegna un frame completo: griglia, agenti e pannello."""
        self.screen.fill(COLOR_BACKGROUND)
        self._draw_grid(show_ground_truth)
        self._draw_agents()
        self._draw_panel()
        pygame.display.flip()


    def _draw_grid(self, show_ground_truth: bool):
        """Disegna la griglia dell'ambiente, colorando le celle in base al loro tipo."""
        # Costruisce la mappa visibile come unione delle mappe locali degli agenti
        from agent import UNKNOWN
        shared_map = np.full((self.env.size, self.env.size), UNKNOWN, dtype=int)
        for a in self.agents:
            if a.state != AgentState.DEAD:
                mask = a.local_map != UNKNOWN
                shared_map[mask] = a.local_map[mask]

        color_map = {
            EMPTY: COLOR_EMPTY,
            WALL: COLOR_WALL,
            WAREHOUSE: COLOR_WAREHOUSE,
            ENTRANCE: COLOR_ENTRANCE,
            EXIT: COLOR_EXIT,
        }
        for r in range(self.env.size):
            for c in range(self.env.size):
                if show_ground_truth:
                    cell_val = self.env.grid[r][c]
                else:
                    cell_val = int(shared_map[r][c])
                color = color_map.get(cell_val, COLOR_UNKNOWN)
                rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)
                # Freccia direzionale su ENTRANCE/EXIT
                if (r, c) in self._arrow_map and cell_val in (ENTRANCE, EXIT):
                    arrow = self._arrow_map[(r, c)]
                    surf = self.arrow_font.render(arrow, True, (255, 255, 255))
                    ax = c * CELL_SIZE + (CELL_SIZE - surf.get_width()) // 2
                    ay = r * CELL_SIZE + (CELL_SIZE - surf.get_height()) // 2
                    self.screen.blit(surf, (ax, ay))

        # Oggetti: ground truth o quelli scoperti dagli agenti
        if show_ground_truth:
            for obj_id, (or_, oc) in self.env._objects.items():
                if obj_id in self.env._claimed:
                    continue  # già raccolto da un agente, non mostrarlo a terra
                cx = oc * CELL_SIZE + CELL_SIZE // 2
                cy = or_ * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(self.screen, COLOR_OBJECT, (cx, cy), CELL_SIZE // 4)
        else:
            # Unione degli oggetti conosciuti da tutti gli agenti (id → pos)
            known_by_id: dict = {}
            for a in self.agents:
                if a.state != AgentState.DEAD:
                    for obj_id, pos in a.known_objects.items():
                        known_by_id.setdefault(obj_id, pos)
            for obj_id, (or_, oc) in known_by_id.items():
                # Salta se in trasporto o già consegnato (non più in env._objects)
                if obj_id in self.env._claimed:
                    continue
                if obj_id not in self.env._objects:
                    continue
                cx = oc * CELL_SIZE + CELL_SIZE // 2
                cy = or_ * CELL_SIZE + CELL_SIZE // 2
                pygame.draw.circle(self.screen, COLOR_OBJECT, (cx, cy), CELL_SIZE // 4)


    def _draw_agents(self):
        """Disegna gli agenti sulla griglia, colorandoli in base al loro ruolo e stato."""
        for agent in self.agents:
            cx = agent.c * CELL_SIZE + CELL_SIZE // 2
            cy = agent.r * CELL_SIZE + CELL_SIZE // 2
            if agent.state == AgentState.DEAD:
                color = COLOR_DEAD
            elif isinstance(agent, Scout):
                color = COLOR_SCOUT
            elif isinstance(agent, Collector):
                color = COLOR_COLLECTOR
            else:
                color = COLOR_RELAY
            pygame.draw.circle(self.screen, color, (cx, cy), CELL_SIZE // 3)
            if agent.carrying is not None:
                pygame.draw.circle(self.screen, COLOR_OBJECT, (cx, cy), CELL_SIZE // 6)


    def _draw_panel(self):
        """Disegna il pannello laterale con le statistiche e lo stato degli agenti."""
        panel_x = self.env.size * CELL_SIZE
        pygame.draw.rect(self.screen, COLOR_PANEL, (panel_x, 0, PANEL_WIDTH, self.env.size * CELL_SIZE))
        y_offset = 10
        for agent in self.agents:
            text = f"Agent {agent.agent_id} ({agent.role})"
            label = self.font.render(text, True, COLOR_BACKGROUND)
            self.screen.blit(label, (panel_x + 10, y_offset))
            y_offset += 20
            text = f"Pos: ({agent.r}, {agent.c})"
            label = self.font.render(text, True, COLOR_BACKGROUND)
            self.screen.blit(label, (panel_x + 10, y_offset))
            y_offset += 20
            text = f"Battery: {agent.battery}"
            label = self.font.render(text, True, COLOR_BACKGROUND)
            self.screen.blit(label, (panel_x + 10, y_offset))
            y_offset += 20
            text = f"Carrying: {agent.carrying if agent.carrying is not None else 'None'}"
            label = self.font.render(text, True, COLOR_BACKGROUND)
            self.screen.blit(label, (panel_x + 10, y_offset))
            y_offset += 20
            strategy = agent.explore_strategy or "relay"
            text = f"Strategy: {strategy}"
            label = self.font.render(text, True, COLOR_BACKGROUND)
            self.screen.blit(label, (panel_x + 10, y_offset))
            y_offset += 30