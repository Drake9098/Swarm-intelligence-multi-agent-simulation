"""Visualizzazione della simulazione con PySide6 (Qt)."""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, QElapsedTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QMouseEvent, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from agent import UNKNOWN, AgentState, Collector, Relay, Scout
from environment import EMPTY, ENTRANCE, EXIT, WALL, WAREHOUSE
from simulation import Simulation, _mesh_communicate

HEADLESS = os.environ.get("HEADLESS", "0") == "1"

BASE_CELL_PX = 24
PANEL_MIN_WIDTH = 300
INITIAL_BATTERY = 500
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 840
DEFAULT_TICKS_PER_SEC = 10
MAP_REPAINT_HZ = 30
TRAIL_MAX_LEN = 14
LERP_MS = 220
HIGHLIGHT_ALPHA = 55


COLOR_EMPTY = QColor(255, 255, 255)
COLOR_WALL = QColor(64, 64, 64)
COLOR_WAREHOUSE = QColor(74, 144, 217)
COLOR_ENTRANCE = QColor(46, 204, 113)
COLOR_EXIT = QColor(231, 76, 60)
COLOR_UNKNOWN = QColor(180, 180, 180)  # fallback raro (mappa locale incoerente)
# Nebbia: terreno reale sotto + velo scuro leggermente trasparente (si vede la mappa)
COLOR_FOG_OVERLAY = QColor(20, 25, 35, 105)
COLOR_BACKGROUND = QColor(240, 240, 240)
COLOR_PANEL_BG = QColor(40, 44, 52)
COLOR_PANEL_TEXT = QColor(230, 230, 235)
COLOR_GRID_LINE = QColor(200, 200, 200)
COLOR_SCOUT = QColor(34, 139, 34)
COLOR_COLLECTOR = QColor(255, 140, 0)
COLOR_RELAY = QColor(160, 80, 220)
COLOR_DEAD = QColor(100, 100, 100)
COLOR_OBJECT = QColor(255, 215, 0)
COLOR_ACCENT = QColor(97, 175, 239)
COLOR_TRAIL = QColor(80, 80, 90, 90)
# Stessa componente mesh: alone chiaro sotto l’agente + anello opaco (l’alpha sul solo QPen spesso non si vede).
COLOR_COMPONENT_HALO = QColor(255, 230, 80, 130)
COLOR_COMPONENT_RING = QColor(255, 165, 0)
# Raggio comunicazione (selezione): riempimento più leggibile + bordo sul perimetro Manhattan
COLOR_COMM_DISK_FILL = QColor(0, 110, 255, 125)
COLOR_COMM_DISK_EDGE = QColor(0, 55, 160, 235)


def _agent_accent_hex(agent) -> str:
    """Colore accento UI per ruolo (come sulla mappa)."""
    if agent.state == AgentState.DEAD:
        return COLOR_DEAD.name()
    if isinstance(agent, Scout):
        return COLOR_SCOUT.name()
    if isinstance(agent, Collector):
        return COLOR_COLLECTOR.name()
    return COLOR_RELAY.name()


def _advance_one_tick(sim: Simulation, max_ticks: int) -> bool:
    if sim.tick >= max_ticks or sim._all_delivered():
        return False
    active = [a for a in sim.agents if a.state != AgentState.DEAD]
    if not active:
        return False
    for a in active:
        a.observe(sim.env)
    _mesh_communicate(active)
    for a in active:
        if isinstance(a, Relay):
            a.decide(sim.env, agents=sim.agents)
        else:
            a.decide(sim.env)
    for a in active:
        a.act(sim.env)
    sim.log.append(sim._build_snapshot(sim.tick))
    sim.tick += 1
    return True


def _agent_id_to_component(agents: list) -> dict[int, frozenset[int]]:
    """Componenti connesse con la stessa regola arco della simulazione (max dei due raggi)."""
    active = [a for a in agents if a.state != AgentState.DEAD]
    n = len(active)
    aid = [a.agent_id for a in active]
    adj: dict[int, set] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = active[i], active[j]
            dist = abs(a.r - b.r) + abs(a.c - b.c)
            if dist <= max(a.comm_radius, b.comm_radius):
                adj[i].add(j)
                adj[j].add(i)
    visited: set[int] = set()
    comp_by_agent: dict[int, frozenset[int]] = {}
    for start in range(n):
        if start in visited:
            continue
        comp_ids: list[int] = []
        queue = [start]
        while queue:
            u = queue.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_ids.append(aid[u])
            for v in adj[u]:
                if v not in visited:
                    queue.append(v)
        fs = frozenset(comp_ids)
        for x in comp_ids:
            comp_by_agent[x] = fs
    return comp_by_agent


class GridMapWidget(QWidget):
    """Griglia mappa ridimensionabile (scala uniforme), trail, lerp, click."""

    def __init__(self, visualizer: "Visualizer"):
        super().__init__()
        self.viz = visualizer
        n = visualizer.env.size
        self.setMinimumSize(n * 8, n * 8)
        self.setMouseTracking(True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), COLOR_BACKGROUND)
        self.setPalette(pal)

        self._lerp_from: dict[int, tuple[int, int]] = {}
        self._lerp_clock = QElapsedTimer()

    def cell_px(self) -> int:
        n = self.viz.env.size
        if n <= 0:
            return BASE_CELL_PX
        return max(4, min(self.width(), self.height()) // n)

    def grid_origin(self) -> tuple[int, int]:
        """Centra la griglia n×n nel widget."""
        n = self.viz.env.size
        cs = self.cell_px()
        gw, gh = n * cs, n * cs
        ox = (self.width() - gw) // 2
        oy = (self.height() - gh) // 2
        return ox, oy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick_at(event.position().toPoint())
        elif event.button() == Qt.MouseButton.RightButton:
            self.viz.selected_agent_id = None
            self.viz.selected_cell = None
            self._notify_detail()
            self.update()
        super().mousePressEvent(event)

    def _pick_at(self, pt: QPoint) -> None:
        n = self.viz.env.size
        cs = self.cell_px()
        ox, oy = self.grid_origin()
        x, y = pt.x() - ox, pt.y() - oy
        if x < 0 or y < 0:
            self.viz.selected_agent_id = None
            self._notify_detail()
            self.update()
            return
        c, r = x // cs, y // cs
        if not (0 <= r < n and 0 <= c < n):
            self.viz.selected_agent_id = None
            self._notify_detail()
            self.update()
            return
        picked = None
        for agent in self.viz.agents:
            if agent.state != AgentState.DEAD and agent.r == r and agent.c == c:
                picked = agent.agent_id
        self.viz.selected_agent_id = picked
        self.viz.selected_cell = (r, c) if picked is None else None
        self._notify_detail()
        self.update()

    def _notify_detail(self) -> None:
        w = self.window()
        if hasattr(w, "_update_detail"):
            w._update_detail()

    def begin_tick_animation(self) -> None:
        self._lerp_from = {a.agent_id: (a.r, a.c) for a in self.viz.agents}
        self._lerp_clock.restart()

    def _compute_seen_by_swarm(self) -> np.ndarray:
        """True se almeno un agente vivo ha osservato la cella (non è in _unobserved)."""
        env = self.viz.env
        agents = self.viz.agents
        n = env.size
        seen = np.zeros((n, n), dtype=bool)
        for a in agents:
            if a.state == AgentState.DEAD:
                continue
            for r in range(n):
                for c in range(n):
                    if (r, c) not in a._unobserved:
                        seen[r, c] = True
        return seen

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        ox, oy = self.grid_origin()
        painter.translate(ox, oy)
        cs = self.cell_px()

        seen_by_swarm = self._compute_seen_by_swarm()
        self._draw_grid(painter, cs, seen_by_swarm)
        self._draw_comm_overlay(painter, cs)
        self._draw_trails(painter, cs)
        self._draw_objects(painter, cs, seen_by_swarm)
        self._draw_agents(painter, cs)

    def _cell_rect(self, r: int, c: int, cs: int) -> QRect:
        return QRect(c * cs, r * cs, cs, cs)

    def _draw_grid(self, painter: QPainter, cs: int, seen_by_swarm: np.ndarray) -> None:
        env = self.viz.env
        agents = self.viz.agents
        show_gt = self.viz.show_ground_truth

        shared_map = np.full((env.size, env.size), UNKNOWN, dtype=int)
        for a in agents:
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
        pen_grid = QPen(COLOR_GRID_LINE)
        pen_grid.setWidth(1)

        for r in range(env.size):
            for c in range(env.size):
                if show_gt:
                    cell_val = int(env.grid[r][c])
                else:
                    cell_val = int(shared_map[r][c])
                rect = self._cell_rect(r, c, cs)
                if not show_gt and not seen_by_swarm[r, c]:
                    gt_val = int(env.grid[r][c])
                    base = color_map.get(gt_val, COLOR_UNKNOWN)
                    painter.fillRect(rect, base)
                    painter.fillRect(rect, COLOR_FOG_OVERLAY)
                else:
                    color = color_map.get(cell_val, COLOR_UNKNOWN)
                    painter.fillRect(rect, color)
                painter.setPen(pen_grid)
                painter.drawRect(rect)

                terrain_visible = show_gt or seen_by_swarm[r, c]
                if (
                    terrain_visible
                    and (r, c) in self.viz._arrow_map
                    and cell_val in (ENTRANCE, EXIT)
                ):
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    font = QFont("Segoe UI Symbol", max(7, cs // 3))
                    painter.setFont(font)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.viz._arrow_map[(r, c)])

    def _draw_comm_overlay(self, painter: QPainter, cs: int) -> None:
        sel = self.viz.selected_agent_id
        if sel is None:
            return
        agent = next((a for a in self.viz.agents if a.agent_id == sel), None)
        if agent is None or agent.state == AgentState.DEAD:
            return
        env = self.viz.env
        rad = agent.comm_radius
        ar, ac = agent.r, agent.c
        for r in range(env.size):
            for c in range(env.size):
                d = abs(r - ar) + abs(c - ac)
                if d > rad:
                    continue
                rect = self._cell_rect(r, c, cs)
                painter.fillRect(rect, COLOR_COMM_DISK_FILL)
                if d == rad:
                    edge = QPen(COLOR_COMM_DISK_EDGE)
                    edge.setWidth(max(2, cs // 14))
                    edge.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
                    painter.setPen(edge)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(rect)

    def _trail_base_color(self, agent_id: int) -> QColor:
        """Stesso schema cromatico dei cerchi agente in _draw_agents."""
        agent = next((a for a in self.viz.agents if a.agent_id == agent_id), None)
        if agent is None:
            return QColor(COLOR_TRAIL)
        if agent.state == AgentState.DEAD:
            return QColor(COLOR_DEAD)
        if isinstance(agent, Scout):
            return QColor(COLOR_SCOUT)
        if isinstance(agent, Collector):
            return QColor(COLOR_COLLECTOR)
        return QColor(COLOR_RELAY)

    def _draw_trails(self, painter: QPainter, cs: int) -> None:
        for aid, hist in self.viz.trail_history.items():
            if len(hist) < 2:
                continue
            base = self._trail_base_color(aid)
            pen = QPen(base)
            pen.setWidth(max(1, cs // 10))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            pts = []
            for i, (r, c) in enumerate(hist):
                cx = c * cs + cs // 2
                cy = r * cs + cs // 2
                pts.append((cx, cy))
            for i in range(1, len(pts)):
                # Alpha crescente verso la posizione attuale (fine storia)
                t = i / max(1, len(pts) - 1)
                alpha = int(45 + 130 * t)
                col = QColor(base)
                col.setAlpha(min(200, alpha))
                pen.setColor(col)
                painter.setPen(pen)
                painter.drawLine(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])

    def _draw_objects(self, painter: QPainter, cs: int, seen_by_swarm: np.ndarray) -> None:
        env = self.viz.env
        agents = self.viz.agents
        r_obj = max(2, cs // 4)

        if self.viz.show_ground_truth:
            for obj_id, (or_, oc) in env._objects.items():
                if obj_id in env._claimed:
                    continue
                cx = oc * cs + cs // 2
                cy = or_ * cs + cs // 2
                painter.setBrush(COLOR_OBJECT)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cx - r_obj, cy - r_obj, 2 * r_obj, 2 * r_obj)
        else:
            known_by_id: dict = {}
            for a in agents:
                if a.state != AgentState.DEAD:
                    for obj_id, pos in a.known_objects.items():
                        known_by_id.setdefault(obj_id, pos)
            for obj_id, (or_, oc) in known_by_id.items():
                if obj_id in env._claimed:
                    continue
                if obj_id not in env._objects:
                    continue
                if not seen_by_swarm[or_, oc]:
                    continue
                cx = oc * cs + cs // 2
                cy = or_ * cs + cs // 2
                painter.setBrush(COLOR_OBJECT)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cx - r_obj, cy - r_obj, 2 * r_obj, 2 * r_obj)

    def _agent_display_rc(self, agent) -> tuple[float, float]:
        cs = self.cell_px()
        t = min(1.0, self._lerp_clock.elapsed() / float(LERP_MS))
        fr = self._lerp_from.get(agent.agent_id)
        if fr is None:
            return float(agent.r), float(agent.c)
        sr, sc = fr
        er, ec = agent.r, agent.c
        return sr + (er - sr) * t, sc + (ec - sc) * t

    def _draw_agents(self, painter: QPainter, cs: int) -> None:
        r_ag = max(3, cs // 3)
        r_carry = max(2, cs // 6)
        comp = _agent_id_to_component(self.viz.agents)
        sel = self.viz.selected_agent_id
        highlight_ids = comp.get(sel, frozenset()) if sel is not None else frozenset()
        halo_r = r_ag + max(5, cs // 4)

        # Sotto: aureola per tutti gli agenti della stessa componente (anche il selezionato)
        for agent in self.viz.agents:
            if agent.agent_id not in highlight_ids or agent.state == AgentState.DEAD:
                continue
            dr, dc = self._agent_display_rc(agent)
            cx = int(dc * cs + cs / 2)
            cy = int(dr * cs + cs / 2)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(COLOR_COMPONENT_HALO)
            painter.drawEllipse(cx - halo_r, cy - halo_r, 2 * halo_r, 2 * halo_r)

        for agent in self.viz.agents:
            dr, dc = self._agent_display_rc(agent)
            cx = int(dc * cs + cs / 2)
            cy = int(dr * cs + cs / 2)
            if agent.state == AgentState.DEAD:
                color = COLOR_DEAD
            elif isinstance(agent, Scout):
                color = COLOR_SCOUT
            elif isinstance(agent, Collector):
                color = COLOR_COLLECTOR
            else:
                color = COLOR_RELAY
            painter.setBrush(color)
            painter.setPen(QPen(color.darker(130)))
            painter.drawEllipse(cx - r_ag, cy - r_ag, 2 * r_ag, 2 * r_ag)

            if agent.agent_id in highlight_ids and agent.state != AgentState.DEAD:
                margin = max(4, cs // 8)
                ring = QPen(COLOR_COMPONENT_RING)
                ring.setWidth(max(3, cs // 9))
                ring.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(ring)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(
                    cx - r_ag - margin,
                    cy - r_ag - margin,
                    2 * (r_ag + margin),
                    2 * (r_ag + margin),
                )

            if agent.carrying is not None:
                painter.setBrush(COLOR_OBJECT)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cx - r_carry, cy - r_carry, 2 * r_carry, 2 * r_carry)


class SwarmMainWindow(QMainWindow):
    def __init__(self, visualizer: "Visualizer"):
        super().__init__()
        self.viz = visualizer
        self._step_once = False
        self._finished = False

        sim = visualizer.sim
        label = getattr(sim, "_config_label", "sim")
        self.setWindowTitle(f"Swarm — {label} | tick {sim.tick}/{visualizer.max_ticks}")
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        self.map_widget = GridMapWidget(visualizer)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(4, 4, 4, 4)
        fl.addWidget(self.map_widget, stretch=1)
        root.addWidget(frame, stretch=1)

        panel = QFrame()
        panel.setMinimumWidth(PANEL_MIN_WIDTH)
        _bg = COLOR_PANEL_BG.name()
        _ac = COLOR_ACCENT.name()
        # Tutte le graffe Qt in f-string: raddoppiate tranne i placeholder { _bg } / { _ac }
        panel.setStyleSheet(
            f"""
            QFrame {{
                background-color: {_bg};
                border-radius: 8px;
            }}
            QLabel {{
                color: #e6e6eb;
                font-size: 12px;
            }}
            QLabel#title {{
                color: {_ac};
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton {{
                padding: 6px 10px;
                border-radius: 4px;
                background: #3d4451;
                color: #e6e6eb;
            }}
            QPushButton:hover {{
                background: #4a5262;
            }}
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: #e6e6eb;
                background: #2b2f38;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background: {_ac};
                border-radius: 2px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #3d4451;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {_ac};
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QLabel#subtitle {{
                color: #7b8490;
                font-size: 11px;
                font-weight: bold;
            }}
            QFrame#agentCard {{
                background-color: #2d333b;
                border: 1px solid #454b5a;
                border-radius: 8px;
            }}
            QLabel#agentCardTitle {{
                color: #e6e6eb;
                font-size: 13px;
            }}
            QLabel#formLabel {{
                color: #7b8490;
                font-size: 11px;
            }}
            QLabel#agentField {{
                color: #d8dee9;
                font-size: 12px;
            }}
            QProgressBar#batteryBar {{
                border: 1px solid #454b5a;
                border-radius: 4px;
                background: #1e2228;
                min-height: 10px;
                max-height: 12px;
                text-align: center;
                font-size: 10px;
                color: #c8ccd4;
            }}
            QProgressBar#batteryBar::chunk {{
                background-color: #98c379;
                border-radius: 3px;
            }}
            """
        )
        pv = QVBoxLayout(panel)
        pv.setSpacing(8)

        title = QLabel("Stato simulazione")
        title.setObjectName("title")
        pv.addWidget(title)

        ctrl = QHBoxLayout()
        self.btn_pause = QPushButton("Pausa")
        self.btn_pause.clicked.connect(self._btn_toggle_pause)
        ctrl.addWidget(self.btn_pause)
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self._btn_step)
        ctrl.addWidget(self.btn_step)
        self.btn_reset = QPushButton("Riavvia")
        self.btn_reset.clicked.connect(self._btn_reset)
        self.btn_reset.setEnabled(bool(visualizer.restart_path and visualizer.restart_config))
        ctrl.addWidget(self.btn_reset)
        pv.addLayout(ctrl)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Velocità:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 25)
        self.speed_slider.setValue(DEFAULT_TICKS_PER_SEC)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.speed_slider, stretch=1)
        self.speed_value_lbl = QLabel(f"{DEFAULT_TICKS_PER_SEC} tick/s")
        self.speed_value_lbl.setMinimumWidth(72)
        speed_row.addWidget(self.speed_value_lbl)
        pv.addLayout(speed_row)

        self.tick_bar = QProgressBar()
        self.tick_bar.setMaximum(max(1, visualizer.max_ticks))
        self.tick_bar.setFormat("Tick %v / %m")
        pv.addWidget(self.tick_bar)

        self.obj_bar = QProgressBar()
        self.obj_bar.setMaximum(10)
        self.obj_bar.setFormat("Oggetti consegnati %v / 10")
        pv.addWidget(self.obj_bar)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.RichText)
        pv.addWidget(self.status_label)

        legend = QLabel(
            "<b>Legenda</b><br>"
            "<span style='color:#228B22'>■</span> Scout &nbsp;"
            "<span style='color:#FF8C00'>■</span> Collector &nbsp;"
            "<span style='color:#A050DC'>■</span> Relay<br>"
            "<span style='color:#FFD700'>●</span> Oggetto &nbsp;"
            "<span style='background:rgba(20,25,35,0.35);'> &nbsp; </span>"
            " Fog of War<br>"
            "<span style='color:#4A90D9'>■</span> Magazzino &nbsp;"
            "<span style='color:#404040'>■</span> Muro"
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        pv.addWidget(legend)

        self.detail_label = QLabel("Clic sulla mappa: seleziona agente o cella.")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.TextFormat.RichText)
        pv.addWidget(self.detail_label)

        agents_heading = QLabel("AGENTI")
        agents_heading.setObjectName("subtitle")
        pv.addWidget(agents_heading)

        self._agent_card_entries: list[dict] = []
        self.agents_scroll = QScrollArea()
        self.agents_scroll.setWidgetResizable(True)
        self.agents_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.agents_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.agents_list_inner = QWidget()
        self.agents_list_layout = QVBoxLayout(self.agents_list_inner)
        self.agents_list_layout.setContentsMargins(0, 0, 4, 0)
        self.agents_list_layout.setSpacing(10)
        self._build_agent_cards()
        self.agents_list_layout.addStretch(1)
        self.agents_scroll.setWidget(self.agents_list_inner)
        pv.addWidget(self.agents_scroll, stretch=1)

        root.addWidget(panel, stretch=0)

        self._setup_shortcuts()

        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._on_sim_timer)
        self._apply_sim_interval()

        self.map_repaint_timer = QTimer(self)
        self.map_repaint_timer.timeout.connect(self.map_widget.update)
        self.map_repaint_timer.start(1000 // MAP_REPAINT_HZ)

        self._refresh_labels()
        self._update_detail()

    def _build_agent_cards(self) -> None:
        """Card per agente: intestazione colorata, form metriche, barra batteria."""
        for agent in sorted(self.viz.agents, key=lambda a: a.agent_id):
            card = QFrame()
            card.setObjectName("agentCard")
            outer = QVBoxLayout(card)
            outer.setContentsMargins(12, 10, 12, 10)
            outer.setSpacing(8)

            head = QHBoxLayout()
            head.setSpacing(8)
            head.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            _chex = _agent_accent_hex(agent)
            dot = QFrame()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(
                f"QFrame {{ background-color: {_chex}; border-radius: 6px; border: none; }}"
            )
            ttl = QLabel(
                f"<b>Agent {agent.agent_id}</b> · "
                f"<span style='color:#abb2bf'>{agent.role}</span>"
            )
            ttl.setObjectName("agentCardTitle")
            ttl.setTextFormat(Qt.TextFormat.RichText)
            st_pill = QLabel(agent.state.name)
            st_pill.setStyleSheet(
                "background: #3d4451; color: #c8ccd4; padding: 2px 8px; "
                "border-radius: 10px; font-size: 11px;"
            )
            head.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
            head.addWidget(ttl, 1, Qt.AlignmentFlag.AlignVCenter)
            head.addWidget(
                st_pill,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
            outer.addLayout(head)

            form = QFormLayout()
            form.setSpacing(6)
            form.setHorizontalSpacing(10)
            form.setContentsMargins(0, 4, 0, 0)

            def _fl(text: str) -> QLabel:
                w = QLabel(text)
                w.setObjectName("formLabel")
                return w

            pos_l = QLabel()
            pos_l.setObjectName("agentField")
            form.addRow(_fl("Position"), pos_l)

            bat = QProgressBar()
            bat.setObjectName("batteryBar")
            bat.setRange(0, INITIAL_BATTERY)
            bat.setFixedHeight(12)
            bat.setTextVisible(True)
            bat.setFormat("%v / %m")
            form.addRow(_fl("Battery"), bat)

            comm_l = QLabel()
            comm_l.setObjectName("agentField")
            form.addRow(_fl("Comm radius"), comm_l)

            carry_l = QLabel()
            carry_l.setObjectName("agentField")
            form.addRow(_fl("Carrying"), carry_l)

            strat_l = QLabel()
            strat_l.setObjectName("agentField")
            strat_l.setWordWrap(True)
            form.addRow(_fl("Strategy"), strat_l)

            outer.addLayout(form)
            self.agents_list_layout.addWidget(card)

            self._agent_card_entries.append(
                {
                    "id": agent.agent_id,
                    "frame": card,
                    "dot": dot,
                    "state_pill": st_pill,
                    "pos": pos_l,
                    "battery": bat,
                    "comm": comm_l,
                    "carry": carry_l,
                    "strat": strat_l,
                }
            )

    def _apply_sim_interval(self) -> None:
        tps = self.speed_slider.value()
        ms = max(1, int(1000 / tps))
        self.sim_timer.start(ms)

    def _on_speed_changed(self, v: int) -> None:
        self.speed_value_lbl.setText(f"{v} tick/s")
        self._apply_sim_interval()

    def _btn_toggle_pause(self) -> None:
        self.viz.paused = not self.viz.paused
        self._sync_pause_button()
        self._refresh_labels()

    def _sync_pause_button(self) -> None:
        self.btn_pause.setText("Riprendi" if self.viz.paused else "Pausa")

    def _btn_step(self) -> None:
        if not self.viz.paused:
            self.viz.paused = True
            self._sync_pause_button()
        self._step_once = True

    def _btn_reset(self) -> None:
        if not self.viz.restart_path or not self.viz.restart_config:
            return
        self.sim_timer.stop()
        new_sim = Simulation(self.viz.restart_path, self.viz.max_ticks, self.viz.restart_config)
        new_sim._config_label = getattr(self.viz.sim, "_config_label", "sim")
        self.viz.sim = new_sim
        self.viz.env = new_sim.env
        self.viz.agents = new_sim.agents
        self.viz.trail_history.clear()
        self.viz.selected_agent_id = None
        self.viz.selected_cell = None
        self._finished = False
        self.map_widget._lerp_from.clear()
        self._step_once = False
        self.viz.paused = False
        self._sync_pause_button()
        self._apply_sim_interval()
        label = getattr(new_sim, "_config_label", "sim")
        self.setWindowTitle(f"Swarm - {label} | tick {new_sim.tick}/{self.viz.max_ticks}")
        self._refresh_labels()
        self._update_detail()
        self.map_widget.update()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._btn_toggle_pause)
        QShortcut(QKeySequence(Qt.Key.Key_G), self, activated=self._toggle_ground_truth)
        QShortcut(QKeySequence(Qt.Key.Key_S), self, activated=self._btn_step)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close)

    def _toggle_ground_truth(self) -> None:
        self.viz.show_ground_truth = not self.viz.show_ground_truth
        self.map_widget.update()
        self._refresh_labels()
        self._update_detail()

    def _on_sim_timer(self) -> None:
        sim = self.viz.sim
        if self._finished:
            return

        if sim.tick >= self.viz.max_ticks or sim._all_delivered():
            self._finished = True
            self.sim_timer.stop()
            self.setWindowTitle(
                f"Swarm — completato | tick {sim.tick} | "
                f"oggetti {10 - sim.env.objects_remaining()}/10"
            )
            self._refresh_labels()
            self.map_widget.update()
            return

        should_step = (not self.viz.paused) or self._step_once
        if should_step:
            self._step_once = False
            self.map_widget.begin_tick_animation()
            ok = _advance_one_tick(sim, self.viz.max_ticks)
            if ok:
                for a in self.viz.agents:
                    self.viz.trail_history.setdefault(a.agent_id, deque(maxlen=TRAIL_MAX_LEN)).append(
                        (a.r, a.c)
                    )
            self.setWindowTitle(
                f"Swarm — tick {sim.tick}/{self.viz.max_ticks} | "
                f"oggetti {10 - sim.env.objects_remaining()}/10"
                + ("  [PAUSA]" if self.viz.paused else "")
            )

        self._refresh_labels()
        self._update_detail()
        self.map_widget.update()

    def _refresh_labels(self) -> None:
        sim = self.viz.sim
        delivered = 10 - sim.env.objects_remaining()
        remaining = sim.env.objects_remaining()
        in_transit = sum(1 for a in self.viz.agents if a.carrying is not None and a.state != AgentState.DEAD)
        pause_txt = "in pausa" if self.viz.paused else "in esecuzione"
        self.tick_bar.setMaximum(max(1, self.viz.max_ticks))
        self.tick_bar.setValue(min(sim.tick, self.viz.max_ticks))
        self.obj_bar.setValue(delivered)
        self.status_label.setText(
            f"Tick: <b>{sim.tick}</b> / {self.viz.max_ticks}<br>"
            f"Consegnati: <b>{delivered}</b> &nbsp;|&nbsp; A terra: <b>{remaining}</b> "
            f"&nbsp;|&nbsp; In trasporto: <b>{in_transit}</b><br>"
            f"Stato: {pause_txt}"
        )
        self._sync_pause_button()

        for ent in self._agent_card_entries:
            ag = next(a for a in self.viz.agents if a.agent_id == ent["id"])
            ent["pos"].setText(f"({ag.r}, {ag.c})")
            ent["battery"].setValue(max(0, ag.battery))
            ent["comm"].setText(str(ag.comm_radius))
            ent["carry"].setText("-" if ag.carrying is None else str(ag.carrying))
            ent["strat"].setText(ag.explore_strategy or "—")
            ent["state_pill"].setText(ag.state.name)
            _hx = _agent_accent_hex(ag)
            ent["dot"].setStyleSheet(
                f"QFrame {{ background-color: {_hx}; border-radius: 6px; border: none; }}"
            )
            if ag.battery <= 120:
                ent["battery"].setStyleSheet(
                    "QProgressBar#batteryBar { border: 1px solid #454b5a; border-radius: 4px; "
                    "background: #1e2228; min-height: 10px; max-height: 12px; font-size: 10px; "
                    "color: #c8ccd4; }"
                    "QProgressBar#batteryBar::chunk { background-color: #e06c75; border-radius: 3px; }"
                )
            else:
                ent["battery"].setStyleSheet(
                    "QProgressBar#batteryBar { border: 1px solid #454b5a; border-radius: 4px; "
                    "background: #1e2228; min-height: 10px; max-height: 12px; font-size: 10px; "
                    "color: #c8ccd4; }"
                    "QProgressBar#batteryBar::chunk { background-color: #98c379; border-radius: 3px; }"
                )
        self._apply_agent_card_selection_style()

    def _apply_agent_card_selection_style(self) -> None:
        ac = COLOR_ACCENT.name()
        for ent in self._agent_card_entries:
            ag = next(a for a in self.viz.agents if a.agent_id == ent["id"])
            if self.viz.selected_agent_id == ag.agent_id:
                ent["frame"].setStyleSheet(
                    f"QFrame#agentCard {{ background-color: #343b4a; border: 2px solid {ac}; "
                    f"border-radius: 8px; }}"
                )
            else:
                ent["frame"].setStyleSheet(
                    "QFrame#agentCard { background-color: #2d333b; border: 1px solid #454b5a; "
                    "border-radius: 8px; }"
                )

    def _update_detail(self) -> None:
        sim = self.viz.sim
        sel = self.viz.selected_agent_id
        if sel is not None:
            agent = next((a for a in self.viz.agents if a.agent_id == sel), None)
            if agent:
                comp = _agent_id_to_component(self.viz.agents)
                peers = sorted(comp.get(sel, frozenset()) - {sel})
                peer_txt = ", ".join(str(x) for x in peers) if peers else "nessuno (isolato)"
                self.detail_label.setText(
                    f"<b>Agente {sel}</b> ({agent.role})<br>"
                    f"Stato: {agent.state.name} &nbsp;|&nbsp; Batteria: {agent.battery}<br>"
                    f"Raggio comunicazione (Manhattan): {agent.comm_radius}<br>"
                    f"<i>Stessa componente connessa (mesh):</i> {peer_txt}<br>"
                )
                self._apply_agent_card_selection_style()
                return
        cell = self.viz.selected_cell
        if cell is not None:
            r, c = cell
            val = int(sim.env.grid[r][c])
            names = {
                EMPTY: "Vuoto",
                WALL: "Muro",
                WAREHOUSE: "Area magazzino",
                ENTRANCE: "Ingresso magazzino",
                EXIT: "Uscita magazzino",
            }
            self.detail_label.setText(
                f"<b>Cella ({r}, {c})</b><br>Tipo (mappa reale): {names.get(val, str(val))}<br>"
                "<span style='color:#888'>Clic destro deseleziona.</span>"
            )
            self._apply_agent_card_selection_style()
            return
        self.detail_label.setText(
            "<b>Selezione</b><br>Clic sinistro su agente o cella. Clic destro deseleziona."
        )
        self._apply_agent_card_selection_style()

    def closeEvent(self, event) -> None:
        self.sim_timer.stop()
        self.map_repaint_timer.stop()
        super().closeEvent(event)


class Visualizer:
    def __init__(
        self,
        env,
        agents: list,
        max_ticks: int,
        show_ground_truth: bool = False,
        *,
        restart_path: Optional[str] = None,
        restart_config: Optional[str] = None,
    ):
        self.env = env
        self.agents = agents
        self.max_ticks = max_ticks
        self.show_ground_truth = show_ground_truth
        self.paused = False
        self.sim: Simulation | None = None
        self.restart_path = restart_path
        self.restart_config = restart_config
        self.trail_history: dict[int, deque] = {}
        self.selected_agent_id: int | None = None
        self.selected_cell: tuple[int, int] | None = None

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
            else:
                self._arrow_map[(er, ec)] = "\u25B6"
                self._arrow_map[(xr, xc)] = "\u25C0"

    def run_simulation(self, simulation: Simulation):
        self.sim = simulation
        app = QApplication.instance()
        own_app = False
        if app is None:
            app = QApplication(sys.argv)
            own_app = True

        win = SwarmMainWindow(self)
        win.show()
        app.exec()

        if own_app:
            del win
        return self.sim.log if self.sim else []

