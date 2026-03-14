"""analysis.py — Metriche post-simulazione, visualizzazioni e confronto A/B."""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # rendering senza display attivo
import matplotlib.pyplot as plt

GRID_SIZE = 25
INITIAL_BATTERY = 500
TOTAL_OBJECTS = 10


# ---------------------------------------------------------------------------
# Calcolo metriche scalari
# ---------------------------------------------------------------------------

def compute_metrics(log: list) -> dict:
    """Calcola tutte le metriche scalari da un log di simulazione.

    Restituisce un dizionario con:
      - objects_delivered       int
      - objects_recovered_pct   float  (percentuale su TOTAL_OBJECTS)
      - total_ticks             int
      - energy_consumed_per_agent  dict  {str(agent_id): int}
      - mean_energy_consumed    float
      - energy_per_object       float | None
      - ticks_per_object        float | None
    """
    if not log:
        return {}

    first = log[0]
    last = log[-1]

    objects_delivered = last["objects_delivered"]
    total_ticks = last["tick"] + 1  # tick è 0-indexed

    initial_batteries = {str(a["id"]): INITIAL_BATTERY for a in first["agents"]}
    final_batteries   = {str(a["id"]): a["battery"]    for a in last["agents"]}

    energy_consumed = {
        aid: initial_batteries[aid] - final_batteries[aid]
        for aid in initial_batteries
    }

    mean_energy = sum(energy_consumed.values()) / len(energy_consumed)

    energy_per_object = (
        round(sum(energy_consumed.values()) / objects_delivered, 2)
        if objects_delivered > 0 else None
    )
    ticks_per_object = (
        round(total_ticks / objects_delivered, 2)
        if objects_delivered > 0 else None
    )

    return {
        "objects_delivered":        objects_delivered,
        "objects_recovered_pct":    round(objects_delivered / TOTAL_OBJECTS * 100, 1),
        "total_ticks":              total_ticks,
        "energy_consumed_per_agent": energy_consumed,
        "mean_energy_consumed":     round(mean_energy, 2),
        "energy_per_object":        energy_per_object,
        "ticks_per_object":         ticks_per_object,
    }


# ---------------------------------------------------------------------------
# Heatmap e timeline energetica
# ---------------------------------------------------------------------------

def build_heatmap(log: list, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Conta quante volte ogni cella è stata occupata (tutti gli agenti, tutti i tick)."""
    heatmap = np.zeros((grid_size, grid_size), dtype=int)
    for snapshot in log:
        for agent in snapshot["agents"]:
            r, c = agent["pos"]
            if 0 <= r < grid_size and 0 <= c < grid_size:
                heatmap[r, c] += 1
    return heatmap


def build_energy_timeline(log: list) -> list:
    """Restituisce la batteria media per tick [mean_tick0, mean_tick1, ...]."""
    timeline = []
    for snapshot in log:
        batteries = [a["battery"] for a in snapshot["agents"]]
        timeline.append(sum(batteries) / len(batteries))
    return timeline


# ---------------------------------------------------------------------------
# Salvataggio file
# ---------------------------------------------------------------------------

def save_results(metrics: dict, instance: str, output_dir: str = ".") -> str:
    """Scrive results_X.json in output_dir. Restituisce il percorso del file."""
    path = os.path.join(output_dir, f"results_{instance}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Grafici PNG
# ---------------------------------------------------------------------------

def plot_heatmap(heatmap: np.ndarray, instance: str, output_dir: str = ".") -> str:
    """Genera e salva heatmap_X.png. Restituisce il percorso del file."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visite")
    ax.set_title(f"Heatmap frequenza di visita — Istanza {instance}")
    ax.set_xlabel("Colonna")
    ax.set_ylabel("Riga")
    path = os.path.join(output_dir, f"heatmap_{instance}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_energy_timeline(timeline: list, instance: str, output_dir: str = ".") -> str:
    """Genera e salva energy_X.png — batteria media nel tempo. Restituisce il percorso del file."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timeline, color="tab:blue", linewidth=1.5)
    ax.set_title(f"Energia media degli agenti nel tempo — Istanza {instance}")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Batteria media rimanente")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f"energy_{instance}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Pipeline completa per una singola istanza
# ---------------------------------------------------------------------------

def run_analysis(log: list, instance: str, output_dir: str = ".",
                 grid_size: int = GRID_SIZE) -> dict:
    """Calcola metriche, salva results_X.json, heatmap_X.png, energy_X.png.

    Restituisce il dizionario delle metriche.
    """
    metrics  = compute_metrics(log)
    heatmap  = build_heatmap(log, grid_size)
    timeline = build_energy_timeline(log)

    json_path = save_results(metrics, instance, output_dir)
    hm_path   = plot_heatmap(heatmap, instance, output_dir)
    en_path   = plot_energy_timeline(timeline, instance, output_dir)

    print(f"[analysis] Istanza {instance}: {json_path}, {hm_path}, {en_path}")
    return metrics


# ---------------------------------------------------------------------------
# Confronto A vs B
# ---------------------------------------------------------------------------

_COMPARE_KEYS = [
    ("objects_delivered",        "Oggetti consegnati"),
    ("objects_recovered_pct",    "Recupero (%)"),
    ("total_ticks",              "Tick totali"),
    ("mean_energy_consumed",     "Energia media consumata"),
    ("energy_per_object",        "Energia / oggetto"),
    ("ticks_per_object",         "Tick / oggetto"),
]


def compare(metrics_a: dict, metrics_b: dict) -> None:
    """Stampa un confronto tabellare tra istanza A e B."""
    print(f"\n{'Metrica':<30} {'Istanza A':>12} {'Istanza B':>12}")
    print("-" * 56)
    for key, label in _COMPARE_KEYS:
        va = metrics_a.get(key, "N/A")
        vb = metrics_b.get(key, "N/A")
        print(f"{label:<30} {str(va):>12} {str(vb):>12}")
    print()


def try_compare_with_other(metrics: dict, instance: str, output_dir: str = ".") -> None:
    """Se esiste results_X.json per l'altra istanza, carica e confronta automaticamente."""
    other = "B" if instance == "A" else "A"
    other_path = os.path.join(output_dir, f"results_{other}.json")
    if not os.path.exists(other_path):
        return
    with open(other_path) as f:
        other_metrics = json.load(f)
    print(f"\n--- Confronto automatico A vs B ---")
    if instance == "A":
        compare(metrics, other_metrics)
    else:
        compare(other_metrics, metrics)
