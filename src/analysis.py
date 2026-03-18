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
    """Restituisce la somma totale di batteria residua per tick."""
    timeline = []
    for snapshot in log:
        batteries = [a["battery"] for a in snapshot["agents"]]
        timeline.append(sum(batteries))
    return timeline


def build_objects_timeline(log: list) -> list:
    """Restituisce gli oggetti consegnati cumulativi per tick."""
    return [snapshot["objects_delivered"] for snapshot in log]


# ---------------------------------------------------------------------------
# Salvataggio file
# ---------------------------------------------------------------------------

def save_results(metrics: dict, instance: str, output_dir: str = ".") -> str:
    """Scrive results.json in output_dir. Restituisce il percorso del file."""
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Grafici PNG
# ---------------------------------------------------------------------------

def plot_heatmap(heatmap: np.ndarray, instance: str, output_dir: str = ".") -> str:
    """Genera e salva heatmap.png. Restituisce il percorso del file."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Visite")
    ax.set_title(f"Heatmap frequenza di visita — Istanza {instance}")
    ax.set_xlabel("Colonna")
    ax.set_ylabel("Riga")
    path = os.path.join(output_dir, "heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Stile visivo condiviso tra i grafici di confronto
_CONFIG_COLORS = {
    "exploration": "tab:blue",
    "collection":  "tab:orange",
    "with_relay":  "tab:green",
}
_INSTANCE_STYLES = {"A": "-", "B": "--"}


def plot_comparison_energy(runs: list, output_path: str) -> str:
    """Grafico con una linea per run: batteria totale residua nel tempo.

    runs — lista di dict {"label": str, "instance": str, "config": str, "timeline": list}
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        color = _CONFIG_COLORS.get(run["config"], "tab:gray")
        ls    = _INSTANCE_STYLES.get(run["instance"], "-")
        ax.plot(run["timeline"], color=color, linestyle=ls,
                linewidth=1.8, label=run["label"])
    ax.set_title("Energia residua totale — confronto configurazioni")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Batteria totale residua")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_comparison_objects(runs: list, output_path: str) -> str:
    """Grafico con una linea per run: oggetti consegnati cumulativi nel tempo.

    runs — lista di dict {"label": str, "instance": str, "config": str, "timeline": list}
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        color = _CONFIG_COLORS.get(run["config"], "tab:gray")
        ls    = _INSTANCE_STYLES.get(run["instance"], "-")
        ax.plot(run["timeline"], color=color, linestyle=ls,
                linewidth=1.8, label=run["label"])
    ax.set_title("Oggetti consegnati nel tempo — confronto configurazioni")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Oggetti consegnati (cumulativo)")
    ax.set_yticks(range(0, TOTAL_OBJECTS + 1))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Grafici subplot 2×N (sharex + sharey)
# ---------------------------------------------------------------------------

def _make_subplots_grid(runs: list, suptitle: str, ylabel: str,
                        yticks: list | None, output_path: str) -> str:
    """Griglia nrows×ncols di subplot (righe = istanze, colonne = config) con sharex e sharey.

    runs — lista di dict {"label", "instance", "config", "timeline"}
    """
    instances = sorted({r["instance"] for r in runs})
    configs   = sorted(
        {r["config"] for r in runs},
        key=lambda c: ["exploration", "collection", "with_relay"].index(c),
    )
    nrows, ncols = len(instances), len(configs)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    run_by_key = {(r["instance"], r["config"]): r for r in runs}

    for row_i, instance in enumerate(instances):
        for col_i, config in enumerate(configs):
            ax  = axes[row_i][col_i]
            run = run_by_key.get((instance, config))
            if run is not None:
                color = _CONFIG_COLORS.get(config, "tab:gray")
                ax.plot(run["timeline"], color=color, linewidth=1.8)
            ax.set_title(f"{instance} – {config.replace('_', ' ')}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if yticks is not None:
                ax.set_yticks(yticks)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_i == nrows - 1:
                ax.set_xlabel("Tick", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_subplots_energy(runs: list, output_path: str) -> str:
    """Griglia subplot per la batteria totale residua (sharex + sharey)."""
    return _make_subplots_grid(
        runs,
        suptitle="Energia residua totale — subplot per configurazione",
        ylabel="Batteria totale residua",
        yticks=None,
        output_path=output_path,
    )


def plot_subplots_objects(runs: list, output_path: str) -> str:
    """Griglia subplot per gli oggetti consegnati (sharex + sharey)."""
    return _make_subplots_grid(
        runs,
        suptitle="Oggetti consegnati nel tempo — subplot per configurazione",
        ylabel="Oggetti consegnati",
        yticks=list(range(0, TOTAL_OBJECTS + 1)),
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Pipeline completa per una singola istanza
# ---------------------------------------------------------------------------

def run_analysis(log: list, instance: str, output_dir: str = ".",
                 grid_size: int = GRID_SIZE) -> dict:
    """Calcola metriche, salva results.json e heatmap.png nel run_dir.

    I grafici aggregati (energia e oggetti) sono generati da run_all.py
    dopo aver raccolto tutti i run. Restituisce il dizionario delle metriche.
    """
    metrics = compute_metrics(log)
    heatmap = build_heatmap(log, grid_size)

    json_path = save_results(metrics, instance, output_dir)
    hm_path   = plot_heatmap(heatmap, instance, output_dir)

    print(f"[analysis] Istanza {instance}: {json_path}, {hm_path}")
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


def try_compare_with_other(metrics: dict, instance: str, experiments_dir: str = ".") -> None:
    """Cerca il run più recente dell'altra istanza in experiments_dir e confronta le metriche."""
    other = "B" if instance == "A" else "A"
    other_instance_dir = os.path.join(experiments_dir, other)
    if not os.path.isdir(other_instance_dir):
        return
    run_folders = sorted(
        [d for d in os.listdir(other_instance_dir)
         if os.path.isdir(os.path.join(other_instance_dir, d))],
        reverse=True,
    )
    for folder in run_folders:
        candidate = os.path.join(other_instance_dir, folder, "results.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                other_metrics = json.load(f)
            print(f"\n--- Confronto automatico A vs B (run: {folder}) ---")
            if instance == "A":
                compare(metrics, other_metrics)
            else:
                compare(other_metrics, metrics)
            return
