"""run_all.py - Esegue tutte le combinazioni (istanza × config) in modalità headless
e genera i grafici di confronto aggregati.

All'avvio svuota experiments/ e riscrive layout fisso (nessun timestamp):
  experiments/<A|B>/<exploration|collection|with_relay>/{simulation_log.json,results.json,heatmap.png}
  experiments/comparison_{energy,objects}.png, subplots_{energy,objects}.png

Uso:
    python run_all.py                  # tutte le 6 combinazioni
    python run_all.py --ticks 300      # limite tick personalizzato
    python run_all.py --instances A    # solo mappa A
    python run_all.py --configs exploration with_relay   # config scelte
"""

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product

# Aggiunge src/ al path per importare analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from analysis import (
    AGG_COMPARISON_ENERGY,
    AGG_COMPARISON_OBJECTS,
    AGG_SUBPLOTS_ENERGY,
    AGG_SUBPLOTS_OBJECTS,
    build_energy_timeline,
    build_objects_timeline,
    clear_experiments_directory,
    plot_comparison_energy,
    plot_comparison_objects,
    plot_subplots_energy,
    plot_subplots_objects,
    run_output_dir,
)

INSTANCES = ["A", "B"]
CONFIGS   = ["exploration", "collection", "with_relay"]


def main():
    parser = argparse.ArgumentParser(description="Batch runner per tutte le simulazioni")
    parser.add_argument("--ticks",     type=int,    default=500,      help="Tick massimi per run (default: 500)")
    parser.add_argument("--instances", nargs="+",   choices=INSTANCES, default=INSTANCES,
                        help="Istanze da eseguire (default: A B)")
    parser.add_argument("--configs",   nargs="+",   choices=CONFIGS,   default=CONFIGS,
                        help="Configurazioni da eseguire (default: tutte e 3)")
    args = parser.parse_args()

    runs  = list(product(args.instances, args.configs))
    total = len(runs)

    base_dir        = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(base_dir, "experiments")

    clear_experiments_directory(experiments_dir)
    print(f"  [experiments] cartella svuotata: {experiments_dir}\n")

    print(f"╔══ Swarm batch runner ══════════════════════════════════╗")
    print(f"  Istanze   : {args.instances}")
    print(f"  Config    : {args.configs}")
    print(f"  Tick max  : {args.ticks}")
    print(f"  Run totali: {total}")
    print(f"╚════════════════════════════════════════════════════════╝\n")

    run_results = []
    batch_start = time.time()

    for idx, (instance, config) in enumerate(runs, 1):
        label = f"[{idx}/{total}] {instance} / {config}"
        print(f"{'─' * 58}")
        print(f"  ▶  {label}")
        print(f"{'─' * 58}")

        cmd = [
            sys.executable, "src/main.py",
            instance, "--headless",
            "--ticks", str(args.ticks),
            "--config", config,
        ]

        t0    = time.time()
        proc  = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0

        status = "✓ OK" if proc.returncode == 0 else f"✗ ERRORE (exit {proc.returncode})"
        run_results.append((label, status, elapsed))
        print(f"\n  {status}  -  {elapsed:.1f}s\n")

    total_elapsed = time.time() - batch_start

    print(f"\n{'═' * 58}")
    print(f"  RIEPILOGO  ({total_elapsed:.1f}s totali)")
    print(f"{'═' * 58}")
    for label, status, elapsed in run_results:
        print(f"  {status:<12}  {elapsed:>6.1f}s  -  {label}")
    print(f"{'═' * 58}\n")

    failures = [r for r in run_results if "ERRORE" in r[1]]
    if failures:
        print(f"  {len(failures)} run falliti. Grafici di confronto non generati.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Grafici di confronto aggregati
    # -----------------------------------------------------------------------
    print("  Generazione grafici di confronto aggregati...")

    energy_runs  = []
    objects_runs = []

    for instance, config in product(args.instances, args.configs):
        run_dir = run_output_dir(experiments_dir, instance, config)
        if not os.path.isdir(run_dir):
            print(f"  [warn] run non trovato: {instance}/{config} - saltato")
            continue
        log_path = os.path.join(run_dir, "simulation_log.json")
        if not os.path.exists(log_path):
            print(f"  [warn] log mancante in {run_dir} - saltato")
            continue
        with open(log_path) as f:
            log = json.load(f)

        label = f"{instance} – {config.replace('_', ' ')}"
        energy_runs.append({
            "label":    label,
            "instance": instance,
            "config":   config,
            "timeline": build_energy_timeline(log),
        })
        objects_runs.append({
            "label":    label,
            "instance": instance,
            "config":   config,
            "timeline": build_objects_timeline(log),
        })

    if energy_runs:
        out_energy       = os.path.join(experiments_dir, AGG_COMPARISON_ENERGY)
        out_objects      = os.path.join(experiments_dir, AGG_COMPARISON_OBJECTS)
        out_energy_grid  = os.path.join(experiments_dir, AGG_SUBPLOTS_ENERGY)
        out_objects_grid = os.path.join(experiments_dir, AGG_SUBPLOTS_OBJECTS)

        plot_comparison_energy(energy_runs,       out_energy)
        plot_comparison_objects(objects_runs,     out_objects)
        plot_subplots_energy(energy_runs,         out_energy_grid)
        plot_subplots_objects(objects_runs,       out_objects_grid)

        print(f"  Energia (overlay)  : {out_energy}")
        print(f"  Oggetti (overlay)  : {out_objects}")
        print(f"  Energia (subplots) : {out_energy_grid}")
        print(f"  Oggetti (subplots) : {out_objects_grid}")

    print(f"\n  Tutti i {total} run completati. Risultati in: experiments/\n")


if __name__ == "__main__":
    main()
