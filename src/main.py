import argparse
import json
import os
from simulation import Simulation
from visualization import Visualizer
from analysis import (
    clear_experiments_directory,
    run_analysis,
    run_output_dir,
    try_compare_with_other,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm simulation")
    parser.add_argument("instance", choices=["A", "B"], help="Istanza da usare (A o B)")
    parser.add_argument("--ticks", type=int, default=500, help="Numero massimo di tick (default: 500)")
    parser.add_argument("--headless", action="store_true", help="Esegui senza interfaccia grafica")
    parser.add_argument("--show-ground-truth", action="store_true", help="Mostra la posizione reale degli oggetti (solo in modalità grafica)")
    parser.add_argument("--config", choices=["exploration", "collection", "with_relay"], default="with_relay", help="Configurazione strategica degli agenti (default: with_relay)")
    parser.add_argument(
        "--reset-experiments",
        action="store_true",
        help="Svuota experiments/ prima del run (non usare dentro run_all: il batch svuota già all'inizio).",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiments_dir = os.path.join(base_dir, "experiments")
    if args.reset_experiments:
        clear_experiments_directory(experiments_dir)
        print(f"[main] experiments/ svuotata: {experiments_dir}")
    run_dir = run_output_dir(experiments_dir, args.instance, args.config)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(base_dir, "json_grids", f"{args.instance}.json")
    sim = Simulation(json_path, max_ticks=args.ticks, config=args.config)
    if args.headless:
        sim.run()
        log = sim.log
    else:
        sim._config_label = args.config.replace("_", " ")
        viz = Visualizer(
            sim.env,
            sim.agents,
            max_ticks=args.ticks,
            show_ground_truth=args.show_ground_truth,
            restart_path=json_path,
            restart_config=args.config,
        )
        log = viz.run_simulation(sim)
        sim = viz.sim

    log_path = os.path.join(run_dir, "simulation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Tick totali: {sim.tick}")
    print(f"Oggetti consegnati: {10 - sim.env.objects_remaining()}/10")
    print(f"Energia media rimanente: {sum(a.battery for a in sim.agents) / len(sim.agents):.1f}")
    print(f"Output salvato in: {run_dir}")

    metrics = run_analysis(log, args.instance, output_dir=run_dir)
    try_compare_with_other(
        metrics, args.instance, experiments_dir=experiments_dir, config=args.config
    )