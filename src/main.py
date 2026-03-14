import argparse
import json
import os
from simulation import Simulation
from visualization import Visualizer
from analysis import run_analysis, try_compare_with_other

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm simulation")
    parser.add_argument("instance", choices=["A", "B"], help="Istanza da usare (A o B)")
    parser.add_argument("--ticks", type=int, default=500, help="Numero massimo di tick (default: 500)")
    parser.add_argument("--headless", action="store_true", help="Esegui senza interfaccia grafica")
    parser.add_argument("--output", type=str, default=None, help="Percorso file JSON per il log (opzionale)")
    parser.add_argument("--show-ground-truth", action="store_true", help="Mostra la posizione reale degli oggetti (solo in modalità grafica)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json_grids", f"{args.instance}.json")
    sim = Simulation(json_path, max_ticks=args.ticks)
    if args.headless:
        sim.run()
        log = sim.log
    else:
        viz = Visualizer(sim.env, sim.agents, max_ticks=args.ticks, show_ground_truth=args.show_ground_truth)
        log = viz.run_simulation(sim)
    
    with open(args.output or f"simulation_log_{args.instance}.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Tick totali: {sim.tick}")
    print(f"Oggetti consegnati: {10 - sim.env.objects_remaining()}/10")
    print(f"Energia media rimanente: {sum(a.battery for a in sim.agents) / len(sim.agents):.1f}")

    metrics = run_analysis(log, args.instance, output_dir=base_dir)
    try_compare_with_other(metrics, args.instance, output_dir=base_dir)