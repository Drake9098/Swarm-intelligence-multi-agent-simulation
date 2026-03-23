# Swarm intelligence - simulazione multi-agente

Simulazione su griglia in cui uno **sciame di agenti** esplora un ambiente parzialmente sconosciuto, **rileva oggetti**, li **raccoglie** e li **consegna nei magazzini**. La conoscenza è **locale** e si **fonde** solo quando gli agenti sono in **comunicazione** (rete a maglia con raggi diversi per ruolo).

## Requisiti

- **Python** 3.10 o superiore (consigliato 3.11+)
- Dipendenze in `requirements.txt` (interfaccia **PySide6**, **NumPy**, **Matplotlib** per analisi e heatmap)

## Installazione

Dalla radice del repository:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Su Linux/macOS: `source .venv/bin/activate`.

## Esecuzione

Esegui i comandi dalla **radice del progetto** (dove si trovano `run_all.py`, `json_grids/`, `src/`).

### Simulazione singola

```bash
python src/main.py A
```

- **Istanza**: `A` o `B` (mappa diversa in `json_grids/<istanza>.json`).
- **`--ticks N`**: limite massimo di tick (default 500).
- **`--headless`**: nessuna finestra grafica; salva solo log e metriche.
- **`--config`**: `exploration` | `collection` | `with_relay` (default `with_relay`).
- **`--show-ground-truth`**: in GUI, mostra le posizioni reali degli oggetti (utile per demo).
- **`--reset-experiments`**: svuota `experiments/` prima del run (non usare dentro `run_all.py`).

Esempi:

```bash
python src/main.py B --headless --ticks 300 --config collection
python src/main.py A --show-ground-truth
```

### Batch e grafici di confronto

```bash
python run_all.py
python run_all.py --ticks 300 --instances A --configs exploration with_relay
```

All’avvio `run_all.py` **svuota** `experiments/`, esegue tutte le combinazioni richieste (istanza × configurazione) in modalità headless e genera grafici aggregati (`comparison_*.png`, `subplots_*.png`).

## Configurazioni della squadra

| Config        | Idea                                                                                     |
| ------------- | ---------------------------------------------------------------------------------------- |
| `exploration` | Più **Scout**, meno **Collector** - enfasi sull’esplorazione.                            |
| `collection`  | Più **Collector** - enfasi sul recupero e sulla consegna.                                |
| `with_relay`  | Squadra con **Relay** che fa da ponte tra Scout e Collector sulla rete di comunicazione. |

Ruoli principali:

- **Scout**: esplora (anche con logica di dispersione tra pari); non raccoglie oggetti.
- **Collector**: pianifica fetch e consegna in magazzino; presidia zone vicino agli ingressi noti.
- **Relay**: si posiziona per migliorare la connettività e propagare le informazioni verso i Collector.

## Struttura del repository

```
json_grids/          # Mappe (griglia, magazzini, posizioni oggetti)
src/
  main.py            # CLI: run singolo + salvataggio log
  simulation.py      # Loop a tick, comunicazione mesh, merge della conoscenza
  environment.py     # Mondo “vero”: griglia, claim/deliver oggetti
  agent.py           # Agent, Scout, Collector, Relay
  pathfinding.py     # A* sulla mappa locale
  visualization.py   # GUI Qt (PySide6)
  analysis.py        # Metriche, heatmap, confronti tra run
run_all.py           # Batch e grafici aggregati
experiments/         # Output dei run (log JSON, risultati, figure)
```

## Output

Ogni run salva sotto `experiments/<A|B>/<config>/` tipicamente:

- `simulation_log.json` - serie temporale di snapshot (tick, agenti, oggetti rimanenti, …)
- `results.json` - metriche riassuntive
- figure generate dall’analisi (es. heatmap, se previste dal flusso)

I grafici aggregati del batch stanno in `experiments/comparison_energy.png`, `comparison_objects.png`, ecc.

## Note

- Il completamento è definito quando **tutti gli oggetti** sono stati consegnati nei magazzini (o si raggiunge il limite di tick / tutti gli agenti sono esauriti).
- Gli agenti consumano **batteria** a ogni movimento; sotto soglia entrano in modalità di **ritorno d’emergenza** verso un ingresso noto.
