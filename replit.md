# NY Lotto 6/59 Ensemble Predictor

## Overview
A NY Lotto (6/59) prediction and analysis tool that uses an ensemble of statistical and machine learning models to suggest lottery numbers based on historical draw data from the NY Open Data portal.

## Architecture
- **app.py** — Flask web server (port 5000), serves the web UI and runs predictions in background threads
- **main.py** — Core prediction logic: data fetching, all models, ensemble voting, heatmap generation

## Models Used
1. **Markov Chain** — Number-to-number transition probabilities from flattened draw stream
2. **Hidden Markov Model (HMM) + Viterbi** — Predicts next sum bin using discrete HMM
3. **Sum Transition Analysis** — Uses historical sum deltas and conditioned number frequencies
4. **Random Forest Regressor** — scikit-learn RF with 600 estimators
5. **Neural Network** — PyTorch feed-forward network (requires torch to be installed)
6. **Frequency Baseline** — Raw historical frequency of each number

## Tech Stack
- Python 3.10
- Flask (web server)
- pandas, numpy (data manipulation)
- scikit-learn (Random Forest)
- torch / PyTorch (Neural Network — optional, falls back gracefully if not installed)
- matplotlib (Markov transition heatmap, Agg backend)
- requests (Socrata API fetching)

## Running
The workflow command is:
```
LD_LIBRARY_PATH=$PYTHON_LD_LIBRARY_PATH MPLBACKEND=Agg PYTHONPATH='' python3 app.py
```

The `LD_LIBRARY_PATH` must be set to `$PYTHON_LD_LIBRARY_PATH` for numpy/scikit-learn C extensions to load properly (provides libstdc++.so.6 from nix).

## Virtual Environment
Located at `./venv/`. Rebuilt from `/nix/store/hd4cc9rh83j291r5539hkf6qd8lgiikb-python3-3.10.8/bin/python3`. The old venv was broken (pointed to a nonexistent Python path).

Key installed packages:
- pandas, numpy, scikit-learn, matplotlib, flask, requests, torch

## Data Source
NY Open Data Socrata API: `https://data.ny.gov/resource/6nbc-h7bj.json`

## Output Files
- `predictions.txt` — Appended log of all prediction runs
- `transition_heatmap.png` — Markov transition probability heatmap (generated per run)
