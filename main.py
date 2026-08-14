# ============================================
# NY LOTTO (6/59) ENSEMBLE: Markov + Viterbi(HMM) + Sum-Transitions + RF + NN + Filters
# Data source: https://data.ny.gov/resource/6nbc-h7bj.json
# ============================================

import math
import random
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict, Counter
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ----------------------------
# Configuration
# ----------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

N_MAX = 59
PICKS = 6

SOCRATA_ENDPOINT = "https://data.ny.gov/resource/6nbc-h7bj.json"
FETCH_LIMIT = 5000        # dataset ~2543 rows currently; 5000 is safe
USE_LAST_N_DRAWS = None   # e.g., 1000 to restrict; None = use all
INCLUDE_BONUS_AS_FEATURE = True

# Ensemble weights (tune these)
W_MARKOV = 2.0
W_HMM    = 1.5
W_SUM    = 1.0
W_RF     = 2.0
W_NN     = 1.5
W_FREQ   = 0.5

# Heatmap settings
SAVE_HEATMAP = True
HEATMAP_FILE = "transition_heatmap.png"

# Output log
PREDICTIONS_FILE = "predictions.txt"

# ----------------------------
# Helpers: parsing + filters
# ----------------------------
def parse_winning_numbers(s: str):
    # Expected like: "02 24 26 36 43 55"
    parts = str(s).strip().split()
    nums = [int(p) for p in parts if p.strip().isdigit()]
    return nums

def decade_bucket(n):
    # "tenth" (1-9, 10-19, ..., 50-59)
    return (n - 1) // 10

def first_digit(n):
    return int(str(n)[0])

def last_digit(n):
    return n % 10

def count_consecutive_pairs(sorted_nums):
    # count adjacent consecutive pairs in sorted list
    c = 0
    for a, b in zip(sorted_nums, sorted_nums[1:]):
        if b == a + 1:
            c += 1
    return c

def max_consecutive_run(sorted_nums):
    # length of longest consecutive run in sorted list
    best = 1
    cur = 1
    for a, b in zip(sorted_nums, sorted_nums[1:]):
        if b == a + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def within_10_number_range(sorted_nums):
    # True if all 6 numbers fit within any 10-number window (max-min <= 9)
    return (sorted_nums[-1] - sorted_nums[0]) <= 9

def all_in_same_decade(sorted_nums):
    # all 6 in same decade bucket
    buckets = {decade_bucket(n) for n in sorted_nums}
    return len(buckets) == 1

def first_last_gap_is(sorted_nums, gaps={7,8,9}):
    return (sorted_nums[-1] - sorted_nums[0]) in gaps

def too_many_same_last_digit(sorted_nums, k=5):
    counts = Counter(last_digit(n) for n in sorted_nums)
    return max(counts.values()) >= k

def too_many_same_first_digit(sorted_nums, k=5):
    counts = Counter(first_digit(n) for n in sorted_nums)
    return max(counts.values()) >= k

def passes_user_filters(nums):
    """
    Filters remembered from prior work:
    1) Remove combos with 5 consecutive numbers.
    2) Remove combos where all 6 numbers fall within the same tenth (decade).
    3) Remove combos where  all numbers are within a 10-number range.
    4) Remove combos with 4 consecutive numbers.
    5) Remove combos where first and last number are exactly 7,8,9 apart.
    6) Remove combos where 5+ numbers share same last digit.
    7) Remove combos where 5+ numbers share same first digit.
    """
    nums = sorted(nums)
    run = max_consecutive_run(nums)
    if run >= 5:
        return False
    if all_in_same_decade(nums):
        return False
    if within_10_number_range(nums):
        return False
    if run >= 4:
        return False
    if first_last_gap_is(nums, {7, 8, 9}):
        return False
    if too_many_same_last_digit(nums, k=5):
        return False
    if too_many_same_first_digit(nums, k=5):
        return False
    return True

# ----------------------------
# Data loading (Socrata)
# ----------------------------
def fetch_ny_lotto(limit=5000):
    """
    Pulls rows from Socrata endpoint.
    Note: Socrata supports parameters like $limit, $order.
    """
    params = {
        "$limit": limit,
        "$order": "draw_date ASC"
    }
    r = requests.get(SOCRATA_ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows)

    # Normalize columns we need
    if "draw_date" not in df.columns or "winning_numbers" not in df.columns:
        raise ValueError(f"Unexpected schema. Columns found: {list(df.columns)}")

    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df = df.dropna(subset=["draw_date", "winning_numbers"]).copy()
    df["nums"] = df["winning_numbers"].apply(parse_winning_numbers)

    # Keep only valid 6-number rows
    df = df[df["nums"].apply(lambda x: isinstance(x, list) and len(x) == PICKS)].copy()

    # Bonus / extra (optional)
    if "bonus" in df.columns:
        df["bonus"] = pd.to_numeric(df["bonus"], errors="coerce")
    else:
        df["bonus"] = np.nan

    if "extra" in df.columns:
        df["extra"] = pd.to_numeric(df["extra"], errors="coerce")
    else:
        df["extra"] = np.nan

    df = df.sort_values("draw_date").reset_index(drop=True)
    return df

# ----------------------------
# Markov transitions (number-to-number) on flattened stream
# ----------------------------
def calculate_transitions_flattened(draws, step=1):
    transitions = defaultdict(Counter)
    stream = [n for d in draws for n in d]
    for i in range(len(stream) - step):
        a = int(stream[i])
        b = int(stream[i + step])
        transitions[a][b] += 1

    # normalize
    probs = {}
    for a, cnts in transitions.items():
        total = sum(cnts.values())
        probs[a] = {b: c / total for b, c in cnts.items()}
    return probs

def markov_candidate_scores(transitions, last_draw):
    """
    Given last draw, produce a score over next numbers by aggregating
    transition probabilities from each last number.
    """
    scores = Counter()
    for a in last_draw:
        if a in transitions:
            for b, p in transitions[a].items():
                scores[b] += p
    return scores

# ----------------------------
# Sum-transition analysis + key-lag discovery
# ----------------------------
def compute_sums(draws):
    return np.array([sum(d) for d in draws], dtype=float)

def discover_key_lags(sums, max_lag=30, top_k=5):
    """
    Simple autocorrelation-based lag discovery.
    Returns lags with highest absolute correlation, excluding lag 0.
    """
    if len(sums) < (max_lag + 5):
        max_lag = max(5, len(sums) // 4)

    s = (sums - sums.mean()) / (sums.std() + 1e-9)
    scores = []
    for lag in range(1, max_lag + 1):
        a = s[:-lag]
        b = s[lag:]
        corr = float(np.corrcoef(a, b)[0, 1])
        scores.append((lag, abs(corr), corr))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def sum_transition_candidates(draws, top_n=20):
    """
    Uses (sum_t -> sum_{t+1}) delta distribution to propose likely next sums,
    then maps those sums to number preferences using historical co-occurrence
    conditioned on similar sums.
    """
    sums = compute_sums(draws)
    deltas = sums[1:] - sums[:-1]
    delta_counts = Counter(int(round(d)) for d in deltas)

    # Most common deltas
    common = [d for d, _ in delta_counts.most_common(10)]
    last_sum = int(round(sums[-1]))
    candidate_sums = [last_sum + d for d in common]

    # Conditioned number frequency: draws whose sum is close to candidate sum
    all_scores = Counter()
    for target_sum in candidate_sums:
        # window: within +/- 3 of target
        idx = [i for i, s in enumerate(sums) if abs(int(round(s)) - target_sum) <= 3]
        if not idx:
            continue
        freq = Counter()
        for i in idx:
            freq.update(draws[i])
        # add as soft scores
        for n, c in freq.items():
            all_scores[n] += c

    # Keep top_n scored numbers
    return Counter(dict(all_scores.most_common(top_n)))

# ----------------------------
# Discrete HMM over sum bins + Viterbi
# ----------------------------
def make_sum_bins(sums, n_bins=12):
    # Quantile bins for stability
    edges = np.quantile(sums, np.linspace(0, 1, n_bins + 1))
    # Ensure strictly increasing
    edges = np.unique(edges)
    if len(edges) < 4:
        # fallback fixed bins
        edges = np.linspace(sums.min(), sums.max(), 8)
    return edges

def sum_to_bin(s, edges):
    # returns bin index in [0, len(edges)-2]
    return int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2))

def estimate_hmm_params(obs_bins, n_states=None, smoothing=1e-3):
    """
    A simple discrete HMM where hidden states are same as observed bins
    (or optionally fewer). For practicality here, we set hidden states = bins.
    Emission ~ identity with smoothing; transitions estimated from sequences.
    """
    n_obs = int(max(obs_bins)) + 1
    n_states = n_states or n_obs

    # If n_states != n_obs, we would need clustering; keep it equal for now.
    if n_states != n_obs:
        n_states = n_obs

    # Start probs
    pi = np.full(n_states, smoothing, dtype=float)
    pi[obs_bins[0]] += 1.0
    pi /= pi.sum()

    # Transition
    A = np.full((n_states, n_states), smoothing, dtype=float)
    for a, b in zip(obs_bins[:-1], obs_bins[1:]):
        A[a, b] += 1.0
    A /= A.sum(axis=1, keepdims=True)

    # Emission (identity-ish)
    B = np.full((n_states, n_obs), smoothing, dtype=float)
    for s in range(n_states):
        if s < n_obs:
            B[s, s] += 1.0
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B

def viterbi_decode(obs, pi, A, B):
    """
    obs: list/array of observed symbols (ints)
    returns: most likely hidden path (ints)
    """
    T = len(obs)
    N = len(pi)
    logpi = np.log(pi + 1e-12)
    logA  = np.log(A + 1e-12)
    logB  = np.log(B + 1e-12)

    dp = np.full((T, N), -np.inf)
    bp = np.zeros((T, N), dtype=int)

    dp[0] = logpi + logB[:, obs[0]]

    for t in range(1, T):
        for j in range(N):
            vals = dp[t - 1] + logA[:, j]
            bp[t, j] = int(np.argmax(vals))
            dp[t, j] = vals[bp[t, j]] + logB[j, obs[t]]

    last = int(np.argmax(dp[T - 1]))
    path = [last]
    for t in range(T - 1, 0, -1):
        last = bp[t, last]
        path.append(int(last))
    path.reverse()
    return path

def hmm_candidate_scores(draws, n_bins=12, top_n=20):
    sums = compute_sums(draws)
    edges = make_sum_bins(sums, n_bins=n_bins)
    obs_bins = [sum_to_bin(s, edges) for s in sums]

    pi, A, B = estimate_hmm_params(obs_bins)
    path = viterbi_decode(obs_bins, pi, A, B)

    # Predict next hidden state as argmax transition from last state
    last_state = path[-1]
    next_state = int(np.argmax(A[last_state]))

    # Map that state/bin to number frequencies among draws whose obs bin == next_state
    freq = Counter()
    for d, b in zip(draws, obs_bins):
        if b == next_state:
            freq.update(d)

    return Counter(dict(freq.most_common(top_n)))

# ----------------------------
# ML feature engineering (draw-level)
# ----------------------------
def draw_features(draw, bonus=np.nan):
    nums = sorted(draw)
    s = sum(nums)
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    gap_mean = float(np.mean(gaps)) if gaps else 0.0
    gap_std  = float(np.std(gaps)) if gaps else 0.0
    consec   = count_consecutive_pairs(nums)
    odd      = sum(1 for n in nums if n % 2 == 1)
    low      = sum(1 for n in nums if n <= 29)
    high     = PICKS - low

    feat = [
        # normalized numbers
        *[n / N_MAX for n in nums],
        # aggregate stats (scaled)
        s / (N_MAX * PICKS),
        gap_mean / N_MAX,
        gap_std / N_MAX,
        consec / 5.0,
        odd / PICKS,
        low / PICKS,
        high / PICKS,
    ]
    if INCLUDE_BONUS_AS_FEATURE:
        feat.append((0.0 if pd.isna(bonus) else float(bonus)) / N_MAX)
    return np.array(feat, dtype=float)

def build_supervised_dataset(df):
    """
    Predict next draw numbers (as 6 regression targets).
    X_t uses features from draw_t; y_t is draw_{t+1}.
    """
    X = []
    y = []
    for i in range(len(df) - 1):
        X.append(draw_features(df.loc[i, "nums"], df.loc[i, "bonus"]))
        y.append(sorted(df.loc[i + 1, "nums"]))
    return np.vstack(X), np.vstack(y)

# ----------------------------
# Models: RF + NN
# ----------------------------
if HAS_TORCH:
    class LotteryNN(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, PICKS)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return torch.clamp(x, 1, N_MAX)

def train_rf_nn(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # RF regressor
    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=RNG_SEED,
        n_jobs=-1,
        max_depth=None
    )
    rf.fit(Xs, y)

    nnm = None
    if HAS_TORCH:
        X_t = torch.tensor(Xs, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        nnm = LotteryNN(in_dim=X.shape[1])
        opt = optim.Adam(nnm.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        nnm.train()
        for epoch in range(400):
            opt.zero_grad()
            pred = nnm(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

    return scaler, rf, nnm

def predict_rf_nn(last_draw, last_bonus, scaler, rf, nnm):
    feat = draw_features(last_draw, last_bonus).reshape(1, -1)
    feat_s = scaler.transform(feat)

    rf_pred = rf.predict(feat_s)[0]
    rf_pred = np.round(rf_pred).astype(int)

    nn_pred = None
    if HAS_TORCH and nnm is not None:
        nnm.eval()
        with torch.no_grad():
            nn_pred = nnm(torch.tensor(feat_s, dtype=torch.float32)).numpy()[0]
        nn_pred = np.round(nn_pred).astype(int)

    return rf_pred, nn_pred

# ----------------------------
# Ensemble voting -> final 6 numbers with filters
# ----------------------------
def sanitize_to_1_59_unique(arr):
    arr = [int(x) for x in arr]
    arr = [min(max(1, x), N_MAX) for x in arr]
    arr = list(set(arr))
    while len(arr) < PICKS:
        k = random.randint(1, N_MAX)
        if k not in arr:
            arr.append(k)
    arr = sorted(arr)[:PICKS]
    return arr

def weighted_vote(score_counters):
    """
    score_counters: list of (Counter, weight)
    returns: Counter of combined scores
    """
    combined = Counter()
    for c, w in score_counters:
        for k, v in c.items():
            combined[k] += w * float(v)
    return combined

def propose_ticket_from_scores(scores, attempts=2000):
    """
    Build candidate tickets from top scored numbers, then apply filters.
    Returns best passing candidate by total score.
    """
    ranked = [n for n, _ in scores.most_common(30)]
    best = None
    best_score = -1

    for _ in range(attempts):
        # sample biased toward top
        candidate = set()
        while len(candidate) < PICKS:
            # pick from top list with bias
            idx = int(np.clip(np.random.exponential(scale=6), 0, len(ranked) - 1))
            candidate.add(ranked[idx])
        cand = sorted(candidate)

        if not passes_user_filters(cand):
            continue

        s = sum(scores[n] for n in cand)
        if s > best_score:
            best_score = s
            best = cand

    # Fallback: just take top-6 even if filters reject (should be rare with enough attempts)
    if best is None:
        best = sorted([n for n, _ in scores.most_common(PICKS)])
    return best, best_score

# ----------------------------
# Visualization
# ----------------------------
def save_transition_heatmap(transitions, out_file):
    # 59x59 matrix of probabilities
    mat = np.zeros((N_MAX, N_MAX), dtype=float)
    for a, mp in transitions.items():
        for b, p in mp.items():
            if 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                mat[a - 1, b - 1] = p

    plt.figure(figsize=(12, 10))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title("NY Lotto (6/59) Markov Transition Probabilities (Flattened Stream)")
    plt.xlabel("Next Number")
    plt.ylabel("Previous Number")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def run_prediction():
    """
    Run the full prediction pipeline.
    Returns a dict with all results, or raises an exception on error.
    """
    df = fetch_ny_lotto(limit=FETCH_LIMIT)
    if USE_LAST_N_DRAWS is not None and USE_LAST_N_DRAWS > 50:
        df = df.tail(USE_LAST_N_DRAWS).reset_index(drop=True)

    draws = df["nums"].tolist()
    last_draw = draws[-1]
    last_bonus = df.loc[len(df) - 1, "bonus"]
    last_date = str(df.loc[len(df)-1, "draw_date"].date())

    # Markov
    transitions = calculate_transitions_flattened(draws, step=1)
    markov_scores = markov_candidate_scores(transitions, last_draw)

    # Sum model + key lags
    sums = compute_sums(draws)
    key_lags = discover_key_lags(sums, max_lag=30, top_k=5)
    sum_scores = sum_transition_candidates(draws, top_n=25)

    # HMM + Viterbi on sums
    hmm_scores = hmm_candidate_scores(draws, n_bins=12, top_n=25)

    # Frequency baseline
    freq_scores = Counter()
    for d in draws:
        freq_scores.update(d)

    # ML
    X, y = build_supervised_dataset(df)
    scaler, rf, nnm = train_rf_nn(X, y)
    rf_pred, nn_pred = predict_rf_nn(last_draw, last_bonus, scaler, rf, nnm)

    rf_pred = sanitize_to_1_59_unique(rf_pred)

    ensemble_inputs = [
        (markov_scores, W_MARKOV),
        (hmm_scores,    W_HMM),
        (sum_scores,    W_SUM),
        (Counter({n: 1.0 for n in rf_pred}), W_RF),
        (freq_scores,   W_FREQ),
    ]

    nn_pred_clean = None
    if nn_pred is not None:
        nn_pred_clean = sanitize_to_1_59_unique(nn_pred)
        ensemble_inputs.append((Counter({n: 1.0 for n in nn_pred_clean}), W_NN))

    combined_scores = weighted_vote(ensemble_inputs)
    final_ticket, final_score = propose_ticket_from_scores(combined_scores, attempts=3000)

    # Save heatmap
    heatmap_path = HEATMAP_FILE if SAVE_HEATMAP else None
    if SAVE_HEATMAP:
        save_transition_heatmap(transitions, HEATMAP_FILE)

    # Save log
    with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Last draw date: {last_date}\n")
        f.write(f"Last draw: {sorted(last_draw)} | Bonus: {last_bonus}\n\n")
        f.write("Key lags (lag, |corr|, corr):\n")
        for lag, absc, corr in key_lags:
            f.write(f"  lag={lag:2d}  |corr|={absc:.4f}  corr={corr:.4f}\n")
        f.write("\n")
        f.write(f"Markov top-12: { [n for n,_ in markov_scores.most_common(12)] }\n")
        f.write(f"HMM(top sum-bin) top-12: { [n for n,_ in hmm_scores.most_common(12)] }\n")
        f.write(f"Sum-transition top-12: { [n for n,_ in sum_scores.most_common(12)] }\n")
        f.write(f"RF prediction: {rf_pred}\n")
        f.write(f"NN prediction: {nn_pred_clean}\n\n")
        f.write(f"FINAL (ensemble+filters): {final_ticket}  | score={final_score:.4f}\n")

    return {
        "total_draws": len(draws),
        "last_date": last_date,
        "last_draw": sorted(last_draw),
        "last_bonus": None if pd.isna(last_bonus) else int(last_bonus),
        "markov_top12": [n for n, _ in markov_scores.most_common(12)],
        "hmm_top12": [n for n, _ in hmm_scores.most_common(12)],
        "sum_top12": [n for n, _ in sum_scores.most_common(12)],
        "rf_pred": rf_pred,
        "nn_pred": nn_pred_clean,
        "has_torch": HAS_TORCH,
        "key_lags": [(int(lag), round(absc, 4), round(corr, 4)) for lag, absc, corr in key_lags],
        "final_ticket": final_ticket,
        "final_score": round(final_score, 4),
        "heatmap_path": heatmap_path,
    }


def main():
    result = run_prediction()
    print(f"\nLoaded draws: {result['total_draws']}")
    print(f"Most recent draw date: {result['last_date']}")
    print(f"Last draw: {result['last_draw']} | Bonus: {result['last_bonus']}")
    print("\n--- Model Outputs ---")
    print("RF predicted:", result["rf_pred"])
    if result["nn_pred"]:
        print("NN predicted:", result["nn_pred"])
    print("Markov top-12:", result["markov_top12"])
    print("HMM top-12:", result["hmm_top12"])
    print("Sum-model top-12:", result["sum_top12"])
    print("\n=== FINAL (ensemble + filters) ===")
    print(result["final_ticket"])
    if result["heatmap_path"]:
        print(f"\nSaved heatmap to: {result['heatmap_path']}")
    print(f"Saved log to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()