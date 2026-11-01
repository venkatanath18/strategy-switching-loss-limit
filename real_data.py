"""Value iteration on real strategy returns.

Two datasets: seven classic strategies from Kenneth French's data library
(1963 onwards), and the published strategies of the Open Source Asset Pricing
project (only months after each paper's sample ends, i.e. what a trader could
have traded). A strategy's stage is its average return over the last 12
months, cut into five equal groups (stage 1 = best record). Stage moves and
monthly PnL are counted from the data, not assumed.
"""

import csv
import io
import os
import urllib.request
import zipfile

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# French file name -> strategies used from that file.
FRENCH_FILES = {
    "F-F_Research_Data_5_Factors_2x3": ["SMB", "HML", "RMW", "CMA"],
    "F-F_Momentum_Factor": ["Mom"],
    "F-F_ST_Reversal_Factor": ["ST_Rev"],
    "F-F_LT_Reversal_Factor": ["LT_Rev"],
}
FRENCH_FIRST_MONTH = 196307

# Saved by fetch_osap.py.
OSAP_FOLDER = os.path.join(DATA_FOLDER, "osap")

DATASETS = ["french", "osap"]

# Every strategy is run at a size where a typical month moves the account by
# TARGET_VOLATILITY percent. PnL is rounded to whole percent.
TARGET_VOLATILITY = 2.0
TRACK_RECORD_MONTHS = 12
MIN_MONTHS = 36
TOTAL_STAGES = 5

LOSS_LIMIT = 10
TOLERANCE = 1e-9
MAX_ITERATIONS = 100000

# Name -> (K, discount) for the policy tables.
CASES = {
    "CHEAP SWITCHING": (0.5, 0.99),
    "EXPENSIVE SWITCHING": (2.0, 0.95),
}

# Grid for the budget-effect sweep at the end.
SWEEP_VOLATILITIES = [2.0, 3.0]
SWEEP_DISCOUNTS = [0.95, 0.99]
SWEEP_COSTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


# ============================================================
# DATA
# Each strategy is (name, month numbers, monthly returns in %).
# ============================================================


def month_number(year, month):
    return year * 12 + month - 1


def download_french():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for name in FRENCH_FILES:
        path = os.path.join(DATA_FOLDER, name + ".csv")
        if os.path.exists(path):
            continue
        print(f"Downloading {name} ...")
        with urllib.request.urlopen(FRENCH_URL + name + "_CSV.zip") as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                with open(path, "wb") as f:
                    f.write(archive.read(archive.namelist()[0]))


def read_french(path):
    # Monthly block only: rows that start with a six-digit YYYYMM date.
    names, rows = None, {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split(",")]
            if names is None:
                if len(parts) > 1 and parts[0] == "" and parts[1]:
                    names = parts[1:]
                continue
            if len(parts[0]) == 6 and parts[0].isdigit():
                rows[int(parts[0])] = [float(value) for value in parts[1:]]
            elif rows:
                break
    return names, rows


def french_series():
    download_french()
    series = []
    for name, strategies in FRENCH_FILES.items():
        columns, rows = read_french(os.path.join(DATA_FOLDER, name + ".csv"))
        for strategy in strategies:
            k = columns.index(strategy)
            dates = sorted(d for d, r in rows.items()
                           if d >= FRENCH_FIRST_MONTH and r[k] > -99)
            months = np.array([month_number(d // 100, d % 100) for d in dates])
            returns = np.array([rows[d][k] for d in dates])
            series.append((strategy, months, returns))
    return series


def osap_series():
    doc_path = os.path.join(OSAP_FOLDER, "signal_doc.csv")
    with open(doc_path, newline="", encoding="utf-8") as f:
        sample_end = {row["Acronym"]: int(float(row["SampleEndYear"]))
                      for row in csv.DictReader(f)
                      if row["Cat.Signal"] == "Predictor" and row["SampleEndYear"]}

    data = {}
    with open(os.path.join(OSAP_FOLDER, "ls_returns.csv"), newline="") as f:
        for row in csv.DictReader(f):
            name, year = row["signalname"], int(row["date"][:4])
            if name in sample_end and year > sample_end[name] and row["ret"]:
                month = month_number(year, int(row["date"][5:7]))
                data.setdefault(name, []).append((month, float(row["ret"])))

    series = []
    for name, rows in sorted(data.items()):
        rows.sort()
        series.append((name, np.array([m for m, _ in rows]),
                       np.array([r for _, r in rows])))
    return series


def load_series(dataset):
    if dataset == "french":
        return french_series()
    if not os.path.exists(os.path.join(OSAP_FOLDER, "ls_returns.csv")):
        print("OSAP data missing: run fetch_osap.py first.")
        return None
    return osap_series()


# ============================================================
# ESTIMATION
# ============================================================


def build_lines(series, target_volatility):
    # One line per strategy and month t, as columns:
    # strategy number, month t, record at t, record at t + 1, PnL in t + 1.
    window = TRACK_RECORD_MONTHS
    lines = []

    for number, (_, months, returns) in enumerate(series):
        if len(returns) < MIN_MONTHS:
            continue
        scaled = returns * (target_volatility / returns.std())
        for t in range(window - 1, len(scaled) - 1):
            # Needs window consecutive months up to t, then month t + 1.
            if months[t + 1] - months[t - window + 1] != window:
                continue
            lines.append((number, months[t],
                          scaled[t - window + 1:t + 1].mean(),
                          scaled[t - window + 2:t + 2].mean(),
                          scaled[t + 1]))

    return np.array(lines)


def stage_cutoffs(records):
    # Five equal groups of track records.
    return np.quantile(records, np.linspace(0, 1, TOTAL_STAGES + 1)[1:-1])


def stage_of(records, cutoffs):
    # Zero-based stage: 0 is stage 1, the best record.
    return TOTAL_STAGES - 1 - np.digitize(records, cutoffs)


def count_lines(now, after, pnl):
    # Count (stage now, stage next month, PnL next month) together, so the
    # chance of each Bellman row is just how often it happened.
    pnl_values = np.arange(pnl.min(), pnl.max() + 1)
    counts = np.zeros((TOTAL_STAGES, TOTAL_STAGES, len(pnl_values)))
    np.add.at(counts, (now, after, pnl - pnl_values[0]), 1)
    return counts, pnl_values


def estimate(series, target_volatility):
    lines = build_lines(series, target_volatility)
    cutoffs = stage_cutoffs(lines[:, 2])
    counts, pnl_values = count_lines(stage_of(lines[:, 2], cutoffs),
                                     stage_of(lines[:, 3], cutoffs),
                                     np.rint(lines[:, 4]).astype(int))
    used = len(np.unique(lines[:, 0]))
    return counts, pnl_values, cutoffs, used


# ============================================================
# MODEL
# Arrays are zero-based: row 0 is stage 1, column z is budget z.
# ============================================================


def build_model(counts, pnl_values):
    # chances[s, s', x] = chance of (next stage s', PnL x) given stage s
    chances = counts / counts.sum(axis=(1, 2), keepdims=True)
    costs = -(chances.sum(axis=1) * pnl_values).sum(axis=1)
    probabilities = chances.sum(axis=2)

    # Budget after PnL x: z' = min(L, z + x). Landing at z' <= 0 closes the
    # account, so C is always allowed but a big loss ends everything.
    budgets = np.arange(LOSS_LIMIT + 1)
    landing = budgets[:, None] + pnl_values[None, :]
    alive = landing > 0
    landing = np.clip(landing, 0, LOSS_LIMIT)

    return chances, costs, probabilities, landing, alive


def baseline(model, replacement_cost, discount):
    # No loss limit: the state is just s, like the machine baseline.
    costs, probabilities = model[1], model[2]
    values = np.zeros(TOTAL_STAGES)

    for _ in range(MAX_ITERATIONS):
        cont = costs + discount * probabilities @ values
        new_values = np.minimum(replacement_cost + cont[0], cont)
        change = np.abs(new_values - values).max()
        values = new_values
        if change < TOLERANCE:
            break
    else:
        raise RuntimeError("Baseline value iteration has not converged")

    cont = costs + discount * probabilities @ values
    return values, replacement_cost + cont[0] <= cont


def action_costs(values, model, replacement_cost, discount):
    chances, costs, _, landing, alive = model

    # future[s, z] = sum over rows (s', x) of chance * V(s', z'), 0 if closed
    landed = values[:, landing] * alive[None, :, :]
    future = np.einsum("abx,bzx->az", chances, landed)
    cont = costs[:, None] + discount * future

    # R: pay K and run a stage-1 strategy this month.
    replace = np.broadcast_to(replacement_cost + cont[0], values.shape)

    return replace, cont


def value_iteration(model, replacement_cost, discount):
    values = np.zeros((TOTAL_STAGES, LOSS_LIMIT + 1))

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = np.minimum(
            *action_costs(values, model, replacement_cost, discount))
        new_values[:, 0] = 0.0  # closed: no more profit or cost
        change = np.abs(new_values - values).max()
        values = new_values
        if change < TOLERANCE:
            break
    else:
        raise RuntimeError("Value iteration has not converged")

    replace, cont = action_costs(values, model, replacement_cost, discount)
    return values, replace <= cont, iteration


def switch_from(replace_column):
    # First stage where R is chosen; TOTAL_STAGES + 1 means never switch.
    stages = np.flatnonzero(replace_column)
    return int(stages[0]) + 1 if len(stages) else TOTAL_STAGES + 1


# ============================================================
# CONSOLE OUTPUT
# ============================================================


def print_estimates(model, pnl_values, cutoffs):
    chances, costs, probabilities = model[:3]

    print(f"Stage cutoffs (average monthly return over the last "
          f"{TRACK_RECORD_MONTHS} months, %): {np.round(cutoffs[::-1], 2)}")

    print("\nStage moves, P(s, s') (rows = stage now, columns = next month):")
    print("stage | " + " ".join(f"{s:6d}" for s in range(1, TOTAL_STAGES + 1)))
    for s in range(TOTAL_STAGES):
        print(f"{s + 1:5d} | " + " ".join(f"{p:6.2f}" for p in probabilities[s]))

    print("\nMonthly PnL by stage (% of the account):")
    print("stage | average | h(s)  | loss >= 3 | 1-in-100 month | worst month")
    for s in range(TOTAL_STAGES):
        pnl = chances[s].sum(axis=0)
        one_in_100 = pnl_values[np.searchsorted(np.cumsum(pnl), 0.01)]
        worst = pnl_values[np.flatnonzero(pnl)[0]]
        loss_3 = pnl[pnl_values <= -3].sum()
        print(f"{s + 1:5d} | {-costs[s]:+7.2f} | {costs[s]:+5.2f} | "
              f"{loss_3:9.1%} | {one_in_100:14d} | {worst:11d}")


def print_policy_table(costs, replace_policy):
    header = " | ".join(f"z={z:2d}" for z in range(LOSS_LIMIT + 1))
    print(f"stage |  h(s) | {header}")
    print("-" * (16 + len(header)))
    for s in range(TOTAL_STAGES):
        actions = " | ".join(
            f"{('X' if z == 0 else 'R' if replace_policy[s, z] else 'C'):^4s}"
            for z in range(LOSS_LIMIT + 1)
        )
        print(f"{s + 1:5d} | {costs[s]:+5.2f} | {actions}")


def switch_regions(replace_column):
    # Each region is [first stage, last stage].
    regions = []
    for index in range(TOTAL_STAGES):
        if not replace_column[index]:
            continue
        stage = index + 1
        if regions and regions[-1][1] == stage - 1:
            regions[-1][1] = stage
        else:
            regions.append([stage, stage])
    return regions


def describe(regions):
    if not regions:
        return "never switch"
    return "R at " + ", ".join(
        f"{first}-{last}" if first != last else f"{first}"
        for first, last in regions
    )


def print_regions(replace_policy):
    print("\nSwitch regions:")
    lines = ["account closed"] + [
        describe(switch_regions(replace_policy[:, z]))
        for z in range(1, LOSS_LIMIT + 1)
    ]

    # Group consecutive budgets with the same switch regions.
    start = 0
    for z in range(1, LOSS_LIMIT + 2):
        if z == LOSS_LIMIT + 1 or lines[z] != lines[start]:
            budgets = f"z={start}" if z - 1 == start else f"z={start}..{z - 1}"
            print(f"{budgets:9s} | {lines[start]}")
            start = z


def print_case(name, model, replacement_cost, discount):
    costs = model[1]
    base_values, base_policy = baseline(model, replacement_cost, discount)
    values, replace_policy, iteration = value_iteration(
        model, replacement_cost, discount)

    print("\n" + "-" * 82)
    print(f"{name}: K = {replacement_cost}%, discount = {discount}")
    print("-" * 82)

    print("BASELINE (no loss limit)")
    print("stage |  h(s) | action")
    for s in range(TOTAL_STAGES):
        print(f"{s + 1:5d} | {costs[s]:+5.2f} | {'R' if base_policy[s] else 'C'}")
    print(f"V(1) = {base_values[0]:.4f}")

    print(f"\nLOSS LIMIT L = {LOSS_LIMIT} (C = keep, R = switch, X = closed)")
    print_policy_table(costs, replace_policy)
    print_regions(replace_policy)
    print(f"V(1, z=1) = {values[0, 1]:.4f}   "
          f"V(1, z={LOSS_LIMIT}) = {values[0, LOSS_LIMIT]:.4f}   "
          f"Iterations: {iteration}")


def print_budget_sweep(series):
    print("\nBudget effect: switch point at z = 1 compared with z = 10")
    print("(earlier = leave a strategy sooner when the budget is low)")
    print(f"{'size, discount':16s} | " + " ".join(f"K={k:<5}" for k in SWEEP_COSTS))

    for volatility in SWEEP_VOLATILITIES:
        counts, pnl_values, _, _ = estimate(series, volatility)
        model = build_model(counts, pnl_values)
        for discount in SWEEP_DISCOUNTS:
            words = []
            for cost in SWEEP_COSTS:
                _, policy, _ = value_iteration(model, cost, discount)
                low = switch_from(policy[:, 1])
                high = switch_from(policy[:, LOSS_LIMIT])
                words.append("earlier" if low < high else
                             "later" if low > high else "same")
            label = f"{volatility}%, {discount}"
            print(f"{label:16s} | " + " ".join(f"{w:7s}" for w in words))


def main():
    for dataset in DATASETS:
        series = load_series(dataset)
        if series is None:
            continue

        counts, pnl_values, cutoffs, used = estimate(series, TARGET_VOLATILITY)
        model = build_model(counts, pnl_values)

        print("\n" + "=" * 82)
        print(f"{dataset.upper()}: {used} strategies, "
              f"{int(counts.sum())} strategy-months, each run at a "
              f"{TARGET_VOLATILITY}% typical monthly move")
        print("=" * 82)
        print_estimates(model, pnl_values, cutoffs)

        for name, (replacement_cost, discount) in CASES.items():
            print_case(name, model, replacement_cost, discount)

        print_budget_sweep(series)


if __name__ == "__main__":
    main()
