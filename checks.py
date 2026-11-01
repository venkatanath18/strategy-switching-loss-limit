"""Robustness checks for real_data.py.

1. Luck tests: does the stage really predict next month's PnL and how often a
   strategy loses money? Months are the independent units.
2. Resampling test: resample whole months of data, rebuild the counted table
   and record how often each C/R cell keeps its action.
3. Split test: build the model on months up to 2007 (stage boundaries
   included) and judge its policy on 2008 onwards.
4. Model assumptions: does the stage alone decide the future, and does what
   switching gets you depend on your own stage?
5. Other loss limits.
"""

import numpy as np

import real_data as rd


# ============================================================
# PARAMETERS
# ============================================================

SEED = 42
RESAMPLES = 200
SPLIT_MONTH = rd.month_number(2008, 1)  # first PnL month of the late period
OTHER_LIMITS = [6, 15]
STABLE = 0.90


# ============================================================
# HELPERS
# ============================================================


def stages_and_pnl(lines, cutoffs):
    now = rd.stage_of(lines[:, 2], cutoffs)
    after = rd.stage_of(lines[:, 3], cutoffs)
    pnl = np.rint(lines[:, 4]).astype(int)
    return now, after, pnl


def model_from(now, after, pnl):
    counts, pnl_values = rd.count_lines(now, after, pnl)
    return rd.build_model(counts, pnl_values)


def monthly_spread(months, groups, values, first, second):
    # For each calendar month: average of values in group `first` minus the
    # average in group `second`. Returns mean, t-value, share positive, months.
    spreads = []
    for month in np.unique(months):
        here = months == month
        a = values[here & (groups == first)]
        b = values[here & (groups == second)]
        if len(a) and len(b):
            spreads.append(a.mean() - b.mean())
    spreads = np.array(spreads)
    t_value = spreads.mean() / (spreads.std(ddof=1) / np.sqrt(len(spreads)))
    return spreads.mean(), t_value, np.mean(spreads > 0), len(spreads)


def policy_value(model, policy, replacement_cost, discount):
    # Value of following a fixed C/R table, using the same Bellman sums.
    values = np.zeros((rd.TOTAL_STAGES, rd.LOSS_LIMIT + 1))
    for _ in range(rd.MAX_ITERATIONS):
        replace, cont = rd.action_costs(values, model, replacement_cost, discount)
        new_values = np.where(policy, replace, cont)
        new_values[:, 0] = 0.0
        change = np.abs(new_values - values).max()
        values = new_values
        if change < rd.TOLERANCE:
            return values
    raise RuntimeError("Policy evaluation has not converged")


def region_count(policy):
    # Largest number of separate switching regions in any budget column.
    return max(
        int(np.sum(np.diff(np.concatenate([[0], policy[:, z].astype(int)])) == 1))
        for z in range(1, rd.LOSS_LIMIT + 1)
    )


# ============================================================
# CHECKS
# ============================================================


def luck_tests(dataset, lines):
    cutoffs = rd.stage_cutoffs(lines[:, 2])
    now, _, pnl = stages_and_pnl(lines, cutoffs)
    months = lines[:, 1]
    losing = (pnl <= -1).astype(float)

    print(f"\n{dataset.upper()}: next-month difference between two stages, "
          "month by month")
    print(f"{'comparison':34s} | difference | t-value | months positive")
    tests = [
        ("average PnL, stage 1 minus 5", lines[:, 4], 0, 4),
        ("average PnL, stage 1 minus 3", lines[:, 4], 0, 2),
        ("average PnL, stage 3 minus 5", lines[:, 4], 2, 4),
        ("chance of losing 1%, stage 5 minus 1", losing, 4, 0),
        ("chance of losing 1%, stage 5 minus 3", losing, 4, 2),
    ]
    for label, values, first, second in tests:
        mean, t_value, positive, count = monthly_spread(
            months, now, values, first, second)
        print(f"{label:34s} | {mean:+10.3f} | {t_value:7.1f} | "
              f"{positive:6.0%} of {count}")


def resampling_test(lines):
    rng = np.random.default_rng(SEED)
    cutoffs = rd.stage_cutoffs(lines[:, 2])
    now, after, pnl = stages_and_pnl(lines, cutoffs)

    months = lines[:, 1]
    order = np.argsort(months, kind="stable")
    unique, starts = np.unique(months[order], return_index=True)
    rows_of_month = np.split(order, starts[1:])

    model = model_from(now, after, pnl)
    full, same = {}, {}
    for name, (cost, discount) in rd.CASES.items():
        full[name] = (rd.value_iteration(model, cost, discount)[1],
                      rd.baseline(model, cost, discount)[1])
        same[name] = [np.zeros(full[name][0].shape), np.zeros(rd.TOTAL_STAGES)]

    for _ in range(RESAMPLES):
        picked = rng.integers(len(unique), size=len(unique))
        rows = np.concatenate([rows_of_month[i] for i in picked])
        model = model_from(now[rows], after[rows], pnl[rows])
        for name, (cost, discount) in rd.CASES.items():
            same[name][0] += rd.value_iteration(model, cost, discount)[1] == full[name][0]
            same[name][1] += rd.baseline(model, cost, discount)[1] == full[name][1]

    print(f"\nResampling test: {RESAMPLES} resamples of {len(unique)} months. "
          "Each cell: full-data action, then % of resamples that agree.")
    for name in rd.CASES:
        policy, base = full[name]
        share = same[name][0] / RESAMPLES
        share_base = same[name][1] / RESAMPLES
        header = " ".join(f"   z={z:<2d}" for z in range(1, rd.LOSS_LIMIT + 1))
        print(f"\n{name}")
        print(f"stage | no limit | {header}")
        for s in range(rd.TOTAL_STAGES):
            cells = " ".join(
                f"{'R' if policy[s, z] else 'C'} {share[s, z]:4.0%}"
                for z in range(1, rd.LOSS_LIMIT + 1)
            )
            print(f"{s + 1:5d} | {'R' if base[s] else 'C'} {share_base[s]:4.0%}   | {cells}")
        unstable = int(np.sum(share[:, 1:] < STABLE)) + int(np.sum(share_base < STABLE))
        print(f"Cells below {STABLE:.0%}: {unstable} of "
              f"{rd.TOTAL_STAGES * (rd.LOSS_LIMIT + 1)}")


def split_test(lines):
    early = lines[:, 1] + 1 < SPLIT_MONTH
    late = ~early

    # Stage boundaries from the early period only: no look-ahead.
    cutoffs = rd.stage_cutoffs(lines[early, 2])
    now, after, pnl = stages_and_pnl(lines, cutoffs)
    early_model = model_from(now[early], after[early], pnl[early])
    late_model = model_from(now[late], after[late], pnl[late])

    print(f"\nSplit test: model built on {int(early.sum())} lines up to 2007, "
          f"judged on {int(late.sum())} lines from 2008.")
    print("Profit of a fresh strategy with a full budget in the 2008 onwards "
          "model, -V(1, z=10):")

    for name, (cost, discount) in rd.CASES.items():
        early_policy = rd.value_iteration(early_model, cost, discount)[1]
        late_values, late_policy = rd.value_iteration(late_model, cost, discount)[:2]
        early_base = rd.baseline(early_model, cost, discount)[1]

        ignore_limit = np.broadcast_to(early_base[:, None], early_policy.shape)
        never_switch = np.zeros(early_policy.shape, dtype=bool)

        best = -late_values[0, rd.LOSS_LIMIT]
        rows = [
            ("best possible (built on 2008 on)", late_policy),
            ("built on months up to 2007", early_policy),
            ("up to 2007, ignoring the limit", ignore_limit),
            ("never switch", never_switch),
        ]
        agree = np.mean(early_policy[:, 1:] == late_policy[:, 1:])
        print(f"\n{name}: the two periods' policies agree on {agree:.0%} of cells")
        for label, policy in rows:
            profit = -policy_value(late_model, policy, cost, discount)[0, rd.LOSS_LIMIT]
            print(f"  {label:34s} {profit:7.3f}  ({profit / best:6.1%} of best)")


def model_assumptions(lines):
    # The 5-stage model assumes (a) the stage alone decides the future and
    # (b) what switching gets you does not depend on your own stage.
    cutoffs = rd.stage_cutoffs(lines[:, 2])
    now, after, pnl = stages_and_pnl(lines, cutoffs)
    counts, pnl_values = rd.count_lines(now, after, pnl)
    probabilities = counts.sum(axis=2) / counts.sum(axis=(1, 2))[:, None]
    average = (counts.sum(axis=1) / counts.sum(axis=(1, 2))[:, None] * pnl_values).sum(axis=1)

    strategy, month = lines[:, 0].astype(int), lines[:, 1].astype(int)
    index = {(k, m): i for i, (k, m) in enumerate(zip(strategy.tolist(), month.tolist()))}

    print("\n(a) Where strategies really are k months later, against the model's P^k:")
    for k in (6, 12):
        model = np.linalg.matrix_power(probabilities, k)
        for s in (0, rd.TOTAL_STAGES - 1):
            later = [index.get((strategy[i], month[i] + k))
                     for i in np.flatnonzero(now == s)]
            later = [now[j] for j in later if j is not None]
            actual = np.bincount(later, minlength=rd.TOTAL_STAGES) / len(later)
            print(f"  from stage {s + 1}, {k:2d} months later: "
                  f"real {np.round(actual, 2)}   model {np.round(model[s], 2)}")

    print("  Average PnL over the next 12 months:")
    for s in (0, rd.TOTAL_STAGES - 1):
        real = []
        for i in np.flatnonzero(now == s):
            path = [index.get((strategy[i], month[i] + j)) for j in range(12)]
            if None not in path:
                real.append(np.mean(lines[path, 4]))
        model = np.mean([np.linalg.matrix_power(probabilities, j)[s] @ average
                         for j in range(12)])
        print(f"  stage {s + 1}: real {np.mean(real):+.3f}   model {model:+.3f}")

    print("\n(b) Next month of a random stage-1 strategy, in the months when your "
          "own strategy is in each stage:")
    hot = {m: pnl[(month == m) & (now == 0)] for m in np.unique(month[now == 0])}
    for s in range(rd.TOTAL_STAGES):
        months = [m for m in month[now == s] if m in hot]
        print(f"  your stage {s + 1}: average {np.mean([hot[m].mean() for m in months]):+.3f}"
              f"   loses 1% or more {np.mean([np.mean(hot[m] <= -1) for m in months]):.1%}")
    print(f"  the model uses: average {pnl[now == 0].mean():+.3f}"
          f"   loses 1% or more {np.mean(pnl[now == 0] <= -1):.1%}")


def other_limits(lines):
    cutoffs = rd.stage_cutoffs(lines[:, 2])
    now, after, pnl = stages_and_pnl(lines, cutoffs)
    print("\nOther loss limits:")
    for limit in OTHER_LIMITS:
        rd.LOSS_LIMIT = limit  # the model code reads the limit from here
        model = model_from(now, after, pnl)
        for name, (cost, discount) in rd.CASES.items():
            policy = rd.value_iteration(model, cost, discount)[1]
            print(f"\nL = {limit}, {name} (most switching regions in a column: "
                  f"{region_count(policy)})")
            rd.print_regions(policy)
    rd.LOSS_LIMIT = 10


def main():
    for dataset in rd.DATASETS:
        series = rd.load_series(dataset)
        if series is not None:
            luck_tests(dataset, rd.build_lines(series, rd.TARGET_VOLATILITY))

    lines = rd.build_lines(rd.load_series("osap"), rd.TARGET_VOLATILITY)
    print("\n" + "=" * 82)
    print("OSAP CHECKS")
    print("=" * 82)
    resampling_test(lines)
    split_test(lines)
    model_assumptions(lines)
    other_limits(lines)


if __name__ == "__main__":
    main()
