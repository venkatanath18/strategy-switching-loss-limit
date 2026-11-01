"""Value iteration for switching trading strategies under a loss limit.

The state is (s, z): s is how crowded the strategy is, z is the budget left,
i.e. how much more the account may lose from its peak. C keeps the strategy,
R switches to a fresh one. The crowding-bump case has a risky middle stage,
which gives two switching regions in some budget columns.
"""

import numpy as np


# ============================================================
# PARAMETERS
# ============================================================

TOTAL_STAGES = 5
LOSS_LIMIT = 10

# Monthly PnL in stage s: +WIN with the case's win probability, otherwise
# -loss[s]. Stage 1 is a fresh strategy, stage 5 a dead one.
WIN = 2

# Name -> (win probability by stage, loss size by stage, K).
# Crowding bump: stage 3 is crowded (it usually wins but can lose 4), and
# stage 4 is calmer once the crowd has left.
CASES = {
    "TOY": ([0.90, 0.75, 0.60, 0.40, 0.20], [1, 2, 3, 3, 3], 4.0),
    "CROWDING BUMP": ([0.90, 0.80, 5 / 6, 0.60, 0.30], [1, 2, 4, 1, 2], 6.0),
}

# Each month a strategy stays in its stage or becomes one stage more crowded.
STAY_PROBABILITY = 0.8

DISCOUNT_FACTOR = 0.95
TOLERANCE = 1e-9
MAX_ITERATIONS = 10000


# ============================================================
# MODEL
# Arrays are zero-based: row 0 is stage 1, column z is budget z.
# ============================================================


def build_model(win_probability, loss, replacement_cost):
    win_probability = np.array(win_probability)
    loss = np.array(loss)

    # h(s): cost of one month at stage s, a win costs -WIN and a loss +loss[s].
    # Rounding removes float noise such as 0.6 * 2 - 0.4 * 3.
    average_pnl = np.round(win_probability * WIN - (1 - win_probability) * loss, 6) + 0.0

    probabilities = np.zeros((TOTAL_STAGES, TOTAL_STAGES))
    for stage in range(TOTAL_STAGES - 1):
        probabilities[stage, stage] = STAY_PROBABILITY
        probabilities[stage, stage + 1] = 1 - STAY_PROBABILITY
    probabilities[-1, -1] = 1.0

    # Budget after a winning or a losing month: z' = min(L, z + PnL).
    budgets = np.arange(LOSS_LIMIT + 1)
    after_win = np.minimum(LOSS_LIMIT, budgets + WIN)
    after_loss = budgets[None, :] - loss[:, None]

    # C is allowed only if a losing month still leaves z >= 0.
    # z = 0 means the account is closed, so nothing is allowed there.
    feasible = (after_loss >= 0) & (budgets[None, :] > 0)

    return {
        "win_probability": win_probability,
        "loss": loss,
        "replacement_cost": replacement_cost,
        "costs": -average_pnl,
        "probabilities": probabilities,
        "after_win": after_win,
        "after_loss": np.maximum(after_loss, 0),
        "feasible": feasible,
    }


def baseline(model):
    # No loss limit: the state is just s, like the machine baseline.
    costs, probabilities = model["costs"], model["probabilities"]
    values = np.zeros(TOTAL_STAGES)

    for _ in range(MAX_ITERATIONS):
        cont = costs + DISCOUNT_FACTOR * probabilities @ values
        replace = model["replacement_cost"] + cont[0]
        new_values = np.minimum(replace, cont)
        change = np.abs(new_values - values).max()
        values = new_values
        if change < TOLERANCE:
            break
    else:
        raise RuntimeError("Baseline value iteration has not converged")

    cont = costs + DISCOUNT_FACTOR * probabilities @ values
    replace = model["replacement_cost"] + cont[0]
    return values, replace <= cont


def action_costs(values, model):
    # expected[s, z] = sum_j P(j|s) V(j, z)
    expected = model["probabilities"] @ values

    # C: cost h(s) now; a win or a loss moves z, and P moves the stage.
    p = model["win_probability"][:, None]
    future = (p * expected[:, model["after_win"]]
              + (1 - p) * np.take_along_axis(expected, model["after_loss"], axis=1))
    cont = model["costs"][:, None] + DISCOUNT_FACTOR * future
    cont = np.where(model["feasible"], cont, np.inf)

    # R: pay K and run a fresh strategy at stage 1 from the start of this
    # month, so R = K + a C month at stage 1 (the machine's K + h(1)).
    replace = np.broadcast_to(model["replacement_cost"] + cont[0], values.shape)

    return replace, cont


def value_iteration(model):
    values = np.zeros((TOTAL_STAGES, LOSS_LIMIT + 1))

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = np.minimum(*action_costs(values, model))
        new_values[:, 0] = 0.0  # closed: no more profit or cost
        change = np.abs(new_values - values).max()
        values = new_values
        if change < TOLERANCE:
            break
    else:
        raise RuntimeError("Value iteration has not converged")

    replace, cont = action_costs(values, model)
    return values, replace <= cont, iteration


# ============================================================
# SWITCH REGIONS
# ============================================================


def replacement_regions(replace_column, feasible_column):
    # Each region is [first, last, forced stages, chosen stages].
    regions = []

    for index in range(TOTAL_STAGES):
        if not replace_column[index]:
            continue

        stage = index + 1
        if not regions or regions[-1][1] != stage - 1:
            regions.append([stage, stage, 0, 0])

        regions[-1][1] = stage
        if feasible_column[index]:
            regions[-1][3] += 1
        else:
            regions[-1][2] += 1

    return regions


def describe(regions):
    if not regions:
        return "never switch"

    parts = []
    for first, last, forced, chosen in regions:
        if chosen == 0:
            kind = "forced"
        elif forced == 0:
            kind = "chosen"
        else:
            kind = f"{chosen} chosen + {forced} forced"
        parts.append(f"{first}-{last} {kind}")

    return "R at " + ", ".join(parts)


# ============================================================
# CONSOLE OUTPUT
# ============================================================


def action_letter(z, replace, feasible):
    if z == 0:
        return "X"
    if not replace:
        return "C"
    return "R" if feasible else "R*"


def print_policy_table(model, replace_policy):
    costs, feasible = model["costs"], model["feasible"]

    header = " | ".join(f"z={z:2d}" for z in range(LOSS_LIMIT + 1))
    print(f"stage | avg PnL | worst loss | {header}")
    print("-" * (31 + len(header)))

    for index in range(TOTAL_STAGES):
        actions = " | ".join(
            f"{action_letter(z, replace_policy[index, z], feasible[index, z]):^4s}"
            for z in range(LOSS_LIMIT + 1)
        )
        print(f"{index + 1:5d} | {-costs[index]:+7.1f} | "
              f"{model['loss'][index]:10d} | {actions}")


def print_regions(replace_policy, feasible):
    print("\nSwitch regions (forced = a losing month would break the "
          "limit, chosen = R is cheaper than allowed C):")

    lines = ["account closed"] + [
        describe(replacement_regions(replace_policy[:, z], feasible[:, z]))
        for z in range(1, LOSS_LIMIT + 1)
    ]

    # Group consecutive budgets with the same switch regions.
    start = 0
    for z in range(1, LOSS_LIMIT + 2):
        if z == LOSS_LIMIT + 1 or lines[z] != lines[start]:
            budgets = f"z={start}" if z - 1 == start else f"z={start}..{z - 1}"
            print(f"{budgets:9s} | {lines[start]}")
            start = z


def print_case(name, model):
    costs = model["costs"]
    base_values, base_policy = baseline(model)
    values, replace_policy, iteration = value_iteration(model)

    print("\n" + "=" * 90)
    print(f"{name}: K = {model['replacement_cost']}")
    print("=" * 90)

    print("BASELINE (no loss limit, state = s)")
    print("stage | avg PnL | action")
    for index in range(TOTAL_STAGES):
        action = "R" if base_policy[index] else "C"
        print(f"{index + 1:5d} | {-costs[index]:+7.1f} | {action}")
    print(f"V(1) = {base_values[0]:.4f}")

    print(f"\nLOSS LIMIT L = {LOSS_LIMIT}: state = (s, z)")
    print_policy_table(model, replace_policy)
    print_regions(replace_policy, model["feasible"])
    print(f"V(1, z=1) = {values[0, 1]:.4f}   "
          f"V(1, z={LOSS_LIMIT}) = {values[0, LOSS_LIMIT]:.4f}   "
          f"Iterations: {iteration}")


def main():
    print("C = keep the strategy, R = switch by choice, R* = switch forced "
          "(a losing month would break the limit), X = account closed")
    print(f"Stages = {TOTAL_STAGES}, loss limit L = {LOSS_LIMIT}, "
          f"discount = {DISCOUNT_FACTOR}")

    for name, (win_probability, loss, replacement_cost) in CASES.items():
        print_case(name, build_model(win_probability, loss, replacement_cost))


if __name__ == "__main__":
    main()
