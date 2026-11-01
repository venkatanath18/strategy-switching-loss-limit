"""Q-learning for the trading models.

Three versions, each further from value iteration:
1. Toy cases: simulated months from the known win, loss and stage-move chances.
2. Real data, one month at a time: each update draws one real month (next
   stage, PnL) of a strategy in the right stage. The learner never sees the
   counted table, so it should reproduce value iteration from samples alone.
3. Real history: an agent walks through the actual months, holding a real
   strategy, and switching means taking a real stage-1 strategy of that month.
   Its policy is then judged on the real history: in sample, and on 2008
   onwards after learning only from months up to 2007.
"""

import random

import numpy as np

import real_data as rd
import value_iteration as toy
from checks import SPLIT_MONTH, model_from, stages_and_pnl


# ============================================================
# Q-LEARNING PARAMETERS
# ============================================================

SEED = 42
STEPS_PER_EPISODE = 100
MAX_EPISODES = 100000
CHECK_EVERY = 2000

# Ties are left out of the agreement, so every clear cell must agree.
MINIMUM_VISITS = 200
MINIMUM_POLICY_AGREEMENT = 1.0
MAXIMUM_POLICY_CHANGE = 0.02
REQUIRED_PASSES = 3

ALPHA_EXPONENT = 0.8

# Cells where value iteration's C and R costs differ by less than this are
# ties: either action is fine, so they are left out of the agreement.
TIE_TOLERANCE = 0.02

# Real history: episodes start at a random month, strategy and budget, and
# last at most HISTORY_MONTHS months.
HISTORY_EPISODES = 100000
HISTORY_MONTHS = 60
EXPLORATION = 0.2
EVALUATION_EPISODES = 50000

REPLACE = 0
CONTINUE = 1
STAGES = range(rd.TOTAL_STAGES)
BUDGETS = range(1, rd.LOSS_LIMIT + 1)


# ============================================================
# ONE MONTH
# Stages are zero-based here: 0 is stage 1.
# ============================================================


def toy_month_sampler(model):
    p_win = model["win_probability"]
    loss = model["loss"]

    def sample(stage, rng):
        pnl = toy.WIN if rng.random() < p_win[stage] else -int(loss[stage])
        if stage < rd.TOTAL_STAGES - 1 and rng.random() >= toy.STAY_PROBABILITY:
            return stage + 1, pnl
        return stage, pnl

    return sample


def data_month_sampler(now, after, pnl):
    # One real month of a strategy that was in the given stage.
    months = [list(zip(after[now == s].tolist(), pnl[now == s].tolist()))
              for s in STAGES]

    def sample(stage, rng):
        return rng.choice(months[stage])

    return sample


def step(budget, action, stage, sample, replacement_cost, rng):
    # R pays K and runs a fresh stage-1 strategy this month.
    month_stage = 0 if action == REPLACE else stage
    next_stage, pnl = sample(month_stage, rng)
    cost = -pnl + (replacement_cost if action == REPLACE else 0.0)
    next_budget = min(rd.LOSS_LIMIT, budget + pnl)
    return next_stage, next_budget, cost


# ============================================================
# REAL HISTORY
# ============================================================


def history_tables(lines, now):
    # For each line: its month, and the same strategy's line one month later
    # (-1 if there is none). For each month: the lines of stage-1 strategies.
    strategy = lines[:, 0].astype(int).tolist()
    month = lines[:, 1].astype(int).tolist()
    index = {(k, m): i for i, (k, m) in enumerate(zip(strategy, month))}
    next_line = [index.get((k, m + 1), -1) for k, m in zip(strategy, month)]

    hot = {}
    for i in np.flatnonzero(now == 0).tolist():
        hot.setdefault(month[i], []).append(i)

    return month, next_line, hot


def history_month(line, action, month, hot, rng):
    # The line whose next month is actually lived: the same strategy for C,
    # a random stage-1 strategy of that month for R. A few early months have
    # no stage-1 strategy, and then the trader keeps the current one.
    if action == REPLACE and month[line] in hot:
        return rng.choice(hot[month[line]]), REPLACE
    return line, CONTINUE


# ============================================================
# POLICY CHECKS
# ============================================================


def get_policy(q_values):
    policy = np.zeros((rd.TOTAL_STAGES, rd.LOSS_LIMIT + 1), dtype=bool)
    for s in STAGES:
        for z in BUDGETS:
            policy[s, z] = q_values[s][z][REPLACE] <= q_values[s][z][CONTINUE]
    return policy


def policy_agreement(policy, exact_policy, clear):
    return np.mean(policy[:, 1:][clear[:, 1:]] == exact_policy[:, 1:][clear[:, 1:]])


def policy_change(policy, old_policy):
    if old_policy is None:
        return 1.0
    return np.mean(policy[:, 1:] != old_policy[:, 1:])


def minimum_visits(visits, feasible):
    return min(
        visits[s][z][a]
        for s in STAGES for z in BUDGETS for a in (REPLACE, CONTINUE)
        if a == REPLACE or feasible(s, z)
    )


def decision_margins(cont, replace):
    # Cost of C minus cost of R in value iteration. The closed column z = 0
    # has both costs infinite; it is never compared.
    with np.errstate(invalid="ignore"):
        return cont - replace


def clear_cells(margins):
    return np.abs(margins) >= TIE_TOLERANCE


# ============================================================
# Q-LEARNING
# ============================================================


def new_tables(feasible):
    q_values = [[[0.0, 0.0] for _ in range(rd.LOSS_LIMIT + 1)] for _ in STAGES]
    visits = [[[0, 0] for _ in range(rd.LOSS_LIMIT + 1)] for _ in STAGES]

    # An infeasible CONTINUE action is never selected.
    for s in STAGES:
        for z in BUDGETS:
            if not feasible(s, z):
                q_values[s][z][CONTINUE] = float("inf")

    return q_values, visits


def update(q_values, visits, stage, budget, action, next_stage, next_budget,
           cost, discount):
    visits[stage][budget][action] += 1
    alpha = visits[stage][budget][action] ** (-ALPHA_EXPONENT)

    # Landing at budget 0 or below closes the account: no future.
    target = cost
    if next_budget > 0:
        target += discount * min(q_values[next_stage][next_budget])

    q_values[stage][budget][action] += alpha * (target - q_values[stage][budget][action])


def q_learning(name, sample, feasible, exact_policy, clear,
               replacement_cost, discount):
    # Each update picks a random stage, budget and action.
    rng = random.Random(SEED)
    q_values, visits = new_tables(feasible)
    previous_policy = None
    consecutive_passes = 0

    for episode in range(1, MAX_EPISODES + 1):
        for _ in range(STEPS_PER_EPISODE):
            stage = rng.randrange(rd.TOTAL_STAGES)
            budget = rng.randint(1, rd.LOSS_LIMIT)
            if feasible(stage, budget):
                action = rng.choice([REPLACE, CONTINUE])
            else:
                action = REPLACE

            next_stage, next_budget, cost = step(
                budget, action, stage, sample, replacement_cost, rng)
            update(q_values, visits, stage, budget, action,
                   next_stage, next_budget, cost, discount)

        if episode % CHECK_EVERY == 0:
            policy = get_policy(q_values)
            agreement = policy_agreement(policy, exact_policy, clear)
            change = policy_change(policy, previous_policy)
            passed = (
                minimum_visits(visits, feasible) >= MINIMUM_VISITS
                and agreement >= MINIMUM_POLICY_AGREEMENT
                and change <= MAXIMUM_POLICY_CHANGE
            )
            consecutive_passes = consecutive_passes + 1 if passed else 0

            print(f"{name:34s} episode={episode:6d}  agreement={agreement:7.2%}  "
                  f"change={change:7.2%}  passes={consecutive_passes}")

            previous_policy = policy
            if consecutive_passes == REQUIRED_PASSES:
                return policy, episode

    return get_policy(q_values), MAX_EPISODES


def history_q_learning(name, lines, now, after, pnl, exact_policy, clear,
                       replacement_cost, discount):
    # An agent walks through real months with exploration EXPLORATION.
    rng = random.Random(SEED)
    q_values, visits = new_tables(lambda *_: True)
    month, next_line, hot = history_tables(lines, now)
    now, after, pnl = now.tolist(), after.tolist(), pnl.tolist()
    previous_policy = None
    steps = 0

    for episode in range(1, HISTORY_EPISODES + 1):
        line = rng.randrange(len(now))
        budget = rng.randint(1, rd.LOSS_LIMIT)

        for _ in range(HISTORY_MONTHS):
            stage = now[line]
            row = q_values[stage][budget]
            if rng.random() < EXPLORATION:
                wanted = rng.choice([REPLACE, CONTINUE])
            else:
                wanted = REPLACE if row[REPLACE] <= row[CONTINUE] else CONTINUE

            used, action = history_month(line, wanted, month, hot, rng)
            cost = -pnl[used] + (replacement_cost if action == REPLACE else 0.0)
            next_budget = min(rd.LOSS_LIMIT, budget + pnl[used])
            update(q_values, visits, stage, budget, action,
                   after[used], next_budget, cost, discount)
            steps += 1

            line = next_line[used]
            budget = next_budget
            if budget <= 0 or line < 0:
                break  # account closed, or the data ends

        if episode % (HISTORY_EPISODES // 10) == 0:
            policy = get_policy(q_values)
            print(f"{name:34s} episode={episode:6d}  "
                  f"agreement={policy_agreement(policy, exact_policy, clear):7.2%}  "
                  f"change={policy_change(policy, previous_policy):7.2%}  "
                  f"months={steps}")
            previous_policy = policy

    return get_policy(q_values), visits


def evaluate_on_history(policy, lines, now, pnl, replacement_cost, discount):
    # Discounted profit of following a fixed C/R table through real months,
    # one value per episode. The starts come from their own generator, so
    # every policy is judged from exactly the same starts.
    starts = random.Random(SEED + 1)
    picks = random.Random(SEED + 2)
    month, next_line, hot = history_tables(lines, now)
    now, pnl = now.tolist(), pnl.tolist()
    profits = np.zeros(EVALUATION_EPISODES)

    for episode in range(EVALUATION_EPISODES):
        line = starts.randrange(len(now))
        budget = starts.randint(1, rd.LOSS_LIMIT)
        weight = 1.0

        for _ in range(HISTORY_MONTHS):
            wanted = REPLACE if policy[now[line], budget] else CONTINUE
            used, action = history_month(line, wanted, month, hot, picks)
            paid = replacement_cost if action == REPLACE else 0.0
            profits[episode] += weight * (pnl[used] - paid)
            weight *= discount

            budget = min(rd.LOSS_LIMIT, budget + pnl[used])
            line = next_line[used]
            if budget <= 0 or line < 0:
                break

    return profits


# ============================================================
# OUTPUT
# ============================================================


def print_comparison(name, policy, exact_policy, clear, margins, forced=None):
    print(f"\n{name}: Q-learning policy, with value iteration where they differ")
    header = " | ".join(f"z={z:2d}" for z in BUDGETS)
    print(f"stage | {header}")
    for s in STAGES:
        cells = []
        for z in BUDGETS:
            letter = "R" if policy[s, z] else "C"
            if forced is not None and not forced(s, z):
                letter = "R*"
            if policy[s, z] != exact_policy[s, z]:
                exact = "R" if exact_policy[s, z] else "C"
                letter += f"({exact}{'' if clear[s, z] else ',tie'})"
            cells.append(f"{letter:^4s}")
        print(f"{s + 1:5d} | " + " | ".join(cells))
    print(f"Agreement on clear cells: {policy_agreement(policy, exact_policy, clear):.2%}"
          f"   ties left out: {int(np.sum(~clear[:, 1:]))}")

    # How close value iteration's own call was in each differing cell.
    for s in STAGES:
        for z in BUDGETS:
            if policy[s, z] != exact_policy[s, z]:
                better = "R" if exact_policy[s, z] else "C"
                print(f"  stage {s + 1}, z={z}: value iteration prefers {better} "
                      f"by {abs(margins[s, z]):.3f}")


def print_evaluation(title, rows):
    # rows: (label, profits per episode); the first row is the reference.
    print(f"\n{title}")
    print(f"Average discounted profit over at most {HISTORY_MONTHS} months from "
          f"{EVALUATION_EPISODES} random starts, and the difference from the first row:")
    reference = rows[0][1]
    for label, profits in rows:
        difference = profits - reference
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        print(f"  {label:38s} {profits.mean():7.3f}   "
              f"{difference.mean():+7.3f} +/- {error:.3f}")


def compare_on_history(title, policies, lines, now, pnl, replacement_cost, discount):
    print_evaluation(title, [
        (label, evaluate_on_history(policy, lines, now, pnl, replacement_cost, discount))
        for label, policy in policies
    ])


def main():
    print("C = keep, R = switch, R* = switch forced; X(Y) means Q-learning "
          "chose X where value iteration chose Y")
    print(f"Seed = {SEED}, tie tolerance = {TIE_TOLERANCE}")

    for name, (win_probability, loss, replacement_cost) in toy.CASES.items():
        model = toy.build_model(win_probability, loss, replacement_cost)
        values, exact_policy, _ = toy.value_iteration(model)
        replace, cont = toy.action_costs(values, model)
        margins = decision_margins(cont, replace)
        clear = clear_cells(margins)

        def feasible(stage, budget, model=model):
            return bool(model["feasible"][stage, budget])

        print("\n" + "=" * 90)
        print(f"1. TOY, {name}: simulated months")
        print("=" * 90)
        policy, episodes = q_learning(
            name, toy_month_sampler(model), feasible, exact_policy, clear,
            replacement_cost, toy.DISCOUNT_FACTOR)
        print(f"Episodes: {episodes}")
        print_comparison(name, policy, exact_policy, clear, margins,
                         forced=feasible)

    lines = rd.build_lines(rd.load_series("osap"), rd.TARGET_VOLATILITY)
    now, after, pnl = stages_and_pnl(lines, rd.stage_cutoffs(lines[:, 2]))
    model = model_from(now, after, pnl)

    # Out of sample: stage boundaries and learning from months up to 2007 only.
    early = lines[:, 1] + 1 < SPLIT_MONTH
    late = ~early
    split_now, split_after, split_pnl = stages_and_pnl(
        lines, rd.stage_cutoffs(lines[early, 2]))
    early_model = model_from(split_now[early], split_after[early], split_pnl[early])

    for name, (replacement_cost, discount) in rd.CASES.items():
        values, exact_policy, _ = rd.value_iteration(model, replacement_cost, discount)
        replace, cont = rd.action_costs(values, model, replacement_cost, discount)
        margins = decision_margins(cont, replace)
        clear = clear_cells(margins)
        base = rd.baseline(model, replacement_cost, discount)[1]
        never = np.zeros(exact_policy.shape, dtype=bool)

        print("\n" + "=" * 90)
        print(f"2. REAL DATA, {name}: one sampled month at a time")
        print("=" * 90)
        policy, episodes = q_learning(
            name, data_month_sampler(now, after, pnl), lambda *_: True,
            exact_policy, clear, replacement_cost, discount)
        print(f"Episodes: {episodes}")
        print_comparison(name, policy, exact_policy, clear, margins)

        print("\n" + "=" * 90)
        print(f"3. REAL HISTORY, {name}: walking through the actual months")
        print("=" * 90)
        history_policy, _ = history_q_learning(
            name, lines, now, after, pnl, exact_policy, clear,
            replacement_cost, discount)
        print_comparison(name, history_policy, exact_policy, clear, margins)

        compare_on_history(
            "Judged on the real history (in sample):",
            [("value iteration (counted table)", exact_policy),
             ("Q-learning on real history", history_policy),
             ("ignore the limit", np.broadcast_to(base[:, None], never.shape)),
             ("never switch", never)],
            lines, now, pnl, replacement_cost, discount)

        early_values, early_policy, _ = rd.value_iteration(
            early_model, replacement_cost, discount)
        early_replace, early_cont = rd.action_costs(
            early_values, early_model, replacement_cost, discount)
        early_clear = clear_cells(decision_margins(early_cont, early_replace))
        early_base = rd.baseline(early_model, replacement_cost, discount)[1]

        print(f"\nLearning from months up to 2007 only ({name}):")
        early_history_policy, _ = history_q_learning(
            name, lines[early], split_now[early], split_after[early],
            split_pnl[early], early_policy, early_clear, replacement_cost, discount)

        compare_on_history(
            "Learned on months up to 2007, judged on 2008 onwards:",
            [("value iteration (table up to 2007)", early_policy),
             ("Q-learning on history up to 2007", early_history_policy),
             ("ignore the limit (up to 2007)",
              np.broadcast_to(early_base[:, None], never.shape)),
             ("never switch", never)],
            lines[late], split_now[late], split_pnl[late],
            replacement_cost, discount)


if __name__ == "__main__":
    main()
