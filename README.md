# Switching Trading Strategies under a Loss Limit

**When should a trader switch away from a trading strategy that is wearing
out, if the firm closes the account after a fixed loss?** This project models
the decision as a Markov decision process whose state includes the loss budget
left, solves it by value iteration, checks the solution with Q-learning, and
estimates it from 211 published long-short strategies.

## 1. The idea

A trading firm gives a trader an account to run one trading strategy.
Strategies wear out: once other traders find and copy a strategy, its profits
shrink. So every month the trader decides whether to keep the current strategy
or to pay a cost and switch to a fresh one. The firm also sets a loss limit:
if the account falls L below its best value so far, the account is closed.

This is the classic machine-replacement problem with the remaining loss budget
added to the state, the way Sauté RL (Sootla et al., 2022) adds a safety
budget to the state. The strategy plays the machine, switching plays
replacement, and the loss the trader is still allowed to take is the budget z.
Losing months use up the budget and winning months refill it, up to the limit.

## 2. Trading terms used below

| Term | Meaning |
|---|---|
| Strategy | A fixed rule that decides which stocks to buy and which to sell short. Each month it has one result: its return, in %. |
| Long / short | Buying (gains when the price rises) / selling borrowed shares and buying them back later (gains when the price falls). |
| Long-short strategy | Buys one group of stocks and shorts another, so it earns the difference between the two groups. Example, momentum: buy the past year's winners, short the past year's losers. |
| PnL | Profit and loss: how much the account gained (+) or lost (−) in a month, in % of the account. |
| Peak | The highest value the account has reached so far. |
| Loss from peak | Peak minus the current value (traders call it the drawdown). |
| Loss limit L | The firm closes the account once the loss from peak reaches L. We use L = 10% of the account. |

## 3. The model

| Part | In this model |
|---|---|
| State | (s, z). The stage s = 1..5 says how well the strategy did over the last 12 months (1 = best, 5 = worst). The budget z = L − loss from peak, 0..10, is how much more the account may lose. |
| Actions | C: keep trading the strategy next month. R: pay K and trade a fresh stage-1 strategy this month, so R = K + one stage-1 month. |
| Running cost | h(s) = −(average PnL in the month after a strategy is in stage s); a profit is a negative cost. |
| Budget update | z' = min(L, z + PnL): a losing month uses budget, a winning month refills it. Gains above the old peak only move the peak up, so z can never exceed L. |
| Switching | Keeps z: the limit is on the account, not on the strategy. |
| Closing | Once z reaches 0 the account is closed, with value 0 from then on. |

The running cost and the budget come from the same monthly PnL, so they cannot
be tuned separately: the average PnL sets h(s), and the actual PnL of the
month moves z.

## 4. Bellman equation

With x = next month's PnL:

```
Q_C(s, z) = h(s) + γ · Σ over (s', x): chance(s', x | s) · V(s', min(L, z + x))
Q_R(s, z) = K + Q_C(1, z)
V(s, z)   = min( Q_C(s, z), Q_R(s, z) ),   V = 0 once z ≤ 0 (account closed)
```

Next month's budget is random, because the month's PnL is random, so the sum
runs over pairs (next stage, PnL).

## 5. Toy model: `value_iteration.py`

Made-up numbers, to see the structure. Each month is a win (+2) or a loss:

| Stage | Chance of +2 | Otherwise | h(s) |
|---|---|---|---|
| 1 | 0.90 | −1 | −1.7 |
| 2 | 0.75 | −2 | −1.0 |
| 3 | 0.60 | −3 | 0 |
| 4 | 0.40 | −3 | +1.0 |
| 5 | 0.20 | −3 | +2.0 |

Each month a strategy stays in its stage (0.8) or moves one stage worse (0.2).
L = 10, K = 4, γ = 0.95. In the toy model, C is allowed only if a losing month
still leaves z ≥ 0; otherwise the switch is forced (R*).

Without a limit: keep stages 1–2, switch at 3–5. With the limit:

```
stage | z=1 | z=2 | z=3 | z=4..10
  1   |  C  |  C  |  C  |   C
  2   |  R* |  R  |  R  |   C
  3   |  R* |  R* |  R  |   R
  4   |  R* |  R* |  R  |   R
  5   |  R* |  R* |  R  |   R
```

- One switch point in every budget column.
- Forced switching wherever a loss could break the limit; this staircase
  shrinks as z grows.
- Chosen early switching at stage 2 for z = 2–3: keeping is allowed, but a
  loss could close the account.
- From z = 4 the policy equals the no-limit policy.
- If every month in a stage had a fixed PnL (+2, +1, 0, −1, −2), the limit
  would never change a decision. The budget matters only because good
  strategies also have bad months.
- The second case in the script, `CROWDING BUMP`, has a crowded middle stage
  (stage 3 usually wins but can lose 4, stage 4 only loses 1, K = 6). Its
  z = 3 column reads C, C, R*, C, R: two switching regions. At z = 4 the
  middle region is chosen rather than forced, and from z = 5 only stage 5
  switches.

## 6. Real-data model: `real_data.py`

### Data

- **Kenneth French data library**: 7 classic long-short strategies, monthly
  from July 1963 to September 2026: size (SMB), value (HML), profitability
  (RMW), investment (CMA), momentum (Mom), short-term reversal (ST_Rev) and
  long-term reversal (LT_Rev).
- **Open Source Asset Pricing** (Chen and Zimmermann): 212 strategies from
  published papers, rebuilt from stock data following each paper's rules,
  monthly to December 2025. Only months after each paper's data period are
  used, i.e. months a trader could actually have traded: 211 strategies and
  61,758 strategy-months. This is the main dataset.

Published strategies are known to earn less after their paper's sample ends
and after publication (McLean and Pontiff, 2016), which is why only the months
after each paper's data period are used.

### From data to model

1. **Size**: scale every strategy so that a typical month moves the account
   2%, which makes wild and calm strategies comparable. Round the PnL to whole
   % so that z stays a whole number. The scaling uses each strategy's
   volatility over its whole sample, which a trader would not have known in
   advance; the stage boundaries in the split test (section 7) use only past
   data.
2. **Stage**: each month, average the last 12 months' PnL. Sort all these
   averages and cut them into 5 equal groups. Boundaries: +0.68, +0.31, +0.03,
   −0.30 % per month.
3. **Lines**: for every strategy and month write (stage now, stage next month,
   PnL next month).
4. **Chances**: chance(s', x | s) = share of the lines starting in stage s
   that went to stage s' with PnL x. Stage moves and PnL are counted together
   because they are linked: from stage 3, a losing month leads to a worse
   stage 50% of the time, a winning month only 5%.
5. **Cost**: h(s) = −(average next-month PnL of the lines starting in
   stage s).
6. **Chosen, not from data**: L = 10%, K, γ and the 2% size.

The worst real months (−14% to −19% of the account) are bigger than the whole
limit, so the toy rule "C only if you survive the worst month" would forbid
all trading. Here C is always allowed, and a month that takes z to 0 or below
closes the account.

### Estimated model (211 strategies)

Stage moves:

```
from \ to   1     2     3     4     5
    1     0.82  0.16  0.01  0.00  0.00
    2     0.16  0.60  0.21  0.03  0.00
    3     0.02  0.21  0.55  0.21  0.02
    4     0.00  0.03  0.21  0.60  0.17
    5     0.00  0.00  0.02  0.16  0.81
```

| Stage | h(s) | Months losing 3% or more | 1-in-100 month | Worst month |
|---|---|---|---|---|
| 1 | −0.45 | 8.3% | −6 | −19 |
| 2 | −0.29 | 5.4% | −5 | −16 |
| 3 | −0.17 | 4.8% | −4 | −15 |
| 4 | −0.05 | 6.0% | −4 | −14 |
| 5 | +0.03 | 10.0% | −6 | −14 |

Is the stage effect luck? Comparing the stages month by month (months as the
independent units): stage 1 earns 0.60% more than stage 5 the next month
(t = 6.3, higher in 67% of months); stage 1 vs 3: +0.43% (t = 7.6); stage 3
vs 5: +0.21% (t = 3.3). In the 7-strategy French data only stage 3 vs 5
passes (t = 4.2), which is why the 211-strategy data is the main dataset.

The stage effect itself is **factor momentum**: strategies that did well over
the past year tend to keep doing well (Gupta and Kelly, 2019; Ehsani and
Linnainmaa, 2022). It is a known result, not a finding of this project. What
is studied here is the switching decision under a loss limit.

### Results (211 strategies)

Case 1, cheap switching (K = 0.5%, γ = 0.99):

```
stage | z=1 | z=2 | z=3..10 | no limit
  1   |  C  |  C  |    C    |    C
  2   |  R  |  R  |    C    |    C
  3   |  R  |  R  |    R    |    R
  4   |  R  |  R  |    R    |    R
  5   |  R  |  R  |    R    |    R
```

With little budget the trader leaves earlier (from stage 2): weaker strategies
lose money more often, any loss can close the account, and switching is
cheap.

Case 2, expensive switching (K = 2%, γ = 0.95):

```
stage | z=1 | z=2..3 | z=4..10 | no limit
  1   |  C  |   C    |    C    |    C
  2   |  C  |   C    |    C    |    C
  3   |  C  |   C    |    C    |    C
  4   |  C  |   C    |    R    |    R
  5   |  C  |   R    |    R    |    R
```

With little budget the trader holds on, and at z = 1 never switches. The
account is likely to close soon, so at z = 1 a fresh strategy is worth only
2.77; after paying K = 2 that leaves 0.77, less than the 0.84 from keeping even
a stage-5 strategy. It is like not buying a new machine for a factory that is
about to close.

In both cases the budget changes the decision only at small budgets; from
z = 3–4 the policy settles to the no-limit switch point. The limit is costly
over a long horizon: in case 1 a fresh strategy is worth 15.6 with a full
budget, against 36.8 with no limit.

Robustness over both datasets:

- Never more than one switching region in a budget column (40 settings
  checked: sizes 2–4%, K from 0.5% to 3%, γ = 0.95 and 0.99).
- As K rises, the low-budget effect moves from "switch earlier" to "switch
  later", never the reverse (64 settings: 2 datasets × 2 sizes × 2 discounts
  × 8 values of K).

## 7. Robustness checks: `checks.py`

### Luck tests

Months are the independent units: for each calendar month, compare the next
month of the strategies in two stages, then test the monthly differences.

| Comparison (211 strategies) | Difference | t-value |
|---|---|---|
| Average PnL, stage 1 minus stage 5 | +0.60% a month | 6.3 |
| Average PnL, stage 1 minus stage 3 | +0.43% | 7.6 |
| Average PnL, stage 3 minus stage 5 | +0.21% | 3.3 |
| Chance of losing 1% or more, stage 5 minus stage 1 | +14 points | 7.8 |
| Chance of losing 1% or more, stage 5 minus stage 3 | +7 points | 4.7 |

All pass. In the 7-strategy French data only the stage 3 vs 5 comparisons
pass (t = 4.2 for PnL, 4.0 for losses).

### Resampling test

Rebuild the counted table 200 times from resampled months (each month keeps
all its strategies, so strategies that move together stay together) and solve
again.

- Stage 1 (keep) and the coldest stages (switch) keep their action in about
  100% of resamples.
- The stage next to the switch point flips often: stage 2 in the cheap case
  (52–89%) and stage 4 in the expensive case (50–81%).
- So **the switch point is known only to within about one stage**. The
  low-budget shifts are supported in 81% of resamples (cheap: switch stage 2
  at z = 1; expensive: keep stage 4 at z = 1). "Never switch at z = 1" in the
  expensive case is weak: stage 5 keeps there in only 56% of resamples.

### Split test

Build the model from months up to 2007 only (stage boundaries included, so
nothing from the future leaks in), then judge its policy with the model of
2008 onwards. Profit of a fresh strategy with a full budget:

| Policy | Cheap switching | Expensive switching |
|---|---|---|
| Best possible (built on 2008 onwards) | 10.41 | 3.86 |
| Built on months up to 2007 | 97% of best | 69% of best |
| Built on months up to 2007, ignoring the limit | 85% | 69% |
| Never switch | 71% | 98% |

With cheap switching, the old policy carries over well and beats ignoring the
limit by 12 points. With expensive switching it does not: after 2008 the hot
strategies' advantage halved (stage 1 earned 0.72% a month up to 2007, 0.36%
after) and they lost money more often (25% of months, then 33%). Paying 2% to
switch stopped paying, and the best policy after 2008 almost never switches.

### Model assumptions

The 5-stage model assumes that the stage alone decides what happens next, and
that what switching gets you does not depend on your own stage.

- **The stage forgets memory.** Twelve months after being in stage 5, 19% of
  strategies are back in stage 1 (the model says 10%) and only 24% are still
  in stage 5 (the model says 31%). Over those 12 months a stage-5 strategy
  really earns +0.14% a month; the model says +0.05%. Hot strategies also fall
  further than the model says: 19% are in stage 5 a year later, against 10%.
  The likely reason is that the stage is a 12-month average, so it matters
  whether a strategy got there through one crash month or a long slide, and
  five stages cannot show that.
- **What switching gets you depends a little on timing.** A hot strategy
  picked while your own strategy is in stages 2–5 earns +0.39% to +0.42% the
  next month, against +0.45% in the model.

### Other loss limits

With L = 6 and L = 15 the patterns stay the same, with one switching region
per budget column. Cheap switching: switch from stage 2 at the lowest budgets
and from stage 3 otherwise. Expensive switching: never switch at z = 1, then
switch only stage 5, and stages 4–5 at large budgets (L = 15).

## 8. Q-learning: `q_learning.py`

Seed 42, 100 random updates per episode, step size visits^(−0.8), a check
every 2,000 episodes, and a stop after 3 checks in a row with at least 200
visits per pair, at most 2% policy change and full agreement with value
iteration. Cells where value iteration's two costs differ by less than 0.02
are ties (either action is fine), so they are left out of the agreement.

| Version | What the learner sees | Agreement with value iteration |
|---|---|---|
| 1. Toy | simulated months | 100% of clear cells, for both cases (the crowding bump's two switching regions included) |
| 2. Real data, one month at a time | one real month of a strategy in the right stage, never the counted table | 100% of clear cells, both cases (7 ties left out with expensive switching) |
| 3. Real history | the actual sequence of months: it holds a real strategy, and switching takes a real stage-1 strategy of that month | 84% (cheap), 81% (expensive) |

Versions 1 and 2 show that Q-learning, which never sees the probabilities,
finds the same policy as value iteration. Version 3 differs because real
history does not follow the 5-stage model exactly (section 7, model
assumptions). The differences repeat with other seeds and with 300,000
episodes, so they are not learning noise.

**Which policy does better on the real history?** Average discounted profit
over at most 60 months from 50,000 random starts. In brackets: the difference
from the table policy, with its standard error.

| Policy | Cheap, same history | Cheap, learned up to 2007, judged from 2008 | Expensive, same history | Expensive, learned up to 2007, judged from 2008 |
|---|---|---|---|---|
| Table policy (value iteration) | 8.51 | 4.93 | 2.54 | 0.20 |
| Q-learning on history | 8.85 (+0.34 ± 0.07) | 4.76 (−0.17 ± 0.07) | 2.41 (−0.13 ± 0.03) | 0.34 (+0.14 ± 0.04) |
| Ignore the limit | 8.53 (+0.02 ± 0.07) | 4.54 (−0.39 ± 0.07) | 2.58 (+0.04 ± 0.03) | 0.20 (0.00) |
| Never switch | 4.61 (−3.91 ± 0.08) | 3.48 (−1.46 ± 0.07) | 2.45 (−0.09 ± 0.03) | 1.92 (+1.72 ± 0.04) |

- **Cheap switching:** out of sample, the limit-aware table policy is the best
  of the four. It beats ignoring the limit by 0.39 and never switching by 1.46.
  The history learner wins on the history it learned from but not on later
  years, so its extra edge looks like fitting the past.
- **Expensive switching:** after 2008 nothing that switches beats never
  switching, as the split test showed. The history learner switches less than
  the table policy, so it loses less.
- Learning from real sequences is not automatically better here. Tabular
  Q-learning still sees only (stage, budget), so it cannot use the memory that
  the 5-stage state forgets. That needs a richer state (next steps).

## 9. What is solid and what is not yet

Solid:

- The stage predicts next month's PnL and how often a strategy loses money
  (luck tests, t = 3.3 to 7.8 on 211 strategies). This is factor momentum, a
  known effect.
- One switching region per budget column in every setting checked, other loss
  limits included, and the K ordering of the low-budget effect.
- Q-learning reproduces value iteration from samples alone (100% of clear
  cells).
- With cheap switching, the limit-aware policy built on months up to 2007
  still works after 2008: 97% of the best possible in the split test, and the
  best of the four policies on the real 2008 onwards history.

Not solid:

- The exact switch stage is known only to within about one stage (resampling
  test).
- With expensive switching, the policy did not carry over to 2008 onwards:
  never switching did better.
- The 5-stage state forgets memory in the data, so the model is a
  simplification of real history.
- K, γ, L and the 2% size are choices, the volatility scaling looks ahead,
  and all strategies in a stage are treated alike.

## 10. Running

```bash
pip install numpy openassetpricing
python3 fetch_osap.py        # download the 212-strategy data (Open Source Asset Pricing)
python3 value_iteration.py   # toy model: TOY and CROWDING BUMP cases
python3 real_data.py         # both datasets; downloads the French files if missing
python3 checks.py            # luck tests, resampling, split test, other limits (~10 s)
python3 q_learning.py        # the three Q-learning versions and their evaluation (a few minutes)
```

`data/` holds the downloaded files (a few MB). It isn't in the repository; the
scripts recreate it.

## 11. Next steps

1. **A richer state for RL.** Add what the 5-stage state forgets, such as how
   long the strategy has been in its stage, or a market-wide condition. A table
   over a bigger state needs far more data, so this is where Q-learning with
   function approximation is genuinely needed.
2. **Let R choose the target stage**, for example the calmer stage 2 instead
   of the hottest stage 1.
3. **A model that changes over time**: re-estimate on a rolling window, since
   the split test shows the odds changed after 2008.

## References

- Chen, A. Y., and Zimmermann, T. (2022). Open Source Cross-Sectional Asset
  Pricing. *Critical Finance Review*.
- Ehsani, S., and Linnainmaa, J. T. (2022). Factor Momentum and the Momentum
  Factor. *Journal of Finance*.
- Gupta, T., and Kelly, B. (2019). Factor Momentum Everywhere. *Journal of
  Portfolio Management*.
- McLean, R. D., and Pontiff, J. (2016). Does Academic Research Destroy Stock
  Return Predictability? *Journal of Finance*.
- Sootla, A., et al. (2022). Sauté RL: Almost Surely Safe Reinforcement
  Learning Using State Augmentation. *ICML*.
- Kenneth R. French, Data Library (Dartmouth).
