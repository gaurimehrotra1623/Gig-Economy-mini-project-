"""
simulate_income.py

Generates synthetic weekly income series for gig workers, calibrated
loosely against real published stats:
- JPMorgan Chase Institute: hourly/platform workers see typical month-to-month
  swings of ~9%, with 1-in-4 months swinging >21% (employer/demand-driven).
- Fairwork India / NITI Aayog: rider monthly earnings roughly in the
  Rs 10,000-25,000 range depending on platform and city, with strong
  festival-season spikes and monsoon dips.

This is a SIMULATION calibrated to real aggregate statistics, not real
individual-level data (which isn't public). Document this clearly in
your writeup.

Each worker's weekly income is built from four additive pieces:
    income = base_level + trend + seasonal_pattern + noise + shock_events

- base_level: worker-specific average earning power
- trend: slow drift up/down (e.g. worker getting more experienced, or
  platform demand secularly declining in their area)
- seasonal_pattern: repeating weekly/annual cycle (festivals, monsoon)
- noise: small random week-to-week fluctuation (normal, "boring" variance)
- shock_events: occasional large, non-seasonal disruptions (illness,
  platform deactivation, demand collapse) -- THIS is the part a good
  model should be able to separate from the seasonal part.
"""

import numpy as np
import pandas as pd


def simulate_worker(
    worker_id: int,
    n_weeks: int = 104,          # 2 years of weekly data
    base_income: float = 6000.0,  # avg weekly income in INR
    trend_per_week: float = 0.0,  # secular drift, INR/week
    seasonal_amplitude: float = 1200.0,
    noise_std: float = 400.0,
    shock_prob: float = 0.04,     # prob of a shock week
    shock_severity_mean: float = 2500.0,
    shock_severity_std: float = 1200.0,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """Simulate one worker's weekly income series."""
    if rng is None:
        rng = np.random.default_rng()

    weeks = np.arange(n_weeks)

    # Trend: slow linear drift
    trend = trend_per_week * weeks

    # Seasonal: annual cycle (52-week period) with festival bump + monsoon dip
    # Roughly: bump around weeks 40-46 (Oct-Nov, festival season),
    # dip around weeks 22-30 (Jul-Aug, monsoon)
    annual_cycle = np.sin(2 * np.pi * (weeks - 13) / 52)  # base sinusoid
    festival_bump = 0.6 * np.exp(-0.5 * ((weeks % 52 - 43) / 4) ** 2)
    monsoon_dip = -0.5 * np.exp(-0.5 * ((weeks % 52 - 26) / 5) ** 2)
    seasonal = seasonal_amplitude * (annual_cycle * 0.4 + festival_bump + monsoon_dip)

    # Regular "boring" noise -- this is NOT risk, just normal randomness
    noise = rng.normal(0, noise_std, size=n_weeks)

    # Shock events -- THIS is the true-risk component: unpredictable,
    # not tied to the calendar, potentially large and negative-skewed
    # (illness/deactivation hurt income; rarely does a "shock" help it)
    shocks = np.zeros(n_weeks)
    shock_weeks = rng.random(n_weeks) < shock_prob
    n_shocks = shock_weeks.sum()
    if n_shocks > 0:
        severities = rng.normal(shock_severity_mean, shock_severity_std, n_shocks)
        severities = np.clip(severities, 0, None)
        shocks[shock_weeks] = -severities  # shocks are income LOSSES

    income = base_income + trend + seasonal + noise + shocks
    income = np.clip(income, 0, None)  # income can't go negative

    df = pd.DataFrame({
        "worker_id": worker_id,
        "week": weeks,
        "income": income,
        "had_shock": shock_weeks.astype(int),  # ground truth, for validation only
    })
    return df


def simulate_population(n_workers: int = 500, n_weeks: int = 104, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a population of gig workers with heterogeneous profiles.
    Three archetypes, roughly reflecting real segments you'd read about
    in the Fairwork/NITI Aayog reports:
      - 'steady':      low seasonal swing, low shock risk (e.g. Urban Company-style
                        service workers with more stable slot bookings)
      - 'seasonal':    high seasonal swing, LOW shock risk (predictable volatility --
                        the interesting case your hypothesis is about)
      - 'volatile':    moderate seasonal swing, HIGH shock risk (genuinely risky)
    """
    rng = np.random.default_rng(seed)
    archetypes = rng.choice(
        ["steady", "seasonal", "volatile"], size=n_workers, p=[0.35, 0.35, 0.30]
    )

    all_dfs = []
    for i, archetype in enumerate(archetypes):
        base = rng.normal(6000, 800)
        if archetype == "steady":
            seasonal_amp, noise_std, shock_p = 400, 300, 0.015
        elif archetype == "seasonal":
            seasonal_amp, noise_std, shock_p = 1800, 350, 0.015
        else:  # volatile
            seasonal_amp, noise_std, shock_p = 900, 400, 0.08

        df = simulate_worker(
            worker_id=i,
            n_weeks=n_weeks,
            base_income=base,
            trend_per_week=rng.normal(0, 5),
            seasonal_amplitude=seasonal_amp,
            noise_std=noise_std,
            shock_prob=shock_p,
            rng=rng,
        )
        df["archetype"] = archetype
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    pop = simulate_population(n_workers=500, n_weeks=104)
    pop.to_csv("data/simulated_income.csv", index=False)
    print(f"Simulated {pop.worker_id.nunique()} workers, {len(pop)} rows total.")
    print(pop.groupby("archetype")["income"].describe())
