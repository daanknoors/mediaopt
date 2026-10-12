"""Candidate recipes and how promising each one is (Expected Improvement)."""

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor

from .cost import BASELINE, media_cost
from .data import COLS
from .model import fit, make_gp, predict


def sample_candidates(df: pd.DataFrame, ceiling: float, n_random: int = 20_000, n_local: int = 5_000,
                      step: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Candidate recipes within the cost ceiling (all values 0-1, so within concentration limits).

    - random: evenly spread over all 14 components (Latin hypercube) -> searches everywhere
    - local:  small random changes (sd = `step`) to the 5 best measured recipes -> searches finely near what works
    """
    rng = np.random.default_rng(seed)
    n_local = n_local if len(df) else 0
    random = qmc.LatinHypercube(d=len(COLS), seed=seed).random(n_random)
    best5 = df.nlargest(5, "y")[COLS].to_numpy()
    local = np.clip(best5[rng.integers(0, 5, n_local)] + rng.normal(0, step, (n_local, len(COLS))), 0, 1)
    cand = pd.DataFrame(np.vstack([random, local]), columns=COLS)
    cand["source"] = ["random"] * n_random + ["local"] * n_local
    return cand[media_cost(cand) <= ceiling].reset_index(drop=True)


def best_so_far(gp, df: pd.DataFrame, ceiling: float) -> float:
    """The model's estimate of the best recipe tested so far (within the ceiling).

    We use the model's estimate rather than the raw measurement, so a lucky measurement
    doesn't set an unreachable bar: predictions are compared with predictions.
    """
    tested = df[media_cost(df) <= ceiling]
    return float(predict(gp, tested)[0].max())


def expected_improvement(mean, sd, best):
    """Average amount by which a recipe beats `best`, counting only gains (zero if worse)."""
    mean, sd = np.asarray(mean), np.asarray(sd)
    z = (mean - best) / sd
    return (mean - best) * norm.cdf(z) + sd * norm.pdf(z)


def score_candidates(gp, cand: pd.DataFrame, best: float) -> pd.DataFrame:
    """Add predicted mean, uncertainty and Expected Improvement to each candidate."""
    mean, sd = predict(gp, cand)
    # predict() includes measurement noise; we want uncertainty about the TRUE growth, so remove it
    sd = np.sqrt(np.maximum(sd**2 - gp.kernel_.k2.noise_level, 1e-12))
    return cand.assign(mean=mean, sd=sd, ei=expected_improvement(mean, sd, best))


def _pretend_measured(gp, x, value):
    """Copy of the GP that 'knows' recipe x gave `value` (settings kept fixed).

    Used to build a batch: after picking a recipe we pretend it was measured at its predicted value,
    so the model becomes confident there and the next pick goes somewhere else ("kriging believer").
    """
    X = np.vstack([gp.X_train_, x])
    y = np.append(gp.y_train_, value)
    alpha = np.append(gp.alpha, np.median(gp.alpha))  # typical replicate noise for the pretend point
    return GaussianProcessRegressor(gp.kernel_, alpha=alpha, optimizer=None).fit(X, y)


def select_batch(gp, cand: pd.DataFrame, best: float, n_exploit: int = 3, n_explore: int = 1) -> pd.DataFrame:
    """Pick n_exploit recipes by highest Expected Improvement, then n_explore by highest uncertainty
    among recipes predicted at least average (mean >= 0; y is standardised, so 0 = average recipe).
    Picks are made one at a time."""
    picks = []
    for role in ["exploit"] * n_exploit + ["explore"] * n_explore:
        scored = score_candidates(gp, cand, best).drop(index=[p.name for p in picks])
        if role == "exploit":
            pick = scored.loc[scored.ei.idxmax()]
        else:
            not_poor = scored[scored["mean"] >= 0]
            pick = not_poor.loc[not_poor.sd.idxmax()]
        picks.append(pick.copy())
        picks[-1]["role"] = role
        gp = _pretend_measured(gp, pick[COLS].to_numpy(float)[None, :], pick["mean"])
    return pd.DataFrame(picks)


def recommend(df: pd.DataFrame, ceiling: float, n_exploit: int = 3, n_explore: int = 1, seed: int = 0):
    """Full pipeline: fit the GP, generate candidates, pick the batch, add the benchmark.

    Returns the batch (with predictions, 90 % intervals, cost) and the bar to beat.
    """
    gp = fit(make_gp(), df)
    best = best_so_far(gp, df, ceiling)
    batch = select_batch(gp, sample_candidates(df, ceiling, seed=seed), best, n_exploit, n_explore)
    bench = BASELINE.reset_index(drop=True).assign(role="benchmark", source="baseline")
    out = pd.concat([batch[COLS + ["role", "source"]], bench], ignore_index=True)
    out = score_candidates(gp, out, best)  # re-score all with the same (real-data) model
    out["low90"] = out["mean"] - 1.645 * out.sd   # 90 % interval for the true growth
    out["high90"] = out["mean"] + 1.645 * out.sd
    out["cost"] = media_cost(out)
    return out, best


def random_batch(df: pd.DataFrame, ceiling: float, n: int = 4, seed: int = 1) -> pd.DataFrame:
    """Baseline strategy: n random recipes within the ceiling (no model used to choose)."""
    cand = sample_candidates(df, ceiling, n_local=0, seed=seed)
    return cand.sample(n, random_state=seed).assign(role="random")
    