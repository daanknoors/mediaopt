"""Models that predict a recipe's cell count (y), each with an uncertainty estimate."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .data import COLS


def make_gp() -> GaussianProcessRegressor:
    """GP: amplitude x Matern similarity (one shared length-scale) + a little extra noise.

    The measured replicate noise of each recipe is added in `fit` (as `alpha`).
    y is already standardised, so no rescaling: the prior mean 0 is the dataset average.
    """
    kernel = (ConstantKernel(1.0, (0.01, 100))
              * Matern(length_scale=1.0, length_scale_bounds=(0.05, 100), nu=2.5)
              + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1)))
    return GaussianProcessRegressor(kernel, n_restarts_optimizer=10, random_state=0)


MODELS = {
    "average (baseline)": DummyRegressor(),
    "random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0),
    "Gaussian process": make_gp(),
}


def fit(model, df: pd.DataFrame):
    """Fit a fresh copy of `model` on the recipes in df."""
    m = clone(model)
    if isinstance(m, GaussianProcessRegressor):
        m.set_params(alpha=df.y_var.clip(lower=1e-3).to_numpy())  # measured noise per recipe
    m.fit(df[COLS].to_numpy(), df.y.to_numpy())
    m.y_sd_ = df.y.std()  # used as the 'uncertainty' of the average model
    return m


def predict(model, df: pd.DataFrame):
    """Predicted mean and standard deviation for each recipe in df."""
    X = df[COLS].to_numpy()
    if isinstance(model, GaussianProcessRegressor):
        return model.predict(X, return_std=True)
    if isinstance(model, RandomForestRegressor):  # spread between the individual trees
        trees = np.stack([tree.predict(X) for tree in model.estimators_])
        return trees.mean(axis=0), trees.std(axis=0)
    return model.predict(X), np.full(len(X), model.y_sd_)


def leave_one_out(model, df: pd.DataFrame) -> pd.DataFrame:
    """Hide each recipe in turn, train on the rest, predict the hidden one."""
    rows = []
    for i in range(len(df)):
        m = fit(model, df.drop(index=i))
        mean, sd = predict(m, df.iloc[[i]])
        rows.append({"observed": df.y.iloc[i], "predicted": mean[0], "sd": sd[0]})
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame) -> dict:
    """Error size, ranking quality, and honesty of the 90 % interval."""
    error = pred.observed - pred.predicted
    return {
        "RMSE": np.sqrt(np.mean(error**2)),
        "rank correlation": spearmanr(pred.observed, pred.predicted).statistic,
        "90% interval coverage": np.mean(np.abs(error) <= 1.645 * pred.sd),
    }
