"""Generalized linear models with a log link, fitted by IRLS from scratch, and pricing metrics."""
import numpy as np
import pandas as pd

BASE_LEVELS = {"DrivAgeBand": "41-50", "VehAgeBand": "1-5", "VehPowerCat": "6",
               "VehBrand": "B1", "VehGas": "Regular", "Region": "Centre"}


def design_matrix(df, categorical, continuous, interactions=()):
    """Dummy-coded design matrix with fixed base levels, so train and test share columns.

    Call set_levels on the training data first.
    """
    cols = {"Intercept": np.ones(len(df))}
    for c in categorical:
        base = BASE_LEVELS.get(c)
        for level in design_matrix.levels[c]:
            if level != base:
                cols[f"{c}[{level}]"] = (df[c].values == level).astype(float)
    for c in continuous:
        cols[c] = df[c].values.astype(float)
    for cat, cont in interactions:
        base = BASE_LEVELS.get(cat)
        for level in design_matrix.levels[cat]:
            if level != base:
                cols[f"{cat}[{level}]:{cont}"] = (df[cat].values == level) * df[cont].values.astype(float)
    return pd.DataFrame(cols)


def set_levels(df, categorical):
    design_matrix.levels = {c: sorted(df[c].unique()) for c in categorical}


def fit_glm(X, y, offset=None, weights=None, family="poisson", tol=1e-10, max_iter=100):
    """IRLS for a log-link GLM. Poisson: working weight w*mu; Gamma: working weight w."""
    X = np.asarray(X, float)
    n, p = X.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, float)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    beta = np.zeros(p)
    beta[0] = np.log(np.average(y, weights=w) / np.average(np.exp(offset), weights=w))
    for it in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(eta)
        W = w * mu if family == "poisson" else w.copy()
        z = eta - offset + (y - mu) / mu
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(X @ beta + offset)
    W = w * mu if family == "poisson" else w
    cov = np.linalg.inv((X.T * W) @ X)
    if family == "gamma":
        phi = np.sum(w * ((y - mu) / mu) ** 2) / (n - p)
        cov = cov * phi
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "iterations": it + 1, "mu": mu}


def poisson_deviance(y, mu, weights=None):
    """Mean Poisson deviance (scaled by 100 to match the literature's 10^-2 units when y is a count)."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    dev = 2 * (term - (y - mu))
    return 100 * (np.average(dev, weights=weights) if weights is not None else dev.mean())


def gini(actual, predicted_rate, exposure):
    """Gini coefficient of the exposure-weighted Lorenz curve, policies ordered by predicted risk."""
    # Ties are broken at random: the raw file is not randomly ordered (claims cluster at the start)
    tie_break = np.random.default_rng(0).random(len(predicted_rate))
    # Rounding removes floating-point noise that would otherwise act as a hidden sort key
    order = np.lexsort((tie_break, np.round(predicted_rate, 10)))
    cum_exp = np.cumsum(exposure[order]) / exposure.sum()
    cum_claims = np.cumsum(actual[order]) / actual.sum()
    return 1 - 2 * np.trapezoid(cum_claims, cum_exp), cum_exp, cum_claims
