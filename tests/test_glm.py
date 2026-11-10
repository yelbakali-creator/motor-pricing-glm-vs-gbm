"""Validation tests on synthetic data (no download needed): IRLS GLM vs statsmodels, balance property, Gini."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.glm import design_matrix, fit_glm, gini, poisson_deviance, set_levels


def _data(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"DrivAgeBand": rng.choice(["18-20", "41-50", "71+"], n),
                       "VehGas": rng.choice(["Diesel", "Regular"], n),
                       "BonusMalusC": rng.uniform(50, 150, n), "Exposure": rng.uniform(0.1, 1.0, n)})
    eta = -3 + 0.02 * (df["BonusMalusC"] - 100) + np.where(df["DrivAgeBand"] == "18-20", 0.5, 0.0)
    df["ClaimNb"] = rng.poisson(df["Exposure"] * np.exp(eta))
    return df


def test_poisson_irls_matches_statsmodels_and_balances():
    df = _data()
    cat = ["DrivAgeBand", "VehGas"]
    set_levels(df, cat)
    X = design_matrix(df, cat, ["BonusMalusC"])
    g = fit_glm(X.values, df["ClaimNb"].values, np.log(df["Exposure"].values))
    ref = sm.GLM(df["ClaimNb"].values, X.values, family=sm.families.Poisson(), offset=np.log(df["Exposure"].values)).fit()
    assert np.max(np.abs(g["beta"] - ref.params)) < 1e-9
    assert abs(g["mu"].sum() - df["ClaimNb"].sum()) < 1e-6         # balance property


def test_gamma_irls_matches_statsmodels():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(5000), rng.normal(size=5000)]
    y = rng.gamma(2.0, np.exp(7 + 0.3 * X[:, 1]) / 2.0)
    w = rng.integers(1, 3, 5000).astype(float)
    g = fit_glm(X, y, weights=w, family="gamma")
    ref = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()), var_weights=w).fit()
    assert np.max(np.abs(g["beta"] - ref.params)) < 1e-9


def test_gini_of_uninformative_model_is_noise_and_deviance_is_zero_at_truth():
    df = _data(seed=2)
    rate = np.full(len(df), df["ClaimNb"].sum() / df["Exposure"].sum())
    assert abs(gini(df["ClaimNb"].values, rate, df["Exposure"].values)[0]) < 0.05
    y = np.array([0.0, 1.0, 2.0])
    assert poisson_deviance(y, y + 1e-12) < 1e-6
