"""Motor pricing on freMTPL2: Poisson GLM vs gradient boosting, SHAP, and a GBM-informed GLM."""
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm

from src.data import add_features, load
from src.glm import design_matrix, fit_glm, gini, poisson_deviance, set_levels

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

SEED = 2026
CAT = ["DrivAgeBand", "VehAgeBand", "VehPowerCat", "VehBrand", "VehGas", "Region"]
CONT = ["BonusMalusC", "LogDensity", "AreaCode"]
SHAPE_TERMS = ["BonusMalusLog", "BonusMalusOver100", "BonusMalusMin", "LogDensitySq"]
# Candidate GLM enhancements suggested by SHAP; the choice is made on a validation split, never on the test set
CANDIDATES = {
    "Base GLM": (CONT, []),
    "+ driver age x bonus-malus": (CONT, [("DrivAgeBand", "BonusMalusC")]),
    "+ non-linear shapes": (CONT + SHAPE_TERMS, []),
    "+ shapes + driver age x bonus-malus": (CONT + SHAPE_TERMS, [("DrivAgeBand", "BonusMalusC")]),
    "+ shapes + both interactions": (CONT + SHAPE_TERMS, [("DrivAgeBand", "BonusMalusC"), ("Region", "BonusMalusC")]),
}
GBM_FEATURES = ["DrivAge", "VehAge", "VehPower", "BonusMalusC", "LogDensity", "AreaCode",
                "VehBrand", "VehGas", "Region"]
SEV_CAT, SEV_CONT = ["DrivAgeBand", "VehAgeBand", "VehGas"], ["BonusMalusC", "AreaCode"]
LARGE_CLAIM_CAP = 50_000


def main():
    df = add_features(load(ROOT / "data"))
    rng = np.random.default_rng(SEED)
    is_test = rng.random(len(df)) < 0.2
    tr, te = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    off_tr, off_te = np.log(tr["Exposure"].values), np.log(te["Exposure"].values)

    # 1. Frequency GLM (IRLS from scratch), validated against statsmodels
    set_levels(tr, CAT)
    Xtr, Xte = design_matrix(tr, CAT, CONT), design_matrix(te, CAT, CONT)
    glm = fit_glm(Xtr.values, tr["ClaimNb"].values, off_tr)
    ref = sm.GLM(tr["ClaimNb"].values, Xtr.values, family=sm.families.Poisson(), offset=off_tr).fit()
    mu_glm = np.exp(Xte.values @ glm["beta"] + off_te)

    # 2. Gradient boosting (LightGBM, Poisson objective, exposure offset), early stopping
    Gtr, Gte = tr[GBM_FEATURES].copy(), te[GBM_FEATURES].copy()
    cat_gbm = ["VehBrand", "VehGas", "Region"]
    for c in cat_gbm:                                   # integer codes, declared categorical to LightGBM
        cats = sorted(Gtr[c].unique())
        Gtr[c] = pd.Categorical(Gtr[c], categories=cats).codes.astype(float)
        Gte[c] = pd.Categorical(Gte[c], categories=cats).codes.astype(float)
    val = np.random.default_rng(SEED + 1).random(len(Gtr)) < 0.1
    params = {"objective": "poisson", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 1000,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "seed": SEED}
    dtr = lgb.Dataset(Gtr[~val], tr["ClaimNb"].values[~val], init_score=off_tr[~val], categorical_feature=cat_gbm)
    dval = lgb.Dataset(Gtr[val], tr["ClaimNb"].values[val], init_score=off_tr[val], reference=dtr)
    gbm = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    mu_gbm = np.exp(gbm.predict(Gte, raw_score=True, num_iteration=gbm.best_iteration) + off_te)

    # 3. SHAP: global importance and the strongest interaction
    sample = Gte.sample(3000, random_state=SEED)
    explainer = shap.TreeExplainer(gbm)
    sv = explainer.shap_values(sample)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=GBM_FEATURES).sort_values(ascending=False)
    inter = explainer.shap_interaction_values(sample.iloc[:1000])
    inter_mean = np.abs(inter).mean(axis=0)
    np.fill_diagonal(inter_mean, 0)
    pairs = pd.DataFrame(inter_mean, index=GBM_FEATURES, columns=GBM_FEATURES)
    top_pairs = (pairs.where(np.triu(np.ones(pairs.shape), 1).astype(bool)).stack()
                 .sort_values(ascending=False).head(5))

    # 4. GBM-informed GLM: candidates compared on the validation split (the GBM's early-stopping split)
    fit_part, val_part = tr[~val], tr[val]
    selection = []
    for name, (cont, inter_terms) in CANDIDATES.items():
        Xf = design_matrix(fit_part, CAT, cont, inter_terms)
        Xv = design_matrix(val_part, CAT, cont, inter_terms)
        g = fit_glm(Xf.values, fit_part["ClaimNb"].values, off_tr[~val])
        mu_v = np.exp(Xv.values @ g["beta"] + off_tr[val])
        selection.append({"candidate": name, "parameters": Xf.shape[1],
                          "validation_poisson_deviance_x100": poisson_deviance(val_part["ClaimNb"].values, mu_v)})
    selection = pd.DataFrame(selection)
    selection.to_csv(TABLES / "glm_enhancement_selection_on_validation.csv", index=False)
    best = selection.loc[selection["validation_poisson_deviance_x100"].idxmin(), "candidate"]
    cont_best, inter_best = CANDIDATES[best]
    Xtr2, Xte2 = design_matrix(tr, CAT, cont_best, inter_best), design_matrix(te, CAT, cont_best, inter_best)
    glm2 = fit_glm(Xtr2.values, tr["ClaimNb"].values, off_tr)                 # refit on the full training set
    mu_glm2 = np.exp(Xte2.values @ glm2["beta"] + off_te)

    # 5. Frequency model comparison on the test set
    mu_hom = te["Exposure"].values * tr["ClaimNb"].sum() / tr["Exposure"].sum()
    y_te, e_te = te["ClaimNb"].values, te["Exposure"].values
    comp = []
    for name, mu, k in [("Homogeneous (no rating factors)", mu_hom, 1), ("Poisson GLM", mu_glm, Xtr.shape[1]),
                        ("GLM enhanced with GBM insights", mu_glm2, Xtr2.shape[1]),
                        ("LightGBM (Poisson)", mu_gbm, np.nan)]:
        comp.append({"model": name, "parameters": k, "test_poisson_deviance_x100": poisson_deviance(y_te, mu),
                     "gini": gini(y_te, mu / e_te, e_te)[0], "predicted_over_actual": mu.sum() / y_te.sum()})
    comp = pd.DataFrame(comp)
    base_dev = comp.loc[0, "test_poisson_deviance_x100"]
    comp["deviance_reduction_vs_homogeneous"] = 1 - comp["test_poisson_deviance_x100"] / base_dev
    comp.to_csv(TABLES / "frequency_model_comparison.csv", index=False)

    validation = pd.DataFrame([
        ["Max |coefficient difference| vs statsmodels", np.max(np.abs(ref.params - glm["beta"]))],
        ["Max |standard error difference| vs statsmodels", np.max(np.abs(ref.bse - glm["se"]))],
        ["Balance: predicted claims (train)", glm["mu"].sum()],
        ["Balance: observed claims (train)", tr["ClaimNb"].sum()],
        ["IRLS iterations", glm["iterations"]],
        ["GBM boosting rounds (early stopping)", gbm.best_iteration],
    ], columns=["check", "value"])
    validation.to_csv(TABLES / "validation.csv", index=False)

    # 6. Severity GLM (Gamma, log link) on capped average claim amounts, and the tariff
    sev = tr[(tr["ClaimNb"] > 0) & (tr["SevCount"] > 0)].copy()
    sev["AvgAmount"] = sev["ClaimTotal"] / sev["SevCount"]
    sev["AvgCapped"] = np.minimum(sev["AvgAmount"], LARGE_CLAIM_CAP)
    large_loading = (sev["AvgAmount"] * sev["SevCount"]).sum() / (sev["AvgCapped"] * sev["SevCount"]).sum()
    set_levels(tr, CAT)
    Xs = design_matrix(sev, SEV_CAT, SEV_CONT)
    sev_glm = fit_glm(Xs.values, sev["AvgCapped"].values, weights=sev["SevCount"].values, family="gamma")
    ref_sev = sm.GLM(sev["AvgCapped"].values, Xs.values, family=sm.families.Gamma(sm.families.links.Log()),
                     var_weights=sev["SevCount"].values).fit()
    sev_diff = np.max(np.abs(ref_sev.params - sev_glm["beta"]))

    tariff = pd.DataFrame({"factor": Xtr.columns, "frequency_relativity": np.exp(glm["beta"]),
                           "se": glm["se"]})
    tariff.loc[0, "factor"] = "Base frequency (claims per year)"
    tariff.to_csv(TABLES / "frequency_tariff_relativities.csv", index=False)
    sev_tab = pd.DataFrame({"factor": Xs.columns, "severity_relativity": np.exp(sev_glm["beta"]), "se": sev_glm["se"]})
    sev_tab.loc[0, "factor"] = "Base severity (EUR, capped)"
    sev_tab.to_csv(TABLES / "severity_relativities.csv", index=False)

    # Pure premium on the test set: GLM frequency x GLM severity x large-loss loading
    Xs_te = design_matrix(te, SEV_CAT, SEV_CONT)
    sev_te = np.exp(Xs_te.values @ sev_glm["beta"]) * large_loading
    pp_glm = mu_glm * sev_te
    pp_gbm = mu_gbm * sev_te
    amt = te["ClaimTotal"].values
    avg_te = np.where(te["SevCount"] > 0, te["ClaimTotal"] / te["SevCount"].clip(lower=1), 0)
    amt_capped = np.minimum(avg_te, LARGE_CLAIM_CAP) * te["SevCount"].values
    capped_calibration = (mu_glm * sev_te / large_loading).sum() / amt_capped.sum()
    share_above_cap = {"train": 1 - (sev["AvgCapped"] * sev["SevCount"]).sum() / (sev["AvgAmount"] * sev["SevCount"]).sum(),
                       "test": 1 - amt_capped.sum() / amt.sum()}
    pure = pd.DataFrame([
        {"model": "GLM frequency x GLM severity", "gini_on_claim_amounts": gini(amt, pp_glm / e_te, e_te)[0],
         "predicted_over_actual_amounts": pp_glm.sum() / amt.sum()},
        {"model": "GBM frequency x GLM severity", "gini_on_claim_amounts": gini(amt, pp_gbm / e_te, e_te)[0],
         "predicted_over_actual_amounts": pp_gbm.sum() / amt.sum()},
    ])
    pure["predicted_over_actual_capped_amounts"] = [capped_calibration, (mu_gbm * sev_te / large_loading).sum() / amt_capped.sum()]
    pure["share_of_amounts_above_cap_train"] = share_above_cap["train"]
    pure["share_of_amounts_above_cap_test"] = share_above_cap["test"]
    pure.to_csv(TABLES / "pure_premium_comparison.csv", index=False)

    # 7. Double lift chart: GBM vs GLM
    ratio = mu_gbm / mu_glm
    dec = pd.qcut(ratio, 10, labels=False, duplicates="drop")
    lift = pd.DataFrame({"decile": dec, "actual": y_te, "glm": mu_glm, "gbm": mu_gbm}).groupby("decile").sum()
    lift_rel = pd.DataFrame({"actual_over_glm": lift["actual"] / lift["glm"], "gbm_over_glm": lift["gbm"] / lift["glm"]})
    lift_rel.to_csv(TABLES / "double_lift_gbm_vs_glm.csv")

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].barh(importance.index[::-1], importance.values[::-1], color=NAVY)
    axes[0].set_title("SHAP global importance (mean |SHAP|, log frequency)")
    top = top_pairs.head(5)
    axes[1].barh([f"{a} x {b}" for a, b in top.index[::-1]], top.values[::-1], color=GOLD)
    axes[1].set_title("Strongest pairwise interactions (SHAP)")
    fig.tight_layout(); fig.savefig(FIGS / "shap_importance_interactions.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    i_bm, i_age = GBM_FEATURES.index("BonusMalusC"), GBM_FEATURES.index("DrivAge")
    sc = axes[0].scatter(sample["BonusMalusC"], sv[:, i_bm], c=sample["DrivAge"], s=6, cmap="viridis", alpha=0.7)
    axes[0].set_xlabel("Bonus-malus"); axes[0].set_ylabel("SHAP value (log frequency)")
    axes[0].set_title("Bonus-malus effect, coloured by driver age")
    plt.colorbar(sc, ax=axes[0], label="Driver age")
    sc2 = axes[1].scatter(sample["DrivAge"], sv[:, i_age], c=sample["BonusMalusC"], s=6, cmap="magma", alpha=0.7)
    plt.colorbar(sc2, ax=axes[1], label="Bonus-malus")
    axes[1].set_xlabel("Driver age"); axes[1].set_ylabel("SHAP value (log frequency)")
    axes[1].set_title("Driver-age effect, coloured by bonus-malus")
    fig.tight_layout(); fig.savefig(FIGS / "shap_dependence.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, mu, col in [("GLM", mu_glm, NAVY), ("Enhanced GLM", mu_glm2, GREEN), ("LightGBM", mu_gbm, RED)]:
        g, cx, cy = gini(y_te, mu / e_te, e_te)
        axes[0].plot(cx, cy, color=col, lw=1.3, label=f"{name} (Gini {g:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GREY, ls=":")
    axes[0].set_xlabel("Cumulative share of exposure (ordered by predicted risk)")
    axes[0].set_ylabel("Cumulative share of claims")
    axes[0].set_title("Lorenz curves, test set")
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(len(lift_rel))
    axes[1].plot(x + 1, lift_rel["actual_over_glm"], color="black", marker="o", label="Actual / GLM")
    axes[1].plot(x + 1, lift_rel["gbm_over_glm"], color=RED, marker="s", label="GBM / GLM")
    axes[1].axhline(1, color=GREY, ls=":")
    axes[1].set_xlabel("Decile of GBM / GLM prediction ratio")
    axes[1].set_title("Double lift chart: GBM vs GLM")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "lorenz_double_lift.png", dpi=150); plt.close(fig)

    t = tariff[tariff["factor"].str.startswith("DrivAgeBand")].copy()
    t["level"] = t["factor"].str.extract(r"\[(.+)\]")
    t = pd.concat([t, pd.DataFrame({"level": ["41-50"], "frequency_relativity": [1.0], "se": [0.0]})])
    order = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
    t = t.set_index("level").loc[order]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(order, t["frequency_relativity"], yerr=1.96 * t["frequency_relativity"] * t["se"], fmt="o-",
                color=NAVY, capsize=3)
    ax.axhline(1, color=GREY, ls=":")
    ax.set_xlabel("Driver age band (base 41-50)"); ax.set_ylabel("Frequency relativity")
    ax.set_title("GLM tariff: driver-age relativities, controlling for bonus-malus")
    fig.tight_layout(); fig.savefig(FIGS / "glm_driver_age_relativities.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 200)
    print(f"Policies: {len(df):,} | train {len(tr):,} | test {len(te):,} | claims {df['ClaimNb'].sum():,.0f}")
    print("\nValidation\n" + validation.to_string(index=False))
    print(f"Severity GLM max |coef diff| vs statsmodels: {sev_diff:.2e} | large-loss loading: {large_loading:.4f}")
    print("\nGLM enhancement selection (validation split)\n" + selection.round(4).to_string(index=False))
    print(f"Selected: {best}")
    print("\nFrequency models (test set)\n" + comp.round(4).to_string(index=False))
    print("\nSHAP importance\n" + importance.round(4).to_string())
    print("\nTop interactions\n" + top_pairs.round(4).to_string())
    print("\nPure premium\n" + pure.round(4).to_string(index=False))
    print("\nDouble lift\n" + lift_rel.round(3).to_string())
    print("\nSeverity relativities\n" + sev_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
