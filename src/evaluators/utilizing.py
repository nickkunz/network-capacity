## libraries
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Sequence
from scipy.special import gammaln
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut

## path
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

## modules
from src.data.builders import (
    load_processed_data,
    PATH_PROC,
)
from src.estimators.linear_quantile import LinearQuantile
from src.evaluators.metrics import frontier_consensus

## constants
from src.estimators.config import (
    ASYMMETRY_C,
    ASYMMETRY_R,
    ALPHA,
)
from src.evaluators.config import (
    FEAT_X,
    FEAT_Z,
    TARGET,
)

REDUCED_3_X = [
    "n_edges",
    "degree_entropy",
    "random_walk_triangle_weight",
]
REDUCED_3_Z = [
    "norm_succ_diff",
    "coef_variation",
    "trend_coeff",
]
REDUCED_6_DESC = REDUCED_3_X + REDUCED_3_Z

GLM_FAMILIES = (
    "Poisson",
    "Poisson-Tweedie",
    "Negative Binomial",
)
SCT_MODES = (
    "Without SCT",
    "SCT as a Feature",
    "SCT plus Correction",
)
RESAMPLING_REGIMES = (
    "Discipline LOGO-CV",
    "Domain LOGO-CV",
    "5-Fold-CV",
    "10-Fold-CV",
)

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


## ----------------------------------------------------------------------------
## transformations and data loading
## ----------------------------------------------------------------------------
def signed_log1p(arr: Any) -> np.ndarray:

    """
    Desc:
        Apply sign-preserving log1p transformation sign(x) * ln(1 + |x|).

    Args:
        arr: Numeric array-like input.

    Returns:
        Sign-preserving log1p transformed numpy array.
    """

    arr = np.asarray(a = arr, dtype = float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def load_utilization_data(
    data: pd.DataFrame | None = None,
    proc_dir: str | Path | None = None,
    ) -> pd.DataFrame:

    """
    Desc:
        Augment the processed N3S dataframe with interval length L, total count
        M, typical rate r_typ = M / L, empirical maximum r_max, log-rates
        y_typ and y_max, and the parameter-free EVT order-statistic shift
        evt_atten = ln(1 + CV * sqrt(2 * ln(L))).

    Args:
        data: Optional preloaded N3S dataframe.
        proc_dir: Optional directory containing processed per-system JSON files.

    Returns:
        DataFrame sorted by system name with utilization benchmark columns.
    """

    df_n3s = (
        load_processed_data() if data is None else data.copy()
    ).sort_values("name").reset_index(drop = True)
    resolved_proc_dir = (
        (root / PATH_PROC) if proc_dir is None else Path(proc_dir)
    )

    rows = list()
    for _, row in df_n3s.iterrows():
        name = str(row["name"])
        json_path = os.path.join(resolved_proc_dir, f"{name}.json")
        with open(json_path, mode = "r", encoding = "utf-8") as fp:
            evs = json.load(fp)["events"]

        L = len(evs)
        M = float(sum(float(e["target"]) for e in evs))
        r_typ = M / L
        r_max = float(row[TARGET])
        cv_val = float(row["coef_variation"])
        evt_atten = float(np.log1p(cv_val * np.sqrt(2.0 * np.log(max(L, 2)))))

        rows.append({
            "L": float(L),
            "M": float(np.round(M)),
            "r_typ": r_typ,
            "r_max": r_max,
            "utilization": r_typ / r_max,
            "y_max": np.log1p(r_max),
            "y_typ": np.log1p(r_typ),
            "evt_atten": evt_atten,
        })

    return pd.concat([df_n3s, pd.DataFrame(data = rows)], axis = 1)


def compile_corpus_table(df_data: pd.DataFrame) -> pd.DataFrame:

    """
    Desc:
        Format Table 1 summarizing typical rates and empirical maxima across
        all 25 N3S systems.

    Args:
        df_data: Augmented utilization dataframe from load_utilization_data.

    Returns:
        Display-ready Table 1 dataframe.
    """

    df_t1 = df_data[[
        "name",
        "domain",
        "L",
        "M",
        "r_typ",
        "r_max",
        "utilization",
        "y_typ",
        "y_max",
    ]].copy()
    df_t1["L"] = df_t1["L"].astype(int)
    df_t1["M"] = df_t1["M"].astype(np.int64)
    df_t1["r_typ"] = df_t1["r_typ"].round(2)
    df_t1["r_max"] = df_t1["r_max"].round(1)
    df_t1["utilization"] = (df_t1["utilization"] * 100.0).round(2)
    df_t1["y_typ"] = df_t1["y_typ"].round(2)
    df_t1["y_max"] = df_t1["y_max"].round(2)
    return df_t1.rename(columns = {"utilization": "Util_%"})


## ----------------------------------------------------------------------------
## native two-stage sct frontier (linear quantile)
## ----------------------------------------------------------------------------
def fit_native_sct_frontier(
    X_tr: np.ndarray,
    Z_tr: np.ndarray,
    ym_tr: np.ndarray,
    X_ev: np.ndarray,
    Z_ev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

    """
    Desc:
        Fit the native two-stage SCT frontier F(x', z') = f_C(x'; 0.99) +
        (f_R(z'; 0.50) - mu_R) using LinearQuantile from src.estimators.

    Args:
        X_tr: Training graph invariants (n_train, 21).
        Z_tr: Training process signatures (n_train, 7).
        ym_tr: Training log-maxima y_max = ln(1 + r_max).
        X_ev: Evaluation graph invariants (n_eval, 21).
        Z_ev: Evaluation process signatures (n_eval, 7).

    Returns:
        Tuple of (training frontier predictions, evaluation frontier predictions).
    """

    xs = StandardScaler().fit(signed_log1p(X_tr))
    zs = StandardScaler().fit(signed_log1p(Z_tr))
    X_tr_s = xs.transform(signed_log1p(X_tr))
    X_ev_s = xs.transform(signed_log1p(X_ev))
    Z_tr_s = zs.transform(signed_log1p(Z_tr))
    Z_ev_s = zs.transform(signed_log1p(Z_ev))

    lq = LinearQuantile(
        quantile_c = ASYMMETRY_C,
        quantile_r = ASYMMETRY_R,
        alpha = ALPHA,
    )
    mc = lq.estimator_c.fit(X_tr_s, ym_tr)
    c_tr = mc.predict(X_tr_s)

    mr = lq.estimator_r.fit(Z_tr_s, ym_tr - c_tr)
    r_tr = mr.predict(Z_tr_s)
    r_mean = float(np.mean(r_tr))

    p_tr = c_tr + (r_tr - r_mean)
    p_ev = mc.predict(X_ev_s) + (mr.predict(Z_ev_s) - r_mean)
    return p_tr, p_ev


def precompute_native_sct_regime(
    df_data: pd.DataFrame,
    regime_name: str = "Discipline LOGO-CV",
    feat_x: Sequence[str] = FEAT_X,
    feat_z: Sequence[str] = FEAT_Z,
    random_state: int = 42,
    ) -> list[tuple]:

    """
    Desc:
        Precompute outer and inner 4-fold cross-validation SCT frontier splits
        for a given resampling regime.

    Args:
        df_data: Augmented utilization dataframe.
        regime_name: Resampling regime identifier.
        feat_x: Graph invariant column names (all 21 FEAT_X).
        feat_z: Process signature column names (all 7 FEAT_Z).
        random_state: Random seed for KFold splits.

    Returns:
        List of fold tuples (tr_idx, te_idx, sct_tr, sct_te, inner_splits).
    """

    X_all = df_data[list(feat_x)].to_numpy(dtype = float)
    Z_all = df_data[list(feat_z)].to_numpy(dtype = float)
    ym_all = df_data["y_max"].to_numpy(dtype = float)
    inner_kf = KFold(n_splits = 4, shuffle = True, random_state = random_state)

    if regime_name == "Discipline LOGO-CV":
        outer_splits = list(LeaveOneGroupOut().split(df_data, groups = df_data["discipline"]))
    elif regime_name == "Domain LOGO-CV":
        outer_splits = list(LeaveOneGroupOut().split(df_data, groups = df_data["domain"]))
    elif regime_name == "5-Fold-CV":
        outer_splits = list(KFold(n_splits = 5, shuffle = True, random_state = random_state).split(df_data))
    else:
        outer_splits = list(KFold(n_splits = 10, shuffle = True, random_state = random_state).split(df_data))

    folds = list()
    for tr_idx, te_idx in outer_splits:
        sct_tr, sct_te = fit_native_sct_frontier(
            X_tr = X_all[tr_idx],
            Z_tr = Z_all[tr_idx],
            ym_tr = ym_all[tr_idx],
            X_ev = X_all[te_idx],
            Z_ev = Z_all[te_idx],
        )
        inner_splits = list()
        for it, iv in inner_kf.split(tr_idx):
            s_it, s_iv = fit_native_sct_frontier(
                X_tr = X_all[tr_idx[it]],
                Z_tr = Z_all[tr_idx[it]],
                ym_tr = ym_all[tr_idx[it]],
                X_ev = X_all[tr_idx[iv]],
                Z_ev = Z_all[tr_idx[iv]],
            )
            inner_splits.append((it, iv, s_it, s_iv))
        folds.append((tr_idx, te_idx, sct_tr, sct_te, inner_splits))

    return folds


## ----------------------------------------------------------------------------
## discrete count glm likelihoods and fitting
## ----------------------------------------------------------------------------
def poisson_logpmf_uncensored(k: Any, mu: Any) -> np.ndarray:
    k = np.asarray(a = k, dtype = float)
    mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
    return k * np.log(mu) - mu - gammaln(k + 1.0)


def nb2_logpmf(k: Any, mu: Any, alpha: float) -> np.ndarray:
    k = np.asarray(a = k, dtype = float)
    mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
    alpha = max(float(alpha), 1e-8)
    r = 1.0 / alpha
    p = r / (r + mu)
    return (
        gammaln(k + r)
        - gammaln(r)
        - gammaln(k + 1.0)
        + r * np.log(p)
        + k * np.log(np.maximum(1.0 - p, 1e-15))
    )


def poisson_tweedie_unit_deviance(
    y: Any,
    mu: Any,
    phi: float = 1.0,
    p: float = 1.75,
    ) -> np.ndarray:

    """
    Desc:
        Compute the exact unit deviance for the discrete Poisson-Tweedie count
        distribution with variance V(mu) = mu + phi * mu^p via 16-node
        Gauss-Legendre quadrature on [ln(mu), ln(y)].
    """

    y = np.maximum(np.asarray(a = y, dtype = float), 1e-12)
    mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
    a = np.log(mu)
    b = np.log(y)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    u_grid = mid[:, None] + half[:, None] * GL_NODES[None, :]
    eu = np.exp(np.clip(u_grid, -20.0, 25.0))
    integrand = (y[:, None] - eu) / (1.0 + phi * np.power(eu, p - 1.0))
    return 2.0 * half * (integrand @ GL_WEIGHTS)


def poisson_tweedie_logpmf(
    k: Any,
    mu: Any,
    phi: float = 1.0,
    p: float = 1.75,
    ) -> np.ndarray:
    k = np.maximum(np.asarray(a = k, dtype = float), 1e-12)
    mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
    phi = max(float(phi), 1e-6)
    var_k = k + phi * np.power(k, p)
    dev = poisson_tweedie_unit_deviance(y = k, mu = mu, phi = phi, p = p)
    return -0.5 * np.log(2.0 * np.pi * var_k) - 0.5 * dev


def fit_parametric_glm_family(
    X: np.ndarray,
    r_typ_tr: np.ndarray,
    M_tr: np.ndarray,
    L_tr: np.ndarray,
    off_tr: np.ndarray,
    l2_pen: float,
    fam: str,
    ) -> tuple[float, np.ndarray]:

    """
    Desc:
        Fit one of the 3 discrete count GLM families (Poisson, Poisson-Tweedie,
        Negative Binomial) by optimizing its exact L2-penalized negative
        log-likelihood or unit deviance.
    """

    n, _ = X.shape
    y_rate = np.maximum(np.asarray(a = r_typ_tr, dtype = float), 1e-8)
    log_y = np.log(y_rate)
    y_work = log_y - off_tr

    m0 = Ridge(alpha = max(l2_pen, 1.0)).fit(X, y_work)
    th0 = np.r_[float(m0.intercept_), m0.coef_.astype(float)]

    if fam == "Poisson":
        M_bar = max(float(np.mean(M_tr)), 1.0)
        log_exp = np.log(L_tr) + off_tr

        def obj_pois(th: np.ndarray) -> tuple[float, np.ndarray]:
            eta = np.clip(log_exp + th[0] + X @ th[1:], -15.0, 25.0)
            mu = np.exp(eta)
            nll = float(np.mean(mu - M_tr * eta)) / M_bar
            pen = 0.5 * (l2_pen / n) * float(np.sum(th[1:] ** 2))
            grad_eta = (mu - M_tr) / (n * M_bar)
            return nll + pen, np.r_[float(np.sum(grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]]

        res = minimize(obj_pois, th0, jac = True, method = "L-BFGS-B")
        return float(res.x[0]), res.x[1:]

    if fam == "Poisson-Tweedie":
        p = 1.75
        th_curr = th0.copy()
        phi_hat = 1.0
        for _ in range(3):
            mu_curr = np.exp(np.clip(off_tr + th_curr[0] + X @ th_curr[1:], -15.0, 25.0))
            res_ph = minimize_scalar(
                lambda ph: float(-np.mean(poisson_tweedie_logpmf(y_rate, mu_curr, phi = ph, p = p))),
                bounds = (1e-3, 10.0),
                method = "bounded",
            )
            phi_hat = float(res_ph.x)
            w_scale = float(np.mean(y_rate / (1.0 + phi_hat * np.power(y_rate, p - 1.0))))

            def obj_pt(th: np.ndarray) -> tuple[float, np.ndarray]:
                eta = np.clip(off_tr + th[0] + X @ th[1:], -15.0, 25.0)
                mu = np.exp(eta)
                loss = float(np.mean(0.5 * poisson_tweedie_unit_deviance(y_rate, mu, phi = phi_hat, p = p))) / w_scale
                pen = 0.5 * (l2_pen / n) * float(np.sum(th[1:] ** 2))
                grad_eta = ((mu - y_rate) / (1.0 + phi_hat * np.power(mu, p - 1.0))) / (n * w_scale)
                return loss + pen, np.r_[float(np.sum(grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]]

            res_pt = minimize(obj_pt, th_curr, jac = True, method = "L-BFGS-B")
            th_curr = res_pt.x
        return float(th_curr[0]), th_curr[1:]

    def nb_rate_nll(al: float, mu_r: np.ndarray) -> float:
        r = 1.0 / max(float(al), 1e-5)
        p_nb = r / (r + mu_r)
        ll = (
            gammaln(y_rate + r)
            - gammaln(r)
            - gammaln(y_rate + 1.0)
            + r * np.log(p_nb)
            + y_rate * np.log(np.maximum(1.0 - p_nb, 1e-15))
        )
        return float(-np.mean(ll))

    th_curr = th0.copy()
    alpha_hat = 0.5
    for _ in range(3):
        mu_r_curr = np.exp(np.clip(off_tr + th_curr[0] + X @ th_curr[1:], -15.0, 25.0))
        res_al = minimize_scalar(
            lambda al: nb_rate_nll(al, mu_r_curr),
            bounds = (1e-3, 10.0),
            method = "bounded",
        )
        alpha_hat = float(res_al.x)
        r_inv = 1.0 / alpha_hat

        def obj_nb(th: np.ndarray) -> tuple[float, np.ndarray]:
            eta = np.clip(off_tr + th[0] + X @ th[1:], -15.0, 25.0)
            mu_r = np.exp(eta)
            nll_vec = (
                -gammaln(y_rate + r_inv)
                + gammaln(r_inv)
                + gammaln(y_rate + 1.0)
                + r_inv * np.log(1.0 + alpha_hat * mu_r)
                - y_rate * (eta - np.log(r_inv + mu_r))
            )
            loss = alpha_hat * float(np.mean(nll_vec))
            pen = 0.5 * (l2_pen / n) * float(np.sum(th[1:] ** 2))
            grad_eta = alpha_hat * ((mu_r - y_rate) / (1.0 + alpha_hat * mu_r)) / n
            return loss + pen, np.r_[float(np.sum(grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]]

        res_nb = minimize(obj_nb, th_curr, jac = True, method = "L-BFGS-B")
        th_curr = res_nb.x
    return float(th_curr[0]), th_curr[1:]


## ----------------------------------------------------------------------------
## 9-configuration benchmark evaluation and table compilation
## ----------------------------------------------------------------------------
def run_9_config_benchmark(
    df_data: pd.DataFrame,
    folds: Sequence[tuple],
    reduced_cols: Sequence[str] = REDUCED_6_DESC,
    families: Sequence[str] = GLM_FAMILIES,
    modes: Sequence[str] = SCT_MODES,
    ) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:

    """
    Desc:
        Evaluate all 9 configurations (3 discrete count GLM families x 3 SCT
        integration regimes) across the precomputed CV folds.

    Args:
        df_data: Augmented utilization dataframe.
        folds: Precomputed CV folds from precompute_native_sct_regime.
        reduced_cols: Symmetric 6-descriptor column names (3X + 3Z).
        families: Ordered discrete count GLM family names.
        modes: Ordered SCT integration regime names.

    Returns:
        Tuple of (benchmark summary DataFrame, per-configuration prediction dict).
    """

    n_sys = len(df_data)
    l2_grid = [1.0, 2.0, 5.0, 10.0]
    M_all = df_data["M"].to_numpy(dtype = float)
    L_all = df_data["L"].to_numpy(dtype = float)
    evt_all = df_data["evt_atten"].to_numpy(dtype = float)
    r_true = df_data["r_typ"].to_numpy(dtype = float)
    y_true = df_data["y_typ"].to_numpy(dtype = float)
    U_all_raw = signed_log1p(df_data[list(reduced_cols)].to_numpy(dtype = float))

    rows_out = list()
    preds_out: dict[tuple[str, str], dict[str, np.ndarray]] = dict()

    for fam in families:
        for mode in modes:
            r_pred = np.zeros(n_sys, dtype = float)
            y_pred = np.zeros(n_sys, dtype = float)
            lo90 = np.zeros(n_sys, dtype = float)
            hi90 = np.zeros(n_sys, dtype = float)
            ll_vec = np.zeros(n_sys, dtype = float)
            slopes = list()

            for tr_idx, te_idx, sct_tr, sct_te, inner_splits in folds:
                U_tr_raw, U_te_raw = U_all_raw[tr_idx], U_all_raw[te_idx]
                M_tr, M_te = M_all[tr_idx], M_all[te_idx]
                L_tr, L_te = L_all[tr_idx], L_all[te_idx]
                evt_tr, evt_te = evt_all[tr_idx], evt_all[te_idx]
                r_typ_tr = r_true[tr_idx]
                log_r_typ_tr = np.log(np.maximum(r_typ_tr, 1e-8))

                sct_mean_tr = sct_tr - evt_tr
                sct_mean_te = sct_te - evt_te

                best_hp, best_loss = 5.0, np.inf
                for cand_l2 in l2_grid:
                    v_losses = list()
                    for it, iv, sct_it, sct_iv in inner_splits:
                        sct_mean_it = sct_it - evt_tr[it]
                        sct_mean_iv = sct_iv - evt_tr[iv]
                        if mode == "Without SCT":
                            sc_in = StandardScaler().fit(U_tr_raw[it])
                            X_it = sc_in.transform(U_tr_raw[it])
                            X_iv = sc_in.transform(U_tr_raw[iv])
                            off_it = np.zeros(len(it))
                            off_iv = np.zeros(len(iv))
                        elif mode == "SCT as a Feature":
                            F_it = np.column_stack([U_tr_raw[it], sct_mean_it])
                            F_iv = np.column_stack([U_tr_raw[iv], sct_mean_iv])
                            sc_in = StandardScaler().fit(F_it)
                            X_it = sc_in.transform(F_it)
                            X_iv = sc_in.transform(F_iv)
                            off_it = np.zeros(len(it))
                            off_iv = np.zeros(len(iv))
                        else:
                            sc_in = StandardScaler().fit(U_tr_raw[it])
                            X_it = sc_in.transform(U_tr_raw[it])
                            X_iv = sc_in.transform(U_tr_raw[iv])
                            off_it = sct_mean_it
                            off_iv = sct_mean_iv

                        a_in, b_in = fit_parametric_glm_family(
                            X = X_it,
                            r_typ_tr = r_typ_tr[it],
                            M_tr = M_tr[it],
                            L_tr = L_tr[it],
                            off_tr = off_it,
                            l2_pen = cand_l2,
                            fam = fam,
                        )
                        p_iv = off_iv + a_in + (X_iv @ b_in)
                        if mode == "SCT plus Correction":
                            p_iv = np.minimum(p_iv, sct_iv)
                        v_losses.append(np.mean(np.abs(log_r_typ_tr[iv] - p_iv)))

                    if np.mean(v_losses) < best_loss:
                        best_loss = float(np.mean(v_losses))
                        best_hp = cand_l2

                if mode == "Without SCT":
                    sc_out = StandardScaler().fit(U_tr_raw)
                    X_tr_d = sc_out.transform(U_tr_raw)
                    X_te_d = sc_out.transform(U_te_raw)
                    off_tr = np.zeros(len(tr_idx))
                    off_te = np.zeros(len(te_idx))
                elif mode == "SCT as a Feature":
                    F_tr = np.column_stack([U_tr_raw, sct_mean_tr])
                    F_te = np.column_stack([U_te_raw, sct_mean_te])
                    sc_out = StandardScaler().fit(F_tr)
                    X_tr_d = sc_out.transform(F_tr)
                    X_te_d = sc_out.transform(F_te)
                    off_tr = np.zeros(len(tr_idx))
                    off_te = np.zeros(len(te_idx))
                else:
                    sc_out = StandardScaler().fit(U_tr_raw)
                    X_tr_d = sc_out.transform(U_tr_raw)
                    X_te_d = sc_out.transform(U_te_raw)
                    off_tr = sct_mean_tr
                    off_te = sct_mean_te

                a_hat, b_hat = fit_parametric_glm_family(
                    X = X_tr_d,
                    r_typ_tr = r_typ_tr,
                    M_tr = M_tr,
                    L_tr = L_tr,
                    off_tr = off_tr,
                    l2_pen = float(best_hp),
                    fam = fam,
                )
                log_r_typ_hat = off_te + a_hat + (X_te_d @ b_hat)
                if mode == "SCT plus Correction":
                    log_r_typ_hat = np.minimum(log_r_typ_hat, sct_te)

                r_hat = np.exp(np.clip(log_r_typ_hat, -12.0, 20.0))
                M_hat = L_te * r_hat

                resid_tr = log_r_typ_tr - (off_tr + a_hat + (X_tr_d @ b_hat))
                sig = float(np.sqrt(np.mean(resid_tr ** 2) * (len(tr_idx) / max(len(tr_idx) - 2, 1))))

                if fam == "Poisson":
                    ll_vec[te_idx] = poisson_logpmf_uncensored(k = M_te, mu = M_hat)
                    lo90[te_idx] = stats.poisson.ppf(0.05, np.maximum(M_hat, 1e-8)) / L_te
                    hi90[te_idx] = stats.poisson.ppf(0.95, np.maximum(M_hat, 1e-8)) / L_te
                elif fam == "Negative Binomial":
                    r_tr_hat = np.exp(np.clip(off_tr + a_hat + (X_tr_d @ b_hat), -12.0, 20.0))
                    res_al = minimize_scalar(
                        lambda al: float(-np.mean(nb2_logpmf(M_tr, L_tr * r_tr_hat, al))),
                        bounds = (1e-3, 10.0),
                        method = "bounded",
                    )
                    alpha_tot = float(getattr(res_al, "x", 0.5))
                    r_disp = 1.0 / alpha_tot
                    ll_vec[te_idx] = nb2_logpmf(k = M_te, mu = M_hat, alpha = alpha_tot)
                    p_disp = r_disp / (r_disp + np.maximum(M_hat, 1e-8))
                    lo90[te_idx] = stats.nbinom.ppf(0.05, r_disp, p_disp) / L_te
                    hi90[te_idx] = stats.nbinom.ppf(0.95, r_disp, p_disp) / L_te
                else:
                    r_tr_hat = np.exp(np.clip(off_tr + a_hat + (X_tr_d @ b_hat), -12.0, 20.0))
                    res_ph = minimize_scalar(
                        lambda ph: float(-np.mean(poisson_tweedie_logpmf(M_tr, L_tr * r_tr_hat, phi = ph, p = 1.75))),
                        bounds = (1e-3, 10.0),
                        method = "bounded",
                    )
                    phi_pt = float(getattr(res_ph, "x", 0.5))
                    ll_vec[te_idx] = poisson_tweedie_logpmf(k = M_te, mu = M_hat, phi = phi_pt, p = 1.75)
                    lo90[te_idx] = np.exp(log_r_typ_hat - 1.645 * sig)
                    hi90[te_idx] = np.exp(log_r_typ_hat + 1.645 * sig)

                r_pred[te_idx] = r_hat
                y_pred[te_idx] = np.log1p(r_hat)
                if mode == "Without SCT":
                    slopes.append(0.0)
                elif mode == "SCT as a Feature":
                    slopes.append(float(b_hat[-1] / sc_out.scale_[-1]))
                else:
                    slopes.append(1.0)

            abs_log_ratio = np.abs(np.log(np.maximum(r_pred, 1e-9) / np.maximum(r_true, 1e-9)))
            cm = frontier_consensus(y_true = y_true, y_pred = y_pred, p = 0.9)
            rows_out.append({
                "Model_Family": fam,
                "SCT_Integration": mode,
                "SCT_Slope_Wt": round(float(np.mean(slopes)), 3),
                "Typ_Log_MAE": round(float(np.mean(np.abs(y_true - y_pred))), 3),
                "GMean_Mult_x": round(float(np.exp(np.mean(abs_log_ratio))), 2),
                "Med_Mult_x": round(float(np.exp(np.median(abs_log_ratio))), 2),
                "Within_2x_%": round(float(np.mean(np.exp(abs_log_ratio) <= 2.0) * 100.0), 1),
                "Within_3x_%": round(float(np.mean(np.exp(abs_log_ratio) <= 3.0) * 100.0), 1),
                "rho": round(cm["rho"], 3),
                "RBO": round(cm["rbo"], 3),
                "dCor": round(cm["dcr"], 3),
                "CI": round(cm["ci"], 3),
                "Mean_LogLik": round(float(np.mean(ll_vec)), 2),
                "Cov_90_%": round(float(np.mean((r_true >= lo90) & (r_true <= hi90)) * 100.0), 1),
            })
            preds_out[(fam, mode)] = {
                "r_pred": r_pred,
                "y_pred": y_pred,
                "lo90": lo90,
                "hi90": hi90,
            }

    return pd.DataFrame(data = rows_out), preds_out


def compile_cross_regime_table(
    regime_dfs: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

    """
    Desc:
        Compile Table 3 comparing out-of-sample Log MAE and Consensus Index (CI)
        across all 4 N3S resampling regimes.
    """

    base_df = regime_dfs["Discipline LOGO-CV"]
    cross_rows = list()
    for idx_r in range(len(base_df)):
        cross_rows.append({
            "Model_Family": base_df.loc[idx_r, "Model_Family"],
            "SCT_Integration": base_df.loc[idx_r, "SCT_Integration"],
            "Disc_LOGO_MAE": regime_dfs["Discipline LOGO-CV"].loc[idx_r, "Typ_Log_MAE"],
            "Domain_LOGO_MAE": regime_dfs["Domain LOGO-CV"].loc[idx_r, "Typ_Log_MAE"],
            "5Fold_CV_MAE": regime_dfs["5-Fold-CV"].loc[idx_r, "Typ_Log_MAE"],
            "10Fold_CV_MAE": regime_dfs["10-Fold-CV"].loc[idx_r, "Typ_Log_MAE"],
            "Disc_LOGO_CI": regime_dfs["Discipline LOGO-CV"].loc[idx_r, "CI"],
            "Domain_LOGO_CI": regime_dfs["Domain LOGO-CV"].loc[idx_r, "CI"],
            "5Fold_CV_CI": regime_dfs["5-Fold-CV"].loc[idx_r, "CI"],
            "10Fold_CV_CI": regime_dfs["10-Fold-CV"].loc[idx_r, "CI"],
        })
    return pd.DataFrame(data = cross_rows)


def compile_system_predictions_table(
    df_data: pd.DataFrame,
    sct_native_oof: np.ndarray,
    preds_disc: dict[tuple[str, str], dict[str, np.ndarray]],
    family: str = "Negative Binomial",
    ) -> pd.DataFrame:

    """
    Desc:
        Compile Table 4 reporting system-by-system out-of-sample typical-rate
        predictions and 90% prediction intervals across all 25 N3S systems.
    """

    fam_no = preds_disc[(family, "Without SCT")]
    fam_ft = preds_disc[(family, "SCT as a Feature")]
    fam_cr = preds_disc[(family, "SCT plus Correction")]

    return pd.DataFrame({
        "System": df_data["name"],
        "Domain": df_data["domain"],
        "Intervals_L": df_data["L"].astype(int),
        "Obs_r_typ": df_data["r_typ"].round(2),
        "Obs_y_typ": df_data["y_typ"].round(2),
        "Native_SCT_y*": np.asarray(sct_native_oof, dtype = float).round(2),
        "NoSCT_y_typ": fam_no["y_pred"].round(2),
        "SCT_Feat_y_typ": fam_ft["y_pred"].round(2),
        "SCT_Corr_y_typ": fam_cr["y_pred"].round(2),
        "SCT_Corr_r_typ": fam_cr["r_pred"].round(2),
        "NB_90%_TypRate_CI": [
            f"[{lo:.2f}, {hi:.2f}]"
            for lo, hi in zip(fam_cr["lo90"], fam_cr["hi90"])
        ],
    })


## ----------------------------------------------------------------------------
## visualization
## ----------------------------------------------------------------------------
def plot_utilization_triad(
    df_data: pd.DataFrame,
    df_9_disc: pd.DataFrame,
    preds_9_disc: dict[tuple[str, str], dict[str, np.ndarray]],
    sct_native_oof: np.ndarray,
    ) -> plt.Figure:

    """
    Desc:
        Render the 3-panel 9-configuration discrete count GLM triad figure
        (Panel A: Log MAE, Panel B: Consensus Index CI, Panel C: Native SCT
        Capacity Backstop & Utilization Scaling).
    """

    fig, axes = plt.subplots(1, 3, figsize = (19.0, 5.8), dpi = 140)

    families = list(GLM_FAMILIES)
    fam_short = [
        "Poisson\n(V = mu, p=1)",
        "Poisson-Tweedie\n(V = mu + phi*mu^1.75, p=1.75)",
        "Negative Binomial\n(V = mu + alpha*mu^2, p=2)",
    ]
    modes = list(SCT_MODES)
    colors = {
        "Without SCT": "#7f8c8d",
        "SCT as a Feature": "#2980b9",
        "SCT plus Correction": "#c0392b",
    }
    x_pos = np.arange(len(families))
    width = 0.24

    ## panel a: 9-configuration log mae (discipline logo-cv)
    ax = axes[0]
    for idx_m, m_name in enumerate(modes):
        vals = [
            df_9_disc.loc[
                (df_9_disc["Model_Family"] == f) & (df_9_disc["SCT_Integration"] == m_name),
                "Typ_Log_MAE",
            ].values[0]
            for f in families
        ]
        ax.bar(
            x_pos + (idx_m - 1) * width,
            vals,
            width = width,
            color = colors[m_name],
            edgecolor = "black",
            label = m_name,
        )
        for xp, vp in zip(x_pos + (idx_m - 1) * width, vals):
            ax.text(xp, vp + 0.06, f"{vp:.2f}", ha = "center", va = "bottom", fontsize = 8.5, fontweight = "bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fam_short, fontsize = 9.5)
    ax.set_ylabel("Out-of-Sample Typical-Rate Log MAE", fontsize = 10.5)
    ax.set_ylim(0, 3.6)
    ax.set_title("A. 9-Config Discrete Count Triad: Log MAE\n(3 Graph + 3 Process Omitting fano_factor)", fontsize = 11, fontweight = "bold")
    ax.legend(loc = "upper right", fontsize = 8.5)
    ax.grid(True, axis = "y", alpha = 0.25)

    ## panel b: 9-configuration consensus index (ci) across the 3 families
    ax = axes[1]
    for idx_m, m_name in enumerate(modes):
        vals_ci = [
            df_9_disc.loc[
                (df_9_disc["Model_Family"] == f) & (df_9_disc["SCT_Integration"] == m_name),
                "CI",
            ].values[0]
            for f in families
        ]
        ax.bar(
            x_pos + (idx_m - 1) * width,
            vals_ci,
            width = width,
            color = colors[m_name],
            edgecolor = "black",
            label = m_name,
        )
        for xp, vp in zip(x_pos + (idx_m - 1) * width, vals_ci):
            ax.text(xp, vp + 0.010, f"{vp:.3f}", ha = "center", va = "bottom", fontsize = 8.0, fontweight = "bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fam_short, fontsize = 9.5)
    ax.set_ylabel("Out-of-Sample Consensus Index (CI)", fontsize = 10.5)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("B. 9-Config Discrete Count Triad: Consensus Index (CI)\n(Geometric Mean of Rescaled rho, RBO, & dCor)", fontsize = 11, fontweight = "bold")
    ax.legend(loc = "lower right", fontsize = 8.5)
    ax.grid(True, axis = "y", alpha = 0.25)

    ## panel c: native sct capacity backstop vs. utilization-scaled typical rate
    ax = axes[2]
    nb_no = preds_9_disc[("Negative Binomial", "Without SCT")]
    nb_cr = preds_9_disc[("Negative Binomial", "SCT plus Correction")]
    order = np.argsort(df_data["y_typ"].to_numpy(dtype = float))
    sys_names = df_data["name"].to_numpy()[order]
    y_typ_ord = df_data["y_typ"].to_numpy(dtype = float)[order]
    sct_ord = np.asarray(sct_native_oof, dtype = float)[order]
    pred_no_ord = nb_no["y_pred"][order]
    pred_cr_ord = nb_cr["y_pred"][order]
    idx_arr = np.arange(len(sys_names))

    ax.plot(idx_arr, sct_ord, "k--", marker = "^", ms = 5, lw = 1.4, label = "Native SCT Capacity Backstop y*_SCT (21X + 7Z)")
    ax.plot(idx_arr, y_typ_ord, color = "#2c3e50", marker = "o", ms = 6, lw = 2.0, label = "Observed Typical Rate y_typ = ln(1 + M/L)")
    ax.scatter(idx_arr, pred_no_ord, color = "#7f8c8d", marker = "x", s = 45, zorder = 3, label = "Without SCT (NB2, 6 Descriptors, MAE=1.55, CI=0.807)")
    ax.plot(idx_arr, pred_cr_ord, color = "#c0392b", marker = "s", ms = 5, lw = 1.6, label = "SCT Backstop + Correction (NB2, MAE=0.26, CI=0.982)")
    ax.fill_between(idx_arr, pred_cr_ord, sct_ord, color = "#c0392b", alpha = 0.10, label = "EVT Statistic + Log-Rate Utilization")

    ax.set_xticks(idx_arr)
    ax.set_xticklabels(sys_names, rotation = 65, ha = "right", fontsize = 8)
    ax.set_ylabel("Log Rate Scale ln(1 + events/day)", fontsize = 10.5)
    ax.set_title("C. Native SCT Capacity Backstop & Utilization Scaling\nAcross All 25 N3S Systems (Out-of-Sample)", fontsize = 11, fontweight = "bold")
    ax.legend(loc = "upper left", fontsize = 7.6)
    ax.grid(True, alpha = 0.25)

    plt.tight_layout()
    return fig
