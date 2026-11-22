## libraries
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib._mathtext as _mathtext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetBox, VPacker
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from io import BytesIO
from html import escape
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Iterator, Sequence
from scipy.special import gammaln
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut
from IPython.display import HTML, Image
from IPython.core.pylabtools import print_figure

## path
root = Path(__file__).resolve().parents[2]

## modules
from src.data.builders import load_processed_data
from src.evaluators.metrics import frontier_consensus
from src.estimators.linear_quantile import LinearQuantile
from src.vectorizers.signatures import ProcessSignatures
from src.visualizers.visualize import export_pdf_scaled
from src.evaluators.helpers import (
    _clean_differences,
    paired_rank_biserial
)

## constants
from src.data.builders import PATH_PROC
from src.estimators.config import (
    ASYMMETRY_C,
    ASYMMETRY_R,
    ALPHA,
)
from src.evaluators.config import (
    FEAT_X,
    FEAT_Z,
    FEAT_XU,
    FEAT_ZU,
    TARGET,
)
FEAT_6U = FEAT_XU + FEAT_ZU

GLM_ESTIMATORS = (
    "Poisson",
    "Poisson-Tweedie",
    "Negative Binomial",
)
SCT_MODALITIES = (
    "SCT Frontier Exclusion",
    "SCT Frontier Anchor",
)
RESAMPLING_REGIMES = (
    "Domain LOGO-CV",
    "Discipline LOGO-CV",
    "5-Fold-CV",
    "10-Fold-CV",
)

UTIL_METRICS = ("MAE", "RMSE", "CI", "GMME")
UTIL_ABLATIONS = (
    "SCT Frontier Anchor",
    "EVT Statistic Exclusion",
    "Utilization Features Exclusion",
    "SCT Frontier Anchor Only",
    "SCT Frontier Exclusion",
    "Arbitrary Ceiling (3x Log-Max)",
    "SCT Feature Match Offset",
    "SCT Frontier Remap",
)

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(deg = 16)


## signed log transform
def signed_log1p(arr: Any) -> np.ndarray:
    arr = np.asarray(a = arr, dtype = float)
    return np.sign(arr) * np.log1p(np.abs(arr))


## load and process utilization data
def load_utilization_data(
    data: pd.DataFrame | None = None,
    proc_dir: str | Path | None = None,
    ) -> pd.DataFrame:

        ## load processed systems sorted by name
        df_n3s = (
            (load_processed_data() if data is None else data.copy())
            .sort_values(by = "name")
            .reset_index(drop = True)
        )
        resolved_proc_dir = (root / PATH_PROC) if proc_dir is None else Path(proc_dir)

        ## read each system's event series
        rows = list()
        for _, row in df_n3s.iterrows():
            name = str(row["name"])
            json_path = resolved_proc_dir / f"{name}.json"
            with json_path.open(mode = "r", encoding = "utf-8") as source:
                evs = json.load(fp = source)["events"]

            ## derive expected rates and evt adjustment from each event series
            L = len(evs)
            M = float(sum(float(e["target"]) for e in evs))
            r_mean = M / L
            r_max = float(row[TARGET])
            cv_val = float(row["coef_variation"])
            evt_atten = float(np.log1p(cv_val * np.sqrt(2.0 * np.log(max(L, 2)))))

            rows.append(
                {
                    "L": float(L),
                    "M": float(np.round(a = M)),
                    "r_mean": r_mean,
                    "r_max": r_max,
                    "utilization": r_mean / r_max,
                    "y_max": np.log1p(r_max),
                    "y_mean": np.log1p(r_mean),
                    "evt_atten": evt_atten,
                }
            )
        return pd.concat(objs = [df_n3s, pd.DataFrame(data = rows)], axis = 1)


## compile corpus table for display
def compile_corpus_table(df_data: pd.DataFrame) -> pd.DataFrame:
    df_t1 = df_data[
        [
            "name",
            "domain",
            "L",
            "M",
            "r_mean",
            "r_max",
            "utilization",
            "y_mean",
            "y_max",
        ]
    ].copy()
    df_t1["L"] = df_t1["L"].astype(dtype = int)
    df_t1["M"] = df_t1["M"].astype(dtype = np.int64)
    df_t1["r_mean"] = df_t1["r_mean"].round(2)
    df_t1["r_max"] = df_t1["r_max"].round(1)
    df_t1["utilization"] = (df_t1["utilization"] * 100.0).round(2)
    df_t1["y_mean"] = df_t1["y_mean"].round(2)
    df_t1["y_max"] = df_t1["y_max"].round(2)
    return df_t1.rename(columns = {"utilization": "Util_%"})


## native two-stage sct frontier (linear quantile)
def fit_native_sct_frontier(
    X_tr: np.ndarray,
    Z_tr: np.ndarray,
    ym_tr: np.ndarray,
    X_ev: np.ndarray,
    Z_ev: np.ndarray,
    quantile_c: float = ASYMMETRY_C,
    ) -> tuple[np.ndarray, np.ndarray]:

        ## fit feature scaling on training systems only
        xs = StandardScaler().fit(X = signed_log1p(arr = X_tr))
        zs = StandardScaler().fit(X = signed_log1p(arr = Z_tr))
        X_tr_s = xs.transform(X = signed_log1p(arr = X_tr))
        X_ev_s = xs.transform(X = signed_log1p(arr = X_ev))
        Z_tr_s = zs.transform(X = signed_log1p(arr = Z_tr))
        Z_ev_s = zs.transform(X = signed_log1p(arr = Z_ev))

        lq = LinearQuantile(
            quantile_c = quantile_c,
            quantile_r = ASYMMETRY_R,
            alpha = ALPHA,
        )

        mc = lq.estimator_c.fit(X = X_tr_s, y = ym_tr)
        c_tr = mc.predict(X = X_tr_s)

        ## fit process effects to the structural residuals
        mr = lq.estimator_r.fit(X = Z_tr_s, y = ym_tr - c_tr)
        r_tr = mr.predict(X = Z_tr_s)
        r_mean = float(np.mean(a = r_tr))

        ## center both predictions using the training process mean
        p_tr = c_tr + (r_tr - r_mean)
        p_ev = mc.predict(X = X_ev_s) + (mr.predict(X = Z_ev_s) - r_mean)

        ## return predictions
        return p_tr, p_ev


def precompute_native_sct_regime(
    df_data: pd.DataFrame,
    regime_name: str = "Domain LOGO-CV",
    feat_x: Sequence[str] = FEAT_X,
    feat_z: Sequence[str] = FEAT_Z,
    random_state: int = 42,
    target: str = "y_max",
    quantile_c: float = ASYMMETRY_C,
    inner_random_state: int | None = None,
    ) -> list[tuple]:

        ## prepare data arrays for all samples
        X_all = df_data[list(feat_x)].to_numpy(dtype = float)
        Z_all = df_data[list(feat_z)].to_numpy(dtype = float)
        ym_all = df_data[target].to_numpy(dtype = float)
        inner_kf = KFold(
            n_splits = 4,
            shuffle = True,
            random_state = random_state if inner_random_state is None else inner_random_state,
        )

        ## determine outer splits based on regime
        if regime_name == "Discipline LOGO-CV":
            outer_splits = list(
                LeaveOneGroupOut().split(X = df_data, groups = df_data["discipline"])
            )
        elif regime_name == "Domain LOGO-CV":
            outer_splits = list(
                LeaveOneGroupOut().split(X = df_data, groups = df_data["domain"])
            )
        elif regime_name == "5-Fold-CV":
            outer_splits = list(
                KFold(n_splits = 5, shuffle = True, random_state = random_state).split(X = df_data)
            )
        elif regime_name == "10-Fold-CV":
            outer_splits = list(
                KFold(n_splits = 10, shuffle = True, random_state = random_state).split(X = df_data)
            )
        else:
            raise ValueError(f"Unknown resampling regime: {regime_name!r}")

        ## loop over outer splits
        folds = list()
        for tr_idx, te_idx in outer_splits:
            sct_tr, sct_te = fit_native_sct_frontier(
                X_tr = X_all[tr_idx],
                Z_tr = Z_all[tr_idx],
                ym_tr = ym_all[tr_idx],
                X_ev = X_all[te_idx],
                Z_ev = Z_all[te_idx],
                quantile_c = quantile_c,
            )
            inner_splits = list()

            ## refit the frontier within each inner fold for penalty selection
            for it, iv in inner_kf.split(X = tr_idx):
                s_it, s_iv = fit_native_sct_frontier(
                    X_tr = X_all[tr_idx[it]],
                    Z_tr = Z_all[tr_idx[it]],
                    ym_tr = ym_all[tr_idx[it]],
                    X_ev = X_all[tr_idx[iv]],
                    Z_ev = Z_all[tr_idx[iv]],
                    quantile_c = quantile_c,
                )
                inner_splits.append((it, iv, s_it, s_iv))
            folds.append((tr_idx, te_idx, sct_tr, sct_te, inner_splits))
        return folds


## discrete count glm likelihoods and fitting
def poisson_tweedie_unit_deviance(
    y: Any,
    mu: Any,
    phi: float = 1.0,
    p: float = 1.75,
    ) -> np.ndarray:

        ## ensure positivity
        y = np.maximum(np.asarray(a = y, dtype = float), 1e-12)
        mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
        
        ## integrate log unit deviance with gauss-legendre quadrature
        a = np.log(mu)
        b = np.log(y)
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        u_grid = mid[:, None] + half[:, None] * GL_NODES[None, :]
        eu = np.exp(np.clip(a = u_grid, a_min = -20.0, a_max = 25.0))
        integrand = (y[:, None] - eu) / (1.0 + phi * np.power(eu, p - 1.0))

        ## compute deviance integrand
        return 2.0 * half * (integrand @ GL_WEIGHTS)


## poisson-tweedie log pmf
def poisson_tweedie_logpmf(
    k: Any,
    mu: Any,
    phi: float = 1.0,
    p: float = 1.75,
    ) -> np.ndarray:

        ## ensure positivity
        k = np.maximum(np.asarray(a = k, dtype = float), 1e-12)
        mu = np.maximum(np.asarray(a = mu, dtype = float), 1e-12)
        phi = max(float(phi), 1e-6)

        ## approximate the count log score with a saddlepoint correction
        var_k = k + phi * np.power(k, p)
        dev = poisson_tweedie_unit_deviance(y = k, mu = mu, phi = phi, p = p)

        ## add a small correction to deviance
        return -0.5 * np.log(2.0 * np.pi * var_k) - 0.5 * dev


## fit parametric glm family
def fit_parametric_glm_family(
    X: np.ndarray,
    r_mean_tr: np.ndarray,
    M_tr: np.ndarray,
    L_tr: np.ndarray,
    off_tr: np.ndarray,
    l2_pen: float,
    fam: str,
    ) -> tuple[float, np.ndarray]:

        ## number of samples
        n, _ = X.shape

        ## preprocess rates
        y_rate = np.maximum(np.asarray(a = r_mean_tr, dtype = float), 1e-8)
        log_y = np.log(y_rate)
        y_work = log_y - off_tr

        ## init the intercept and coefficients from offset-adjusted log rates
        m0 = Ridge(alpha = max(l2_pen, 1.0)).fit(X = X, y = y_work)
        th0 = np.r_[float(m0.intercept_), m0.coef_.astype(dtype = float)]

        ## fit glm for the mean
        if fam == "Poisson":
            M_bar = max(float(np.mean(a = M_tr)), 1.0)
            
            ## include observation length as poisson exposure
            log_exp = np.log(L_tr) + off_tr

            def obj_pois(th: np.ndarray) -> tuple[float, np.ndarray]:
                eta = np.clip(a = log_exp + th[0] + X @ th[1:], a_min = -15.0, a_max = 25.0)
                mu = np.exp(eta)
                nll = float(np.mean(a = mu - M_tr * eta)) / M_bar
                pen = 0.5 * (l2_pen / n) * float(np.sum(a = th[1:] ** 2))
                grad_eta = (mu - M_tr) / (n * M_bar)
                return nll + pen, np.r_[
                    float(np.sum(a = grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]
                ]

            res = minimize(fun = obj_pois, x0 = th0, jac = True, method = "L-BFGS-B")
            return float(res.x[0]), res.x[1:]

        ## alternate rate-based dispersion and penalized deviance fitting
        if fam == "Poisson-Tweedie":
            p = 1.75
            th_curr = th0.copy()
            phi_hat = 1.0
            for _ in range(3):
                mu_curr = np.exp(
                    np.clip(
                        a = off_tr + th_curr[0] + X @ th_curr[1:], a_min = -15.0, a_max = 25.0
                    )
                )
                res_ph = minimize_scalar(
                    fun = lambda ph: float(
                        -np.mean(
                            a = poisson_tweedie_logpmf(k = y_rate, mu = mu_curr, phi = ph, p = p)
                        )
                    ),
                    bounds = (1e-3, 10.0),
                    method = "bounded",
                )
                phi_hat = float(res_ph.x)
                w_scale = float(
                    np.mean(a = y_rate / (1.0 + phi_hat * np.power(y_rate, p - 1.0)))
                )

                ## objective for poisson-tweedie
                def obj_pt(th: np.ndarray) -> tuple[float, np.ndarray]:
                    eta = np.clip(a = off_tr + th[0] + X @ th[1:], a_min = -15.0, a_max = 25.0)
                    mu = np.exp(eta)
                    loss = (
                        float(
                            np.mean(
                                a = 0.5
                                * poisson_tweedie_unit_deviance(
                                    y = y_rate, mu = mu, phi = phi_hat, p = p
                                )
                            )
                        )
                        / w_scale
                    )
                    pen = 0.5 * (l2_pen / n) * float(np.sum(a = th[1:] ** 2))
                    grad_eta = ((mu - y_rate) / (1.0 + phi_hat * np.power(mu, p - 1.0))) / (
                        n * w_scale
                    )
                    return loss + pen, np.r_[
                        float(np.sum(a = grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]
                    ]

                res_pt = minimize(fun = obj_pt, x0 = th_curr, jac = True, method = "L-BFGS-B")
                th_curr = res_pt.x
            return float(th_curr[0]), th_curr[1:]

        ## estimate negative binomial dispersion from mean rates
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
            return float(-np.mean(a = ll))

        ## alternate dispersion and coefficient updates for three iterations
        th_curr = th0.copy()
        alpha_hat = 0.5
        for _ in range(3):
            mu_r_curr = np.exp(
                np.clip(a = off_tr + th_curr[0] + X @ th_curr[1:], a_min = -15.0, a_max = 25.0)
            )
            res_al = minimize_scalar(
                fun = lambda al: nb_rate_nll(al = al, mu_r = mu_r_curr),
                bounds = (1e-3, 10.0),
                method = "bounded",
            )
            alpha_hat = float(res_al.x)
            r_inv = 1.0 / alpha_hat

            def obj_nb(th: np.ndarray) -> tuple[float, np.ndarray]:
                eta = np.clip(a = off_tr + th[0] + X @ th[1:], a_min = -15.0, a_max = 25.0)
                mu_r = np.exp(eta)
                nll_vec = (
                    -gammaln(y_rate + r_inv)
                    + gammaln(r_inv)
                    + gammaln(y_rate + 1.0)
                    + r_inv * np.log(1.0 + alpha_hat * mu_r)
                    - y_rate * (eta - np.log(r_inv + mu_r))
                )
                loss = alpha_hat * float(np.mean(a = nll_vec))
                pen = 0.5 * (l2_pen / n) * float(np.sum(a = th[1:] ** 2))
                grad_eta = alpha_hat * ((mu_r - y_rate) / (1.0 + alpha_hat * mu_r)) / n
                return loss + pen, np.r_[
                    float(np.sum(a = grad_eta)), X.T @ grad_eta + (l2_pen / n) * th[1:]
                ]

            res_nb = minimize(fun = obj_nb, x0 = th_curr, jac = True, method = "L-BFGS-B")
            th_curr = res_nb.x
        return float(th_curr[0]), th_curr[1:]


## six-configuration utilization benchmark
def train_utilization(
    df_data: pd.DataFrame,
    folds: Sequence[tuple],
    reduced_cols: Sequence[str] = FEAT_6U,
    families: Sequence[str] = GLM_ESTIMATORS,
    modes: Sequence[str] = SCT_MODALITIES,
    cap_frontier: bool = True,
    evt_offset: bool = True,
    ) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:

        ## prepare data arrays for training
        n_sys = len(df_data)
        l2_grid = [1.0, 2.0, 5.0, 10.0]
        M_all = df_data["M"].to_numpy(dtype = float)
        L_all = df_data["L"].to_numpy(dtype = float)
        evt_all = (
            df_data["evt_atten"].to_numpy(dtype = float)
            if evt_offset
            else np.zeros(shape = n_sys)
        )
        r_true = df_data["r_mean"].to_numpy(dtype = float)
        U_all_raw = signed_log1p(
            arr = df_data[list(reduced_cols)].to_numpy(dtype = float)
        )

        ## init output container
        preds_out: dict[tuple[str, str], dict[str, Any]] = dict()

        ## loop over families and modes
        for fam in families:
            for mode in modes:
                r_pred = np.zeros(shape = n_sys, dtype = float)
                y_pred = np.zeros(shape = n_sys, dtype = float)
                fold_fits = list()

                for tr_idx, te_idx, sct_tr, sct_te, inner_splits in folds:
                    U_tr_raw, U_te_raw = U_all_raw[tr_idx], U_all_raw[te_idx]
                    M_tr = M_all[tr_idx]
                    L_tr = L_all[tr_idx]
                    evt_tr, evt_te = evt_all[tr_idx], evt_all[te_idx]
                    r_mean_tr = r_true[tr_idx]
                    log_r_mean_tr = np.log(np.maximum(r_mean_tr, 1e-8))

                    ## subtract evt adj to form the anchored log-rate
                    sct_mean_tr = sct_tr - evt_tr
                    sct_mean_te = sct_te - evt_te

                    ## select penalty by inner-fold absolute log-rate error
                    best_hp, best_loss = 5.0, np.inf
                    for cand_l2 in l2_grid:
                        v_losses = list()
                        for it, iv, sct_it, sct_iv in inner_splits:
                            sct_mean_it = sct_it - evt_tr[it]
                            sct_mean_iv = sct_iv - evt_tr[iv]
                            if mode == "SCT Frontier Exclusion":
                                sc_in = StandardScaler().fit(X = U_tr_raw[it])
                                X_it = sc_in.transform(X = U_tr_raw[it])
                                X_iv = sc_in.transform(X = U_tr_raw[iv])
                                off_it = np.zeros(shape = len(it))
                                off_iv = np.zeros(shape = len(iv))
                            else:
                                sc_in = StandardScaler().fit(X = U_tr_raw[it])
                                X_it = sc_in.transform(X = U_tr_raw[it])
                                X_iv = sc_in.transform(X = U_tr_raw[iv])
                                off_it = sct_mean_it
                                off_iv = sct_mean_iv

                            a_in, b_in = fit_parametric_glm_family(
                                X = X_it,
                                r_mean_tr = r_mean_tr[it],
                                M_tr = M_tr[it],
                                L_tr = L_tr[it],
                                off_tr = off_it,
                                l2_pen = cand_l2,
                                fam = fam,
                            )
                            p_iv = off_iv + a_in + (X_iv @ b_in)
                            if mode == "SCT Frontier Anchor" and cap_frontier:
                                p_iv = np.minimum(p_iv, sct_iv)
                            v_losses.append(np.mean(a = np.abs(log_r_mean_tr[iv] - p_iv)))

                        if np.mean(a = v_losses) < best_loss:
                            best_loss = float(np.mean(a = v_losses))
                            best_hp = cand_l2

                    if mode == "SCT Frontier Exclusion":
                        sc_out = StandardScaler().fit(X = U_tr_raw)
                        X_tr_d = sc_out.transform(X = U_tr_raw)
                        X_te_d = sc_out.transform(X = U_te_raw)
                        off_tr = np.zeros(shape = len(tr_idx))
                        off_te = np.zeros(shape = len(te_idx))
                    else:
                        sc_out = StandardScaler().fit(X = U_tr_raw)
                        X_tr_d = sc_out.transform(X = U_tr_raw)
                        X_te_d = sc_out.transform(X = U_te_raw)
                        off_tr = sct_mean_tr
                        off_te = sct_mean_te

                    ## refit on outer training fold with the selected penalty
                    a_hat, b_hat = fit_parametric_glm_family(
                        X = X_tr_d,
                        r_mean_tr = r_mean_tr,
                        M_tr = M_tr,
                        L_tr = L_tr,
                        off_tr = off_tr,
                        l2_pen = float(best_hp),
                        fam = fam,
                    )
                    log_r_mean_hat = off_te + a_hat + (X_te_d @ b_hat)
                    ## retain fitted coefficients and scales
                    fold_fits.append(
                        {
                            "fold": len(fold_fits),
                            "train_idx": tr_idx.copy(),
                            "test_idx": te_idx.copy(),
                            "l2": float(best_hp),
                            "intercept": float(a_hat),
                            "coef": b_hat.copy(),
                            "feature_scale": sc_out.scale_.copy(),
                        }
                    )

                    ## cap anchored estimates at the held-out frontier
                    if mode == "SCT Frontier Anchor" and cap_frontier:
                        log_r_mean_hat = np.minimum(log_r_mean_hat, sct_te)

                    r_hat = np.exp(np.clip(a = log_r_mean_hat, a_min = -12.0, a_max = 20.0))

                    ## restore predictions to the original system order
                    r_pred[te_idx] = r_hat
                    y_pred[te_idx] = np.log1p(r_hat)

                preds_out[(fam, mode)] = {
                    "r_pred": r_pred,
                    "y_pred": y_pred,
                    "fold_fits": fold_fits,
                }
        return compile_utilization_benchmark(df_data = df_data, predictions = preds_out), preds_out


## score out-of-fold predictions at full precision, leaving rounding to display
def compile_utilization_benchmark(
    df_data: pd.DataFrame,
    predictions: dict[tuple[str, str], dict[str, Any]],
    ) -> pd.DataFrame:

        ## prepare observed rates and log-rates
        r_true = df_data["r_mean"].to_numpy(dtype = float)
        y_true = df_data["y_mean"].to_numpy(dtype = float)

        ## score each family and estimation regime
        rows = list()
        for (fam, mode), prediction in predictions.items():
            r_pred = prediction["r_pred"]
            y_pred = prediction["y_pred"]
            abs_log_ratio = np.abs(
                np.log(np.maximum(r_pred, 1e-9) / np.maximum(r_true, 1e-9))
            )
            cm = frontier_consensus(y_true = y_true, y_pred = y_pred, p = 0.9)
            rows.append(
                {
                    "Estimator": fam,
                    "Estimation Regime": mode,
                    "MAE": float(np.mean(a = np.abs(y_true - y_pred))),
                    "RMSE": float(np.sqrt(np.mean(a = (y_true - y_pred) ** 2))),
                    "GMME": float(np.exp(np.mean(a = abs_log_ratio))),
                    "Within 2x (%)": float(np.mean(a = np.exp(abs_log_ratio) <= 2.0) * 100.0),
                    "Within 3x (%)": float(np.mean(a = np.exp(abs_log_ratio) <= 3.0) * 100.0),
                    "rho": cm["rho"],
                    "RBO": cm["rbo"],
                    "dCor": cm["dcr"],
                    "CI": cm["ci"],
                }
            )
        return pd.DataFrame(data = rows)


## compile utilization metrics from predictions
def compile_utilization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ) -> dict[str, float]:

        ## compute pointwise log-rate errors and multiplicative ratios
        error = y_true - y_pred
        ratio = np.maximum(np.expm1(y_pred), 1e-9) / np.maximum(np.expm1(y_true), 1e-9)
        return {
            "MAE": float(np.mean(a = np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(a = error**2))),
            "CI": frontier_consensus(y_true = y_true, y_pred = y_pred)["ci"],
            "GMME": float(np.exp(np.mean(a = np.abs(np.log(ratio))))),
        }


## train utilization with resampling
def train_utilization_resampling(
    data: pd.DataFrame,
    n_repeats: int = 30,
    random_state: int = 42,
    ) -> dict[str, Any]:

        ## check repeats and init storage
        if n_repeats < 1:
            raise ValueError("n_repeats must be positive")
        folds, results, predictions, repeated = dict(), dict(), dict(), dict()

        ## loop over resampling regimes
        for regime in RESAMPLING_REGIMES:
            print(f"Training utilization: {regime}")
            folds[regime] = precompute_native_sct_regime(
                df_data = data,
                regime_name = regime,
                random_state = random_state,
            )
            results[regime], predictions[regime] = train_utilization(
                df_data = data,
                folds = folds[regime],
            )
            if regime in RESAMPLING_REGIMES[2:]:
                for repeat in range(n_repeats):
                    if repeat == 0:
                        fitted = predictions[regime]
                    else:

                        ## vary the outer split, keeping inner split seed fixed
                        repeat_folds = precompute_native_sct_regime(
                            df_data = data,
                            regime_name = regime,
                            random_state = random_state + repeat,
                            inner_random_state = random_state,
                        )
                        _, fitted = train_utilization(
                            df_data = data,
                            folds = repeat_folds,
                            modes = SCT_MODALITIES[1:],
                        )
                    for family in GLM_ESTIMATORS:
                        repeated[(regime, repeat, family)] = fitted[(family, SCT_MODALITIES[1])][
                            "y_pred"
                        ]
        return {
            "folds": folds,
            "results": results,
            "predictions": predictions,
            "repeated": repeated,
        }


## compile frontier from folds
def compile_utilization_frontier(
    folds: Sequence[tuple],
    n_obs: int,
    ) -> np.ndarray:

        ## compile the frontier from folds
        frontier = np.full(shape = n_obs, fill_value = np.nan)
        counts = np.zeros(shape = n_obs, dtype = int)
        for _, test_idx, _, prediction, _ in folds:
            frontier[test_idx] = prediction
            counts[test_idx] += 1
        
        ## require one finite held-out frontier estimate per system
        if not np.all(counts == 1) or not np.isfinite(frontier).all():
            raise ValueError("Frontier folds must cover each system exactly once")
        return frontier


## train controls using precomputed folds
def train_utilization_controls(
    data: pd.DataFrame,
    resampling: dict[str, Any],
    n_repeats: int = 30,
    random_state: int = 42,
    ) -> dict[tuple[str, str, str], list[np.ndarray]]:

        ## validate repeats and prepare control inputs
        if n_repeats < 1 or len(data) < 2:
            raise ValueError("Controls require positive repeats and at least two systems")
        inputs = data.assign(constant = 1.0, evt_feature = np.expm1(data["evt_atten"]))
        predictions = dict()

        ## loop over resampling regimes
        for regime, folds in resampling["folds"].items():
            print(f"Training utilization controls: {regime}")

            ## carry over the primary exclusion and anchor predictions
            for family in GLM_ESTIMATORS:
                for mode in SCT_MODALITIES:
                    predictions[(regime, family, mode)] = [
                        resampling["predictions"][regime][(family, mode)]["y_pred"]
                    ]

            ## replace the sct frontier with an arbitrary ceiling
            ceiling_folds = list()
            maxima = data["y_max"].to_numpy(dtype = float)
            for train_idx, test_idx, _, _, inner in folds:

                ## derive each arbitrary ceiling from its training fold only
                ceiling = 3.0 * maxima[train_idx].max()
                inner_folds = list()
                for fit_idx, valid_idx, _, _ in inner:
                    inner_ceiling = 3.0 * maxima[train_idx[fit_idx]].max()
                    inner_folds.append(
                        (
                            fit_idx,
                            valid_idx,
                            np.full(shape = len(fit_idx), fill_value = inner_ceiling),
                            np.full(shape = len(valid_idx), fill_value = inner_ceiling),
                        )
                    )
                ceiling_folds.append(
                    (
                        train_idx,
                        test_idx,
                        np.full(shape = len(train_idx), fill_value = ceiling),
                        np.full(shape = len(test_idx), fill_value = ceiling),
                        inner_folds,
                    )
                )

            ## fit a median-rate offset using the full frontier feature set
            matched_folds = precompute_native_sct_regime(
                df_data = inputs,
                regime_name = regime,
                random_state = random_state,
                target = "y_mean",
                quantile_c = 0.50,
            )

            ## pair each ablation with its folds, features, cap, and evt offset
            settings = (
                (UTIL_ABLATIONS[1], folds, FEAT_6U, True, False),
                (UTIL_ABLATIONS[2], folds, ["constant"], True, True),
                (UTIL_ABLATIONS[3], folds, ["constant"], True, False),
                (UTIL_ABLATIONS[5], ceiling_folds, FEAT_6U, True, True),
                (
                    UTIL_ABLATIONS[6],
                    matched_folds,
                    FEAT_6U + ["evt_feature"],
                    False,
                    False,
                ),
            )

            ## refit the anchored configuration under each ablation
            for name, control_folds, features, cap, offset in settings:
                _, fitted = train_utilization(
                    df_data = inputs,
                    folds = control_folds,
                    reduced_cols = features,
                    modes = SCT_MODALITIES[1:],
                    cap_frontier = cap,
                    evt_offset = offset,
                )
                for family in GLM_ESTIMATORS:
                    predictions[(regime, family, name)] = [
                        fitted[(family, SCT_MODALITIES[1])]["y_pred"]
                    ]

            ## remap held-out frontiers across systems by random derangement
            frontier = compile_utilization_frontier(folds = folds, n_obs = len(data))
            rng = np.random.default_rng(seed = random_state)
            for family in GLM_ESTIMATORS:
                predictions[(regime, family, UTIL_ABLATIONS[-1])] = list()
            for _ in range(n_repeats):
                permutation = rng.permutation(x = len(data))

                ## reject permutations for any system paired with its own frontier
                while np.any(permutation == np.arange(len(data))):
                    permutation = rng.permutation(x = len(data))
                values = frontier[permutation]

                ## substitute remapped frontiers into the outer and inner folds
                remapped = [
                    (
                        train_idx,
                        test_idx,
                        values[train_idx],
                        values[test_idx],
                        [
                            (
                                fit_idx,
                                valid_idx,
                                values[train_idx[fit_idx]],
                                values[train_idx[valid_idx]],
                            )
                            for fit_idx, valid_idx, _, _ in inner
                        ],
                    )
                    for train_idx, test_idx, _, _, inner in folds
                ]
                _, fitted = train_utilization(
                    df_data = inputs,
                    folds = remapped,
                    modes = SCT_MODALITIES[1:],
                )
                for family in GLM_ESTIMATORS:

                    ## retain every remap so metrics can be scored before averaging
                    predictions[(regime, family, UTIL_ABLATIONS[-1])].append(
                        fitted[(family, SCT_MODALITIES[1])]["y_pred"]
                    )
        return predictions


## load prior and later temporal windows
def load_utilization_windows(
    data: pd.DataFrame,
    fractions: Sequence[float] = (0.25, 0.50, 0.75),
    proc_dir: str | Path | None = None,
    ) -> dict[float, pd.DataFrame]:

        ## reload each event series in chronological order
        directory = root / PATH_PROC if proc_dir is None else Path(proc_dir)
        series = dict()
        for _, row in data.iterrows():
            with (directory / f"{row['name']}.json").open(
                mode = "r", encoding = "utf-8"
            ) as source:
                events = pd.DataFrame(data = json.load(fp = source)["events"])
            sort_by = "date" if "date" in events else "day"
            events = events.sort_values(by = sort_by).reset_index(drop = True)

            ## recompute full-series process signatures
            signatures = ProcessSignatures(
                data = events, sort_by = [sort_by], target = TARGET
            ).all()
            counts = events[TARGET].to_numpy(dtype = float)

            ## verify reconstructed series against the supplied full-series inputs
            if not (
                np.allclose(
                    a = [signatures[feature] for feature in FEAT_Z],
                    b = row[FEAT_Z].to_numpy(dtype = float),
                    rtol = 1e-9,
                    atol = 1e-12,
                )
                and np.isclose(a = counts.max(), b = row["r_max"])
                and np.isclose(a = counts.mean(), b = row["r_mean"])
            ):
                raise ValueError(f"Full-series inputs disagree for {row['name']}")
            series[row["name"]] = (events, sort_by)

        ## split each series into prior and later windows at each fraction
        frames = dict()
        for fraction in fractions:
            rows = list()
            for name in data["name"]:
                events, sort_by = series[name]
                n_early = int(np.floor(fraction * len(events)))
                if n_early < 3 or len(events) - n_early < 3:
                    raise ValueError(
                        f"Window split {fraction} leaves too few intervals for {name}"
                    )

                ## use early process inputs and disjoint later rate targets
                early, later = events.iloc[:n_early], events.iloc[n_early:]
                signatures = ProcessSignatures(
                    data = early, sort_by = [sort_by], target = TARGET
                ).all()
                rate = float(later[TARGET].mean())
                rows.append(
                    {
                        **signatures,
                        "L_A": float(n_early),
                        "r_mean_A": float(early[TARGET].mean()),
                        "y_max": float(np.log1p(early[TARGET].max())),
                        "evt_atten": float(
                            np.log1p(
                                signatures["coef_variation"]
                                * np.sqrt(2.0 * np.log(max(n_early, 2)))
                            )
                        ),
                        "L": float(len(later)),
                        "M": float(np.round(a = later[TARGET].sum())),
                        "r_mean": rate,
                        "y_mean": float(np.log1p(rate)),
                    }
                )

            ## retain static graph features while replacing temporal inputs and targets
            frame = data.drop(columns = [TARGET, "r_max", "utilization"]).copy()
            for column, values in pd.DataFrame(data = rows).items():
                frame[column] = values.to_numpy()
            frames[fraction] = frame
        return frames


## train utilization across temporal windows
def train_utilization_windows(
    frames: dict[float, pd.DataFrame],
    random_state: int = 42,
    ) -> dict[tuple[float, str, bool], dict]:

        ## loop over window fractions and resampling regimes
        predictions = dict()
        for fraction, data in frames.items():
            for regime in RESAMPLING_REGIMES:
                print(f"Training utilization windows: {fraction:.0%}, {regime}")

                ## precompute frontier folds from prior-window inputs
                folds = precompute_native_sct_regime(
                    df_data = data,
                    regime_name = regime,
                    random_state = random_state,
                )

                ## refit each cap setting, including its inner penalty selection
                for capped in (True, False):
                    _, predictions[(fraction, regime, capped)] = train_utilization(
                        df_data = data,
                        folds = folds,
                        cap_frontier = capped,
                        modes = SCT_MODALITIES if capped else SCT_MODALITIES[1:],
                    )
        return predictions


## compile utilization predictions and fold coefficients
def compile_utilization_predictions(
    data: pd.DataFrame,
    resampling: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:

        ## prepare observed targets and their dispersion
        target = data["y_mean"].to_numpy(dtype = float)
        observed = data["r_mean"].to_numpy(dtype = float)
        target_iqr = float(np.diff(np.percentile(a = target, q = [25, 75]))[0])
        target_variance = float(np.var(a = target))

        ## init output containers
        contraction, targets, coefficients, metrics, predictions = [], [], [], [], []

        ## loop over resampling regimes
        for regime, folds in resampling["folds"].items():
            frontier = compile_utilization_frontier(folds = folds, n_obs = len(data))

            ## compute target contraction once per regime, independently of glm family
            slack = frontier - data["evt_atten"].to_numpy(dtype = float) - target
            slack_iqr = float(np.diff(np.percentile(a = slack, q = [25, 75]))[0])
            slack_variance = float(np.var(a = slack))
            contraction.append(
                {
                    "Resampling Regime": regime,
                    "N": len(data),
                    "Observed IQR": target_iqr,
                    "Slack IQR": slack_iqr,
                    "IQR Contraction (%)": 100 * (1 - slack_iqr / target_iqr),
                    "Observed Variance": target_variance,
                    "Slack Variance": slack_variance,
                    "Variance Contraction (%)": 100
                    * (1 - slack_variance / target_variance),
                }
            )

            ## record the per-system frontier decomposition
            for index, name in enumerate(data["name"]):
                targets.append(
                    {
                        "Resampling Regime": regime,
                        "System": name,
                        "Frontier": frontier[index],
                        "EVT": data["evt_atten"].iloc[index],
                        "Observed_LogRate": target[index],
                        "Slack": slack[index],
                    }
                )

            ## score each family and configuration
            for (family, mode), prediction in resampling["predictions"][regime].items():
                labels = {
                    "Resampling Regime": regime,
                    "Estimator": family,
                    "Configuration": mode,
                }

                ## compute multiplicative errors in rate space
                ratio = np.exp(
                    np.abs(
                        np.log(
                            np.maximum(prediction["r_pred"], 1e-9)
                            / np.maximum(observed, 1e-9)
                        )
                    )
                )
                scores = compile_utilization_metrics(
                    y_true = target, y_pred = prediction["y_pred"]
                )
                metrics.append(
                    {
                        **labels,
                        **{metric: value for metric, value in scores.items()},
                        "Within 2x (%)": 100 * np.mean(a = ratio <= 2),
                        "Within 3x (%)": 100 * np.mean(a = ratio <= 3),
                    }
                )

                ## record per-system predictions
                for index, name in enumerate(data["name"]):
                    predictions.append(
                        {
                            **labels,
                            "System": name,
                            "Domain": data["domain"].iloc[index],
                            "Observed_Rate": observed[index],
                            "Predicted_Rate": prediction["r_pred"][index],
                            "Observed_LogRate": target[index],
                            "Predicted_LogRate": prediction["y_pred"][index],
                            "Multiplicative_Error": ratio[index],
                        }
                    )

                ## extract actual benchmark coefficients without fitting a surrogate model
                for fit in prediction["fold_fits"]:
                    for index, feature in enumerate(FEAT_6U):
                        coefficients.append(
                            {
                                **labels,
                                "Fold": fit["fold"],
                                "Feature": feature,
                                "Coefficient": fit["coef"][index],
                                "L2": fit["l2"],
                                "Training_SD": fit["feature_scale"][index],
                                "N_train": len(fit["train_idx"]),
                                "N_test": len(fit["test_idx"]),
                            }
                        )

        ## summarize coefficient magnitudes across folds
        fold_coefficients = pd.DataFrame(data = coefficients)
        summaries = list()
        for (regime, family, feature), group in fold_coefficients.groupby(
            by = ["Resampling Regime", "Estimator", "Feature"],
            sort = False,
            observed = True,
        ):
            excluded = group.loc[
                group["Configuration"] == SCT_MODALITIES[0], "Coefficient"
            ].to_numpy()
            anchored = group.loc[
                group["Configuration"] == SCT_MODALITIES[1], "Coefficient"
            ].to_numpy()
            magnitude_excluded = float(np.mean(a = np.abs(excluded)))
            magnitude_anchored = float(np.mean(a = np.abs(anchored)))
            summaries.append(
                {
                    "Resampling Regime": regime,
                    "Estimator": family,
                    "Feature": feature,
                    "N_folds": len(excluded),
                    "Exclusion Mean": excluded.mean(),
                    "Anchor Mean": anchored.mean(),
                    "Exclusion Fold SD": excluded.std(ddof = 1),
                    "Anchor Fold SD": anchored.std(ddof = 1),
                    "Exclusion Mean Absolute Coefficient": magnitude_excluded,
                    "Anchor Mean Absolute Coefficient": magnitude_anchored,
                    "Magnitude_Reduction": magnitude_excluded - magnitude_anchored,
                    "Magnitude Reduction (%)": 100 * (1 - magnitude_anchored / magnitude_excluded)
                    if magnitude_excluded > 1e-12
                    else np.nan,
                }
            )

        ## assemble output tables
        results = pd.DataFrame(data = metrics)
        system_predictions = pd.DataFrame(data = predictions)
        return {
            "corpus": compile_corpus_table(df_data = data),
            "benchmark": compile_utilization_benchmark(
                df_data = data,
                predictions = resampling["predictions"][RESAMPLING_REGIMES[0]],
            ),
            "resampling": results.pivot(
                index = ["Estimator", "Configuration"],
                columns = "Resampling Regime",
                values = [metric for metric in UTIL_METRICS],
            ),
            "systems": system_predictions.loc[
                (system_predictions["Resampling Regime"] == RESAMPLING_REGIMES[0])
                & (system_predictions["Estimator"] == GLM_ESTIMATORS[-1])
            ].reset_index(drop = True),
            "contraction": pd.DataFrame(data = contraction),
            "targets": pd.DataFrame(data = targets),
            "coefficients": pd.DataFrame(data = summaries),
            "fold_coefficients": fold_coefficients,
            "predictive": results,
            "predictions": system_predictions,
        }


## holm step-down adjusted p-values
def holm_adjust(pvals: Any) -> np.ndarray:
    pvals = np.asarray(a = pvals, dtype = float)
    order = np.argsort(a = pvals)
    scaled = pvals[order] * np.arange(start = len(pvals), stop = 0, step = -1)
    adjusted = np.maximum.accumulate(scaled)
    adjusted[adjusted > 1] = 1
    result = np.empty_like(prototype = adjusted)
    result[order] = adjusted
    return result


## test equivalence across resampling regimes
def stat_utilization_equivalence(
    data: pd.DataFrame,
    resampling: dict[str, Any],
    decimals: int = 2,
    ) -> dict[str, pd.DataFrame]:

        ## prepare targets and domain labels
        target = data["y_mean"].to_numpy(dtype = float)
        domain = data["domain"].to_numpy()
        rows = list()

        ## score each regime and family within domains
        for regime in RESAMPLING_REGIMES:
            for family in GLM_ESTIMATORS:

                ## use single logo-cv predictions and every k-fold-cv repeat
                predictions = (
                    [resampling["predictions"][regime][(family, SCT_MODALITIES[1])]["y_pred"]]
                    if regime in RESAMPLING_REGIMES[:2]
                    else [
                        prediction
                        for (name, _, model), prediction in resampling["repeated"].items()
                        if name == regime and model == family
                    ]
                )
                for group in sorted(data["domain"].unique()):
                    selected = domain == group

                    ## score each repeat within a domain before averaging metrics
                    scores = [
                        compile_utilization_metrics(
                            y_true = target[selected], y_pred = prediction[selected]
                        )
                        for prediction in predictions
                    ]
                    rows.append(
                        {
                            "Resampling Regime": regime,
                            "Estimator": family,
                            "Domain": group,
                            **{
                                metric: np.mean(a = [score[metric] for score in scores])
                                for metric in UTIL_METRICS
                            },
                        }
                    )

        ## use domain logo-cv as the baseline
        domain_results = pd.DataFrame(data = rows)
        baseline = domain_results.loc[
            domain_results["Resampling Regime"] == RESAMPLING_REGIMES[0]
        ].set_index(keys = ["Estimator", "Domain"])
        tables = list()

        ## loop over metrics
        for metric in UTIL_METRICS:

            ## average each family across domains at the baseline
            model_means = baseline.groupby(level = "Estimator", sort = False, observed = True)[
                metric
            ].mean()
            dispersion = float(np.diff(np.percentile(a = model_means, q = [25, 75]))[0])

            ## round the baseline inter-model iqr upward to set the equivalence margin
            delta = float(
                np.ceil(max(dispersion, 1e-6) * 10**decimals - 1e-10) / 10**decimals
            )
            comparisons = list()

            ## compare each alternative regime with the baseline
            for regime in (
                RESAMPLING_REGIMES[1],
                RESAMPLING_REGIMES[3],
                RESAMPLING_REGIMES[2],
            ):
                comparison = domain_results.loc[
                    domain_results["Resampling Regime"] == regime
                ].set_index(keys = ["Estimator", "Domain"])

                ## align comparisons by model family and domain before pairing
                diff = _clean_differences(
                    diff = comparison[metric].reindex(index = baseline.index) - baseline[metric]
                )

                ## test both equivalence bounds and retain the larger p-value
                upper = (
                    1.0
                    if np.all(diff - delta == 0)
                    else float(stats.wilcoxon(x = diff - delta, alternative = "less").pvalue)
                )
                lower = (
                    1.0
                    if np.all(diff + delta == 0)
                    else float(stats.wilcoxon(x = diff + delta, alternative = "greater").pvalue)
                )
                comparisons.append(
                    {
                        "Metric": metric,
                        "Comparison": regime,
                        "Pairs": len(diff),
                        "delta": delta,
                        "Median Δ": np.median(a = diff),
                        "Maximum Absolute Δ": np.max(a = np.abs(diff)),
                        "Rank-biserial r": paired_rank_biserial(diff = diff),
                        "TOST p": max(upper, lower),
                    }
                )

            ## adjust p-values within each metric
            table = pd.DataFrame(data = comparisons)
            table["Holm-adj. p"] = holm_adjust(pvals = table["TOST p"])
            table["Eq."] = table["Holm-adj. p"] < 0.05
            tables.append(table)

        ## compute per-system error differences from the baseline
        differences = list()
        for regime in RESAMPLING_REGIMES[2:]:
            for family in GLM_ESTIMATORS:
                errors = [
                    np.abs(target - prediction)
                    for (name, _, model), prediction in resampling["repeated"].items()
                    if name == regime and model == family
                ]
                baseline_error = np.abs(
                    target
                    - resampling["predictions"][RESAMPLING_REGIMES[0]][
                        (family, SCT_MODALITIES[1])
                    ]["y_pred"]
                )
                differences.append(np.mean(a = errors, axis = 0) - baseline_error)
        differences = np.stack(arrays = differences, axis = 1)
        return {
            "equivalence": pd.concat(objs = tables, ignore_index = True),
            "domain_metrics": domain_results,
            "resampling_differences": pd.DataFrame(
                data = {
                    "System": data["name"],
                    "Domain": data["domain"],
                    "Discipline": data["discipline"],
                    "Median": np.median(a = differences, axis = 1),
                    "Q1": np.percentile(a = differences, q = 25, axis = 1),
                    "Q3": np.percentile(a = differences, q = 75, axis = 1),
                }
            ),
        }


## bootstrap utilization metrics
def stat_utilization_bootstrap(
    y_true: np.ndarray,
    predictions: Sequence[np.ndarray],
    indices: np.ndarray,
    n_boot_ci: int = 1000,
    ) -> dict[str, np.ndarray]:

        ## init metric distributions
        distributions = {metric: list() for metric in UTIL_METRICS}

        ## resample fixed predictions with shared system indices across all remaps
        for prediction in predictions:
            error = y_true - prediction
            log_ratio = np.abs(
                np.log(
                    np.maximum(np.expm1(prediction), 1e-9)
                    / np.maximum(np.expm1(y_true), 1e-9)
                )
            )

            ## score each bootstrap sample, limiting ci to a subset of draws
            distributions["MAE"].append(np.abs(error)[indices].mean(axis = 1))
            distributions["RMSE"].append(np.sqrt((error**2)[indices].mean(axis = 1)))
            distributions["GMME"].append(np.exp(log_ratio[indices].mean(axis = 1)))
            distributions["CI"].append(
                np.array(
                    object = [
                        frontier_consensus(
                            y_true = y_true[sample], y_pred = prediction[sample]
                        )["ci"]
                        for sample in indices[:n_boot_ci]
                    ]
                )
            )

        ## average metric distributions after scoring each remap
        return {
            metric: np.mean(a = values, axis = 0) for metric, values in distributions.items()
        }


## compile domain-level metric cells
def compile_utilization_domain_cells(
    y_true: np.ndarray,
    predictions: dict[str, list[np.ndarray]],
    domains: np.ndarray,
    ) -> dict[str, np.ndarray]:

        ## coerce inputs and init cells
        y_true = np.asarray(a = y_true, dtype = float)
        domains = np.asarray(a = domains)
        cells = {metric: [] for metric in UTIL_METRICS}

        ## keep family and domain ordering identical across paired comparisons
        for family in GLM_ESTIMATORS:
            for domain in sorted(set(domains)):
                mask = domains == domain
                scores = [
                    compile_utilization_metrics(
                        y_true = y_true[mask],
                        y_pred = np.asarray(a = prediction, dtype = float)[mask],
                    )
                    for prediction in predictions[family]
                ]

                ## average metrics across repeats
                for metric in UTIL_METRICS:
                    cells[metric].append(
                        float(np.mean(a = [score[metric] for score in scores]))
                    )
        return {metric: np.asarray(a = values) for metric, values in cells.items()}


## test anchor superiority with a one-sided wilcoxon test
def stat_utilization_superiority(
    metric: str,
    comparator_cells: np.ndarray,
    anchor_cells: np.ndarray,
    ) -> dict[str, Any]:

        ## orient every metric so positive differences favor the anchor
        diff = (
            anchor_cells - comparator_cells
            if metric == "CI"
            else comparator_cells - anchor_cells
        )

        ## treat numerical near-ties as zero before ranking differences
        diff = np.where(np.abs(diff) < 1e-10, 0.0, diff)

        ## rank nonzero differences for the rank-biserial effect size
        nonzero = diff[diff != 0.0]
        ranks = stats.rankdata(a = np.abs(nonzero))
        return {
            "Pairs": diff.size,
            "Comparator_Mean": float(np.mean(a = comparator_cells)),
            "Anchor Mean": float(np.mean(a = anchor_cells)),
            "Median Δ": float(np.median(a = diff)),
            "Pairs Favoring Anchor": int(np.sum(a = diff > 0)),
            "Rank-biserial r": float(
                (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
            )
            if nonzero.size
            else np.nan,
            "One-sided p": 1.0
            if not np.any(diff)
            else float(stats.wilcoxon(x = diff, alternative = "greater").pvalue),
        }


## holm correction for superiority tests
def stat_utilization_superiority_holm(
    frame: pd.DataFrame,
    family_mask: pd.Series,
    suffix: str,
    alpha: float = 0.05,
    ) -> pd.DataFrame:

        ## init corrected p-value and decision columns
        result = frame.copy(deep = True)
        column = "Holm-adj. p" if suffix == "Primary" else "Holm-adj. p (Complete)"
        decision = "Sp." if suffix == "Primary" else "Superior (Complete)"
        result[column] = np.nan

        ## apply holm correction within each metric and selected comparison family
        for metric in result["Metric"].unique():
            selected = family_mask & (result["Metric"] == metric)
            if selected.any():
                result.loc[selected, column] = holm_adjust(
                    pvals = result.loc[selected, "One-sided p"]
                )

        ## declare superiority when significant and favoring the anchor
        result[decision] = (
            (
                (result[column] < alpha)
                & (result["Median Δ"] > 0)
            )
            .astype(dtype = "boolean")
            .where(cond = family_mask)
        )
        return result


## compile superiority tests and support summaries
def compile_utilization_superiority(
    data: pd.DataFrame,
    controls: dict[tuple[str, str, str], list[np.ndarray]],
    frames: dict[float, pd.DataFrame],
    windows: dict[tuple[float, str, bool], dict],
    tables: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:

        ## build domain cells for ablation and frontier controls
        settings = []
        for regime in RESAMPLING_REGIMES:
            cells = {
                configuration: compile_utilization_domain_cells(
                    y_true = data["y_mean"].to_numpy(dtype = float),
                    predictions = {
                        family: controls[(regime, family, configuration)]
                        for family in GLM_ESTIMATORS
                    },
                    domains = data["domain"].to_numpy(),
                )
                for configuration in UTIL_ABLATIONS
            }
            settings.append(("Ablation and frontier controls", "", regime, cells))

        ## build domain cells for temporal separation windows
        for fraction, frame in frames.items():
            for regime in RESAMPLING_REGIMES:
                cells = {
                    configuration: compile_utilization_domain_cells(
                        y_true = frame["y_mean"].to_numpy(dtype = float),
                        predictions = {
                            family: [
                                windows[(fraction, regime, True)][(family, configuration)][
                                    "y_pred"
                                ]
                            ]
                            for family in GLM_ESTIMATORS
                        },
                        domains = frame["domain"].to_numpy(),
                    )
                    for configuration in SCT_MODALITIES
                }
                settings.append(("Temporal separation", f"{fraction:.0%}", regime, cells))

        ## test the anchor against each comparator and metric
        rows = []
        for analysis, window, regime, cells in settings:
            for comparator, comparison in cells.items():
                if comparator == SCT_MODALITIES[1]:
                    continue
                for metric in UTIL_METRICS:
                    rows.append(
                        {
                            "Analysis": analysis,
                            "Prior-Window Share": window,
                            "Resampling Regime": regime,
                            "Comparator": comparator,
                            "Metric": metric,
                            **stat_utilization_superiority(
                                metric = metric,
                                comparator_cells = comparison[metric],
                                anchor_cells = cells[SCT_MODALITIES[1]][metric],
                            ),
                        }
                    )

        ## correct p-values within each analysis
        corrected = []
        for analysis, group in pd.DataFrame(data = rows).groupby(
            by = "Analysis", sort = False, observed = True
        ):

            ## correct primary comparisons separately from the complete analysis
            primary = (group["Resampling Regime"] == RESAMPLING_REGIMES[0]) & group[
                "Prior-Window Share"
            ].isin(
                values = ["", "50%"]
            )
            group = stat_utilization_superiority_holm(
                frame = group, family_mask = primary, suffix = "Primary"
            )
            group = stat_utilization_superiority_holm(
                frame = group,
                family_mask = pd.Series(data = True, index = group.index),
                suffix = "Complete",
            )
            corrected.append(group)
        superiority = pd.concat(objs = corrected)

        ## summarize support for each comparator and metric across settings
        support = []
        for (analysis, comparator, metric), group in superiority.groupby(
            by = ["Analysis", "Comparator", "Metric"], sort = False, observed = True
        ):
            support.append(
                {
                    "Analysis": analysis,
                    "Comparator": comparator,
                    "Metric": metric,
                    "N_settings": len(group),
                    "Median Δ Min": group["Median Δ"].min(),
                    "Median Δ Max": group["Median Δ"].max(),
                    "Settings Favoring Anchor": int((group["Median Δ"] > 0).sum()),
                    "Settings With Superiority": int(group["Superior (Complete)"].sum()),
                    "Maximum Holm-adj. p (Complete)": group["Holm-adj. p (Complete)"].max(),
                }
            )

        ## remove legacy test fields from descriptive bootstrap tables
        result = {}
        for name in (
            "ablation",
            "ablation_contrasts",
            "temporal_contrasts",
            "window_contrasts",
        ):
            result[name] = (
                tables[name]
                .drop(
                    columns = [
                        "Wilcoxon_p",
                        "Holm_p_Primary",
                        "Holm_p_Complete",
                        "Supported",
                    ],
                    errors = "ignore",
                )
                .rename(
                    columns = {
                        "CI_Excludes_Zero": "Interval Excludes Zero",
                        "Anchor_Wins": "Systems With Lower Anchor Error",
                    }
                )
            )
            if "Systems With Lower Anchor Error" in result[name]:
                result[name]["Systems With Lower Anchor Error"] = result[name][
                    "Systems With Lower Anchor Error"
                ].astype(dtype = "Int64")

        ## append superiority tests and support summaries
        result.update(
            {
                "superiority": superiority,
                "ablation_tests": superiority.loc[
                    superiority["Analysis"] == "Ablation and frontier controls"
                ].reset_index(drop = True),
                "temporal_tests": superiority.loc[
                    superiority["Analysis"] == "Temporal separation"
                ].reset_index(drop = True),
                "support": pd.DataFrame(data = support),
            }
        )
        return result


## compare configurations with paired bootstrap intervals
def stat_utilization_comparisons(
    y_true: np.ndarray,
    predictions: dict[str, list[np.ndarray]],
    indices: np.ndarray,
    n_boot_ci: int = 1000,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        ## init output containers
        points, boots, errors = dict(), dict(), dict()
        summaries = list()

        ## score each configuration and its bootstrap distribution
        for configuration, values in predictions.items():
            scores = [
                compile_utilization_metrics(y_true = y_true, y_pred = value) for value in values
            ]
            points[configuration] = {
                metric: np.mean(a = [score[metric] for score in scores])
                for metric in UTIL_METRICS
            }
            boots[configuration] = stat_utilization_bootstrap(
                y_true = y_true,
                predictions = values,
                indices = indices,
                n_boot_ci = n_boot_ci,
            )
            errors[configuration] = np.mean(
                a = [np.abs(y_true - value) for value in values], axis = 0
            )

            ## summarize point estimates with percentile intervals
            summary = {"Configuration": configuration}
            for metric in UTIL_METRICS:
                lower, upper = np.percentile(a = boots[configuration][metric], q = [2.5, 97.5])
                summary.update(
                    {
                        metric: points[configuration][metric],
                        f"{metric}_lo": lower,
                        f"{metric}_hi": upper,
                    }
                )

            ## compute shares within multiplicative error thresholds
            log_ratios = np.asarray(
                a = [
                    np.abs(
                        np.log(
                            np.maximum(np.expm1(value), 1e-9)
                            / np.maximum(np.expm1(y_true), 1e-9)
                        )
                    )
                    for value in values
                ]
            )
            for factor in (2, 3):
                summary[f"Within {factor}x (%)"] = 100 * np.mean(a = log_ratios <= np.log(factor))
            summaries.append(summary)

        ## contrast each comparator with the anchor
        rows = list()
        anchor = SCT_MODALITIES[1]
        for configuration in predictions:
            if configuration == anchor:
                continue
            diff = errors[configuration] - errors[anchor]
            for metric in UTIL_METRICS:
                delta = points[configuration][metric] - points[anchor][metric]

                ## form paired intervals from shared bootstrap draws
                lower, upper = np.percentile(
                    a = boots[configuration][metric] - boots[anchor][metric], q = [2.5, 97.5]
                )
                rows.append(
                    {
                        "Comparator": configuration,
                        "Metric": metric,
                        "Comparator_Value": points[configuration][metric],
                        "Anchor_Value": points[anchor][metric],
                        "Delta": delta,
                        "Delta_CI_Lower": lower,
                        "Delta_CI_Upper": upper,
                        "Favors Anchor": bool(delta < 0 if metric == "CI" else delta > 0),
                        "Interval Excludes Zero": bool(lower > 0 or upper < 0),
                        "Systems With Lower Anchor Error": int(np.sum(a = diff > 0))
                        if metric == "MAE"
                        else np.nan,
                    }
                )
        return pd.DataFrame(data = summaries), pd.DataFrame(data = rows)


## compile utilization ablations and temporal controls
def compile_utilization_controls(
    data: pd.DataFrame,
    controls: dict[tuple[str, str, str], list[np.ndarray]],
    frames: dict[float, pd.DataFrame],
    windows: dict[tuple[float, str, bool], dict],
    n_boot: int = 10000,
    n_boot_ci: int = 1000,
    random_state: int = 42,
    ) -> dict[str, pd.DataFrame]:

        ## validate bootstrap counts
        if not 1 <= n_boot_ci <= n_boot:
            raise ValueError("Bootstrap counts must satisfy 1 <= n_boot_ci <= n_boot")

        ## reuse the same system draws across configurations and temporal windows
        indices = np.random.default_rng(seed = random_state).integers(
            low = 0,
            high = len(data),
            size = (n_boot, len(data)),
        )

        ## init output containers
        target = data["y_mean"].to_numpy(dtype = float)
        ablation_rows, ablation_tests, window_rows, window_tests, cap_rows = (
            [],
            [],
            [],
            [],
            [],
        )

        ## compare ablations within each regime and family
        for regime in RESAMPLING_REGIMES:
            for family in GLM_ESTIMATORS:
                print(f"Compiling utilization controls: {regime}, {family}")
                predictions = {
                    name: controls[(regime, family, name)] for name in UTIL_ABLATIONS
                }
                summary, tests = stat_utilization_comparisons(
                    y_true = target,
                    predictions = predictions,
                    indices = indices,
                    n_boot_ci = n_boot_ci,
                )
                ablation_rows.append(
                    summary.assign(**{"Resampling Regime": regime, "Estimator": family})
                )
                ablation_tests.append(
                    tests.assign(**{"Resampling Regime": regime, "Estimator": family})
                )

        ## compare anchor and exclusion within each temporal window
        for fraction, frame in frames.items():
            target_window = frame["y_mean"].to_numpy(dtype = float)
            for regime in RESAMPLING_REGIMES:
                for family in GLM_ESTIMATORS:
                    predictions = {
                        mode: [windows[(fraction, regime, True)][(family, mode)]["y_pred"]]
                        for mode in SCT_MODALITIES
                    }
                    summary, tests = stat_utilization_comparisons(
                        y_true = target_window,
                        predictions = predictions,
                        indices = indices,
                        n_boot_ci = n_boot_ci,
                    )
                    labels = {
                        "Prior-Window Share": f"{fraction:.0%}",
                        "Resampling Regime": regime,
                        "Estimator": family,
                    }
                    window_rows.append(summary.assign(**labels))
                    window_tests.append(tests.assign(**labels))

                    ## contrast cap-enabled and cap-disabled anchored predictions
                    capped = predictions[SCT_MODALITIES[1]][0]
                    uncapped = windows[(fraction, regime, False)][(family, SCT_MODALITIES[1])][
                        "y_pred"
                    ]

                    ## compare separately fitted cap settings, which may select different penalties
                    changed = int(np.sum(a = ~np.isclose(a = capped, b = uncapped)))
                    for setting, prediction in (
                        ("Cap enabled", capped),
                        ("Cap disabled", uncapped),
                    ):
                        scores = compile_utilization_metrics(
                            y_true = target_window, y_pred = prediction
                        )
                        cap_rows.append(
                            {
                                **labels,
                                "Setting": setting,
                                "N_predictions_changed": changed,
                                **{
                                    metric: value
                                    for metric, value in scores.items()
                                },
                            }
                        )

        ## combine summaries and contrasts
        ablations = pd.concat(objs = ablation_rows, ignore_index = True)
        temporal = pd.concat(objs = window_rows, ignore_index = True)
        tests_ablation = pd.concat(objs = ablation_tests, ignore_index = True)
        tests_temporal = pd.concat(objs = window_tests, ignore_index = True)

        ## select the primary domain logo-cv negative binomial results
        primary_ablation = (ablations["Resampling Regime"] == RESAMPLING_REGIMES[0]) & (
            ablations["Estimator"] == GLM_ESTIMATORS[-1]
        )
        primary_tests = tests_ablation.loc[
            (tests_ablation["Resampling Regime"] == RESAMPLING_REGIMES[0])
            & (tests_ablation["Estimator"] == GLM_ESTIMATORS[-1])
            & (tests_ablation["Metric"] == "MAE")
        ]
        ablation = (
            ablations.loc[primary_ablation]
            .merge(
                right = primary_tests[
                    [
                        "Comparator",
                        "Delta",
                        "Delta_CI_Lower",
                        "Delta_CI_Upper",
                    ]
                ],
                how = "left",
                left_on = "Configuration",
                right_on = "Comparator",
            )
            .drop(columns = "Comparator")
        )

        ## select the primary 50% window results
        temporal_primary = (temporal["Prior-Window Share"] == "50%") & (
            temporal["Resampling Regime"] == RESAMPLING_REGIMES[0]
        )
        tests_primary = (tests_temporal["Prior-Window Share"] == "50%") & (
            tests_temporal["Resampling Regime"] == RESAMPLING_REGIMES[0]
        )

        ## assemble output tables
        frame = frames[0.50]
        tables = {
            "ablation": ablation,
            "ablation_complete": ablations,
            "temporal": temporal.loc[temporal_primary].reset_index(drop = True),
            "temporal_contrasts": tests_temporal.loc[tests_primary].reset_index(drop = True),
            "temporal_complete": temporal,
            "cap_sensitivity": pd.DataFrame(data = cap_rows),
            "ablation_contrasts": tests_ablation,
            "window_contrasts": tests_temporal,
            "window_data": pd.DataFrame(
                data = {
                    "System": frame["name"],
                    "Domain": frame["domain"],
                    "L_A": frame["L_A"].astype(dtype = int),
                    "L_B": frame["L"].astype(dtype = int),
                    "r_mean_A": frame["r_mean_A"],
                    "r_mean_B": frame["r_mean"],
                    "B/A": frame["r_mean"] / frame["r_mean_A"],
                }
            ),
            "carry_forward": pd.DataFrame(
                data = [
                    {
                        metric: value
                        for metric, value in compile_utilization_metrics(
                            y_true = frame["y_mean"].to_numpy(dtype = float),
                            y_pred = np.log1p(frame["r_mean_A"].to_numpy(dtype = float)),
                        ).items()
                    }
                ]
            ),
        }

        ## append superiority tests
        tables.update(
            compile_utilization_superiority(
                data = data,
                controls = controls,
                frames = frames,
                windows = windows,
                tables = tables,
            )
        )
        return tables


## significance code from p-value
def _sig_code(p_value: float) -> str:
    if not (pd.notna(obj = p_value) and np.isfinite(float(p_value))):
        return ""
    value = float(p_value)
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return ""


## format utilization table for display
def format_utilization_table(
    frame: pd.DataFrame,
    title: str,
    columns: Sequence[str] | None = None,
    groups: Sequence[str] = (),
    note: str = "",
    details: bool = False,
    decimals: int = 2,
    ) -> HTML:

        ## format a copy so cached numerical results retain full precision
        view = frame.copy(deep = True)

        ## flatten multi-index columns and reset the index
        if isinstance(view.columns, pd.MultiIndex):
            view.columns = [" / ".join(map(str, column)) for column in view.columns]
        if not isinstance(view.index, pd.RangeIndex):
            view = view.reset_index()

        ## combine point estimates and interval bounds for display
        for metric in ("MAE", "RMSE", "CI", "GMME", "Delta"):
            lower, upper = (
                ("Delta_CI_Lower", "Delta_CI_Upper")
                if metric == "Delta"
                else (f"{metric}_lo", f"{metric}_hi")
            )
            if all(column in view for column in (metric, lower, upper)):
                view[f"{metric} [95% CI]"] = [
                    ""
                    if pd.isna(value)
                    else (
                        f"{value:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
                        if pd.notna(lo) and pd.notna(hi)
                        else f"{value:.{decimals}f}"
                    )
                    for value, lo, hi in zip(view[metric], view[lower], view[upper])
                ]
                view = view.drop(columns = [metric, lower, upper])

        ## insert significance codes beside each decision column
        sig_pairs = (
            ("Holm-adj. p", "Sp.", "Sig."),
            ("Holm-adj. p", "Eq.", "Sig."),
            ("Holm-adj. p (Complete)", "Superior (Complete)", "Sig. (Complete)"),
        )
        for holm_col, decision_col, sig_col in sig_pairs:
            if holm_col in view and sig_col not in view:
                sig_values = view[holm_col].apply(func = _sig_code)
                if decision_col in view:
                    view.insert(
                        loc = list(view.columns).index(decision_col),
                        column = sig_col,
                        value = sig_values,
                    )
                else:
                    view[sig_col] = sig_values

        ## select requested columns, retaining their significance codes
        if columns is not None:
            selected_columns = list(columns)
            for holm_col, decision_col, sig_col in sig_pairs:
                if (
                    holm_col in selected_columns
                    and decision_col in selected_columns
                    and sig_col not in selected_columns
                ):
                    selected_columns.insert(
                        selected_columns.index(decision_col), sig_col
                    )
            view = view.loc[:, selected_columns]

        ## order metric columns consistently
        metric_order = {"CI": 0, "MAE": 1, "RMSE": 2, "GMME": 3}
        metric_columns = {
            column: str(column).split(" / ")[0].split(" [")[0]
            for column in view
        }
        metric_columns = {
            column: metric for column, metric in metric_columns.items() if metric in metric_order
        }
        if metric_columns:
            ordered = list(view.columns)
            first = next(index for index, column in enumerate(ordered) if column in metric_columns)
            metrics = sorted(metric_columns, key = lambda column: metric_order[metric_columns[column]])
            remaining = [column for column in ordered if column not in metric_columns]
            view = view.loc[:, remaining[:first] + metrics + remaining[first:]]
        if "Metric" in view:
            categories = list(metric_order) + [
                metric for metric in view["Metric"].unique() if metric not in metric_order
            ]
            view["Metric"] = pd.Categorical(values = view["Metric"], categories = categories, ordered = True)

            ## order metric groups and preserve the caller's row order within each
            view = view.sort_values(by = "Metric", kind = "stable")

        ## tidy feature labels and set the grouping index
        if "Feature" in view:
            view["Feature"] = view["Feature"].str.replace(pat = "_", repl = " ")
        if groups:
            view = view.set_index(keys = list(groups))
            if "Metric" not in view.index.names:
                view = view.sort_index()
            view.index.names = [str(name) for name in groups]

        ## assign formatters by column type
        formatters = {}
        for column in view:
            label = str(column)
            if pd.api.types.is_bool_dtype(arr_or_dtype = view[column]):
                formatters[label] = lambda value: "Yes" if value else "No"
            elif label.endswith(" p") or " p (" in label:
                view[column] = [
                    ""
                    if not (pd.notna(obj = value) and np.isfinite(float(value)))
                    else "<0.001"
                    if float(value) < 0.001
                    else f"{float(value):.3f}"
                    for value in view[column]
                ]
            elif pd.api.types.is_integer_dtype(arr_or_dtype = view[column]):
                formatters[label] = "{:,.0f}"
            elif pd.api.types.is_float_dtype(arr_or_dtype = view[column]):
                formatters[label] = f"{{:,.{decimals}f}}"

        ## style the table
        styled = view.style.format(formatter = formatters, na_rep = "", escape = "html")
        styled = styled.format_index(escape = "html", axis = "index")
        if not groups:
            styled = styled.hide(axis = "index")
        styled = styled.set_caption(caption = escape(s = title)).set_table_styles(
            table_styles = [
                {
                    "selector": "caption",
                    "props": [
                        ("text-align", "left"),
                        ("font-weight", "600"),
                        ("padding", "8px 0"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [("text-align", "left"), ("vertical-align", "top")],
                },
                {"selector": "td", "props": [("white-space", "nowrap")]},
            ]
        )

        ## wrap the table in scrollable html with an optional note and details
        body = (
            '<div style="max-width:100%;overflow-x:auto">'
            + styled.to_html(max_rows = len(view), max_columns = len(view.columns))
            + "</div>"
        )
        if note:
            body += f"<p><small>{escape(s=note)}</small></p>"
        if details:
            body = f"<details><summary>{escape(s=title)} ({len(view):,} rows)</summary>{body}</details>"
        return HTML(data = body)


## publication layout and typography
MM_TO_IN = 1.0 / 25.4
TARGET_WIDTH_MM = 183.0
DOUBLE_COL_IN = TARGET_WIDTH_MM * MM_TO_IN
PUBLICATION_FONT_FAMILY = "Arial"
PUBLICATION_FONT_SIZE = 7.0

## base colors and strokes
BG_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
SPINE_COLOR = "#000000"
SPINE_WIDTH = 0.5
DASHED_GUIDE_COLOR = "#8A8A8A"
DASHED_GUIDE_LW = 0.6
LEGEND_EDGE_COLOR = "#D0D0D0"

## marker diameters
MARKER_D_PRIMARY = 5.0
MARKER_D_SECONDARY = 4.0

## series colors
COL_OBSERVED = "#000000"
COL_CR = "#4A9A60"
COL_CR_FILL = "#DDEFE3"
COL_NO = "#9B2F2F"
COL_NO_FILL = "#EEC6C6"
COL_BAND3 = "#B35F00"
COL_BAND3_FILL = "#F0E8B0"
COL_DYNAMICAL = "#A12A5A"
COL_DYNAMICAL_FILL = "#FCDFEF"
COL_STRUCTURAL = "#2944A2"
COL_STRUCTURAL_FILL = "#D6E0FA"

## series labels
LBL_NO = "Estimated Log-Rate Using SCT Frontier Exclusion"
LBL_CR = "Estimated Log-Rate Using SCT Frontier Anchor"
LBL_NO_SHORT = "SCT Frontier Exclusion"
LBL_CR_SHORT = "SCT Frontier Anchor"

## matplotlib rc parameters
_UTILIZATION_RC = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "font.family": PUBLICATION_FONT_FAMILY,
    "font.sans-serif": [PUBLICATION_FONT_FAMILY],
    "font.size": PUBLICATION_FONT_SIZE,
    "axes.labelsize": PUBLICATION_FONT_SIZE,
    "axes.titlesize": PUBLICATION_FONT_SIZE,
    "xtick.labelsize": PUBLICATION_FONT_SIZE,
    "ytick.labelsize": PUBLICATION_FONT_SIZE,
    "legend.fontsize": PUBLICATION_FONT_SIZE,
    "figure.titlesize": PUBLICATION_FONT_SIZE,
    "axes.linewidth": SPINE_WIDTH,
    "lines.linewidth": 1.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": PUBLICATION_FONT_FAMILY,
    "mathtext.sf": PUBLICATION_FONT_FAMILY,
    "mathtext.it": f"{PUBLICATION_FONT_FAMILY}:italic",
    "mathtext.bf": f"{PUBLICATION_FONT_FAMILY}:bold",
    "mathtext.default": "it",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
}


## style utilization axis
def _style_utilization_axis(
    ax: Axes,
    show_left: bool = True,
    show_bottom: bool = True,
    ) -> None:

        ## hide the top and right spines
        ax.set_facecolor(BG_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(show_left)
        ax.spines["bottom"].set_visible(show_bottom)

        ## set spine and tick appearance
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE_COLOR)
            ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.tick_params(
            axis = "both",
            which = "major",
            color = SPINE_COLOR,
            labelcolor = TEXT_COLOR,
            labelsize = PUBLICATION_FONT_SIZE,
            length = 2.5,
            width = SPINE_WIDTH,
            pad = 1.6,
        )
        ax.grid(False)
        ax.set_axisbelow(True)


## legend padding box
class _UtilizationLegendPad(OffsetBox):

    ## store the reference text
    def __init__(self, text: Text) -> None:
        super().__init__()
        self._text = text

    ## compute the padding box
    def get_bbox(self, renderer: RendererBase) -> mpl.transforms.Bbox:
        ## balance legend padding using rendered glyph height and descent
        prop = self._text.get_fontproperties()
        _, h_lp, d_lp = renderer.get_text_width_height_descent("lp", prop, ismath = False)
        _, h_cap, _ = renderer.get_text_width_height_descent("H", prop, ismath = False)
        return mpl.transforms.Bbox.from_bounds(
            0.0, 0.0, 0.0, max(d_lp - (h_lp - d_lp - h_cap), 0.0)
        )

    ## draw nothing
    def draw(self, renderer: RendererBase) -> None:
        pass


## style utilization legend
def _style_utilization_legend(
    ax: Axes,
    handles: Sequence[Any] | None = None,
    loc: str = "best",
    ncol: int = 1,
    **kwargs: Any,
    ) -> Legend:

        ## set publication legend defaults, allowing overrides
        leg_kwargs = dict(
            loc = loc,
            ncol = ncol,
            fontsize = PUBLICATION_FONT_SIZE,
            frameon = True,
            facecolor = BG_COLOR,
            framealpha = 0.92,
            edgecolor = LEGEND_EDGE_COLOR,
            borderpad = 0.28,
            labelspacing = 0.2,
            handletextpad = 0.4,
            handlelength = 1.35,
            columnspacing = 0.8,
        )
        leg_kwargs.update(kwargs)

        ## draw the legend
        if handles is not None:
            leg = ax.legend(handles = handles, **leg_kwargs)
        else:
            leg = ax.legend(**leg_kwargs)
        leg.get_frame().set_linewidth(SPINE_WIDTH)

        ## pad the legend box to balance glyph spacing
        if leg.texts:
            box = leg._legend_box
            box._children[-1] = VPacker(
                pad = 0.0,
                sep = 0.0,
                children = [_UtilizationLegendPad(leg.texts[0]), box._children[-1]],
            )
            box.set_figure(leg.get_figure())

        ## disable snapping on line handles
        for h in leg.legend_handles:
            if isinstance(h, Line2D):
                h.set_snap(False)
        return leg


## apply publication text styling
def _apply_utilization_text(
    figure: Figure,
    scale: float = 1.0,
    ) -> None:

        ## apply the publication font family, size, and weight to all text
        for text_artist in figure.findobj(match = Text):

            ## preserve annotation sizes after scaling to publication width
            target_fs = getattr(text_artist, "_pub_fontsize", PUBLICATION_FONT_SIZE) / scale
            text_artist.set_fontfamily(PUBLICATION_FONT_FAMILY)
            text_artist.set_fontname(PUBLICATION_FONT_FAMILY)
            text_artist.set_fontstyle("normal")
            txt = text_artist.get_text().strip()
            text_artist.set_fontsize(target_fs)

            ## bold panel labels and flagged annotations
            if txt in {"a", "b", "c", "d"} or getattr(text_artist, "_pub_bold", False):
                text_artist.set_fontweight("bold")
            else:
                text_artist.set_fontweight("normal")


## marker size with matched visual area
def _utilization_marker_size(
    marker: str | mpl.path.Path,
    d: float = MARKER_D_PRIMARY,
    lw: float = 0.0,
    ) -> float:

        ## subtract the edge width from the face diameter
        d_face = d - lw
        if isinstance(marker, mpl.path.Path):
            v = marker.vertices[:-1] * 0.5 / np.max(np.abs(marker.vertices))

            ## normalize polygon area to match the circular marker size
            unit_area = 0.5 * abs(
                np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1))
            )
            return np.pi / 4.0 * d_face**2 / unit_area

        ## match diamond area to circles, otherwise use the squared diameter
        if marker == "D":
            return np.pi / 4.0 * d_face**2
        return d_face**2


## prepare figure for publication export
def _prepare_utilization_figure(
    fig: Figure,
    target_width_mm: float = 183.0,
    pad_in: float = 0.1,
    ) -> mpl.transforms.Bbox:

        ## measure the padded tight bounding box
        _apply_utilization_text(fig, scale = 1.0)
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = mpl.transforms.Bbox.from_extents(
            tight.x0 - pad_in, tight.y0 - pad_in, tight.x1 + pad_in, tight.y1 + pad_in
        )
        width_in = bbox.width
        height_in = bbox.height

        ## compensate text and stroke sizes for the final cropped figure width
        scale = target_width_mm / (width_in * 25.4)
        _apply_utilization_text(fig, scale = scale)
        for artist in fig.findobj():
            if isinstance(artist, mpl.collections.PathCollection):
                artist.set_sizes(artist.get_sizes() / scale**2)
                artist.set_linewidths(
                    np.atleast_1d(artist.get_linewidths()).astype(float) / scale
                )
            elif isinstance(artist, Line2D):
                artist.set_markersize(artist.get_markersize() / scale)
                artist.set_markeredgewidth(artist.get_markeredgewidth() / scale)

        ## rescale remaining patch, line, and collection strokes
        for artist in fig.findobj():
            if isinstance(artist, mpl.collections.PathCollection):
                continue
            if isinstance(artist, (mpl.patches.Patch, Line2D)):
                lw = artist.get_linewidth()
                if lw:
                    artist.set_linewidth(lw / scale)
            elif isinstance(artist, mpl.collections.Collection):
                artist.set_linewidths(
                    np.atleast_1d(artist.get_linewidths()).astype(float) / scale
                )

        ## keep visible strokes within the publication width limits
        minimum_line_width = 0.25 / scale
        maximum_line_width = 1.0 / scale
        for artist in fig.findobj():
            if hasattr(artist, "set_linewidths") and hasattr(artist, "get_linewidths"):
                lws = np.atleast_1d(artist.get_linewidths()).astype(float)
                if lws.size:
                    lws = np.where(
                        (lws > 0) & (lws < minimum_line_width), minimum_line_width, lws
                    )
                    lws = np.where(lws > maximum_line_width, maximum_line_width, lws)
                    artist.set_linewidths(lws)
            elif hasattr(artist, "set_linewidth") and hasattr(artist, "get_linewidth"):
                lw = artist.get_linewidth()
                if lw is not None and lw > 0:
                    if lw < minimum_line_width:
                        artist.set_linewidth(minimum_line_width)
                    elif lw > maximum_line_width:
                        artist.set_linewidth(maximum_line_width)
        fig.canvas.draw()
        return bbox


## publication style context
@contextmanager
def _utilization_style() -> Iterator[None]:
    shrink_factor = _mathtext.SHRINK_FACTOR
    try:
        _mathtext.SHRINK_FACTOR = 0.73
        with mpl.rc_context(rc = _UTILIZATION_RC):
            yield

    ## restore the global mathtext setting even if rendering fails
    finally:
        _mathtext.SHRINK_FACTOR = shrink_factor


## render utilization figure
def _render_utilization_figure(
    figure: Figure,
    raster: bool = False,
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:

        ## prepare the figure within publication bounds
        bounds = _prepare_utilization_figure(fig = figure)

        ## write the publication pdf as fig{index}.pdf when requested
        if index is not None and output_dir is not None:
            export_pdf_scaled(
                fig = figure,
                index = index,
                output_dir = output_dir,
                bbox = bounds,
            )

        ## render the high-resolution preview into memory
        if raster:
            buffer = BytesIO()
            figure.savefig(
                fname = buffer,
                format = "png",
                dpi = 600.0,
                facecolor = figure.get_facecolor(),
                bbox_inches = bounds,
                pad_inches = 0.0,
            )
            result = Image(data = buffer.getvalue(), width = 720)
        else:
            result = Image(data = print_figure(fig = figure, fmt = "png", bbox_inches = "tight"))
        plt.close(fig = figure)
        return result


## plot sct frontier and utilization slack
def plot_utilization_frontier(
    data: pd.DataFrame,
    resampling: dict[str, Any],
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:

        ## load held-out frontiers and negative binomial predictions
        df_all = data.rename(columns = {"evt_atten": "evt_stat"})
        sct_native_oof = compile_utilization_frontier(
            folds = resampling["folds"][RESAMPLING_REGIMES[0]], n_obs = len(data)
        )
        nb_no = resampling["predictions"][RESAMPLING_REGIMES[0]][
            GLM_ESTIMATORS[-1], SCT_MODALITIES[0]
        ]
        nb_cr = resampling["predictions"][RESAMPLING_REGIMES[0]][
            GLM_ESTIMATORS[-1], SCT_MODALITIES[1]
        ]

        ## draw the figure within the publication style
        with _utilization_style():
            fig1, ax = plt.subplots(1, 1, figsize = (DOUBLE_COL_IN, 3.1))
            fig1.patch.set_facecolor(BG_COLOR)

            ## order systems by their held-out frontier estimates
            order = np.argsort(sct_native_oof)

            ## wrap long discipline labels
            _disc_wrap = {
                "International Development": "International\nDevelopment",
                "Interaction Networks": "Interaction\nNetworks",
                "Knowledge Graphs": "Knowledge\nGraphs",
            }
            disc_names = [
                _disc_wrap.get(str(d), str(d))
                for d in df_all["discipline"].to_numpy()[order]
            ]

            ## order arrays by frontier
            y_mean_ord = df_all["y_mean"].to_numpy(float)[order]
            sct_ord = sct_native_oof[order]
            evt_ord = df_all["evt_stat"].to_numpy(float)[order]
            pred_no_ord = nb_no["y_pred"][order]
            pred_cr_ord = nb_cr["y_pred"][order]

            ## clip the displayed evt segment at the frontier
            evt_top_ord = np.minimum(y_mean_ord + evt_ord, sct_ord)

            ## report systems with negative utilization slack
            util_slack_ord = sct_ord - evt_ord - y_mean_ord
            n_neg_slack = int(np.sum(util_slack_ord < 0))
            if n_neg_slack:
                print(
                    f"Figure 1: {n_neg_slack} system(s) with negative utilization slack (observed + EVT > SCT); EVT segment capped at the frontier: "
                    + ", ".join(
                        (
                            f"{d.replace(chr(10), ' ')} ({u:+.2f})"
                            for d, u in zip(disc_names, util_slack_ord)
                            if u < 0
                        )
                    )
                )

            ## set bar colors and widths
            idx_arr = np.arange(len(disc_names))
            COL_SLACK_UPPER = "#C4C8CE"
            COL_SLACK_LOWER = "#E9EBEE"
            SLACK_BAR_W_PT = 8.0
            COL_FRONTIER = "#1F4E79"
            FRONTIER_LW = 2.8
            slack_bar_w = 0.36

            ## draw stacked utilization and evt slack bars with outlines
            bars_slack_upper = ax.bar(
                idx_arr,
                sct_ord - evt_top_ord,
                bottom = evt_top_ord,
                width = slack_bar_w,
                facecolor = mpl.colors.to_rgba(COL_SLACK_UPPER, 0.85),
                edgecolor = "none",
                linewidth = 0.0,
                zorder = 2,
            )
            bars_slack_lower = ax.bar(
                idx_arr,
                evt_top_ord - y_mean_ord,
                bottom = y_mean_ord,
                width = slack_bar_w,
                facecolor = mpl.colors.to_rgba(COL_SLACK_LOWER, 0.72),
                edgecolor = "none",
                linewidth = 0.0,
                zorder = 2,
            )
            bars_outline = ax.bar(
                idx_arr,
                sct_ord - y_mean_ord,
                bottom = y_mean_ord,
                width = slack_bar_w,
                facecolor = "none",
                edgecolor = TEXT_COLOR,
                linewidth = 0.5,
                zorder = 2,
            )

            ## draw caps at the frontier and observed log-rate
            caps_frontier = ax.bar(
                idx_arr,
                0.0,
                bottom = sct_ord,
                width = slack_bar_w,
                facecolor = COL_FRONTIER,
                edgecolor = "none",
                linewidth = 0.0,
                zorder = 4,
            )
            caps_observed = ax.bar(
                idx_arr,
                0.0,
                bottom = y_mean_ord,
                width = slack_bar_w,
                facecolor = COL_OBSERVED,
                edgecolor = "none",
                linewidth = 0.0,
                zorder = 4,
            )

            ## exclude caps from autoscaling and disable snapping
            for cap_k in [*caps_frontier.patches, *caps_observed.patches]:
                cap_k.sticky_edges.y.clear()
            for bar_k in [
                *bars_slack_upper.patches,
                *bars_slack_lower.patches,
                *bars_outline.patches,
                *caps_frontier.patches,
                *caps_observed.patches,
            ]:
                bar_k.set_snap(False)

            ## overlay anchored and excluded estimates
            ax.scatter(
                idx_arr,
                pred_cr_ord,
                color = COL_CR,
                edgecolors = "none",
                marker = "o",
                s = _utilization_marker_size("o"),
                zorder = 5,
            )
            ax.scatter(
                idx_arr,
                pred_no_ord,
                color = COL_NO,
                marker = "x",
                s = _utilization_marker_size("x", lw = 1.0),
                linewidths = 1.0,
                zorder = 6,
            )

            ## build legend handles
            legend_handles_1 = [
                Patch(
                    facecolor = COL_FRONTIER,
                    edgecolor = "none",
                    linewidth = 0.0,
                    label = "SCT Frontier",
                ),
                Patch(
                    facecolor = mpl.colors.to_rgba(COL_SLACK_LOWER, 0.72),
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                    label = "EVT Statistic",
                ),
                Patch(
                    facecolor = mpl.colors.to_rgba(COL_SLACK_UPPER, 0.85),
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                    label = "Utilization Slack",
                ),
                Patch(
                    facecolor = COL_OBSERVED,
                    edgecolor = "none",
                    linewidth = 0.0,
                    label = "Observed Log-Rate",
                ),
                Line2D(
                    [0],
                    [0],
                    marker = "o",
                    color = "w",
                    markerfacecolor = COL_CR,
                    markeredgecolor = "none",
                    markersize = np.sqrt(_utilization_marker_size("o")),
                    label = LBL_CR,
                ),
                Line2D(
                    [0],
                    [0],
                    marker = "x",
                    color = "w",
                    markeredgecolor = COL_NO,
                    markeredgewidth = 1.0,
                    markersize = np.sqrt(_utilization_marker_size("x", lw = 1.0)),
                    label = LBL_NO,
                ),
            ]

            ## style axes and ticks
            _style_utilization_axis(ax)
            ax.set_xlim(-0.65, len(disc_names) - 0.35)
            ax.set_xticks(idx_arr)
            ax.set_xticklabels(
                disc_names,
                rotation = 90,
                rotation_mode = "anchor",
                ha = "right",
                va = "center",
                fontsize = 7.0,
                color = TEXT_COLOR,
            )
            ax.tick_params(
                axis = "x", pad = 2.6, length = 3.5, width = SPINE_WIDTH, color = SPINE_COLOR
            )
            ax.set_ylim(ax.get_ylim()[0], 20.5)
            ax.set_yticks(np.arange(0.0, 20.1, 5.0))
            ax.spines["left"].set_bounds(ax.get_ylim()[0], 20.0)
            ax.set_ylabel("Log-Rate", fontsize = 7.0)

            ## add legend
            leg1 = _style_utilization_legend(
                ax, handles = legend_handles_1, loc = "upper left", ncol = 2
            )
            for h in leg1.legend_handles:
                h.set_snap(False)

            ## measure the page scale and points per data unit after layout
            plt.tight_layout()
            _apply_utilization_text(fig1, scale = 1.0)
            fig1.canvas.draw()
            page_scale = TARGET_WIDTH_MM / (
                (fig1.get_tightbbox(fig1.canvas.get_renderer()).width + 2.0 * 0.1) * 25.4
            )
            ax.set_ylim(ax.get_ylim())
            pt_per_x = (
                ax.get_window_extent().width * 72.0 / fig1.dpi / np.diff(ax.get_xlim())[0]
            )
            pt_per_y = (
                ax.get_window_extent().height * 72.0 / fig1.dpi / np.diff(ax.get_ylim())[0]
            )

            ## resize bars and caps to fixed widths in points
            w_x = SLACK_BAR_W_PT / page_scale / pt_per_x
            cap_w_x = w_x + bars_outline.patches[0].get_linewidth() / pt_per_x
            cap_h_y = FRONTIER_LW / page_scale / pt_per_y
            for caps_k, y_k in [(caps_frontier, sct_ord), (caps_observed, y_mean_ord)]:
                for cap_k, x_c, y_c in zip(caps_k.patches, idx_arr, y_k):
                    cap_k.set_bounds(
                        x_c - 0.5 * cap_w_x, y_c - 0.5 * cap_h_y, cap_w_x, cap_h_y
                    )
            for bar_k in [
                *bars_slack_upper.patches,
                *bars_slack_lower.patches,
                *bars_outline.patches,
            ]:
                x_c = bar_k.get_x() + 0.5 * bar_k.get_width()
                bar_k.set_width(w_x)
                bar_k.set_x(x_c - 0.5 * w_x)

            ## match legend cap handles to the frontier line width
            for h in leg1.legend_handles:
                if h.get_label() in ("SCT Frontier", "Observed Log-Rate"):
                    h_c = h.get_y() + 0.5 * h.get_height()
                    h.set_y(h_c - 0.5 * FRONTIER_LW / page_scale)
                    h.set_height(FRONTIER_LW / page_scale)
                    h.set_x(h.get_x() - 0.5 * 0.5)
                    h.set_width(h.get_width() + 0.5)
            return _render_utilization_figure(figure = fig1, index = index, output_dir = output_dir, raster = True)


## plot utilization estimation quality
def plot_utilization_quality(
    tables: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    resampling: dict[str, Any],
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:

        ## prepare benchmark results and labels
        df_all = data
        df_6_main = tables["benchmark"].copy()
        families = list(GLM_ESTIMATORS)
        fam_short = families
        modes = [LBL_NO, LBL_CR]
        colors = {LBL_NO: COL_NO, LBL_CR: COL_CR}
        x_pos = np.arange(len(families))
        SI_DOMAINS = sorted(data["domain"].unique())

        ## load the mae equivalence margin
        DELTA_MAE = float(
            tables["equivalence"]
            .loc[tables["equivalence"]["Metric"] == "MAE", "delta"]
            .iloc[0]
        )
        dom_vec = data["domain"].to_numpy()
        disc_vec = data["discipline"].astype(str).to_numpy()
        y_mean_si = data["y_mean"].to_numpy(float)

        ## compare repeated-split errors with domain-held-out errors per system
        si_diff = np.stack(
            [
                np.mean(
                    [
                        np.abs(y_mean_si - values)
                        for (regime, repeat, model), values in resampling[
                            "repeated"
                        ].items()
                        if regime == comparison and model == family
                    ],
                    axis = 0,
                )
                - np.abs(
                    y_mean_si
                    - resampling["predictions"][RESAMPLING_REGIMES[0]][
                        family, SCT_MODALITIES[1]
                    ]["y_pred"]
                )
                for comparison in RESAMPLING_REGIMES[2:]
                for family in families
            ],
            axis = 1,
        )

        ## summarize per-system differences
        si_med = np.median(si_diff, axis = 1)
        si_q1, si_q3 = np.percentile(si_diff, [25, 75], axis = 1)

        ## draw the figure within the publication style
        with _utilization_style():
            MARKER_D_SECONDARY = 4.0

            ## relabel estimation regimes for display
            df_6_main["Estimation Regime"] = df_6_main["Estimation Regime"].map(
                lambda m: (
                    LBL_NO
                    if "Exclusion" in m or "Absence" in m
                    else LBL_CR
                    if "Anchor" in m
                    else m
                )
            )

            ## compute figure and gridspec geometry
            fig_w2, fig_h_ac = (DOUBLE_COL_IN, 3.5)
            bottom_ac, top_ac = (0.16, 0.94)
            hspace_2, wspace_2 = (0.23, 0.2513)
            D_H_IN = (2.0 + wspace_2) * (top_ac - bottom_ac) * fig_h_ac / len(SI_DOMAINS)
            D_GAP_IN, D_BOTTOM_IN = (0.65, 0.95)
            fig_h2 = D_BOTTOM_IN + D_H_IN + D_GAP_IN + (1.0 - bottom_ac) * fig_h_ac
            bottom_2 = (D_BOTTOM_IN + D_H_IN + D_GAP_IN) / fig_h2
            top_2 = bottom_2 + (top_ac - bottom_ac) * fig_h_ac / fig_h2
            bc_w_frac = 1.0
            bc_h_frac = 0.95
            hspace_bc = (2.0 + hspace_2) / bc_h_frac - 2.0
            wspace_gs2 = 2.0 * (2.0 + wspace_2) / (1.0 + bc_w_frac) - 2.0
            H_fig2 = top_2 - bottom_2
            H_in2 = H_fig2 * fig_h2
            W_col_fig2 = H_in2 / fig_w2
            W_total_fig2 = (2.0 + wspace_2) * W_col_fig2
            left_2 = (1.0 - W_total_fig2) / 2.0
            right_2 = left_2 + W_total_fig2

            ## create the figure and panel axes
            fig2 = plt.figure(figsize = (fig_w2, fig_h2), facecolor = BG_COLOR)
            gs2 = fig2.add_gridspec(
                2,
                2,
                left = left_2,
                right = right_2,
                bottom = bottom_2,
                top = top_2,
                wspace = wspace_gs2,
                hspace = hspace_2,
                width_ratios = [1.0, bc_w_frac],
            )
            ax_a = fig2.add_subplot(gs2[:, 0])
            gs2_bc = gs2[:, 1].subgridspec(2, 1, hspace = hspace_bc)
            ax_b = fig2.add_subplot(gs2_bc[0])
            ax_c = fig2.add_subplot(gs2_bc[1])
            ax_a.set_box_aspect(1.0)

            ## compare consensus and error across model families
            ax = ax_a

            ## define family markers
            _tri_R = 2 / np.sqrt(3)
            tri_marker = mpl.path.Path(
                [(0, _tri_R), (-1, -_tri_R / 2), (1, -_tri_R / 2), (0, _tri_R)],
                [
                    mpl.path.Path.MOVETO,
                    mpl.path.Path.LINETO,
                    mpl.path.Path.LINETO,
                    mpl.path.Path.CLOSEPOLY,
                ],
            )
            fam_markers = {
                "Poisson": "o",
                "Poisson-Tweedie": tri_marker,
                "Negative Binomial": "s",
            }

            ## plot exclusion and anchor estimates for each family
            cluster_ext = {LBL_NO: [[], []], LBL_CR: [[], []]}
            for f_name in families:
                row_u = df_6_main[
                    (df_6_main["Estimator"] == f_name)
                    & (df_6_main["Estimation Regime"] == LBL_NO)
                ].iloc[0]
                row_b = df_6_main[
                    (df_6_main["Estimator"] == f_name)
                    & (df_6_main["Estimation Regime"] == LBL_CR)
                ].iloc[0]
                x_u, y_u = (float(row_u["MAE"]), float(row_u["CI"]))
                x_b, y_b = (float(row_b["MAE"]), float(row_b["CI"]))
                ax.scatter(
                    x_u,
                    y_u,
                    marker = fam_markers[f_name],
                    color = COL_NO,
                    edgecolors = "none",
                    s = _utilization_marker_size(fam_markers[f_name]),
                    snap = False,
                    zorder = 5,
                )
                ax.scatter(
                    x_b,
                    y_b,
                    marker = fam_markers[f_name],
                    color = COL_CR,
                    edgecolors = "none",
                    s = _utilization_marker_size(fam_markers[f_name]),
                    snap = False,
                    zorder = 6,
                )
                cluster_ext[LBL_NO][0].append(x_u)
                cluster_ext[LBL_NO][1].append(y_u)
                cluster_ext[LBL_CR][0].append(x_b)
                cluster_ext[LBL_CR][1].append(y_b)

            ## outline each configuration's range with labeled corners
            box_labels = {
                LBL_NO: {
                    "(Min, Max)": ((-7, 0), "right", "center"),
                    "(Max, Min)": ((5, 0), "left", "center"),
                },
                LBL_CR: {
                    "(Min, Max)": ((4, 4), "center", "bottom"),
                    "(Max, Min)": ((5, 0), "left", "center"),
                },
            }
            for _reg_lbl, _reg_col, _reg_fill in [
                (LBL_NO, COL_NO, (*mpl.colors.to_rgb(COL_NO), 0.18)),
                (LBL_CR, COL_CR, (*mpl.colors.to_rgb(COL_CR), 0.18)),
            ]:
                _xs, _ys = cluster_ext[_reg_lbl]
                _x0, _x1, _y0, _y1 = (min(_xs), max(_xs), min(_ys), max(_ys))
                ax.add_patch(
                    Rectangle(
                        (_x0, _y0),
                        _x1 - _x0,
                        _y1 - _y0,
                        facecolor = _reg_fill,
                        edgecolor = _reg_col,
                        lw = 0.6,
                        linestyle = (0, (4, 2)),
                        zorder = 1.5,
                    )
                )
                for _txt, _xy in (("(Min, Max)", (_x0, _y1)), ("(Max, Min)", (_x1, _y0))):
                    _off, _ha, _va = box_labels[_reg_lbl][_txt]
                    ax.annotate(
                        _txt,
                        _xy,
                        xytext = _off,
                        textcoords = "offset points",
                        fontsize = 7.0,
                        color = _reg_col,
                        ha = _ha,
                        va = _va,
                        zorder = 4,
                    )

            ## draw an arrow from exclusion to anchor
            row_from = df_6_main[
                (df_6_main["Estimator"] == "Negative Binomial")
                & (df_6_main["Estimation Regime"] == LBL_NO)
            ].iloc[0]
            arrow_to = (max(cluster_ext[LBL_CR][0]), min(cluster_ext[LBL_CR][1]))
            ax.annotate(
                "",
                xy = arrow_to,
                xytext = (float(row_from["MAE"]), float(row_from["CI"])),
                arrowprops = dict(
                    arrowstyle = "->",
                    color = TEXT_COLOR,
                    lw = 0.5,
                    alpha = 0.8,
                    shrinkA = MARKER_D_PRIMARY / 2.0 + 3.5,
                    shrinkB = MARKER_D_PRIMARY / 2.0 + 3.5,
                ),
                zorder = 3,
            )

            ## build legend handles
            legend_handles_2a_reg = [
                Patch(
                    facecolor = COL_CR,
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                    label = LBL_CR_SHORT,
                ),
                Patch(
                    facecolor = COL_NO,
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                    label = LBL_NO_SHORT,
                ),
            ]
            legend_handles_2a_fam = [
                Line2D(
                    [0],
                    [0],
                    color = TEXT_COLOR,
                    marker = fam_markers[f],
                    markersize = np.sqrt(_utilization_marker_size(fam_markers[f])),
                    markeredgewidth = 0.0,
                    lw = 0.0,
                    label = f,
                )
                for f in families
            ]

            ## style axes and add legends
            _style_utilization_axis(ax)
            ax.set_xlim(0.0, 4.1)
            ax.set_xticks(np.arange(0.0, 4.1, 1.0))
            ax.set_xticklabels([f"{t:.1f}" for t in np.arange(0.0, 4.1, 1.0)])
            ax.spines["bottom"].set_bounds(0.0, 4.0)
            ax.set_ylim(0.5, 1.05)
            ax.set_xlabel("Mean Absolute Error (MAE)", fontsize = 7.0)
            ax.set_ylabel("Consensus Index (CI)", fontsize = 7.0)
            ax.set_title("a", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)
            ax.add_artist(
                _style_utilization_legend(
                    ax, handles = legend_handles_2a_reg, loc = "upper right"
                )
            )
            _style_utilization_legend(ax, handles = legend_handles_2a_fam, loc = "lower left")
            bar_w2 = 0.22 / bc_w_frac

            ## show paired consensus scores for each model family
            ax = ax_b
            for idx_m, m_name in enumerate(modes):
                vals_ci = [
                    df_6_main.loc[
                        (df_6_main["Estimator"] == f)
                        & (df_6_main["Estimation Regime"] == m_name),
                        "CI",
                    ].values[0]
                    for f in families
                ]
                bar_x = x_pos + (0.5 - idx_m) * bar_w2
                ax.bar(
                    bar_x,
                    vals_ci,
                    width = bar_w2,
                    color = colors[m_name],
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                )
                for xp, vp in zip(bar_x, vals_ci):
                    ax.text(
                        xp,
                        vp + 0.025,
                        f"{vp:.2f}",
                        ha = "center",
                        va = "bottom",
                        fontsize = 7.0,
                        color = TEXT_COLOR,
                    )

            ## style axes and ticks
            _style_utilization_axis(ax)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(fam_short, fontsize = 7.0)
            ax.set_ylabel("Consensus Index (CI)", fontsize = 7.0)
            ax.set_ylim(0.0, 1.18)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_title("b", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

            ## show paired absolute errors for each model family
            ax = ax_c
            for idx_m, m_name in enumerate(modes):
                vals = [
                    df_6_main.loc[
                        (df_6_main["Estimator"] == f)
                        & (df_6_main["Estimation Regime"] == m_name),
                        "MAE",
                    ].values[0]
                    for f in families
                ]
                bar_x = x_pos + (0.5 - idx_m) * bar_w2
                ax.bar(
                    bar_x,
                    vals,
                    width = bar_w2,
                    color = colors[m_name],
                    edgecolor = TEXT_COLOR,
                    linewidth = 0.5,
                )
                for xp, vp in zip(bar_x, vals):
                    ax.text(
                        xp,
                        vp + 0.08,
                        f"{vp:.2f}",
                        ha = "center",
                        va = "bottom",
                        fontsize = 7.0,
                        color = TEXT_COLOR,
                    )

            ## style axes and ticks
            _style_utilization_axis(ax)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(fam_short, fontsize = 7.0)
            ax.set_ylabel("Mean Absolute Error (MAE)", fontsize = 7.0)
            ax.set_ylim(0.0, 4.2)
            ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
            ax.set_yticklabels(["0.0", "1.0", "2.0", "3.0", "4.0"])
            ax.spines["left"].set_bounds(0.0, 4.0)
            ax.set_title("c", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

            ## set domain colors and labels
            DOMAIN_COLORS = {
                "Earth & Physical Sciences": "#C07D3A",
                "Life Sciences & Medicine": "#8A66B5",
                "Technology & Information": "#2C6E91",
                "Trade & Institutions": "#B8860B",
                "Transportation & Infrastructure": "#6B4E31",
            }
            DOMAIN_LABELS = {
                "Transportation & Infrastructure": "Transport & Infrastructure"
            }
            _disc_wrap_d = {
                "International Development": "International\nDevelopment",
                "Interaction Networks": "Interaction\nNetworks",
                "Knowledge Graphs": "Knowledge\nGraphs",
            }

            ## format signed differences with a true minus sign
            _d_signed = lambda v: f"{round(v, 2) + 0.0:+.2f}".replace("-", "−")
            ylim_d = 1.0

            ## create the domain panel gridspec
            gs2_d = fig2.add_gridspec(
                1,
                len(SI_DOMAINS),
                left = left_2,
                right = right_2,
                bottom = D_BOTTOM_IN / fig_h2,
                top = (D_BOTTOM_IN + D_H_IN) / fig_h2,
                wspace = 0.0,
            )

            ## group per-system error shifts by domain with equivalence-margin guides
            axes_d = []
            for k_d, dom_d in enumerate(SI_DOMAINS):
                ax = fig2.add_subplot(gs2_d[0, k_d], sharey = axes_d[0] if axes_d else None)
                axes_d.append(ax)
                col_d = DOMAIN_COLORS[dom_d]
                idx_d = np.where(dom_vec == dom_d)[0]
                idx_d = idx_d[np.argsort(df_all["y_max"].to_numpy(float)[idx_d])]
                xs_d = np.arange(len(idx_d))

                ## shade the equivalence margin and draw reference lines
                ax.axhspan(
                    -DELTA_MAE,
                    DELTA_MAE,
                    facecolor = COL_CR_FILL,
                    alpha = 0.75,
                    edgecolor = "none",
                    zorder = 0,
                )
                for yy_d in (-DELTA_MAE, DELTA_MAE):
                    ax.axhline(
                        yy_d,
                        color = DASHED_GUIDE_COLOR,
                        linestyle = "--",
                        lw = DASHED_GUIDE_LW,
                        zorder = 1,
                    )
                ax.axhline(0.0, color = TEXT_COLOR, lw = 0.5, zorder = 1)

                ## draw the domain median and per-system iqr
                dom_median_d = float(np.median(si_diff[idx_d]))
                ax.axhline(dom_median_d, color = col_d, linestyle = "--", lw = 0.8, zorder = 2)
                ax.errorbar(
                    xs_d,
                    si_med[idx_d],
                    yerr = [si_med[idx_d] - si_q1[idx_d], si_q3[idx_d] - si_med[idx_d]],
                    fmt = "o",
                    color = col_d,
                    ecolor = col_d,
                    markersize = np.sqrt(_utilization_marker_size("o", MARKER_D_SECONDARY)),
                    markeredgewidth = 0,
                    elinewidth = 1.0,
                    capsize = 0,
                    zorder = 3,
                )

                ## style axes, hiding inner y-axes
                _style_utilization_axis(ax, show_left = k_d == 0)
                if k_d > 0:
                    ax.tick_params(axis = "y", which = "both", left = False, labelleft = False)
                    ax.axvline(-0.5, color = SPINE_COLOR, lw = SPINE_WIDTH, zorder = 4)
                ax.set_xlim(-0.5, len(idx_d) - 0.5)
                ax.set_xticks(xs_d)
                ax.set_xticklabels(
                    [_disc_wrap_d.get(disc_vec[i], disc_vec[i]) for i in idx_d],
                    rotation = 90,
                    rotation_mode = "anchor",
                    ha = "right",
                    va = "center",
                    fontsize = 7.0,
                    color = col_d,
                )
                ax.tick_params(axis = "x", pad = 2.6)
                ax.set_ylim(-ylim_d, ylim_d)
                ax.set_title(
                    DOMAIN_LABELS.get(dom_d, dom_d),
                    color = col_d,
                    fontsize = 7.0,
                    fontweight = "bold",
                    pad = 4,
                )
                ax.title._pub_bold = True

                ## annotate the domain median difference
                _dmae_txt = ax.text(
                    0.5,
                    0.95,
                    f"Δ MAE = {_d_signed(dom_median_d)}",
                    transform = ax.transAxes,
                    ha = "center",
                    va = "top",
                    fontsize = 7.0,
                    color = TEXT_COLOR,
                    fontweight = "bold",
                    zorder = 5,
                )
                _dmae_txt._pub_bold = True

            ## label the shared y-axis and margin guides
            ax = axes_d[0]
            ax.set_ylabel("Δ MAE (k-Fold − Domain LOGO-CV)", fontsize = 7.0)
            ax.set_yticks(np.arange(-ylim_d, ylim_d + 1e-09, 0.5))
            ax.yaxis.set_major_formatter("{x:.1f}")
            for yy_d, s_d in ((DELTA_MAE, "+$\\delta$"), (-DELTA_MAE, "−$\\delta$")):
                ax.annotate(
                    s_d,
                    (0.0, yy_d),
                    xycoords = ("axes fraction", "data"),
                    xytext = (-2.0, 0.0),
                    textcoords = "offset points",
                    ha = "right",
                    va = "center",
                    fontsize = 7.0,
                    color = DASHED_GUIDE_COLOR,
                )

            ## add panel label
            ax.annotate(
                "d",
                (0.0, 1.0),
                xycoords = ("axes fraction", ax.title),
                xytext = (0.0, 4.0),
                textcoords = "offset points",
                ha = "left",
                va = "baseline",
                fontsize = 7.0,
                fontweight = "bold",
            )

            ## add legend
            _style_utilization_legend(
                axes_d[-1],
                loc = "lower right",
                ncol = 2,
                handles = [
                    Line2D(
                        [0],
                        [0],
                        color = "#777777",
                        marker = "o",
                        markersize = np.sqrt(
                            _utilization_marker_size("o", MARKER_D_SECONDARY)
                        ),
                        markeredgewidth = 0.0,
                        lw = 1.0,
                        label = "Discipline Median [IQR]",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color = "#777777",
                        linestyle = "--",
                        lw = 0.8,
                        label = "Domain Median",
                    ),
                ],
            )

            ## align y-labels and render
            fig2.align_ylabels([ax_a, axes_d[0]])
            return _render_utilization_figure(figure = fig2, index = index, output_dir = output_dir)


## plot target contraction and coefficient attribution
def plot_utilization_attribution(
    tables: dict[str, pd.DataFrame],
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:
        targets = tables["targets"]
        df_all = targets.loc[targets["Resampling Regime"] == RESAMPLING_REGIMES[0]].rename(
            columns = {"Observed_LogRate": "y_mean", "EVT": "evt_stat"}
        )
        sct_native_oof = df_all["Frontier"].to_numpy(float)
        REDUCED_6_DESC = FEAT_6U

        ## load held-out negative binomial coefficients by fold
        folds = tables["fold_coefficients"]
        folds = folds.loc[
            (folds["Resampling Regime"] == RESAMPLING_REGIMES[0])
            & (folds["Estimator"] == GLM_ESTIMATORS[-1])
        ]
        coefs_no, coefs_cr = [
            folds.loc[folds["Configuration"] == mode]
            .pivot(index = "Fold", columns = "Feature", values = "Coefficient")[FEAT_6U]
            .to_numpy(float)
            for mode in SCT_MODALITIES
        ]

        ## draw the figure within the publication style
        with _utilization_style():
            MARKER_D_SECONDARY = 4.0

            ## use computer modern for math labels
            with mpl.rc_context({"mathtext.fontset": "cm"}):
                fig3, axes_mech = plt.subplots(1, 2, figsize = (DOUBLE_COL_IN, 3.3))
                fig3.patch.set_facecolor(BG_COLOR)

                ## compare target dispersion through the capacity adjustment stages
                ax = axes_mech[0]
                y_mean_vec = df_all["y_mean"].to_numpy(float)
                evt_vec = df_all["evt_stat"].to_numpy(float)
                util_slack_vec = sct_native_oof - evt_vec - y_mean_vec

                ## define exclusion and anchor targets centered on their medians
                stages_b = [
                    (
                        "Exclusion Target",
                        y_mean_vec - np.median(y_mean_vec),
                        COL_NO,
                        COL_NO,
                        0.18,
                    ),
                    (
                        "Anchor Target",
                        util_slack_vec - np.median(util_slack_vec),
                        COL_CR,
                        COL_CR,
                        0.18,
                    ),
                ]

                ## set density and box geometry
                x_grid_b = np.linspace(-10.0, 10.0, 500)
                y_span_orig = 1.96
                y_base_green = 0.0
                kde_peak_h = 0.55 / y_span_orig
                row_y = np.array(
                    [y_base_green, y_base_green + 1.5 * kde_peak_h], dtype = float
                )
                y_box_offset = 0.1 / y_span_orig
                cap_half_h = 0.045 / y_span_orig
                h_bar = 0.32
                y_top_a = row_y.max() + kde_peak_h + 0.17
                box_half_h = 0.5 * (h_bar / (len(REDUCED_6_DESC) + 0.1)) * y_top_a
                q25_list, q75_list = ([], [])

                ## draw a density and box plot for each target
                for r_i, (lbl_b, vals_b, col_b, fill_b, alph_b) in zip(row_y, stages_b):
                    kde_b = stats.gaussian_kde(vals_b, bw_method = 0.38)
                    dens_b = kde_b(x_grid_b)
                    dens_scaled = dens_b / np.max(dens_b) * kde_peak_h
                    ax.fill_between(
                        x_grid_b,
                        r_i,
                        r_i + dens_scaled,
                        color = fill_b,
                        alpha = alph_b,
                        lw = 0.0,
                        zorder = 3,
                    )
                    ax.plot(x_grid_b, r_i + dens_scaled, color = col_b, lw = 0.6, zorder = 4)
                    ax.plot(
                        x_grid_b,
                        np.full_like(x_grid_b, r_i),
                        color = col_b,
                        lw = 0.6,
                        alpha = 0.75,
                        zorder = 4,
                    )

                    ## draw whiskers, fliers, box, and median
                    q10, q25, q50, q75, q90 = np.percentile(vals_b, [10, 25, 50, 75, 90])
                    q25_list.append(q25)
                    q75_list.append(q75)
                    y_box = r_i + y_box_offset
                    ax.plot([q10, q25], [y_box, y_box], color = TEXT_COLOR, lw = 0.5, zorder = 5)
                    ax.plot([q75, q90], [y_box, y_box], color = TEXT_COLOR, lw = 0.5, zorder = 5)
                    ax.plot(
                        [q10, q10],
                        [y_box - cap_half_h, y_box + cap_half_h],
                        color = TEXT_COLOR,
                        lw = 0.5,
                        zorder = 5,
                    )
                    ax.plot(
                        [q90, q90],
                        [y_box - cap_half_h, y_box + cap_half_h],
                        color = TEXT_COLOR,
                        lw = 0.5,
                        zorder = 5,
                    )
                    fliers = vals_b[(vals_b < q10) | (vals_b > q90)]
                    ax.scatter(
                        fliers,
                        np.full_like(fliers, y_box),
                        s = _utilization_marker_size("o", MARKER_D_SECONDARY),
                        color = col_b,
                        edgecolor = "none",
                        alpha = 0.85,
                        zorder = 1,
                    )
                    ax.fill_between(
                        [q25, q75],
                        y_box - box_half_h,
                        y_box + box_half_h,
                        color = col_b,
                        edgecolor = TEXT_COLOR,
                        lw = 0.5,
                        alpha = 0.9,
                        zorder = 6,
                    )
                    ax.plot(
                        [q50, q50],
                        [y_box - box_half_h, y_box + box_half_h],
                        color = TEXT_COLOR,
                        lw = 0.5,
                        solid_capstyle = "butt",
                        zorder = 7,
                    )

                ## connect interquartile ranges between targets
                ax.fill_betweenx(
                    [row_y[0] + y_box_offset, row_y[1] + y_box_offset],
                    [q25_list[0], q25_list[1]],
                    [q75_list[0], q75_list[1]],
                    color = COL_CR_FILL,
                    alpha = 0.55,
                    lw = 0.0,
                    zorder = 2,
                )
                ax.plot(
                    q25_list,
                    row_y + y_box_offset,
                    color = DASHED_GUIDE_COLOR,
                    linestyle = "--",
                    lw = DASHED_GUIDE_LW,
                    zorder = 4,
                )
                ax.plot(
                    q75_list,
                    row_y + y_box_offset,
                    color = DASHED_GUIDE_COLOR,
                    linestyle = "--",
                    lw = DASHED_GUIDE_LW,
                    zorder = 4,
                )

                ## annotate the iqr contraction
                iqr_unb = q75_list[0] - q25_list[0]
                iqr_bnd = q75_list[1] - q25_list[1]
                iqr_pct = (1.0 - iqr_bnd / iqr_unb) * 100.0
                ann_y = 0.5 * (row_y[0] + row_y[1]) + y_box_offset + 0.05
                ann_t = (ann_y - (row_y[0] + y_box_offset)) / (row_y[1] - row_y[0])
                ann_x_right = q75_list[0] + ann_t * (q75_list[1] - q75_list[0])
                ann_x_text = 2.4
                ann_arrow_drop_pt = 0.4
                ann_arrow_tr = mpl.transforms.offset_copy(
                    ax.transData, fig = fig3, y = -ann_arrow_drop_pt, units = "points"
                )
                ax.annotate(
                    "",
                    xy = (ann_x_right, ann_y),
                    xytext = (ann_x_text, ann_y),
                    xycoords = ann_arrow_tr,
                    textcoords = ann_arrow_tr,
                    arrowprops = dict(
                        arrowstyle = "->", color = TEXT_COLOR, lw = 0.5, alpha = 0.8, shrinkA = 1.0
                    ),
                    zorder = 8,
                )
                ann_iqr = ax.text(
                    ann_x_text,
                    ann_y,
                    f"{iqr_pct:.0f}% IQR Contraction",
                    ha = "left",
                    va = "center_baseline",
                    fontsize = 7.0,
                    fontweight = "bold",
                    zorder = 8,
                )
                ann_iqr._pub_bold = True

                ## style axes and ticks
                _style_utilization_axis(ax)
                ax.spines["bottom"].set_zorder(5)
                ax.spines["left"].set_zorder(5)
                ax.set_ylim(0.0, y_top_a)
                dens_fracs = np.array([0.0, 0.5, 1.0])
                ax.set_yticks(
                    np.concatenate([r_i + dens_fracs * kde_peak_h for r_i in row_y])
                )
                ax.set_yticklabels([f"{f:.1f}" for _ in row_y for f in dens_fracs])
                ax.set_xlim(-10.0, 10.0)
                ax.set_xticks([-10, -5, 0, 5, 10])
                ax.set_xticklabels(["−10", "−5", "0", "5", "10"])
                ax.set_box_aspect(1.0)
                ax.set_xlabel("Median-Centered Log-Rate Dispersion", fontsize = 7.0)
                ax.set_ylabel("Relative Density", fontsize = 7.0)
                ax.set_title("a", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

                ## compare standardized coefficients from the actual held-out fits
                ax = axes_mech[1]

                ## label features with their notation
                desc_pretty = [
                    "Number of Edges\n" + "$x^{\\prime}_{u,1}\\;(x^{\\prime}_{2})$",
                    "Degree Entropy\n" + "$x^{\\prime}_{u,2}\\;(x^{\\prime}_{13})$",
                    "Random Walk Triangle Weight\n"
                    + "$x^{\\prime}_{u,3}\\;(x^{\\prime}_{18})$",
                    "Normalized Successive Difference\n"
                    + "$z^{\\prime}_{u,1}\\;(z^{\\prime}_{4})$",
                    "Coefficient of Variation\n"
                    + "$z^{\\prime}_{u,2}\\;(z^{\\prime}_{2})$",
                    "Trend Coefficient\n" + "$z^{\\prime}_{u,3}\\;(z^{\\prime}_{6})$",
                ]

                ## reverse feature order and average coefficients across folds
                desc_pretty = desc_pretty[::-1]
                coefs_no = coefs_no[:, ::-1]
                coefs_cr = coefs_cr[:, ::-1]
                y_idx = np.arange(len(desc_pretty))
                mean_no, mean_cr = (coefs_no.mean(axis = 0), coefs_cr.mean(axis = 0))

                ## draw paired coefficient bars
                ax.barh(
                    y_idx + h_bar / 2,
                    mean_no,
                    height = h_bar,
                    color = COL_NO,
                    edgecolor = TEXT_COLOR,
                    alpha = 0.88,
                    linewidth = 0.5,
                    label = LBL_NO_SHORT,
                )
                ax.barh(
                    y_idx - h_bar / 2,
                    mean_cr,
                    height = h_bar,
                    color = COL_CR,
                    edgecolor = TEXT_COLOR,
                    alpha = 0.9,
                    linewidth = 0.5,
                    label = LBL_CR_SHORT,
                )

                ## annotate coefficient values
                for y_i, m_n, m_c in zip(y_idx, mean_no, mean_cr):
                    off_n = m_n + 0.05 if m_n >= 0 else m_n - 0.05
                    off_c = m_c + 0.05 if m_c >= 0 else m_c - 0.05
                    ax.text(
                        off_n,
                        y_i + h_bar / 2,
                        f"{m_n:+.2f}".replace("-", "−"),
                        va = "center_baseline",
                        ha = "left" if m_n >= 0 else "right",
                        fontsize = 7.0,
                        color = TEXT_COLOR,
                    )
                    ax.text(
                        off_c,
                        y_i - h_bar / 2,
                        f"{m_c:+.2f}".replace("-", "−"),
                        va = "center_baseline",
                        ha = "left" if m_c >= 0 else "right",
                        fontsize = 7.0,
                        color = TEXT_COLOR,
                    )

                ## shade dynamical and structural feature groups
                ax.axvline(0.0, color = TEXT_COLOR, linestyle = "-", lw = 0.5)
                ax.axhline(
                    2.5, color = DASHED_GUIDE_COLOR, linestyle = "--", lw = DASHED_GUIDE_LW
                )
                ax.axhspan(
                    -0.55, 2.5, facecolor = COL_DYNAMICAL_FILL, edgecolor = "none", zorder = 0
                )
                ax.axhspan(
                    2.5,
                    len(desc_pretty) - 0.45,
                    facecolor = COL_STRUCTURAL_FILL,
                    edgecolor = "none",
                    zorder = 0,
                )
                ax_yt = ax.get_yaxis_transform()
                for y_lo, y_hi, grp, col_grp in [
                    (-0.45, 2.4, "Dynamical", COL_DYNAMICAL),
                    (2.6, len(desc_pretty) - 0.55, "Structural", COL_STRUCTURAL),
                ]:
                    ax.text(
                        1.04,
                        0.5 * (y_lo + y_hi),
                        grp,
                        transform = ax_yt,
                        fontsize = 7.0,
                        color = col_grp,
                        ha = "center",
                        va = "center",
                        rotation = 270,
                    )

                ## style axes and ticks
                _style_utilization_axis(ax)
                ax.set_xlim(-2.0, 2.0)
                ax.set_xticks([-2, -1, 0, 1, 2])
                ax.set_xticklabels(["−2", "−1", "0", "1", "2"])
                ax.set_yticks(y_idx)
                ax.set_yticklabels(desc_pretty, fontsize = 7.0)
                ax.invert_yaxis()
                ax.set_ylim(len(desc_pretty) - 0.45, -0.55)
                ax.set_box_aspect(1.0)
                ax.set_xlabel(
                    "Standardized Negative Binomial Log-Rate Coefficient", fontsize = 7.0
                )
                ax.set_title("b", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

                ## add shared legend to panel a
                h_3b, l_3b = ax.get_legend_handles_labels()
                _style_utilization_legend(
                    axes_mech[0],
                    handles = [
                        h_3b[l_3b.index(LBL_CR_SHORT)],
                        h_3b[l_3b.index(LBL_NO_SHORT)],
                    ],
                    loc = "upper right",
                )
                plt.tight_layout()
                return _render_utilization_figure(figure = fig3, index = index, output_dir = output_dir)


## plot utilization calibration
def plot_utilization_calibration(
    tables: dict[str, pd.DataFrame],
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:
        predictions = tables["predictions"]
        predictions = predictions.loc[
            (predictions["Resampling Regime"] == RESAMPLING_REGIMES[0])
            & (predictions["Estimator"] == GLM_ESTIMATORS[-1])
        ]
        df_all = predictions.loc[predictions["Configuration"] == SCT_MODALITIES[0]].rename(
            columns = {"Observed_LogRate": "y_mean", "Observed_Rate": "r_mean"}
        )

        ## collect predictions for each configuration
        nb_no, nb_cr = [
            {
                "y_pred": predictions.loc[
                    predictions["Configuration"] == mode, "Predicted_LogRate"
                ].to_numpy(float),
                "r_pred": predictions.loc[
                    predictions["Configuration"] == mode, "Predicted_Rate"
                ].to_numpy(float),
            }
            for mode in SCT_MODALITIES
        ]
        preds_6_main = {
            (GLM_ESTIMATORS[-1], LBL_NO): nb_no,
            (GLM_ESTIMATORS[-1], LBL_CR): nb_cr,
        }

        ## draw the figure within the publication style
        with _utilization_style():
            fig4, axes4 = plt.subplots(1, 2, figsize = (DOUBLE_COL_IN, 3.25))
            fig4.patch.set_facecolor(BG_COLOR)

            ## compare observed and estimated log-rates
            ax = axes4[0]
            y_true_vec = df_all["y_mean"].to_numpy(float)
            y_no_vec = nb_no["y_pred"]
            y_cr_vec = nb_cr["y_pred"]
            grid_y = np.linspace(0.0, 15.0, 200)

            ## construct multiplicative bands in rate space before log transformation
            r_grid = np.maximum(np.expm1(grid_y), 0.0)
            y_band_2x_lo = np.log1p(r_grid / 2.0)
            y_band_2x_hi = np.log1p(r_grid * 2.0)
            y_band_3x_lo = np.log1p(r_grid / 3.0)
            y_band_3x_hi = np.log1p(r_grid * 3.0)

            ## shade error bands and draw the parity line
            ax.fill_between(
                grid_y,
                y_band_3x_lo,
                y_band_3x_hi,
                color = COL_BAND3_FILL,
                alpha = 0.55,
                lw = 0.0,
                zorder = 0,
            )
            ax.fill_between(
                grid_y,
                y_band_2x_lo,
                y_band_2x_hi,
                color = COL_CR_FILL,
                alpha = 0.75,
                lw = 0.0,
                zorder = 1,
            )
            ax.plot(
                grid_y,
                grid_y,
                color = DASHED_GUIDE_COLOR,
                linestyle = "--",
                lw = DASHED_GUIDE_LW,
                zorder = 3.5,
            )

            ## overlay excluded and anchored estimates
            ax.scatter(
                y_true_vec,
                y_no_vec,
                color = COL_NO,
                marker = "x",
                s = _utilization_marker_size("x", lw = 1.0),
                linewidths = 1.0,
                zorder = 4,
            )
            ax.scatter(
                y_true_vec,
                y_cr_vec,
                color = COL_CR,
                edgecolors = "none",
                marker = "o",
                s = _utilization_marker_size("o"),
                zorder = 3,
            )

            ## style axes and ticks
            _style_utilization_axis(ax)
            ax.set_xlim(0, 15)
            ax.set_ylim(0, 15)
            ax.set_xticks(np.arange(0, 16, 3))
            ax.set_yticks(np.arange(0, 16, 3))
            ax.set_box_aspect(1.0)
            ax.set_xlabel("Observed Log-Rate", fontsize = 7.0)
            ax.set_ylabel("Estimated Log-Rate", fontsize = 7.0)
            ax.set_title("a", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

            ## build legend handles
            legend_handles_4a_bands = [
                Line2D(
                    [0],
                    [0],
                    color = DASHED_GUIDE_COLOR,
                    linestyle = "--",
                    lw = DASHED_GUIDE_LW,
                    label = "1:1 Parity",
                ),
                Patch(
                    facecolor = COL_CR_FILL,
                    alpha = 0.75,
                    edgecolor = "none",
                    label = "2x Error Band",
                ),
                Patch(
                    facecolor = COL_BAND3_FILL,
                    alpha = 0.55,
                    edgecolor = "none",
                    label = "3x Error Band",
                ),
            ]
            legend_handles_4a_reg = [
                Line2D(
                    [0],
                    [0],
                    marker = "o",
                    color = "w",
                    markerfacecolor = COL_CR,
                    markeredgecolor = "none",
                    markersize = np.sqrt(_utilization_marker_size("o")),
                    label = LBL_CR_SHORT,
                ),
                Line2D(
                    [0],
                    [0],
                    marker = "x",
                    color = "w",
                    markeredgecolor = COL_NO,
                    markeredgewidth = 1.0,
                    markersize = np.sqrt(_utilization_marker_size("x", lw = 1.0)),
                    label = LBL_NO_SHORT,
                ),
            ]

            ## add legends
            ax.add_artist(
                _style_utilization_legend(
                    ax, handles = legend_handles_4a_bands, loc = "upper left"
                )
            )
            _style_utilization_legend(ax, handles = legend_handles_4a_reg, loc = "lower right")

            ## show cumulative fractions within each multiplicative error threshold
            ax = axes4[1]
            r_true_vec = df_all["r_mean"].to_numpy(float)

            ## shade error bands
            ax.axvspan(
                1.0, 2.0, facecolor = COL_CR_FILL, alpha = 0.75, edgecolor = "none", zorder = 0
            )
            ax.axvspan(
                2.0, 3.0, facecolor = COL_BAND3_FILL, alpha = 0.55, edgecolor = "none", zorder = 0
            )

            ## draw the empirical cdf of multiplicative errors
            ecdf_specs = [
                ("Negative Binomial", LBL_CR, COL_CR, "-", 1.0, LBL_CR_SHORT),
                ("Negative Binomial", LBL_NO, COL_NO, "--", 1.0, LBL_NO_SHORT),
            ]
            for fam_k, mode_k, col_k, ls_k, lw_k, lbl_k in ecdf_specs:
                r_p = preds_6_main[fam_k, mode_k]["r_pred"]
                mult_err = np.sort(
                    np.exp(
                        np.abs(
                            np.log(np.maximum(r_p, 1e-09) / np.maximum(r_true_vec, 1e-09))
                        )
                    )
                )
                ecdf_y = np.arange(1, len(mult_err) + 1) / len(mult_err)
                x_step = np.r_[1.0, mult_err, 128.0]
                y_step = np.r_[0.0, ecdf_y, 1.0]
                ax.step(
                    x_step,
                    y_step,
                    where = "post",
                    color = col_k,
                    linestyle = ls_k,
                    lw = lw_k,
                    label = lbl_k,
                    zorder = 3,
                )
                if mode_k == LBL_CR:
                    mult_err_cr = mult_err
                else:
                    mult_err_no = mult_err

            ## mark the 2x and 3x thresholds
            ax.axvline(
                2.0, color = DASHED_GUIDE_COLOR, linestyle = "--", lw = DASHED_GUIDE_LW, zorder = 2
            )
            ax.axvline(
                3.0, color = DASHED_GUIDE_COLOR, linestyle = "--", lw = DASHED_GUIDE_LW, zorder = 2
            )

            ## style axes and ticks
            geom_ticks = [1, 2, 3, 4, 8, 16, 32]
            _style_utilization_axis(ax)
            ax.set_xscale("log", base = 2)
            ax.set_xlim(1.0, 32.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.linspace(0.0, 1.0, 11))
            ax.yaxis.set_major_formatter("{x:.1f}")
            ax.set_box_aspect(1.0)
            ax.set_xticks(geom_ticks)
            ax.set_xticklabels([f"{t}x" for t in geom_ticks], fontsize = 7.0)
            ax.set_xlabel("Multiplicative Error", fontsize = 7.0)
            ax.set_ylabel("Cumulative Proportion of Systems", fontsize = 7.0)
            ax.set_title("b", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)
            _style_utilization_legend(ax, loc = "lower right")
            plt.tight_layout()

            ## set a fixed gap between panels at export width
            AB_GAP_MM = 18.5
            fig4_gap_frac = AB_GAP_MM / TARGET_WIDTH_MM
            fig_w4 = fig4.get_figwidth()

            ## adjust panel spacing using the rendered export width
            for _ in range(3):
                fig4.canvas.draw()
                pos_a4, pos_b4 = (axes4[0].get_position(), axes4[1].get_position())
                gap_in = (pos_b4.x0 - pos_a4.x1) * fig_w4
                export_w_in = fig4.get_tightbbox(fig4.canvas.get_renderer()).width + 0.2
                shift_in = (fig4_gap_frac * export_w_in - gap_in) / (1.0 - fig4_gap_frac)
                axes4[1].set_position(pos_b4.translated(shift_in / fig_w4, 0.0))

            ## annotate system counts within each threshold
            fig4.canvas.draw()
            fig4_count_gap_pt = 1.5
            fig4_count_fontsize = 6.0
            renderer_4 = fig4.canvas.get_renderer()
            px_per_pt = fig4.dpi / 72.0
            ax_h_px = ax.get_window_extent(renderer_4).height
            fig4_count_labels = [
                (mult_err_cr, COL_CR, {2.0: 4.0, 3.0: 0.0}),
                (mult_err_no, COL_NO, {2.0: 1.0, 3.0: 0.0}),
            ]
            for errs_k, col_k, drop_by_thr in fig4_count_labels:
                for thr, drop_pt in drop_by_thr.items():
                    n_thr = int(np.sum(errs_k <= thr))
                    ann = ax.annotate(
                        f"{n_thr} / {len(errs_k)} Systems",
                        (thr, 0.5),
                        xytext = (-fig4_count_gap_pt, 0.0),
                        textcoords = "offset points",
                        rotation = 90,
                        ha = "right",
                        va = "top",
                        fontsize = fig4_count_fontsize,
                        color = col_k,
                        fontweight = "bold",
                        zorder = 4,
                    )
                    ann._pub_fontsize = fig4_count_fontsize
                    ann._pub_bold = True

                    ## place each count just below the cdf curve
                    x_left_px = ann.get_window_extent(renderer_4).x0 - 0.5 * px_per_pt
                    x_left = ax.transData.inverted().transform((x_left_px, 0.0))[0]
                    y_curve = np.sum(errs_k <= x_left) / len(errs_k)
                    ann.xy = (
                        thr,
                        y_curve - (fig4_count_gap_pt + 0.5 + drop_pt) * px_per_pt / ax_h_px,
                    )
            return _render_utilization_figure(figure = fig4, index = index, output_dir = output_dir)


## plot utilization ablations and temporal controls
def plot_utilization_controls(
    tables: dict[str, pd.DataFrame],
    index: int | None = None,
    output_dir: Path | None = None,
    ) -> Image:
        df_fig5 = tables["ablation"]
        ABL_CONFIGS = UTIL_ABLATIONS

        ## select primary temporal window results
        temporal = tables["temporal_complete"]
        temporal = temporal.loc[
            (temporal["Resampling Regime"] == RESAMPLING_REGIMES[0])
            & (temporal["Estimator"] == GLM_ESTIMATORS[-1])
        ]
        TH_FRACS = sorted(
            (
                float(value.rstrip("%")) / 100.0
                for value in temporal["Prior-Window Share"].unique()
            )
        )

        ## index mae estimates and intervals by window and configuration
        s1_mae = {
            (float(row["Prior-Window Share"].rstrip("%")) / 100.0, row["Configuration"]): row[
                "MAE"
            ]
            for _, row in temporal.iterrows()
        }
        s1_int = {
            (float(row["Prior-Window Share"].rstrip("%")) / 100.0, row["Configuration"]): (
                row["MAE_lo"],
                row["MAE_hi"],
            )
            for _, row in temporal.iterrows()
        }

        ## draw the figure within the publication style
        with _utilization_style():
            import re

            ## create the figure
            fig5, axes5 = plt.subplots(1, 2, figsize = (DOUBLE_COL_IN, 3.25))
            fig5.patch.set_facecolor(BG_COLOR)

            ## show ablation estimates with paired system-bootstrap intervals
            ax = axes5[0]

            ## assign ablation colors
            ABL_COLORS = dict(
                zip(
                    ABL_CONFIGS,
                    [
                        COL_CR,
                        "#0072B2",
                        "#E69F00",
                        "#CC79A7",
                        COL_NO,
                        "#009E73",
                        "#56B4E9",
                        "#6A6A6A",
                    ],
                )
            )
            fig5_color = lambda cfg: ABL_COLORS[cfg]

            ## plot each configuration with its intervals
            for _, row in df_fig5.iterrows():
                col = fig5_color(row["Configuration"])
                ax.errorbar(
                    row["MAE"],
                    row["CI"],
                    xerr = [
                        [row["MAE"] - row["MAE_lo"]],
                        [row["MAE_hi"] - row["MAE"]],
                    ],
                    yerr = [[row["CI"] - row["CI_lo"]], [row["CI_hi"] - row["CI"]]],
                    fmt = "o",
                    color = col,
                    ecolor = mpl.colors.to_rgba(col, 0.45),
                    markersize = MARKER_D_PRIMARY,
                    markeredgewidth = 0,
                    elinewidth = 0.8,
                    capsize = 0,
                    zorder = 3,
                )

            ## style axes and ticks
            _style_utilization_axis(ax)
            ax.set_xlim(0.0, 4.1)
            ax.set_xticks(np.arange(0.0, 4.1, 1.0))
            ax.set_xticklabels([f"{t:.1f}" for t in np.arange(0.0, 4.1, 1.0)])
            ax.set_ylim(0.5, 1.04)
            ax.set_yticks(np.arange(0.5, 1.01, 0.1))
            ax.spines["left"].set_bounds(0.5, 1.0)
            ax.spines["bottom"].set_bounds(0.0, 4.0)
            ax.set_box_aspect(1.0)
            ax.set_xlabel("Mean Absolute Error (MAE)", fontsize = 7.0)
            ax.set_ylabel("Consensus Index (CI)", fontsize = 7.0)
            ax.set_title("a", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

            ## build legend handles ordered by consensus
            fig5_label = lambda cfg: re.sub("(\\d+)\\.0x", "\\1x", cfg)
            legend_handles_6_cfg = [
                Line2D(
                    [0],
                    [0],
                    color = fig5_color(cfg),
                    marker = "o",
                    markersize = np.sqrt(_utilization_marker_size("o")),
                    markeredgewidth = 0.0,
                    lw = 0.0,
                    label = fig5_label(cfg),
                )
                for cfg in df_fig5.sort_values("CI", ascending = False)["Configuration"]
            ]
            _style_utilization_legend(ax, handles = legend_handles_6_cfg, loc = "lower left")

            ## compare anchored and unanchored errors across temporal splits
            ax = axes5[1]
            s1_row = {f: i for i, f in enumerate(sorted(TH_FRACS))}

            ## plot intervals and estimates for each window
            for f in TH_FRACS:
                x_cr, x_no = (
                    s1_mae[f, "SCT Frontier Anchor"],
                    s1_mae[f, "SCT Frontier Exclusion"],
                )
                for cfg, col in (
                    ("SCT Frontier Anchor", COL_CR),
                    ("SCT Frontier Exclusion", COL_NO),
                ):
                    tint = 0.45 * np.array(mpl.colors.to_rgb(col)) + 0.55
                    ax.plot(
                        s1_int[f, cfg],
                        [s1_row[f]] * 2,
                        color = tint,
                        lw = 0.8,
                        solid_capstyle = "butt",
                        zorder = 2.5,
                    )
                ax.scatter(
                    [x_no],
                    [s1_row[f]],
                    color = COL_NO,
                    marker = "x",
                    s = _utilization_marker_size("x", lw = 1.0),
                    linewidths = 1.0,
                    zorder = 3,
                )
                ax.scatter(
                    [x_cr],
                    [s1_row[f]],
                    color = COL_CR,
                    edgecolors = "none",
                    marker = "o",
                    s = _utilization_marker_size("o"),
                    zorder = 4,
                )

            ## style axes and ticks
            _style_utilization_axis(ax)
            s1_xmax = float(
                np.ceil((max((hi for _, hi in s1_int.values())) + 0.1) * 2.0) / 2.0
            )
            ax.set_xlim(0.0, s1_xmax)
            ax.set_xticks(np.arange(0.0, s1_xmax + 0.01, 0.5))
            ax.xaxis.set_major_formatter("{x:.1f}")
            ax.set_ylim(-1.0, len(TH_FRACS) - 1 + 0.6)
            ax.set_yticks([s1_row[f] for f in TH_FRACS])
            ax.set_yticklabels([f"{f:.0%}" for f in TH_FRACS], fontsize = 7.0)
            ax.set_box_aspect(1.0)
            ax.set_xlabel("Mean Absolute Error (MAE)", fontsize = 7.0)
            ax.set_ylabel("Prior-Window Proportion of Each Series", fontsize = 7.0)
            ax.set_title("b", loc = "left", fontsize = 7.0, fontweight = "bold", pad = 4)

            ## build legend handles
            legend_handles_s1_est = [
                Line2D(
                    [0],
                    [0],
                    marker = "o",
                    color = "w",
                    markerfacecolor = COL_CR,
                    markeredgecolor = "none",
                    markersize = np.sqrt(_utilization_marker_size("o")),
                    label = LBL_CR_SHORT,
                ),
                Line2D(
                    [0],
                    [0],
                    marker = "x",
                    color = "w",
                    markeredgecolor = COL_NO,
                    markeredgewidth = 1.0,
                    markersize = np.sqrt(_utilization_marker_size("x", lw = 1.0)),
                    label = LBL_NO_SHORT,
                ),
            ]
            _style_utilization_legend(ax, handles = legend_handles_s1_est, loc = "lower right")
            plt.tight_layout()

            ## set a fixed gap between panels at export width
            AB_GAP_MM = 18.5
            fig5_gap_frac = AB_GAP_MM / TARGET_WIDTH_MM
            fig_w4 = fig5.get_figwidth()

            ## adjust panel spacing using the rendered export width
            for _ in range(3):
                fig5.canvas.draw()
                pos_a4, pos_b4 = (axes5[0].get_position(), axes5[1].get_position())
                gap_in = (pos_b4.x0 - pos_a4.x1) * fig_w4
                export_w_in = fig5.get_tightbbox(fig5.canvas.get_renderer()).width + 0.2
                shift_in = (fig5_gap_frac * export_w_in - gap_in) / (1.0 - fig5_gap_frac)
                axes5[1].set_position(pos_b4.translated(shift_in / fig_w4, 0.0))
            fig5.canvas.draw()
            return _render_utilization_figure(figure = fig5, index = index, output_dir = output_dir)
