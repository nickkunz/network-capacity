## libraries
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin

## constants
from src.estimators.config import (
    ASYMMETRY_C,
    ASYMMETRY_R,
    ALPHA
)

## laws regression sklearn regressors
class LinearLAWS(BaseEstimator):
    def __init__(self, 
        tau_c: float = ASYMMETRY_C,
        tau_r: float = ASYMMETRY_R, 
        alpha: float = ALPHA, 
        max_iter: int = 1000,
        tol: float = 0.0001
        ) -> None:

        self.tau_c = tau_c
        self.tau_r = tau_r
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.estimator_c = BaseLAWS(
            tau = tau_c, 
            alpha = alpha,
            fit_intercept = True,
            max_iter = max_iter,
            tol = tol
        )
        self.estimator_r = BaseLAWS(
            tau = tau_r, 
            alpha = alpha, 
            fit_intercept = False, 
            max_iter = max_iter, 
            tol = tol
        )

## laws regression sklearn framework
class BaseLAWS(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        tau: float,
        alpha: float,
        fit_intercept: bool,
        max_iter: int,
        tol: float
        ) -> None:
        self.tau = tau
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    ## sklearn fit interface
    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaseLAWS":

        X = np.asarray(X, dtype = np.float64)
        y = np.asarray(y, dtype = np.float64).reshape(-1)

        n, p = X.shape

        ## add intercept and initialize parameters
        if self.fit_intercept:
            X_ = np.c_[np.ones(n), X]
            p_ = p + 1
            beta = np.zeros(p_, dtype = np.float64)
            beta[0] = np.mean(y)
        else:
            X_ = X
            p_ = p
            beta = np.zeros(p_, dtype = np.float64)

        for outer in range(self.max_iter):

            beta_old = beta.copy()

            ## compute residuals
            r = y - X_ @ beta

            ## expectile weights
            w = np.where(r >= 0, self.tau, 1.0 - self.tau)

            ## coordinate descent
            for j in range(p_):

                ## partial residual
                r_j = y - X_ @ beta + X_[:, j] * beta[j]

                ## normalized weighted statistics (1/n scaling)
                num = np.mean(w * X_[:, j] * r_j)
                den = np.mean(w * X_[:, j] ** 2)

                if den < 1e-12:
                    beta[j] = 0.0
                    continue

                ## do not penalize intercept
                skip_penalty = self.fit_intercept and j == 0

                if self.alpha > 0 and not skip_penalty:
                    beta[j] = np.sign(num) * max(abs(num) - self.alpha, 0.0) / den
                else:
                    beta[j] = num / den

            ## convergence check
            delta = np.max(np.abs(beta - beta_old))
            if delta < self.tol:
                break

        ## store parameters
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.n_iter_ = outer + 1
        return self

    ## sklearn predict interface
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.asarray(X, dtype = np.float64)
        return X @ self.coef_ + self.intercept_