"""Scaling law fitting utilities."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit


def ratio_model(PT, alpha, beta):
    """rho = 1 - exp(-alpha * (P/T)^beta)"""
    P, T = PT
    return 1 - np.exp(-alpha * (P / T) ** beta)


def constant_model(PT, alpha, beta):
    """rho = 1 - exp(-alpha * P^beta) -- no T dependence"""
    P, _ = PT
    return 1 - np.exp(-alpha * (P ** beta))


def logratio_model(PT, alpha, beta):
    """rho = sigmoid(alpha * log(P/T) + beta)"""
    P, T = PT
    return expit(alpha * np.log(P / T) + beta)


def fit_and_evaluate(model_fn, P, T, rho, p0, bounds):
    """Fit a model and return params, R^2, RMSE."""
    try:
        popt, pcov = curve_fit(model_fn, (P, T), rho, p0=p0,
                               bounds=bounds, maxfev=10000)
        pred = model_fn((P, T), *popt)
        ss_res = np.sum((rho - pred) ** 2)
        ss_tot = np.sum((rho - rho.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / len(rho)))
        return {
            "params": {f"p{i}": float(v) for i, v in enumerate(popt)},
            "r2": float(r2),
            "rmse": rmse,
        }
    except Exception as e:
        return {"error": str(e)}


def minimum_proxy_size(alpha, beta, target_m, threshold=0.85):
    """Compute minimum proxy size P* for a given target size and threshold.

    Uses the ratio model: rho = 1 - exp(-alpha * (P/T)^beta) >= threshold
    => P >= T * (-log(1-threshold) / alpha)^(1/beta)
    """
    ratio = (-np.log(1 - threshold) / alpha) ** (1 / beta)
    return target_m * ratio
