#!/usr/bin/env python3
"""
Fit scaling laws to the correlation matrix.

Candidate forms:
  1. Constant:    rho = 1 - exp(-alpha * P^beta)  [threshold is absolute]
  2. Ratio:       rho = 1 - exp(-alpha * (P/T)^beta)  [threshold scales with T]
  3. Log-ratio:   rho = sigmoid(alpha * log(P/T) + beta)

With the paper's 1B column included as a free data point, we have a 3x3 grid
(3 proxy sizes x 3 target sizes) = 9 observations for fitting.

Output: results/scaling_law_fits.json
"""
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # sigmoid
import argparse
from pathlib import Path


def load_data(matrix: dict) -> tuple:
    """Extract P, T, rho arrays from the correlation matrix."""
    P_vals, T_vals, rho_vals = [], [], []
    for key, val in matrix.items():
        if "spearman_r" in val and "proxy_size_m" in val and "target_size_m" in val:
            P_vals.append(val["proxy_size_m"])
            T_vals.append(val["target_size_m"])
            rho_vals.append(val["spearman_r"])
    return np.array(P_vals, dtype=float), np.array(T_vals, dtype=float), np.array(rho_vals)


def fit_model(model_fn, P, T, rho, p0, bounds):
    """Fit a model and return parameters, errors, R^2, and RMSE."""
    try:
        popt, pcov = curve_fit(
            model_fn, (P, T), rho, p0=p0, bounds=bounds, maxfev=10000
        )
        pred = model_fn((P, T), *popt)
        ss_res = np.sum((rho - pred) ** 2)
        ss_tot = np.sum((rho - rho.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, float(r2), float(np.sqrt(ss_res / len(rho)))
    except Exception as e:
        return None, None, None, str(e)


def loo_rmse(model_fn, P, T, rho, p0, bounds):
    """Leave-one-out RMSE."""
    errors = []
    for i in range(len(rho)):
        mask = np.arange(len(rho)) != i
        try:
            popt, _, _, rmse_val = fit_model(
                model_fn, P[mask], T[mask], rho[mask], p0, bounds
            )
            if popt is not None:
                pred_i = model_fn((P[i : i + 1], T[i : i + 1]), *popt)[0]
                errors.append((rho[i] - pred_i) ** 2)
        except Exception:
            pass
    return float(np.sqrt(np.mean(errors))) if errors else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    if args.synthetic:
        # Fake correlation matrix for pipeline testing
        matrix = {
            f"P{p}M_T{t}M": {
                "proxy_size_m": p,
                "target_size_m": t,
                "spearman_r": min(0.95, 0.5 + 0.3 * (p / t) ** 0.5),
            }
            for p in [1, 15, 30]
            for t in [60, 150, 1000]
        }
    else:
        matrix = json.load(open("results/correlation_matrix.json"))

    P, T, rho = load_data(matrix)
    print(f"Fitting scaling laws to {len(P)} data points")
    print(f"P values: {sorted(set(P))} M params")
    print(f"T values: {sorted(set(T))} M params")
    print(f"rho range: [{rho.min():.3f}, {rho.max():.3f}]")

    results = {}

    # Model 1: Constant threshold (P only, no T dependence)
    def model_constant(PT, alpha, beta):
        P, _ = PT
        return 1 - np.exp(-alpha * (P**beta))

    popt, perr, r2, rmse = fit_model(
        model_constant, P, T, rho, [0.1, 1.0], ([0, 0], [np.inf, np.inf])
    )
    loo = loo_rmse(
        model_constant, P, T, rho, [0.1, 1.0], ([0, 0], [np.inf, np.inf])
    )
    results["constant"] = {
        "formula": "rho = 1 - exp(-alpha * P^beta)",
        "interpretation": "threshold is absolute (independent of T)",
        "params": (
            {"alpha": float(popt[0]), "beta": float(popt[1])}
            if popt is not None
            else None
        ),
        "r2": r2,
        "rmse": rmse,
        "loo_rmse": loo,
    }

    # Model 2: Ratio model (P/T dependence)
    def model_ratio(PT, alpha, beta):
        P, T = PT
        return 1 - np.exp(-alpha * (P / T) ** beta)

    popt, perr, r2, rmse = fit_model(
        model_ratio, P, T, rho, [1.0, 0.5], ([0, 0], [np.inf, np.inf])
    )
    loo = loo_rmse(
        model_ratio, P, T, rho, [1.0, 0.5], ([0, 0], [np.inf, np.inf])
    )
    implication = "fit failed"
    if popt is not None:
        # Estimate: for T=500M, what's the min proxy?
        # rho = 1 - exp(-alpha * (P/T)^beta) >= 0.85
        # => P/T >= (-log(0.15)/alpha)^(1/beta)
        ratio_threshold = (-np.log(0.15) / popt[0]) ** (1 / popt[1])
        implication = f"For T=500M, min proxy ~ {ratio_threshold * 500:.0f}M"

    results["ratio"] = {
        "formula": "rho = 1 - exp(-alpha * (P/T)^beta)",
        "interpretation": "threshold scales linearly with T when beta=1",
        "params": (
            {"alpha": float(popt[0]), "beta": float(popt[1])}
            if popt is not None
            else None
        ),
        "r2": r2,
        "rmse": rmse,
        "loo_rmse": loo,
        "implication": implication,
    }

    # Model 3: Log-ratio (sigmoid)
    def model_logratio(PT, alpha, beta):
        P, T = PT
        return expit(alpha * np.log(P / T) + beta)

    popt, perr, r2, rmse = fit_model(
        model_logratio,
        P,
        T,
        rho,
        [2.0, 2.0],
        ([-np.inf, -np.inf], [np.inf, np.inf]),
    )
    loo = loo_rmse(
        model_logratio,
        P,
        T,
        rho,
        [2.0, 2.0],
        ([-np.inf, -np.inf], [np.inf, np.inf]),
    )
    results["logratio"] = {
        "formula": "rho = sigmoid(alpha * log(P/T) + beta)",
        "interpretation": "S-curve in log(P/T) space",
        "params": (
            {"alpha": float(popt[0]), "beta": float(popt[1])}
            if popt is not None
            else None
        ),
        "r2": r2,
        "rmse": rmse,
        "loo_rmse": loo,
    }

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':15} {'R^2':>8} {'RMSE':>8} {'LOO-RMSE':>10}")
    print("-" * 60)
    for name, res in results.items():
        r2_str = f"{res['r2']:.4f}" if isinstance(res.get("r2"), float) else "N/A"
        rmse_str = (
            f"{res['rmse']:.4f}" if isinstance(res.get("rmse"), float) else "N/A"
        )
        loo_str = (
            f"{res['loo_rmse']:.4f}"
            if isinstance(res.get("loo_rmse"), float)
            else "N/A"
        )
        print(f"{name:15} {r2_str:>8} {rmse_str:>8} {loo_str:>10}")
    print("=" * 60)

    # Best model by LOO RMSE
    valid = {
        k: v for k, v in results.items() if isinstance(v.get("loo_rmse"), float)
    }
    if valid:
        best = min(valid, key=lambda k: valid[k]["loo_rmse"])
        print(f"\nBest fit by LOO RMSE: {best}")
        print(f"  {results[best]['formula']}")
        print(f"  {results[best]['interpretation']}")

    json.dump(results, open("results/scaling_law_fits.json", "w"), indent=2)
    print("\nSaved results/scaling_law_fits.json")


if __name__ == "__main__":
    main()
