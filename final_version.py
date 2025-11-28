#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quant-ready SDE framework for short-rate models with Monte Carlo simulation and calibration.

Implements:
- Base SDEModel with vectorized Euler–Maruyama and Milstein schemes
- Vasicek, CIR, and Ho–Lee models
- Practical, robust calibration (mean + terminal vol fit)
- Safe CIR square-root handling and Ho–Lee indexing clamp
- Clean __main__ demo (lightweight)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Iterable


# -----------------------------
# Base class
# -----------------------------
class SDEModel:
    """
    Base class for Stochastic Differential Equation (SDE) models.

    Subclasses must implement:
      - drift(r, t) -> np.ndarray
      - diffusion(r, t) -> np.ndarray
      - diffusion_derivative(r, t) -> np.ndarray
      - update_params(params: np.ndarray) -> None
      - get_bounds() -> Tuple[Tuple[float, float], ...]

    Key methods:
      - simulate_path: vectorized MC with Euler or Milstein
      - calibrate: practical calibration to time series (mean path + terminal vol)
      - plot_paths: quick viewer
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    # --- Model pieces to override ---
    def drift(self, r: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def diffusion_derivative(self, r: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def update_params(self, params: np.ndarray) -> None:
        raise NotImplementedError

    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        raise NotImplementedError

    # --- Core engine ---
    def simulate_path(
        self,
        r0: float,
        total_time: float,
        total_steps: int,
        number_of_paths: int,
        scheme: str = "euler",
        project_nonnegative: bool = False,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Vectorized Monte Carlo simulation.

        Args:
            r0: initial short rate
            total_time: horizon (in "time units")
            total_steps: number of steps
            number_of_paths: MC paths
            scheme: 'euler' or 'milstein'
            project_nonnegative: if True, clamp paths >= 0 after each step
            random_state: optional seed

        Returns:
            np.ndarray of shape (number_of_paths, total_steps + 1)
        """
        if total_steps <= 0:
            raise ValueError("total_steps must be >= 1")
        if number_of_paths <= 0:
            raise ValueError("number_of_paths must be >= 1")

        if random_state is not None:
            rng = np.random.default_rng(random_state)
            normal = rng.normal
        else:
            normal = np.random.normal

        dt = total_time / total_steps
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((number_of_paths, total_steps + 1), dtype=float)
        paths[:, 0] = r0

        for i in range(1, total_steps + 1):
            t = (i - 1) * dt
            r_prev = paths[:, i - 1]
            dW = sqrt_dt * normal(size=number_of_paths)

            mu = self.drift(r_prev, t)
            sig = self.diffusion(r_prev, t)

            if scheme == "euler":
                r_new = r_prev + mu * dt + sig * dW
            elif scheme == "milstein":
                sig_r = self.diffusion_derivative(r_prev, t)
                r_new = r_prev + mu * dt + sig * dW + 0.5 * sig * sig_r * (dW**2 - dt)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            if project_nonnegative:
                r_new = np.maximum(r_new, 0.0)

            paths[:, i] = r_new

        return paths

    # --- Practical calibration ---
    def calibrate(
        self,
        data: np.ndarray,
        initial_guess: Iterable[float],
        number_of_paths: int = 200,
        scheme: str = "euler",
        random_state: int | None = 123,
        weight_mean: float = 1.0,
        weight_vol: float = 1.0,
    ):
        """
        Practical, robust calibration to a short-rate time series.

        Objective:
            MSE between model mean path and data + penalty on terminal volatility difference.

        Args:
            data: array-like, observed short-rate path (length T+1)
            initial_guess: starting parameters for optimizer
            number_of_paths: MC paths inside objective
            scheme: 'euler' or 'milstein'
            random_state: RNG seed
            weight_mean: weight for mean-path MSE
            weight_vol:  weight for terminal vol penalty

        Returns:
            scipy OptimizeResult
        """
        data = np.asarray(data, dtype=float).ravel()
        T = len(data) - 1
        if T <= 0:
            raise ValueError("Need at least 2 observations for calibration (T >= 1).")

        bounds = self.get_bounds()
        init = np.asarray(list(initial_guess), dtype=float)
        if len(init) != len(bounds):
            raise ValueError("initial_guess length must match number of parameters.")

        def objective(params: np.ndarray) -> float:
            # Update model parameters
            self.update_params(params)

            # Simulate model
            paths = self.simulate_path(
                r0=data[0],
                total_time=T,
                total_steps=T,
                number_of_paths=number_of_paths,
                scheme=scheme,
                # For CIR-like models, it's helpful to keep non-negativity during calibration
                project_nonnegative=isinstance(self, CIRModel),
                random_state=random_state,
            )
            model_mean = paths.mean(axis=0)
            model_std = paths.std(axis=0)

            # Mean path fit across the whole horizon
            mse_mean = np.mean((model_mean - data) ** 2)

            # Terminal volatility fit
            terminal_vol_penalty = (model_std[-1] - data.std()) ** 2

            return weight_mean * mse_mean + weight_vol * terminal_vol_penalty

        res = minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        if not res.success:
            # Keep parameters at the best point found anyway:
            self.update_params(res.x)
            raise RuntimeError(f"Calibration failed: {res.message}")

        self.update_params(res.x)
        return res

    # --- Plot helper ---
    @staticmethod
    def plot_paths(paths: np.ndarray, title: str = "Simulated Paths", alpha: float = 0.35):
        plt.figure(figsize=(10, 5))
        plt.plot(paths.T, alpha=alpha)
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Short rate")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Vasicek model
# -----------------------------
class VasicekModel(SDEModel):
    """
    dr_t = alpha*(theta - r_t) dt + sigma dW_t
    """

    def drift(self, r: np.ndarray, t: float) -> np.ndarray:
        theta, alpha, sigma = self.params["theta"], self.params["alpha"], self.params["sigma"]
        return alpha * (theta - r)

    def diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(r, self.params["sigma"], dtype=float)

    def diffusion_derivative(self, r: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(r, dtype=float)

    def update_params(self, params: np.ndarray) -> None:
        self.params["theta"], self.params["alpha"], self.params["sigma"] = map(float, params)

    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Loose bounds: theta in [-1,1], alpha>0, sigma>0
        return ((-1.0, 1.0), (1e-6, None), (1e-8, None))


# -----------------------------
# CIR model
# -----------------------------
class CIRModel(SDEModel):
    """
    dr_t = alpha*(theta - r_t) dt + sigma * sqrt(max(r_t, 0)) dW_t

    Notes:
    - Uses full-truncation-style sqrt handling to avoid invalid values.
    - Milstein derivative is computed safely with a small floor.
    """

    def drift(self, r: np.ndarray, t: float) -> np.ndarray:
        theta, alpha, sigma = self.params["theta"], self.params["alpha"], self.params["sigma"]
        return alpha * (theta - r)

    def diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        return self.params["sigma"] * np.sqrt(np.maximum(r, 0.0))

    def diffusion_derivative(self, r: np.ndarray, t: float) -> np.ndarray:
        safe_r = np.maximum(r, 1e-8)
        return 0.5 * self.params["sigma"] / np.sqrt(safe_r)

    def update_params(self, params: np.ndarray) -> None:
        self.params["theta"], self.params["alpha"], self.params["sigma"] = map(float, params)

    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Feller condition is not enforced here (for simplicity), but alpha,sigma>0
        return ((-1.0, 1.0), (1e-6, None), (1e-8, None))


# -----------------------------
# Ho–Lee model
# -----------------------------
class HoLeeModel(SDEModel):
    """
    dr_t = theta(t) dt + sigma(t) dW_t

    This implementation supports a piecewise-constant parameterization with
    clamped indices to avoid out-of-range errors.
    """

    def drift(self, r: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(r, self.params["theta_t"](t), dtype=float)

    def diffusion(self, r: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(r, self.params["sigma_t"](t), dtype=float)

    def diffusion_derivative(self, r: np.ndarray, t: float) -> np.ndarray:
        return np.zeros_like(r, dtype=float)

    def update_params(self, params: np.ndarray) -> None:
        n = int(len(params) // 2)
        theta_arr = np.asarray(params[:n], dtype=float)
        sigma_arr = np.asarray(params[n:], dtype=float)
        # Clamp indexing
        def theta_t(tt: float) -> float:
            idx = min(max(int(tt), 0), n - 1)
            return float(theta_arr[idx])

        def sigma_t(tt: float) -> float:
            idx = min(max(int(tt), 0), n - 1)
            return float(sigma_arr[idx])

        self.params["theta_t"] = theta_t
        self.params["sigma_t"] = sigma_t
        self.params["n_params"] = n

    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        # Expect 'n_params' in params to size the piecewise-constant vectors
        n = int(self.params.get("n_params", 8))
        # theta free-ish, sigma > 0
        return tuple(((-1.0, 1.0) for _ in range(n))) + tuple(((1e-8, None) for _ in range(n)))


# -----------------------------
# Demo / Quick test
# -----------------------------
def _synthetic_short_rate(T: int, r0: float = 0.02, drift: float = 0.0, vol: float = 0.01, seed: int = 7) -> np.ndarray:
    """
    Generate a simple synthetic short-rate series (discretized OU-like), length T+1.
    """
    rng = np.random.default_rng(seed)
    r = np.empty(T + 1, dtype=float)
    r[0] = r0
    dt = 1.0
    for t in range(1, T + 1):
        r[t] = r[t - 1] + drift * dt + vol * np.sqrt(dt) * rng.standard_normal()
    return r


if __name__ == "__main__":
    # Lightweight demo so the file can be run safely.
    np.random.seed(39)

    # Synthetic "observed" short-rate path
    T = 120
    data = _synthetic_short_rate(T=T, r0=0.025, drift=0.0001, vol=0.0075, seed=11)

    # ---------------- Vasicek ----------------
    vas = VasicekModel(params={"theta": 0.03, "alpha": 0.1, "sigma": 0.02})
    init_vas = np.array([0.02, 0.2, 0.01], dtype=float)
    print("Calibrating Vasicek...")
    res_vas = vas.calibrate(data, init_vas, number_of_paths=300, scheme="euler", random_state=123)
    print("  Vasicek params:", vas.params, "\n")

    vas_paths = vas.simulate_path(
        r0=data[0],
        total_time=T,
        total_steps=T,
        number_of_paths=200,
        scheme="euler",
        random_state=321,
    )
    vas.plot_paths(vas_paths, title="Vasicek – Simulated Paths")

    # ---------------- CIR ----------------
    cir = CIRModel(params={"theta": 0.03, "alpha": 0.2, "sigma": 0.02})
    init_cir = np.array([0.02, 0.2, 0.01], dtype=float)
    print("Calibrating CIR...")
    try:
        res_cir = cir.calibrate(data, init_cir, number_of_paths=300, scheme="euler", random_state=123)
        print("  CIR params:", cir.params, "\n")
    except RuntimeError as e:
        print("  CIR calibration failed (keeping best found):", e)
        print("  CIR params (best so far):", cir.params, "\n")

    cir_paths = cir.simulate_path(
        r0=data[0],
        total_time=T,
        total_steps=T,
        number_of_paths=200,
        scheme="milstein",
        project_nonnegative=True,
        random_state=222,
    )
    cir.plot_paths(cir_paths, title="CIR – Simulated Paths (Milstein, projected ≥0)")

    # ---------------- Ho–Lee (piecewise constant) ----------------

    n_segments = 6  # lower dimensional → more stable calibration
    ho = HoLeeModel(
        params={
            "theta_t": (lambda t: 0.0),
            "sigma_t": (lambda t: 0.01),
            "n_params": n_segments,
        }
    )

    # Initial guess: small drift, 1% sigma across segments
    init_ho = np.concatenate([
        np.full(n_segments, 0.0),  # theta guess per segment
        np.full(n_segments, 0.01)  # sigma guess per segment
    ])

    # Tighter, practical bounds (per segment):
    # θ(t) kept small; σ(t) bounded away from 0 to avoid degenerate diffusion
    ho.get_bounds = lambda: tuple(((-0.005, 0.005) for _ in range(n_segments))) + \
                            tuple(((0.001, 0.05) for _ in range(n_segments)))

    print(f"Calibrating Ho–Lee with {n_segments} segments (stronger vol weight)...")
    try:
        res_ho = ho.calibrate(
            data,
            init_ho,
            number_of_paths=300,
            scheme="euler",
            random_state=456,
            weight_mean=1.0,
            weight_vol=5.0,  # ↑ emphasize volatility so σ̂(t) won’t collapse
        )
        print("  Ho–Lee calibrated (first 3 θ, σ):",
              [ho.params["theta_t"](i) for i in range(min(3, n_segments))],
              [ho.params["sigma_t"](i) for i in range(min(3, n_segments))], "\n")
    except RuntimeError as e:
        print("  Ho–Lee calibration failed (keeping best found):", e, "\n")

    # Simulate and plot with calibrated params
    ho_paths = ho.simulate_path(
        r0=data[0],
        total_time=T,
        total_steps=T,
        number_of_paths=200,
        scheme="euler",
        random_state=999,
    )
    ho.plot_paths(ho_paths, title="Ho–Lee – Simulated Paths (regularized)")
