# PFFP2024

A small, quant-ready Python framework for simulating and calibrating short-rate models. It runs fast, makes plots, and keeps the code clean and modular. Project for "Python in finance, finance in Python" course at Jagiellonian University

Models

Vasicek (OU) – mean-reverting, rates can go negative

CIR – mean-reverting with √r volatility, non-negative rates

Ho–Lee – no mean reversion; piecewise-constant θ(t), σ(t) for flexibility

Key features

Vectorized Monte Carlo simulation (Euler–Maruyama + Milstein)

Calibration to a given rate series: fits the mean path and terminal volatility

Safe CIR handling (full-truncation √r and optional projection to r≥0)

Regularized Ho–Lee (bounded drift and minimum volatility to avoid “straight lines”)

Clean API: base SDEModel + subclasses (VasicekModel, CIRModel, HoLeeModel)

Reproducible runs: seed control, tidy __main__ demo, quick plots
