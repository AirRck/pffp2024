import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple
import pandas as pd
# Set random seed for reproducibility
#np.random.seed(84)

class SDEModel:
    def __init__(self, params: dict):
        self.params = params

    def drift(self, r: float, t: float) -> float:
        raise NotImplementedError("Drift function is not implemented")

    def diffusion(self, r: float, t: float) -> float:
        raise NotImplementedError("Diffusion function is not implemented")

    def simulate_path(self, r0: float, total_time: float, total_steps: int, number_of_paths: int, scheme: str = 'euler') -> np.ndarray:
        dt = total_time / total_steps
        paths = np.zeros((number_of_paths, total_steps + 1))
        paths[:, 0] = r0

        for i in range(1, total_steps + 1):
            t = (i - 1) * dt
            W = np.random.normal(0, 1, number_of_paths)

            if scheme == 'euler':
                for j in range(number_of_paths):
                    dW = np.sqrt(dt) * W[j]
                    paths[j, i] = (paths[j, i-1] +
                                   self.drift(paths[j, i-1], t) * dt +
                                   self.diffusion(paths[j, i-1], t) * dW)
            elif scheme == 'milstein':
                for j in range(number_of_paths):
                    dW = np.sqrt(dt) * W[j]
                    paths[j, i] = (paths[j, i-1] +
                                   self.drift(paths[j, i-1], t) * dt +
                                   self.diffusion(paths[j, i-1], t) * dW +
                                   0.5 * self.diffusion(paths[j, i-1], t) *
                                   self.diffusion_derivative(paths[j, i-1], t) * (dW**2 - dt))
        return paths

    def diffusion_derivative(self, r: float, t: float) -> float:
        raise NotImplementedError("Diffusion derivative is not implemented")

    def calibrate(self, data: np.ndarray):
        raise NotImplementedError("Calibration method is not implemented")

    def plot_paths(self, paths: np.ndarray, title: str = "Simulated Paths"):
        plt.figure(figsize=(12, 6))
        for i in range(paths.shape[0]):
            plt.plot(paths[i])
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Interest Rate')
        plt.show()


class VasicekModel(SDEModel):
    def drift(self, r: float, t: float) -> float:
        theta, alpha, sigma = self.params['theta'], self.params['alpha'], self.params['sigma']
        return theta - alpha * r

    def diffusion(self, r: float, t: float) -> float:
        sigma = self.params['sigma']
        return sigma

    def diffusion_derivative(self, r: float, t: float) -> float:
        return 0

    def calibrate(self, data: np.ndarray):
        def objective(params):
            theta, alpha, sigma = params
            self.params['theta'], self.params['alpha'], self.params['sigma'] = theta, alpha, sigma
            simulated_paths = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1, number_of_paths=1)
            simulated_data = simulated_paths[0]
            return np.sum((simulated_data - data)**2)

        initial_guess = [np.mean(data), 0, np.std(data)]
        result = minimize(objective, initial_guess, bounds=[(None, None), (0, None), (0, None)])
        self.params['theta'], self.params['alpha'], self.params['sigma'] = result.x


class CIRModel(SDEModel):
    def drift(self, r: float, t: float) -> float:
        theta, alpha, sigma = self.params['theta'], self.params['alpha'], self.params['sigma']
        return theta - alpha * r

    def diffusion(self, r: float, t: float) -> float:
        sigma = self.params['sigma']
        return sigma * np.sqrt(r)

    def diffusion_derivative(self, r: float, t: float) -> float:
        sigma = self.params['sigma']
        return 0.5 * sigma / np.sqrt(r)

    def calibrate(self, data: np.ndarray):
        def objective(params):
            theta, alpha, sigma = params
            self.params['theta'], self.params['alpha'], self.params['sigma'] = theta, alpha, sigma
            simulated_paths = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1, number_of_paths=1)
            simulated_data = simulated_paths[0]
            return np.sum((simulated_data - data)**2)

        initial_guess = [np.mean(data), 0, np.std(data)]
        result = minimize(objective, initial_guess, bounds=[(None, None), (0, None), (0, None)])
        self.params['theta'], self.params['alpha'], self.params['sigma'] = result.x


class HoLeeModel(SDEModel):
    def drift(self, r: float, t: float) -> float:
        theta_t = self.params['theta_t'](t)
        return theta_t

    def diffusion(self, r: float, t: float) -> float:
        sigma_t = self.params['sigma_t'](t)
        return sigma_t

    def diffusion_derivative(self, r: float, t: float) -> float:
        return 0

    def calibrate(self, data: np.ndarray):
        def objective(params):
            # Here we assume theta_t and sigma_t are piecewise constant functions of time
            theta_t = params[:len(data)]
            sigma_t = params[len(data):]
            self.params['theta_t'] = lambda t: theta_t[int(t)]
            self.params['sigma_t'] = lambda t: sigma_t[int(t)]
            simulated_paths = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1, number_of_paths=1)
            simulated_data = simulated_paths[0]
            return np.sum((simulated_data - data)**2)

        initial_guess = np.concatenate([np.mean(data) * np.ones(len(data)), np.std(data) * np.ones(len(data))])
        result = minimize(objective, initial_guess, method='Nelder-Mead')
        self.params['theta_t'] = lambda t: result.x[:len(data)][int(t)]
        self.params['sigma_t'] = lambda t: result.x[len(data):][int(t)]


def rates_generator(T=100, mean_rate=0.02, volatility=0.005):
    """
    Generate synthetic time series of interest rates

    Parameters:
    - T (int): Number of time periods.
    - mean_rate (float): Mean interest rate.
    - volatility (float): Volatility of interest rates.

    Returns:
    - None
    """


    # Generate synthetic interest rates
    time = np.arange(T)
    rates = np.zeros(T)
    rates[0] = mean_rate

    for t in range(1, T):
        rates[t] = rates[t-1] + volatility * np.random.normal()

    # Plot the synthetic interest rates
    plt.figure(figsize=(12, 6))
    plt.plot(time, rates, label='Interest Rates')
    plt.title('Synthetic Time Series of Interest Rates')
    plt.xlabel('Time')
    plt.ylabel('Interest Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
    return rates

# Load the data
data_url = 'https://raw.githubusercontent.com/wrcarpenter/Interest-Rate-Models/main/Data/ho-lee-tree.csv'
data = pd.read_csv(data_url, header=None)
rates = data.iloc[1:, 0].values  # Skip the first row
# rates = rates_generator(T=200, mean_rate=0.01, volatility=0.015)

# Instantiate models with initial parameters
vasicek_model = VasicekModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
cir_model = CIRModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
ho_lee_model = HoLeeModel(params={'theta_t': lambda t: 0.03, 'sigma_t': lambda t: 0.015})

# Calibrate models to data
vasicek_model.calibrate(rates)
cir_model.calibrate(rates)
ho_lee_model.calibrate(rates)

# Simulate paths
total_time = len(rates) - 1
total_steps = total_time
number_of_paths = 50

vasicek_paths = vasicek_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
cir_paths = cir_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
ho_lee_paths = ho_lee_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
vasicek_paths_2 = vasicek_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')
cir_paths_2 = cir_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')
ho_lee_paths_2 = ho_lee_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')

# Plot simulated paths
vasicek_model.plot_paths(vasicek_paths, title="Vasicek Model Simulated Paths")
cir_model.plot_paths(cir_paths, title="CIR Model Simulated Paths")
ho_lee_model.plot_paths(ho_lee_paths, title="Ho-Lee Model Simulated Paths")
vasicek_model.plot_paths(vasicek_paths_2, title="Vasicek Model Simulated Paths")
cir_model.plot_paths(cir_paths_2, title="CIR Model Simulated Paths")
ho_lee_model.plot_paths(ho_lee_paths_2, title="Ho-Lee Model Simulated Paths")