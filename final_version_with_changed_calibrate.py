import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
import pandas as pd


class SDEModel:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def drift(self, r: float, t: float) -> float:
        raise NotImplementedError("Drift function is not implemented")

    def diffusion(self, r: float, t: float) -> float:
        raise NotImplementedError("Diffusion function is not implemented")

    def simulate_path(self, r0: float, total_time: float, total_steps: int, number_of_paths: int,
                      scheme: str = 'euler') -> np.ndarray:
        dt = total_time / total_steps
        paths = np.zeros((number_of_paths, total_steps + 1))
        paths[:, 0] = r0

        for i in range(1, total_steps + 1):
            t = (i - 1) * dt
            W = np.random.normal(0, 1, number_of_paths)

            if scheme == 'euler':
                for j in range(number_of_paths):
                    dW = np.sqrt(dt) * W[j]
                    paths[j, i] = (paths[j, i - 1] +
                                   self.drift(paths[j, i - 1], t) * dt +
                                   self.diffusion(paths[j, i - 1], t) * dW)
            elif scheme == 'milstein':
                for j in range(number_of_paths):
                    dW = np.sqrt(dt) * W[j]
                    diffusion_value = self.diffusion(paths[j, i - 1], t)
                    paths[j, i] = (paths[j, i - 1] +
                                   self.drift(paths[j, i - 1], t) * dt +
                                   diffusion_value * dW +
                                   0.5 * diffusion_value *
                                   self.diffusion_derivative(paths[j, i - 1], t) * (dW ** 2 - dt))
        return paths

    def diffusion_derivative(self, r: float, t: float) -> float:
        raise NotImplementedError("Diffusion derivative is not implemented")

    # def calibrate(self, data: np.ndarray, initial_guess: np.ndarray):
    #     def loss_function(params):
    #         self.update_params(params)
    #         simulated_data = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1,
    #                                             number_of_paths=100, scheme='euler')
    #         simulated_mean = np.mean(simulated_data, axis=0)
    #         return np.sum((simulated_mean - data) ** 2)
    #
    #     bounds = self.get_bounds()
    #     result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    #     self.update_params(result.x)

    # def loss_function(params):
    #     # Unpack parameters
    #     sigma_est = params[0]
    #
    #     # Simulate paths based on selected method
    #     if self.method == 'euler-maruyama':
    #         simulated_data = self._simulate_paths_euler_maruyama(T=1.0, num_paths=1000, num_steps=100, dt=0.01)
    #     elif self.method == 'milstein':
    #         simulated_data = self._simulate_paths_milstein(T=1.0, num_paths=1000, num_steps=100, dt=0.01)
    #     else:
    #         raise ValueError("Invalid method. Choose 'euler-maruyama' or 'milstein'.")
    #
    #     simulated_volatility = np.std(simulated_data, axis=0)
    #
    #     # Define the loss function as sum of squared differences
    #     loss = np.sum((simulated_volatility - observed_data) ** 2)
    #
    #     return loss
    #
    # # Initial guess for parameters
    # initial_guess = [self.sigma]
    #
    # # Use scipy minimize to find optimal parameters
    # result = minimize(loss_function, initial_guess, method='Nelder-Mead')
    #
    # # Update model parameters
    # self.sigma = result.x[0]
    #
    # return result.fun

    def calibrate(self, data: np.ndarray, initial_guess: np.ndarray):
        """
                Calibrates the model parameters to match given data using optimization.

                Args:
                    data (np.ndarray): Observed data to calibrate the model against.
                    initial_guess (np.ndarray): Initial guess for the model parameters.

                """

        def loss_function(params):
            self.update_params(params)
            simulated_data = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1,
                                                 number_of_paths=100)
            simulated_mean = np.mean(simulated_data, axis=0)
            simulated_volatility = np.std(simulated_data, axis=0)
            loss = np.sum((simulated_mean - data) ** 2)
            return loss

        result = minimize(loss_function, initial_guess, bounds=self.get_bounds(), method='L-BFGS-B')
        self.update_params(result.x)

    def update_params(self, params: np.ndarray):
        raise NotImplementedError("Parameter update function is not implemented")

    def get_bounds(self) -> Tuple[Tuple[float, float]]:
        raise NotImplementedError("Bounds function is not implemented")

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
        return self.params['sigma']

    def diffusion_derivative(self, r: float, t: float) -> float:
        return 0

    def update_params(self, params: np.ndarray):
        self.params['theta'], self.params['alpha'], self.params['sigma'] = params

    def get_bounds(self) -> Tuple[Tuple[float, float]]:
        return [(None, None), (0, None), (0, None)]


class CIRModel(SDEModel):
    def drift(self, r: float, t: float) -> float:
        theta, alpha, sigma = self.params['theta'], self.params['alpha'], self.params['sigma']
        return theta - alpha * r

    def diffusion(self, r: float, t: float) -> float:
        return self.params['sigma'] * np.sqrt(r)

    def diffusion_derivative(self, r: float, t: float) -> float:
        return 0.5 * self.params['sigma'] / np.sqrt(r)

    def update_params(self, params: np.ndarray):
        self.params['theta'], self.params['alpha'], self.params['sigma'] = params

    def get_bounds(self) -> Tuple[Tuple[float, float]]:
        return [(None, None), (0, None), (0, None)]


class HoLeeModel2(SDEModel):
    def drift(self, r: float, t: float) -> float:
        return self.params['theta_t'](t)

    def diffusion(self, r: float, t: float) -> float:
        return self.params['sigma_t'](t)

    def diffusion_derivative(self, r: float, t: float) -> float:
        return 0

    def update_params(self, params: np.ndarray):
        data_len = len(params) // 2
        theta_t = params[:data_len]
        sigma_t = params[data_len:]
        self.params['theta_t'] = lambda t: theta_t[int(t)]
        self.params['sigma_t'] = lambda t: sigma_t[int(t)]

    def get_bounds(self) -> Tuple[Tuple[float, float]]:
        data_len = len(self.params['initial_guess']) // 2
        return [(None, None)] * data_len + [(0, None)] * data_len

    def simulate_path(self, r0: float, total_time: float, total_steps: int, number_of_paths: int,
                      scheme: str = 'milstein') -> np.ndarray:
        dt = total_time / total_steps
        paths = np.zeros((number_of_paths, total_steps + 1))
        paths[:, 0] = r0

        for t in range(1, total_steps + 1):
            W = np.random.normal(0, 1, number_of_paths)
            dW = np.sqrt(dt) * W

            if scheme == 'euler':
                paths[:, t] = paths[:, t - 1] + self.drift(paths[:, t - 1], t) * dt + self.diffusion(paths[:, t - 1],
                                                                                                     t) * dW
            elif scheme == 'milstein':
                dW_squared = dW * dW
                paths[:, t] = paths[:, t - 1] + self.drift(paths[:, t - 1], t) * dt + self.diffusion(paths[:, t - 1],
                                                                                                     t) * dW + 0.5 * self.diffusion(
                    paths[:, t - 1], t) ** 2 * (dW_squared - dt)

        return paths


def rates_generator(T=100, mean_rate=0.02, volatility=0.005) -> np.ndarray:
    time = np.arange(T)
    rates = np.zeros(T)
    rates[0] = mean_rate

    for t in range(1, T):
        rates[t] = rates[t - 1] + volatility * np.random.normal()

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
rates = data.iloc[1:-1, -2].values

# Instantiate models
vasicek_model = VasicekModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
cir_model = CIRModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
ho_lee_model_2 = HoLeeModel2(
    params={'theta_t': lambda t: 0.03, 'sigma_t': lambda t: 0.015, 'initial_guess': np.ones(len(rates) * 2)})

# Calibrate
mean_rate = np.mean(rates)
volatility = np.std(rates)
initial_guess_vasicek = np.array([mean_rate, 0.1 * volatility, volatility])
initial_guess_cir = np.array([mean_rate, 0.1 * volatility, volatility])
initial_guess_ho_lee = np.concatenate([np.mean(rates) * np.ones(len(rates)), np.std(rates) * np.ones(len(rates))])

vasicek_model.calibrate(rates, initial_guess_vasicek)
cir_model.calibrate(rates, initial_guess_cir)
ho_lee_model_2.calibrate(rates, initial_guess_ho_lee)

total_time = len(rates) - 1
total_steps = total_time
number_of_paths = 10

# Simulate paths
vasicek_paths = vasicek_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps,
                                            number_of_paths=number_of_paths)
cir_paths = cir_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps,
                                    number_of_paths=number_of_paths)
ho_lee_paths = ho_lee_model_2.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps,
                                            number_of_paths=number_of_paths)

# Plot paths
vasicek_model.plot_paths(vasicek_paths, title="Vasicek Model Simulated Paths")
cir_model.plot_paths(cir_paths, title="CIR Model Simulated Paths")
ho_lee_model_2.plot_paths(ho_lee_paths, title="Ho-Lee Model Simulated Paths")
