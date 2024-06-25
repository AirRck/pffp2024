import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Dict, Any
import pandas as pd

class SDEModel:
    """
        Base class for Stochastic Differential Equation (SDE) models.

        Attributes:
            params (dict): Parameters of the SDE model.

        Methods:
            drift(r, t):
                Computes the drift term of the SDE.
            diffusion(r, t):
                Computes the diffusion term of the SDE.
            simulate_path(r0, total_time, total_steps, number_of_paths, scheme='euler'):
                Simulates paths of the SDE using Euler-Maruyama or Milstein scheme.
            diffusion_derivative(r, t):
                Computes the derivative of the diffusion term with respect to r.
            calibrate(data, initial_guess):
                Calibrates the model parameters to given data.
            update_params(params):
                Updates the model parameters with new values.
            get_bounds():
                Returns the parameter bounds for optimization.
            plot_paths(paths, title='Simulated Paths'):
                Plots the simulated paths of the model.
        """

    def __init__(self, params: Dict[str, Any]):
        """
               Initialize an SDE model with parameters.

               Args:
                   params (dict): Dictionary containing initial parameters of the model.
               """
        self.params = params

    def drift(self, r: float, t: float) -> float:
        """
                Placeholder for drift function. Must be implemented in subclass.

                Args:
                    r (float): Current value of the process.
                    t (float): Current time.

                Returns:
                    float: Drift at time t given state r.
                """
        raise NotImplementedError("Drift function is not implemented")

    def diffusion(self, r: float, t: float) -> float:
        """
                Placeholder for diffusion function. Must be implemented in subclass.

                Args:
                    r (float): Current value of the process.
                    t (float): Current time.

                Returns:
                    float: Diffusion at time t given state r.
                """
        raise NotImplementedError("Diffusion function is not implemented")

    def simulate_path(self, r0: float, total_time: float, total_steps: int, number_of_paths: int, scheme: str = 'euler') -> np.ndarray:
        """
               Simulates paths of the SDE model using specified numerical scheme.

               Args:
                   r0 (float): Initial value of the process.
                   total_time (float): Total time horizon for simulation.
                   total_steps (int): Number of time steps.
                   number_of_paths (int): Number of paths to simulate.
                   scheme (str, optional): Numerical scheme to use ('euler' or 'milstein'). Defaults to 'euler'.

               Returns:
                   np.ndarray: Array of simulated paths with shape (number_of_paths, total_steps + 1).
               """
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
                    diffusion_value = self.diffusion(paths[j, i-1], t)
                    paths[j, i] = (paths[j, i-1] +
                                   self.drift(paths[j, i-1], t) * dt +
                                   diffusion_value * dW +
                                   0.5 * diffusion_value *
                                   self.diffusion_derivative(paths[j, i-1], t) * (dW**2 - dt))
        return paths

    def diffusion_derivative(self, r: float, t: float) -> float:
        """
               Placeholder for diffusion derivative function. Must be implemented in subclass.

               Args:
                   r (float): Current value of the process.
                   t (float): Current time.

               Returns:
                   float: Derivative of diffusion with respect to r at time t given state r.
               """
        raise NotImplementedError("Diffusion derivative is not implemented")

    def calibrate(self, data: np.ndarray, initial_guess: np.ndarray):
        """
                Calibrates the model parameters to match given data using optimization.

                Args:
                    data (np.ndarray): Observed data to calibrate the model against.
                    initial_guess (np.ndarray): Initial guess for the model parameters.

                """
        def objective(params):
            self.update_params(params)
            simulated_paths = self.simulate_path(r0=data[0], total_time=len(data) - 1, total_steps=len(data) - 1, number_of_paths=1)
            simulated_data = simulated_paths[0]
            return np.mean((simulated_data - data)**2)

        result = minimize(objective, initial_guess, bounds=self.get_bounds(), method='L-BFGS-B')
        self.update_params(result.x)

    def update_params(self, params: np.ndarray):
        """
               Updates the model parameters with new values.

               Args:
                   params (np.ndarray): New parameter values.
               """
        raise NotImplementedError("Parameter update function is not implemented")

    def get_bounds(self) -> Tuple[Tuple[float, float]]:
        """
                Returns the bounds for optimization of model parameters.

                Returns:
                    Tuple[Tuple[float, float]]: Bounds for each parameter.
                """
        raise NotImplementedError("Bounds function is not implemented")

    def plot_paths(self, paths: np.ndarray, title: str = "Simulated Paths"):
        """
                Plots the simulated paths of the model.

                Args:
                    paths (np.ndarray): Array of simulated paths.
                    title (str, optional): Title of the plot. Defaults to "Simulated Paths".
                """
        plt.figure(figsize=(12, 6))
        for i in range(paths.shape[0]):
            plt.plot(paths[i])
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Interest Rate')
        plt.show()

class VasicekModel(SDEModel):
    """
        Vasicek Model class, inheriting from SDEModel.

        Methods:
            drift(r, t):
                Computes the drift term of the Vasicek model.
            diffusion(r, t):
                Computes the diffusion term of the Vasicek model.
            diffusion_derivative(r, t):
                Computes the derivative of the diffusion term with respect to r for the Vasicek model.
            update_params(params):
                Updates the model parameters with new values for the Vasicek model.
            get_bounds():
                Returns the parameter bounds for optimization for the Vasicek model.
        """
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
    """
       CIR Model class, inheriting from SDEModel.

       Methods:
           drift(r, t):
               Computes the drift term of the CIR model.
           diffusion(r, t):
               Computes the diffusion term of the CIR model.
           diffusion_derivative(r, t):
               Computes the derivative of the diffusion term with respect to r for the CIR model.
           update_params(params):
               Updates the model parameters with new values for the CIR model.
           get_bounds():
               Returns the parameter bounds for optimization for the CIR model.
       """
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

class HoLeeModel(SDEModel):
    """
       Ho-Lee Model class, inheriting from SDEModel.

       Methods:
           drift(r, t):
               Computes the drift term of the Ho-Lee model.
           diffusion(r, t):
               Computes the diffusion term of the Ho-Lee model.
           diffusion_derivative(r, t):
               Computes the derivative of the diffusion term with respect to r for the Ho-Lee model.
           update_params(params):
               Updates the model parameters with new values for the Ho-Lee model.
           get_bounds():
               Returns the parameter bounds for optimization for the Ho-Lee model.
       """
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

def rates_generator(T=100, mean_rate=0.02, volatility=0.005) -> np.ndarray:
    """
        Generate synthetic time series of interest rates.

        Parameters:
            T (int): Number of time periods.
            mean_rate (float): Mean interest rate.
            volatility (float): Volatility of interest rates.

        Returns:
            np.ndarray: Array of synthetic interest rates.
        """
    time = np.arange(T)
    rates = np.zeros(T)
    rates[0] = mean_rate

    for t in range(1, T):
        rates[t] = rates[t-1] + volatility * np.random.normal()

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

# Instantiate models
vasicek_model = VasicekModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
cir_model = CIRModel(params={'theta': 0.05, 'alpha': 0.1, 'sigma': 0.02})
ho_lee_model = HoLeeModel(params={'theta_t': lambda t: 0.03, 'sigma_t': lambda t: 0.015, 'initial_guess': np.ones(len(rates) * 2)})

# Calibrate
mean_rate = np.mean(rates)
volatility = np.std(rates)
print(mean_rate)
print(volatility)
initial_guess = [mean_rate, 0.1 * volatility, volatility]
initial_guess_vasicek = np.array([mean_rate, 0.1 * volatility, volatility])
initial_guess_cir = np.array([mean_rate, 0.1 * volatility, volatility])
# initial_guess_vasicek = np.array([0.03, 0.1, 0.02])
# initial_guess_cir = np.array([0.03, 0.1, 0.02])
initial_guess_ho_lee = np.concatenate([np.mean(rates) * np.ones(len(rates)), np.std(rates) * np.ones(len(rates))])

vasicek_model.calibrate(rates, initial_guess_vasicek)
cir_model.calibrate(rates, initial_guess_cir)
ho_lee_model.calibrate(rates, initial_guess_ho_lee)

total_time = len(rates) - 1
total_steps = total_time
number_of_paths = 50

vasicek_paths = vasicek_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
cir_paths = cir_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
ho_lee_paths = ho_lee_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='euler')
vasicek_paths_2 = vasicek_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')
cir_paths_2 = cir_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')
ho_lee_paths_2 = ho_lee_model.simulate_path(r0=rates[0], total_time=total_time, total_steps=total_steps, number_of_paths=number_of_paths, scheme='milstein')

# Plot
vasicek_model.plot_paths(vasicek_paths, title="Vasicek Model Simulated Paths (Euler)")
cir_model.plot_paths(cir_paths, title="CIR Model Simulated Paths (Euler)")
ho_lee_model.plot_paths(ho_lee_paths, title="Ho-Lee Model Simulated Paths (Euler)")
vasicek_model.plot_paths(vasicek_paths_2, title="Vasicek Model Simulated Paths (Milstein)")
cir_model.plot_paths(cir_paths_2, title="CIR Model Simulated Paths (Milstein)")
ho_lee_model.plot_paths(ho_lee_paths_2, title="Ho-Lee Model Simulated Paths (Milstein)")
