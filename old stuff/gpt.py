import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Step 1: Define the ODE y' = ky
def model(t, y, k):
    return k * y

# Step 2: Define a function to solve the ODE and get the solution for the given k
def solve_ode(k, t_span, y0):
    solution = solve_ivp(model, t_span, [y0], args=(k,), t_eval=t_span)
    return solution.y[0]

# Step 3: Define a function to fit the solution to experimental data
def fit_model(t_data, y_data):
    # Define a function that uses solve_ode to compute y for a given k
    def ode_solution(t, k):
        return solve_ode(k, t, y_data[0])[0]

    # Use curve_fit to estimate the best k
    popt, _ = curve_fit(ode_solution, t_data, y_data, p0=[0.1])  # p0 is an initial guess for k
    return popt

# Step 4: Generate some example data (for testing)
t_data = np.linspace(0, 10, 100)
k_true = 0.3
y0 = 5
y_data = solve_ode(k_true, t_data, y0) + np.random.normal(0, 0.5, size=t_data.shape)  # Add some noise

# Step 5: Fit the model to the noisy data
k_fit = fit_model(t_data, y_data)
print(f"Estimated k: {k_fit[0]}")

# Step 6: Plot the data and the fitted curve
plt.plot(t_data, y_data, 'o', label='Noisy data')
t_fitted = np.linspace(0, 10, 200)
y_fitted = solve_ode(k_fit[0], t_fitted, y0)
plt.plot(t_fitted, y_fitted, label=f'Fitted model (k={k_fit[0]:.2f})')
plt.xlabel('Time')
plt.ylabel('y')
plt.legend()
plt.show()
