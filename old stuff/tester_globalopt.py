import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
import pickle

def kinetic(t, y, k1, k2, k3, k4, k5, k6, k7):
    K = np.array([[0,  k3,    k6,    k7, 0],
                  [0, -k2-k3, 0,     0,  k1],
                  [0,  0,    -k5-k6, 0,  k4],
                  [0,  0,     0,    -k7, 0],
                  [0,  k2,    k5,    0, -k1-k4]])
    return K @ y


def modelfunc2(t, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, 0, q0, y0], t_eval=t[0:t_dark],
                     args=[k1, k2, k3, k4, k5, k6, k7])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1],
                     args=[0, k2, k3, 0, k5, k6, 0])
    return sol1, sol2


dataset = h5py.File('blinking/10 Feb 2025/Intensity study.h5', 'r')

p0_fast = [1 / 12.1, 1 / 9.48, 1 / 8.97, 1 / 1.92, 1 / 0.3, 1 / 1.61, 1/1, 0.9, 0.1, 0]


def partint(dataset, partnum=1, p0=p0_fast):
    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    print(particle_.attrs['Description'])

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 10e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    norm_pulsephotons = norm_pulsephotons[7:]

    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * 0.25
    t = np.linspace(0, endpoint, datapoints)
    t_plot = np.linspace(0, endpoint, datapoints * 10)

    norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
    t_dark = np.argmin(norm_pulsephotons)
    # t_dark = 1275
    print(t_dark)
    norm_pulsephotons = norm_pulsephotons / norm_pulsephotons[0]

    def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset):
        sol1, sol2 = modelfunc2(t, k1, k2, 0, k4, k5, k6, k7, y0, q0, t_dark)#+int(t_dark_offset*1e10))
        return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3]+sol2.y[4]))


    def objective(params):
        k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset = params
        model = fitfunc(t_plot, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset)
        return np.sum((model - norm_pulsephotons) ** 2)

    bounds = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10),
              (0, 0.5), (-1e-10, 1e-10)]
    result = dual_annealing(objective, bounds, x0=p0, maxiter=100)
    popt = result.x

    tau1 = 1 / popt[0]
    tau2 = 1 / popt[1]
    tau3 = 1 / popt[2]
    tau4 = 1 / popt[3]
    tau5 = 1 / popt[4]
    tau6 = 1 / popt[5]
    tau7 = 1 / popt[6]
    y0 = popt[7]
    q0 = popt[8]

    print(f'Tau1 = {tau1:.2f} s')
    print(f'Tau2 = {tau2:.2f} s')
    print(f'Tau3 = {tau3:.2f} s')
    print(f'Tau4 = {tau4:.2f} s')
    print(f'Tau5 = {tau5:.2f} s')
    print(f'Tau6 = {tau6:.2f} s')
    print(f'Tau7 = {tau7:.2f} s')
    print(f'Y0 = {y0:.2f} cps')
    print(f'Q0 = {q0:.2f} cps')

    model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])

    return norm_pulsephotons, model, t_plot[:-1], tau1, tau2, tau3, tau4, tau5, tau6, tau7, y0, q0


eight, eightfit, t_eight, *args = partint(dataset, 1, p0=p0_fast)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(t_eight, eight, '-', label='7800 mE')
ax1.plot(t_eight, eightfit, '-', label='7800 mE')
residuals = eightfit - eight
ax2.plot(t_eight[::10], residuals[::10], '.', label='Residuals')


plt.show()
