import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

dataset_thyl = h5py.File('blinking/12 Feb 2025/Thylakoid intensity study new.h5', 'r')


def onetrace(partnum):

    particle_ = dataset_thyl[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 10e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    norm_pulsephotons /= np.mean(norm_pulsephotons[:6])
    return norm_pulsephotons[6:200]


def avtrace(partnums):
    return np.mean([onetrace(partnum) for partnum in partnums], axis=0)


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[-k4, k3,    k4, 0],
                  [0,  -k2-k3, 0,  k1],
                  [0,   0,    -k4, 0],
                  [k4,  k2,    0, -k1]])
    return K @ y


def modelfunc2(t, k1, k2, k3, k4, y0, q0, t_dark):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, q0, y0], t_eval=t[0:t_dark], args=[k1, k2, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1], args=[0, k2, k3, 0])
    return np.concatenate((sol1.y[2] + sol1.y[3], sol2.y[2] + sol2.y[3]))


p0_fast = [1 / 3.1, 1 / 0.48, 1 / 2.97, 1 / 0.61, 0.4, 0.7, 0]
p0_slow = [1 / 6.1, 1 / 0.48, 1 / 2.97, 1 / 1.61, 0.4, 0.7, 0]


def partint(partnums, p0=p0_fast):

    trace = avtrace(partnums)
    datapoints = len(trace) + 1
    endpoint = datapoints * 0.25
    t = np.linspace(0, endpoint, datapoints)
    t_plot = np.linspace(0, endpoint, datapoints * 10)
    trace = np.interp(t_plot, t[:-1], trace)[:-1]
    t_dark = np.argmin(trace)

    # def fitfunc(t, k1, k2, k3, k4, y0, q0, t_dark_offset):
    #     return modelfunc2(t, k1, k2, k3, k4, y0, q0, t_dark+int(t_dark_offset*1e9))
    #
    # popt, pcov, *extra = curve_fit(fitfunc, t_plot, trace,
    #                                p0=p0, bounds=([0, 0, 0, 0, 0, 0, -40e-9],
    #                                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 40e-9]))
    #
    # tau1 = 1 / popt[0]
    # tau2 = 1 / popt[1]
    # tau3 = 1 / popt[2]
    # tau4 = 1 / popt[3]
    #
    # print(f'Tau1 = {tau1:.2f} s')
    # print(f'Tau2 = {tau2:.2f} s')
    # print(f'Tau3 = {tau3:.2f} s')
    # print(f'Tau4 = {tau4:.2f} s')
    # print(popt[4])
    # print(popt[5])
    # print(popt[6]*1e9)
    #
    # model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

    # return trace, model, t_plot[:-1]
    return trace, trace, t_plot[:-1]

# trace_5600 = avtrace([29, 30, 31, 32])
# trace_4000 = avtrace([25, 27, 28])
# trace_2000 = avtrace([22, 23, 24])
# trace_1000 = avtrace([12, 13, 14, 15])
# trace_500 = avtrace([17, 18, 19])
# trace_250 = avtrace([20, 21])
trace_5600, fit_5600, t_plot =  partint([29, 30, 31, 32])
trace_4000, fit_4000, t_plot_4000 =  partint([25, 27, 28])
trace_2000, fit_2000, t_plot_2000 =  partint([22, 23, 24], p0_slow)
trace_1000, fit_1000, t_plot_1000 =  partint([12, 13, 14, 15], p0_slow)
trace_500, fit_500, t_plot_500 =  partint([17, 18, 19], p0_slow)
trace_250, fit_250, t_plot_250 =  partint([20, 21], p0_slow)

plt.plot(t_plot, trace_5600, label='4051 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(t_plot, fit_5600, label='4051 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(t_plot_4000, trace_4000, label='2893 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(t_plot_4000, fit_4000, label='2893 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(t_plot_2000, trace_2000, label='1447 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(t_plot_2000, fit_2000, label='1447 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(t_plot_1000, trace_1000, label='723 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(t_plot_500, trace_500, label='362 mmol photons m$^{-2}$ s$^{-1}$')
plt.plot(t_plot_250, trace_250, label='181 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(trace_1000, label='725 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(trace_500, label='362 mmol photons m$^{-2}$ s$^{-1}$')
# plt.plot(trace_250, label='181 mmol photons m$^{-2}$ s$^{-1}$')

plt.legend()
plt.show()