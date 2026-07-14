import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

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
    return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3]+sol2.y[4]))


dataset = h5py.File('blinking/12 Feb 2025/Thylakoid intensity study new.h5', 'r')

# p0_fast = [1 / 3.6, 1 / 0.6, 1 / 4.6, 1/10, 1/10, 1/100, 1/0.6, 0.8, 0.3, 0]
# p0_fast = [1 / 3.1, 1 / 0.48, 1 / 2.97, 1 / 11.92, 1 / 10, 1 / 0.61, 0.4, 0.7, 0]
p0_fast = [1 / 12.1, 1 / 9.48, 1 / 8.97, 1 / 1.92, 1 / 0.3, 1 / 10.61, 1/1, 0.7, 0.7, 0]


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
    norm_pulsephotons = norm_pulsephotons[6:]

    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * 0.25
    t = np.linspace(0, endpoint, datapoints)
    # t_plot = np.linspace(0, endpoint, datapoints * 10)
    t_plot = t

    # norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
    t_dark = np.argmin(norm_pulsephotons)
    print(t_dark)
    norm_pulsephotons = norm_pulsephotons / norm_pulsephotons[0]

    def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset):
        return modelfunc2(t, k1, k2, 0, k4, k5, k6, k7, y0, q0, t_dark)#+int(t_dark_offset*1e10))

    popt, pcov, *extra = curve_fit(fitfunc, t_plot, norm_pulsephotons,
                                   p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0, -1e-10],
                                   [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1e-10]),
                                   verbose=1, xtol=2.23e-16, ftol=2.23e-16, gtol=2.23e-16)

    # popt = [0.3, 1, 0.05, 0.01, 1, 1, 0.000001, 1, 0, 0]
    # popt = [1 / 3.1, 1 / 0.48, 1 / 4.97, 1 / 11.92, 1 / 10.05, 1 / 0.61, 1/2.57, 0.39, 0.748, popt[8]]
    # popt = [1 / 12.1, 1 / 9.48, 1 / 8.97, 1 / 1.92, 1 / 0.3, 1 / 10.61, 0.9, 0.7, 0]
    # popt[2] = 1/10
    # popt[3] = 0
    # popt[4] = 0
    tau1 = 1 / popt[0]
    tau2 = 1 / popt[1]
    tau3 = 1 / popt[2]
    tau4 = 1 / popt[3]
    tau5 = 1 / popt[4]
    tau6 = 1 / popt[5]
    tau7 = 1 / popt[6]

    print(f'Tau1 = {tau1:.2f} s')
    print(f'Tau2 = {tau2:.2f} s')
    print(f'Tau3 = {tau3:.2f} s')
    print(f'Tau4 = {tau4:.2f} s')
    print(f'Tau5 = {tau5:.2f} s')
    print(f'Tau6 = {tau6:.2f} s')
    print(f'Tau7 = {tau7:.2f} s')
    print(popt[7])
    print(popt[8])
    print(popt[9]*1e10)

    model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])

    return norm_pulsephotons, model, t_plot[:-1]


eight, eightfit, t_eight = partint(dataset, 26, p0=p0_fast)
# five, fivefit, t_five = partint(dataset, 7, p0=p0_fast)
# two, twofit, t_two = partint(dataset, 11, p0=p0_fast)
# one, onefit, t_one = partint(dataset, 10, p0=p0_fast)
# half, halffit, t_half = partint(dataset, 14, p0=p0_fast)
# quart, quartfit, t_quart = partint(dataset, 17, p0=p0_fast)
# eighth, eighthfit, t_eighth = partint(dataset, 19, p0=p0_fast)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(t_eight, eight, '-', label='7800 mE')
ax1.plot(t_eight, eightfit, '-', label='7800 mE')
# plt.plot(t_five, five, '-', label='5000 mE')
# plt.plot(t_two, two, '--', label='2000 mE')
# plt.plot(t_two, twofit, '-', label='2000 mE', color='C1')
# plt.plot(t_half, half, '-', label='500 mE')
# plt.plot(t_quart, quart, '-', label='250 mE')
# plt.plot(t_half, halffit, '-', label='500 mE', color='C2')

residuals = eightfit - eight
ax2.plot(t_eight, residuals, '.', label='Residuals')

rss = np.sum(residuals ** 2)
numparams = 8
bic = len(eight) * np.log(rss / len(eight)) + numparams * np.log(len(eight))
print(rss)
print(bic)

# plt.legend()
ax2.set_xlabel('Time (s)')
ax1.set_ylabel('Norm. Fluorescence')
ax2.set_ylabel('Residuals')
plt.show()