import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import h5py


def kinetic(t, y, k1, k2, k3, k4, k5, k6, k7):
    K = np.array([[-k4, k3,    0,     k7],
                  [ 0, -k2-k3, k6,    0],
                  [ 0,  0,    -k5-k6, k1],
                  [ k4-k7, k2,    k5,   -k1]])
    return K @ y


t_dark = -1068
# t_dark = -870
# t_dark = -1020
# t_dark = -1040
# t_dark = -1000

sol1 = None
sol2 = None
sol3 = None


def modelfunc2(t, k1, k2, k3, k4, k5, k6, k7, y0, trip0, t_dark):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, trip0, y0], t_eval=t[0:t_dark],
                     args=[k1, k2, k3, k4, k5, k6, k7])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1],
                     args=[0, k2, k3, k4, k5, 0, 0])
    return np.concatenate((sol1.y[3], sol2.y[3]))


def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, y0):
    return modelfunc2(t, k1, k2, k3, k4, k5, k6, k7, y0, 0, t_dark)


dataset = h5py.File('blinking/Power study.h5', 'r')

abstimes = dataset['Particle 1']['Absolute Times (ns)']

difftime = np.diff(abstimes)
boundary_photons = np.where(difftime > 10e6)[0]
boundary_times = abstimes[boundary_photons]
boundary_times_start = abstimes[boundary_photons + 1]
boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

pulsephotons = np.diff(boundary_photons)
norm_pulsephotons = pulsephotons / ms_pulse[1:]
norm_pulsephotons = norm_pulsephotons[6:-6]
print(len(norm_pulsephotons))

t_plot = np.linspace(0, 40, 1470)
t = np.linspace(0, 40, 147)
# t_plot = np.linspace(0, 50, 1660)
# t = np.linspace(0, 50, 166)
# t_plot = np.linspace(0, 70, 2630)
# t = np.linspace(0, 70, 263)
# t_plot = np.linspace(0, 110, 4260)
# t = np.linspace(0, 110, 426)
# t_plot = np.linspace(0, 190, 7460)
# t = np.linspace(0, 190, 746)

norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
# popt, pcov = curve_fit(fitfunc, t_plot, norm_pulsephotons, p0=[100, 1/1.7, 1/3.6, 1/120, 1000, 10/2.3, 1/70, 788],
#                        bounds=([0, 0, 0, 0, 0, 0, 0, 0],
#                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))

popt = [1000, 1/1.7, 1/3.6, 1/120, 10000, 10/2.3, 1/90, 788, 0]

print(popt)

model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
tau1 = 1 / popt[0]
tau2 = 1 / popt[1]
tau3 = 1 / popt[2]
tau4 = 1 / popt[3]
tau5 = 1 / popt[4]
tau6 = 1 / popt[5]
tau7 = 1 / popt[6]
y0 = popt[7]
# trip0 = popt[7]

print(f'Tau1 = {tau1:.1f} s')
print(f'Tau2 = {tau2:.1f} s')
print(f'Tau3 = {tau3:.1f} s')
print(f'Tau4 = {tau4:.1f} s')
print(f'Tau5 = {tau5:.4f} s')
print(f'Tau6 = {tau6:.1f} s')
print(f'Tau7 = {tau6:.1f} s')
print(f'Y0 = {y0:.1f} cps')
# print(f'trip0 = {trip0:.1f} cps')

t_plot = t_plot[:-1]
fig, ax1 = plt.subplots(1, 1, sharex=True)
ax1.plot(t_plot, norm_pulsephotons)
ax1.plot(t_plot, model)
plt.show()

# plt.plot(t_plot, np.concatenate((sol1.y[0], sol2.y[0])))
# plt.plot(t_plot, np.concatenate((sol1.y[1], sol2.y[1])))
# plt.plot(t_plot, np.concatenate((sol1.y[2], sol2.y[2])))
# plt.plot(t_plot, np.concatenate((sol1.y[3], sol2.y[3])))
# plt.show()

