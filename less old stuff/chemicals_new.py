import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import h5py


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenced
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


sol1 = None
sol2 = None
sol3 = None


def modelfunc2(t, k1, k2, k3, k4, q0, y0, t_dark):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, q0, y0], t_eval=t[0:t_dark], args=[k1, k2, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1], args=[0, k2, 0, 0])
    return np.concatenate((sol1.y[2], sol2.y[2]))


dataset = h5py.File('blinking/10 Feb 2025/Chemical study.h5', 'r')

particle = dataset['Particle 4']
abstimes = particle['Absolute Times (ns)']
print(particle.attrs['Description'])

difftime = np.diff(abstimes)
boundary_photons = np.where(difftime > 10e6)[0]
boundary_times = abstimes[boundary_photons]
boundary_times_start = abstimes[boundary_photons + 1]
boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

pulsephotons = np.diff(boundary_photons)
norm_pulsephotons = pulsephotons / ms_pulse[1:]
norm_pulsephotons = norm_pulsephotons[6:]
# norm_pulsephotons /= np.mean(norm_pulsephotons[:6])
# norm_pulsephotons = norm_pulsephotons[6:-50]
# norm_pulsephotons = norm_pulsephotons[7:-310]
print(len(norm_pulsephotons))

datapoints = len(norm_pulsephotons) + 1
endpoint = datapoints * 0.25

t = np.linspace(0, endpoint, datapoints)
t_plot = np.linspace(0, endpoint, datapoints * 10)

norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
t_dark = np.argmin(norm_pulsephotons)


def fitfunc(t, k1, k2, k3, k4, q0, y0, t_dark_offset):
    return modelfunc2(t, k1, k2, k3, k4, q0, y0, t_dark+int(t_dark_offset*1e9))


popt, pcov = curve_fit(fitfunc, t_plot, norm_pulsephotons, p0=[1/10, 1/10, 1/10, 1/10, 300, 600, 0],
                       bounds=([0, 0, 0, 0, 0, 0, -40e-9],
                               [10, 10, 10, 10, 1000, 1000, 200e-9]))
#
# popt = [1/60, 1/20, 1/20, 1/100, 1/120, 1448, -20e-9]
print(popt)

model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
tau1 = 1 / popt[0]
tau2 = 1 / popt[1]
tau3 = 1 / popt[2]
tau4 = 1 / popt[3]
y0 = popt[4]
t_dark_offset = popt[5]*1e9

print(f'Tau1 = {tau1:.1f} s')
print(f'Tau2 = {tau2:.1f} s')
print(f'Tau3 = {tau3:.1f} s')
print(f'Tau4 = {tau4:.1f} s')
print(f'Y0 = {y0:.1f} cps')
print(t_dark_offset)

t_plot = t_plot[:-1]
fig, ax1 = plt.subplots(1, 1, sharex=True)
ax1.plot(t_plot, norm_pulsephotons)
ax1.plot(t_plot, model)
plt.show()

# plt.plot(t_plot, np.concatenate((sol1.y[0], sol2.y[0])))
# plt.plot(t_plot, np.concatenate((sol1.y[1], sol2.y[1])))
# plt.plot(t_plot, np.concatenate((sol1.y[2], sol2.y[2])))
# plt.show()

