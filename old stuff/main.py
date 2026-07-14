import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import h5py


def kinetic(t, y, k1, k2, k3, k4, k5):
    K = np.array([[-k5,  k3,    k4],
                  [k5, -k2-k3, k1],
                  [0,  k2,   -k1-k4]])
    return K @ y


# sol = solve_ivp(kinetic, [t[0], t[-1]], [0, 0, 1000], t_eval=t, args=[2, 1, 0.1, 0.5])
# ynoise = np.random.poisson(sol.y[1])
# plt.plot(sol.t, sol.y[0])
# plt.plot(sol.t, sol.y[1])
# plt.plot(sol.t, sol.y[2])
# plt.plot(sol.t, sol.y[3])
# plt.show()

t_dark = 395
# t_light = 480
t_light = 1400
# t_dark = 800
# t_light = 1200
# t_dark = 80
# t_light = 120

sol1 = None
sol2 = None
sol3 = None

floatdiff = np.finfo('double').eps

def modelfunc(t, k1, k2, k3, k4, k5, k1d, y0, t_dark, t_light):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, y0], t_eval=t[0:t_dark], args=[k1, k2, k3, k4, k5])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light]], sol1.y[:, -1], t_eval=t[t_dark:t_light], args=[0, k2, k3, 0, k5])
    sol3 = solve_ivp(kinetic, [t[t_light], t[-1]], sol2.y[:, -1], t_eval=t[t_light:-1], args=[k1, k2, k3, k4, k5])
    return np.concatenate((sol1.y[2], sol2.y[2], sol3.y[2]))


def modelfunc2(t, k1, k2, k3, k4, k5, k1d, y0, t_dark):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, y0], t_eval=t[0:t_dark], args=[k1, k2, k3, k4, k5])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1], args=[0, k2, k3, 0, k5])
    # sol3 = solve_ivp(kinetic, [t[t_light], t[-1]], sol2.y[:, -1], t_eval=t[t_light:-1], args=[k1, k2, k3, k4, k5])
    return np.concatenate((sol1.y[2], sol2.y[2]))


def fitfunc(t, k1, k2, k3, k4, k1d, k5, y0, td, tl):
    # return modelfunc(t, k1, k2, k3, k4, k5, 0, y0, int(td), int(tl))
    # return modelfunc2(t, k1, k2, k3, k4, 0, k1d, y0, t_dark + int(td / floatdiff))
    return modelfunc2(t, k1, k2, k3, k4, k5, k1d, y0, 403)


def plotfunc(t, k1, k2, k3, k4, k2d, k5, y0):
    return modelfunc(t, k1, k2, k3, k4, k5, k2d, y0, t_dark*10, t_light*10)


# def plotfunc(t, k1, k2, k3, k4, k2d, y0):
#     sol1 = solve_ivp(kinetic, [t[0], t[39]], [0, 0, y0], t_eval=t, args=[k1, k2, k3, k4], dense_output=True)
#     sol2 = solve_ivp(kinetic, [t[40], t[49]], sol1.y[:, -1], t_eval=t, args=[0, k2d, 0, 0], dense_output=True)
#     sol3 = solve_ivp(kinetic, [t[50], t[98]], sol2.y[:, -1], t_eval=t, args=[k1, k2, k3, k4], dense_output=True)
#     return np.concatenate((sol1.y[3], sol2.y[3], sol3.y[3]))

# y = fitfunc(t, 2, 1, 0.1, 0.5, 1, 1000)

# dataset = h5py.File('blinking/Test 3000 mE 2.h5', 'r')
dataset = h5py.File('blinking/Power study.h5', 'r')

abstimes = dataset['Particle 1']['Absolute Times (ns)']

# print(abstimes[0])
# print((abstimes[28018]-abstimes[0])/1e6)
# print((abstimes[28019]-abstimes[28018])/1e6)
difftime = np.diff(abstimes)
boundary_photons = np.where(difftime > 10e6)[0]
boundary_times = abstimes[boundary_photons]
boundary_times_start = abstimes[boundary_photons + 1]
boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
# print(boundary_times)
ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
# print(ms_pulse)
# print(boundary_times_start)

pulsephotons = np.diff(boundary_photons)
norm_pulsephotons = pulsephotons / ms_pulse[1:]
norm_pulsephotons = norm_pulsephotons[6:-6]
# t_plot = np.linspace(0, 25, 870)
t_plot = np.linspace(0, 40, 1470)
# t_plot = np.linspace(0, 22, 750)
print(len(norm_pulsephotons))
# t = np.linspace(0, 25, 87)
t = np.linspace(0, 40, 147)
# t = np.linspace(0, 22, 75)
norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
# ynoise = np.random.poisson(y)
# print(len(t))
# print(len(norm_pulsephotons))
popt, pcov = curve_fit(fitfunc, t_plot, norm_pulsephotons, p0=[1, 3, 0.1, 0.5, 0, 2, 800, 0, 1400],
                       bounds=([0, 0, 0, 0, 0, 0, 0, -20*floatdiff, 0],
                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 20*floatdiff, np.inf]))

# popt = [2.9e-01, 6.1e-01, 1.7e-01, 8.09e-01, 5.49e+02, 2e+00, 1.5e+03, 8e+02, 1.2e+03]
print(popt)

model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
tau1 = 1 / popt[0]
tau2 = 1 / popt[1]
tau3 = 1 / popt[2]
tau4 = 1 / popt[3]
tau1d = 1 / popt[4]
tau5 = 1 / popt[5]
y0 = 1 / popt[6]

print(f'Tau1 = {tau1:.1f} s')
print(f'Tau2 = {tau2:.1f} s')
print(f'Tau3 = {tau3:.1f} s')
print(f'Tau4 = {tau4:.1f} s')
print(f'Tau1_dark = {tau1d:.1f} s')
print(f'Tau5 = {tau5:.1f} s')
print(f'Y0 = {y0:.1f} cps')

# t = t[:-1]
t_plot = t_plot[:-1]
fig, ax1 = plt.subplots(1, 1, sharex=True)
ax1.plot(t_plot, norm_pulsephotons)
ax1.plot(t_plot, model)
# ax2.plot(t, model-norm_pulsephotons)
plt.show()

plt.plot(t_plot, np.concatenate((sol1.y[0], sol2.y[0])))
plt.plot(t_plot, np.concatenate((sol1.y[1], sol2.y[1])))
plt.plot(t_plot, np.concatenate((sol1.y[2], sol2.y[2])))
plt.show()

