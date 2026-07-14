import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


def one_exp(x, A, b, C):
    return A * np.exp(b * x) + C


def two_exp(x, A, b, C, d, E):
    return A * np.exp(b * x) + C * np.exp(d * x) + 0


def three_exp(x, A, b, C, d, E, f, G):
    return A * np.exp(b * x) + C * np.exp(d * x) + E * np.exp(f * x) + G


def fitter_2exp(x, y):
    param, pcov = curve_fit(two_exp, x, y, maxfev=80000, p0=[1000, -10, 1000, -10, 0])
    fitted = two_exp(x, param[0], param[1], param[2], param[3], param[4])

    sumamp = param[0] + param[2]

    amp1 = param[0] / sumamp
    amp2 = param[2] / sumamp
    tau1 = -1 / param[1]
    tau2 = -1 / param[3]
    offset = param[4]

    # print(param)
    return fitted, amp1, amp2, tau1, tau2, offset


def fitter_3exp(x, y):
    param, pcov = curve_fit(three_exp, x, y, p0=[400, -1/0.2, 800, -1/0.8, 3000, -1/4.2, 0])
    fitted = three_exp(x, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

    sumamp = param[0] + param[2] + param[4]
    print(sumamp)

    amp1 = param[0] / sumamp
    amp2 = param[2] / sumamp
    amp3 = param[4] / sumamp
    tau1 = -1 / param[1]
    tau2 = -1 / param[3]
    tau3 = -1 / param[5]
    offset = param[6]

    # print(param)
    return fitted, amp1, amp2, amp3, tau1, tau2, tau3, offset


def kinetic(t, y, k1, k2, k3):
    K = np.array([[0,  k3,    0],
                  [0, -k2-k3, k1],
                  [0,  k2,   -k1]])
    return K @ y


def modelfunc2(t, k1, k2, k3, y0):
    sol = solve_ivp(kinetic, [t[0], t[-1]], [0, 0, y0], t_eval=t[0:-1], args=[k1, k2, k3])
    return sol.y[2]


dataset = h5py.File('blinking/8 Feb 2025/Intensity study kinetic new.h5', 'r')

time = np.arange(3, step=0.0001)
hist = modelfunc2(time, 10, 20, 10, 800)

time = time[:-1]

# fitted, amp1, amp2, amp3, tau1, tau2, tau3, offset = fitter_3exp(time, hist)
fitted, amp1, amp2, tau1, tau2, offset = fitter_2exp(time, hist)

print(f'Amp1 = {amp1 * 100:.1f} %')
print(f'K1 = {1/tau1:.3f} s^-1')
print(f'Amp2 = {amp2 * 100:.1f} %')
print(f'K2 = {1/tau2:.3f} s^-1')
# print(f'Amp3 = {amp3 * 100:.1f} %')
# print(f'Tau3 = {tau3:.3f} s')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(time, hist, color='xkcd:medium gray')
ax1.plot(time, fitted)
ax2.plot(time, fitted-hist)
plt.show()