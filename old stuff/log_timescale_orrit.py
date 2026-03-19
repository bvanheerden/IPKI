import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


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
    # tau2 = -1 / param[3]
    # offset = param[4]

    # print(param)
    return fitted, amp1, tau1, amp2


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


dataset = h5py.File('blinking/8 Feb 2025/Intensity study kinetic new.h5', 'r')

particle_ = dataset['Particle 12']
abstimes = particle_['Absolute Times (ns)'][:]
print(particle_.attrs['Description'])

print(abstimes)
abstimes = abstimes[abstimes > 2.1e9]
abstimes = abstimes - abstimes[0]
# bins = np.logspace(7, np.log10(abstimes[-1]), 1000)
bins = np.arange(abstimes[-1], step=1e7)
hist, bin_edges = np.histogram(abstimes, bins)
# hist = hist / np.diff(bin_edges)
bin_edges /= 1e9  # convert to seconds
time = bin_edges[:-1]

fitted, amp1, amp2, amp3, tau1, tau2, tau3, offset = fitter_3exp(time, hist)

print(f'Amp1 = {amp1 * 100:.1f} %')
print(f'Tau1 = {tau1:.3f} s')
print(f'Amp2 = {amp2 * 100:.1f} %')
print(f'Tau2 = {tau2:.3f} s')
print(f'Amp3 = {amp3 * 100:.1f} %')
print(f'Tau3 = {tau3:.3f} s')
# simulate = three_exp(time, 10, -100, 100, -0.01, 0, 0, 0)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# plt.loglog(bin_edges[:-1], hist)
# plt.plot(time, simulate)
ax1.plot(time, hist, color='xkcd:medium gray')
ax1.plot(time, fitted)
ax2.plot(time, fitted-hist)
# plt.yscale('log')
# plt.xscale('log')
# plt.xlim((2e9, 60e9))
plt.show()