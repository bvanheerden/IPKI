import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def two_exp(x, A, b, C, d, E):
    return A * np.exp(b * x) + C * np.exp(d * x) + E


def three_exp(x, A, b, C, d, E, f, G):
    return A * np.exp(b * x) + C * np.exp(d * x) + E * np.exp(f * x) + G


def fitter_2exp(x, y, k1):
    param, pcov = curve_fit(two_exp, x, y, p0=[1.84e4, k1, 4.63e5, -2.5, -2.20e3],
                            bounds=([-np.inf, k1-0.001, -np.inf, -np.inf, -np.inf],
                                    [np.inf, k1+0.001, np.inf, np.inf, np.inf]), max_nfev=1000)
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
    param, pcov = curve_fit(three_exp, x, y, p0=[7.55e3, -1.07e-1, 2.04e3, -2.72, 2.65e3, -5.4e-1, 2],
                            bounds = ([0, -np.inf, 0, -np.inf, 0, -np.inf, 0],
                                      [np.inf, 0, np.inf, 0, np.inf, np.inf, 1000]))
    fitted = three_exp(x, param[0], param[1], param[2], param[3], param[4], param[5], param[6])

    sumamp = param[0] + param[2] + param[4]

    amp1 = param[0] / sumamp
    amp2 = param[2] / sumamp
    amp3 = param[4] / sumamp
    tau1 = -1 / param[1]
    tau2 = -1 / param[3]
    tau3 = -1 / param[5]
    offset = param[6]

    # print(param)
    return fitted, amp1, amp2, amp3, tau1, tau2, tau3, offset


bleach_amp1 = []
bleach_amp2 = []
bleach_amp3 = []
bleach_tau1 = []
bleach_tau2 = []
bleach_tau3 = []
bleach_avtau = []
bleach_offset = []
bleach_maxint = []
bleach_minint = []

blink_av_amp2 = []
blink_av_tau2 = []

powers = [3125, 2122, 1061, 511]
for power in powers:
    print(power)

    bleach = np.loadtxt(f'blinking/{power} mE/trace1.csv', delimiter=',', skiprows=1)

    x = bleach[:, 0]
    y = bleach[:, 1]

    fitted, amp1, amp2, amp3, tau1, tau2, tau3, offset = fitter_3exp(x, y)

    avtau = (amp1 * tau1 ** 2 + amp2 * tau2 ** 2 + amp3 * tau3 ** 2) / (amp1 * tau1  + amp2 * tau2  + amp3 * tau3 )

    bleach_amp1.append(amp1)
    bleach_amp2.append(amp2)
    bleach_amp3.append(amp3)
    bleach_tau1.append(tau1)
    bleach_tau2.append(tau2)
    bleach_tau3.append(tau3)
    bleach_avtau.append(avtau)
    bleach_offset.append(offset)
    bleach_maxint.append(y[0])
    bleach_minint.append(y[-1])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(x, y)
    ax1.plot(x, fitted)
    ax2.plot(x, fitted-y)
    print(f'Amp1 = {amp1*100:.1f} %')
    print(f'Tau1 = {tau1:.1f} s')
    print(f'Amp2 = {amp2*100:.1f} %')
    print(f'Tau2 = {tau2:.1f} s')
    print(f'Amp3 = {amp3*100:.1f} %')
    print(f'Tau3 = {tau3:.1f} s')
    plt.show()

    blink_amp1 = []
    blink_amp2 = []
    blink_tau1 = []
    blink_tau2 = []
    for trace in ['2']:
    # for trace in ['2', '3', '4']:
        blink = np.loadtxt('blinking/3125 mE/trace' + trace + '.csv', delimiter=',', skiprows=1)
        for time in [101, 301, 501]:
            x = blink[time:time+99, 0]
            y = blink[time:time+99, 1]

            fitted, amp1_bl, amp2_bl, tau1_bl, tau2_bl, offset_bl = fitter_2exp(x, y, -1/max(tau1, tau2, tau3))
            blink_amp1.append(amp1_bl)
            blink_amp2.append(amp2_bl)
            blink_tau1.append(tau1_bl)
            blink_tau2.append(tau2_bl)
            # print(f'Amp1 = {amp1_bl*100:.1f} %')
            # print(f'Tau1 = {tau1_bl:.1f} s')
            # print(f'Amp2 = {amp2_bl*100:.1f} %')
            # print(f'Tau2 = {tau2_bl:.1f} s')

            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            # ax1.plot(x, y)
            # ax1.plot(x, fitted)
            # ax2.plot(x, fitted-y)
            # plt.show()

    blink_av_amp2.append(np.mean(blink_amp2))
    blink_av_tau2.append(np.mean(blink_tau2))


bleach_maxint = np.array(bleach_maxint)
bleach_minint = np.array(bleach_minint)

# plt.bar(powers, bleach_amp1, width=200)
# plt.bar(powers, bleach_amp3, width=200)
# plt.bar(powers, bleach_amp2, width=200)
# plt.bar(powers, (bleach_maxint - bleach_minint) / bleach_maxint, width=200)
plt.bar(powers, bleach_avtau, width=200)
# plt.plot(powers, (bleach_maxint - bleach_minint) / bleach_maxint)
plt.show()


# print(offset)
#
# print(f'Amp1 = {amp1*100:.1f} %')
# print(f'Tau1 = {tau1:.1f} s')
# print(f'Amp2 = {amp2*100:.1f} %')
# print(f'Tau2 = {tau2:.1f} s')
# print(f'Amp3 = {amp3*100:.1f} %')
# print(f'Tau3 = {tau3:.1f} s')
