import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import pickle
from scipy.optimize import dual_annealing
from concurrent.futures import ProcessPoolExecutor

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


dataset = h5py.File('blinking/12 Feb 2025/Thylakoid intensity study new.h5', 'r')



def partint(dataset, partnum, p0, t_dark=0):
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
    t_plot = np.linspace(0, endpoint, datapoints * 10)

    norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
    t_dark = np.argmin(norm_pulsephotons)
    # t_dark = 1275
    # print(t_dark)
    norm_pulsephotons = norm_pulsephotons / norm_pulsephotons[0]

    def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset):
        sol1, sol2 = modelfunc2(t, k1, k2, 0, k4, k5, k6, k7, y0, q0,
                                t_dark+int(t_dark_offset*1e10))
        return np.concatenate((sol1.y[3]+sol1.y[4], sol2.y[3]+sol2.y[4]))

    # popt, pcov, *extra = curve_fit(fitfunc, t_plot, norm_pulsephotons,
    #                                p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0, 0.0, 0, -10e-10],
    #                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.5, 10e-10]),
    #                                verbose=1, xtol=2.23e-16, ftol=2.23e-16, gtol=2.23e-16)
    def objective(params):
        k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset = params
        model = fitfunc(t_plot, k1, k2, k3, k4, k5, k6, k7, y0, q0, t_dark_offset)
        return np.sum((model - norm_pulsephotons) ** 2)

    bounds = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 1),
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
    print(popt[9]*1e10)

    model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])

    return norm_pulsephotons, model, t_plot[:-1], tau1, tau2, tau3, tau4, tau5, tau6, tau7, y0, q0, partnum

def process_particle(i):
    return partint(dataset, i, p0)


# five, fivefit, t_five = partint(dataset, 7, p0=p0_fast)
# two, twofit, t_two = partint(dataset, 11, p0=p0_fast)
# one, onefit, t_one = partint(dataset, 10, p0=p0_fast)
# half, halffit, t_half = partint(dataset, 14, p0=p0_fast)
# quart, quartfit, t_quart = partint(dataset, 17, p0=p0_fast)
# eighth, eighthfit, t_eighth = partint(dataset, 19, p0=p0_fast)

# sol1, sol2 = modelfunc2(t_eight, 1/12.27, 1/5.22, 0, 1/1.16, 1/0.59, 1/0.99, 1/0.14, 0.763, 0.2643, 80)
# q1 = np.concatenate((sol1.y[1], sol2.y[1]))
# q2 = np.concatenate((sol1.y[2], sol2.y[2]))

p0 = [1 / 10.1, 1 / 3.48, 1 / 2.97, 1 / 4.92, 1 / 0.7, 1 / 11.61, 1/0.2, 0.6, 0.4, 0]
# eight, eightfit, t_eight, *args = partint(dataset, 12, p0=p0, t_dark=943)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(t_eight, eight, '-', label='7800 mE')
# ax1.plot(t_eight, eightfit, '-', label='7800 mE')
# residuals = eightfit - eight
# ax2.plot(t_eight[::10], residuals[::10], '.', label='Residuals')
# plt.show()

# ax1.plot(t_eight[:-1], q1, '-', label='q1')
# ax1.plot(t_eight[:-1], q2, '-', label='q2')
# plt.plot(t_five, five, '-', label='5000 mE')
# plt.plot(t_two, two, '--', label='2000 mE')
# plt.plot(t_two, twofit, '-', label='2000 mE', color='C1')
# plt.plot(t_half, half, '-', label='500 mE')
# plt.plot(t_quart, quart, '-', label='250 mE')
# plt.plot(t_half, halffit, '-', label='500 mE', color='C2')

# rss = np.sum(residuals[::10] ** 2)
# numparams = 8
# numpoints = len(eight) / 10
# bic = numpoints * np.log(rss / numpoints) + numparams * np.log(numpoints)
# print(rss)
# print(bic)

# plt.legend()
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Norm. Fluorescence')

tau1s = np.array([])
tau2s = np.array([])
tau3s = np.array([])
tau4s = np.array([])
tau5s = np.array([])
tau6s = np.array([])
tau7s = np.array([])
y0s = np.array([])
q0s = np.array([])



powers = np.array([5600, 4000, 2000, 1000, 500, 250])
compute = True
if compute:

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_particle, range(12, 33))
    for norm, model, t, tau1, tau2, tau3, tau4, tau5, tau6, tau7, y0, q0, partnum in results:
        tau1s = np.append(tau1s, 1/tau1)
        tau2s = np.append(tau2s, 1/tau2)
        tau3s = np.append(tau3s, 1/tau3)
        tau4s = np.append(tau4s, 1/tau4)
        tau5s = np.append(tau5s, 1/tau5)
        tau6s = np.append(tau6s, 1/tau6)
        tau7s = np.append(tau7s, 1/tau7)
        y0s = np.append(y0s, y0)
        q0s = np.append(q0s, q0)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(t, norm, '-')
        ax1.plot(t, model, '-')
        residuals = model - norm
        ax2.plot(t[::10], residuals[::10], '.')
        plt.savefig(f'partint_{partnum}.png')
        plt.close()

    tau1avs = np.array([np.mean(tau1s[17:21]),
                        np.mean(tau1s[np.array([13, 15, 16])]),
                        np.mean(tau1s[22:25]),
                        np.mean(tau1s[:4]),
                        np.mean(tau1s[5:8]),
                        np.mean(tau1s[8:10])])
    tau4avs = np.array([np.mean(tau4s[17:21]),
                        np.mean(tau4s[np.array([13, 15, 16])]),
                        np.mean(tau4s[22:25]),
                        np.mean(tau4s[:4]),
                        np.mean(tau4s[5:8]),
                        np.mean(tau4s[8:10])])
    tau7avs = np.array([np.mean(tau7s[17:21]),
                        np.mean(tau7s[np.array([13, 15, 16])]),
                        np.mean(tau7s[22:25]),
                        np.mean(tau7s[:4]),
                        np.mean(tau7s[5:8]),
                        np.mean(tau7s[8:10])])
    # Calculate the standard errors
    tau1_se = np.array([np.std(tau1s[17:21], ddof=0) / np.sqrt(1),
                        np.std(tau1s[np.array([13, 15, 16])], ddof=0) / np.sqrt(1),
                        np.std(tau1s[22:25], ddof=0) / np.sqrt(1),
                        np.std(tau1s[:4], ddof=0) / np.sqrt(1),
                        np.std(tau1s[5:8], ddof=0) / np.sqrt(1),
                        np.std(tau1s[8:10], ddof=0) / np.sqrt(1)])

    tau4_se = np.array([np.std(tau4s[17:21], ddof=0) / np.sqrt(1),
                        np.std(tau4s[np.array([13, 15, 16])], ddof=0) / np.sqrt(1),
                        np.std(tau4s[22:25], ddof=0) / np.sqrt(1),
                        np.std(tau4s[:4], ddof=0) / np.sqrt(1),
                        np.std(tau4s[5:8], ddof=0) / np.sqrt(1),
                        np.std(tau4s[8:10], ddof=0) / np.sqrt(1)])

    tau7_se = np.array([np.std(tau7s[17:21], ddof=0) / np.sqrt(1),
                        np.std(tau7s[np.array([13, 15, 16])], ddof=0) / np.sqrt(1),
                        np.std(tau7s[22:25], ddof=0) / np.sqrt(1),
                        np.std(tau7s[:4], ddof=0) / np.sqrt(1),
                        np.std(tau7s[5:8], ddof=0) / np.sqrt(1),
                        np.std(tau7s[8:10], ddof=0) / np.sqrt(1)])

    variables_to_pickle = {
        'tau1avs': tau1avs,
        'tau4avs': tau4avs,
        'tau7avs': tau7avs,
        'tau1_se': tau1_se,
        'tau4_se': tau4_se,
        'tau7_se': tau7_se
    }

    with open('../variables.pkl', 'wb') as f:
        pickle.dump(variables_to_pickle, f)

# Unpickle the variables
with open('../variables.pkl', 'rb') as f:
    variables = pickle.load(f)

# Access the variables
tau1avs = variables['tau1avs']
tau4avs = variables['tau4avs']
tau7avs = variables['tau7avs']
tau1_se = variables['tau1_se']
tau4_se = variables['tau4_se']
tau7_se = variables['tau7_se']

z1 = np.polyfit(powers, tau1avs, 1)
p1 = np.poly1d(z1)
print(1/p1(2))
z4 = np.polyfit(powers, tau4avs, 1)
p4 = np.poly1d(z4)
print(1/p4(2))
z7 = np.polyfit(powers, tau7avs, 1)
p7 = np.poly1d(z7)
print(1/p7(2))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# Plot the scatter plots
ax1.errorbar(powers, tau1avs, yerr=tau1_se, fmt='o', capsize=4, label=r'$k_1$', color='C0')
ax2.errorbar(powers, tau4avs, yerr=tau4_se, fmt='o', capsize=4, label=r'$k_4$', color='C1')
ax3.errorbar(powers, tau7avs, yerr=tau7_se, fmt='o', capsize=4, label=r'$k_7$', color='C2')

ax1.plot(powers, p1(powers), color='C0')
ax2.plot(powers, p4(powers), color='C1')
ax3.plot(powers, p7(powers), color='C2')

# Set labels
ax1.set_ylabel(r'$k_1$')
ax2.set_ylabel(r'$k_4$')
ax3.set_ylabel(r'$k_7$')

# Set the x-axis label
ax3.set_xlabel('Power (mE)')

plt.show()