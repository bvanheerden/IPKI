import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
from scipy.optimize import curve_fit

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek'

timestep = 0.05  # time step of intensity trace in seconds
p0 = [1 / 20, 1 / 1.55, 1 / 550.7, 1 / 17.5, 1 / 54, 1 / 500.61, 1/0.1, 1/1, 0.9, 0.1]

dataset = h5py.File(os.path.join(data_dir, '12 April 2025', '2500uW new.h5'), 'r')


def kinetic(t, y, k1, k2, k3, k4, k5, k6, k7, k8):
    K = np.array([[-k5,  k3,  k8,  k4],  # Bleached
                  [0, -k2-k3, 0, k1],  # Quenced
                  [0, 0, -k8, 0],  # UnQuenched 2
                  [k5,  k2, 0, -k1-k4]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0, t_dark, t_light, t_dark2, t_light2, t_dark3):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, y0], t_eval=t[0:t_dark+1],
                     args=[k1, k2, 0, k4, k5, 0, 0, k8])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0, k5, 0, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2, 0, k4, k5, 0, 0, k8])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[t_light2+1]], sol3.y[:, -1], t_eval=t[t_dark2:t_light2+1],
                     args=[0, k2, 0, 0, k5, 0, 0, 0])
    sol5 = solve_ivp(kinetic, [t[t_light2], t[t_dark3+1]], sol4.y[:, -1], t_eval=t[t_light2:t_dark3+1],
                     args=[k1, k2, 0, k4, k5, 0, 0, k8])
    sol6 = solve_ivp(kinetic, [t[t_dark3], t[-1]], sol5.y[:, -1], t_eval=t[t_dark3:-1],
                     args=[0, k2, 0, 0, k5, 0, 0, 0])
    return sol1, sol2, sol3, sol4, sol5, sol6


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    # print(ms_pulse)
    # print(pulsephotons[0])
    norm_pulsephotons /= np.mean(norm_pulsephotons[6])
    return norm_pulsephotons[:]


def avtrace(dataset, partnums):
    partnums_ = [onetrace(dataset, partnum) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    return np.mean(partnums_, axis=0)


def fittrace(dataset, partnums):

    norm_pulsephotons = avtrace(dataset, partnums)[6:]
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + 300
    t_dark2 = (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + 300
    t_dark3 = np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.
    print(t_light)


    def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0):
        sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))

    def objective(params):
        k1, k2, k3, k4, k5, k6, k7, k8, y0, q0 = params
        model = fitfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0)
        return np.sum((model - norm_pulsephotons) ** 2)

    # bounds = [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10), (0, 1), (0, 1)]
    # result = dual_annealing(objective, bounds, x0=p0, maxiter=40)
    # popt = result.x

    popt, pcov, *extra = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
                                   [10, 10, 10, 10, 10, 10, 10, 10, 1, 1]),
                                   verbose=2)

    # popt = [1/10, 1/1.58, 1/2.05, 1/28.77, 1/0.2, 1/1.68, 1/0.52, 1/4, 0.8, 0.2, 0.2, 0.03, 0]

    tau1 = 1 / popt[0]
    tau2 = 1 / popt[1]
    tau3 = 1 / popt[2]
    tau4 = 1 / popt[3]
    tau5 = 1 / popt[4]
    tau6 = 1 / popt[5]
    tau7 = 1 / popt[6]
    tau8 = 1 / popt[7]
    y0 = popt[8]
    q0 = popt[9]

    print(f'Tau1 = {tau1:.2f} s')
    print(f'Tau2 = {tau2:.2f} s')
    print(f'Tau3 = {tau3:.2f} s')
    print(f'Tau4 = {tau4:.2f} s')
    print(f'Tau5 = {tau5:.2f} s')
    print(f'Tau6 = {tau6:.2f} s')
    print(f'Tau7 = {tau7:.2f} s')
    print(f'Tau8 = {tau8:.2f} s')
    print(f'Y0 = {y0:.2f} cps')
    print(f'Q0 = {q0:.2f} cps')
    # print(popt[12])

    t_plot = t
    # t_plot = np.linspace(0, endpoint, datapoints*10)
    # model_coarse = fitfunc(t, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9],
    #                        popt[10], popt[11])
    # t_dark = 10*t_dark
    model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
    # norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]

    return norm_pulsephotons, model, t_plot[:-1], tau1, tau2, tau8, tau4, tau5, tau6, tau7, y0, q0


norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [1, 3, 5, 6, 7, 8, 9])
plt.plot(t_225, norm_pulsephotons_225[:], '--')
plt.plot(t_225, model_225[:], '-')

# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [2])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')

# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [3])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')

# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [4])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')

# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [5])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')
# #
# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [6])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')
#
# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [7])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')
#
# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [8])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')
#
# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [9])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')

# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [10])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')

# plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Normalized photon count')
plt.legend()
plt.tight_layout()
# plt.xlim(0, 140)
plt.show()

