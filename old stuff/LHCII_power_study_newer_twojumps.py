import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing
from scipy.optimize import curve_fit

# data_dir = 'E:\SMS\Measurements\Bertus\LHCII\PAM\Modulate and gate'
data_dir = '../blinking'

timestep = 0.125  # time step of intensity trace in seconds
p0 = [1 / 20, 1 / 1.55, 1 / 550.7, 1 / 17.5, 1 / 54, 1 / 500.61, 1/0.1, 1/1, 0.9, 0.1]

dataset_150 = h5py.File(os.path.join(data_dir, '13 March 2025', '150 uW.h5'), 'r')
dataset = h5py.File(os.path.join(data_dir, '13 March 2025', '225 uW.h5'), 'r')
dataset_338 = h5py.File(os.path.join(data_dir, '13 March 2025', '338 uW.h5'), 'r')
dataset_506 = h5py.File(os.path.join(data_dir, '13 March 2025', '506 uW.h5'), 'r')
dataset_760 = h5py.File(os.path.join(data_dir, '13 March 2025', '760 uW.h5'), 'r')
dataset_1140 = h5py.File(os.path.join(data_dir, '13 March 2025', '1140 uW.h5'), 'r')
dataset_1700 = h5py.File(os.path.join(data_dir, '13 March 2025', '1700 uW.h5'), 'r')


def kinetic(t, y, k1, k2, k3, k4, k5, k6, k7, k8):
    K = np.array([[-k5,  k3,  k8,  k4],  # Bleached
                  [0, -k2-k3, 0, k1],  # Quenced
                  [0, 0, -k8, 0],  # UnQuenched 2
                  [k5,  k2, 0, -k1-k4]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0, t_dark, t_light, t_dark2):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, y0], t_eval=t[0:t_dark+1],
                     args=[k1, k2, 0, k4, 0, 0, 0, k8])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0, 0, 0, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2, 0, k4, 0, 0, 0, k8])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[-1]], sol3.y[:, -1], t_eval=t[t_dark2:-1],
                     args=[0, k2, 0, 0, 0, 0, 0, 0])
    return sol1, sol2, sol3, sol4


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])

    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 10e6)[0]
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

    t_dark = np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//2])
    t_light = t_dark + 160
    t_dark2 = np.argmin(norm_pulsephotons)
    # print(t_dark, t_light, t_dark2)

    def fitfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, y0, q0):
        sol1, sol2, sol3, sol4 = modelfunc(t, k1, k2, k3, k4, k5, k6, k7, k8, 0.63, 0.36, t_dark, t_light,
                                           t_dark2)#+int(
        # t_dark_offset*1e10))
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:]))

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

    return norm_pulsephotons, model, t_plot[:-1] / 8, tau1, tau2, tau8, tau4, tau5, tau6, tau7, y0, q0

norm_pulsephotons_150, model_150, t_150, *params_150 = fittrace(dataset_150, [1, 2, 3])
plt.plot(t_150, norm_pulsephotons_150[:], '--')
plt.plot(t_150, model_150[:], label='150 uW', color='C0')

norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [1, 2, 3])
# plt.plot(t_225, norm_pulsephotons_225[:], '--')
# plt.plot(t_225, model_225[:], label='225 uW', color='C1')
#
norm_pulsephotons_338, model_338, t_338, *params_338 = fittrace(dataset_338, [1, 2, 3])
plt.plot(t_338, norm_pulsephotons_338[:], '--')
plt.plot(t_338, model_338[:], label='338 uW', color='C1')
# #
norm_pulsephotons_506, model_506, t_506, *params_506 = fittrace(dataset_506, [1, 2, 3])
# plt.plot(t_506, norm_pulsephotons_506[:], '--')
# plt.plot(t_506, model_506[:], label='506 uW', color='C3')
# # plt.plot(norm_pulsephotons_506[:] - model_506[:])
# #
norm_pulsephotons_760, model_760, t_760, *params_760 = fittrace(dataset_760, [1, 2, 3])
plt.plot(t_760, norm_pulsephotons_760[:], '--')
plt.plot(t_760, model_760[:], label='760 uW', color='C2')
# #
norm_pulsephotons_1140, model_1140, t_1140, *params_1140 = fittrace(dataset_1140, [1, 2, 3])
# plt.plot(t_1140, norm_pulsephotons_1140[:], '--')
# plt.plot(t_1140, model_1140[:], label='1140 uW', color='C5')
# #
# fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=(2, 1))
norm_pulsephotons_1700, model_1700, t_1700, *params_1700 = fittrace(dataset_1700, [1, 2, 3])
plt.plot(t_1700, norm_pulsephotons_1700[:], '--')
plt.plot(t_1700, model_1700[:], label='1700 uW', color='C3')
# ax1.plot(t_1700, norm_pulsephotons_1700[:], '--')
# ax1.plot(t_1700, model_1700[:], label='1700 uW', color='C6')
# ax2.plot(norm_pulsephotons_1700[:]-model_1700[:], '.')

# plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Normalized photon count')
plt.legend()
plt.tight_layout()
# plt.xlim(0, 140)
plt.show()

powers = [150, 225, 338, 506, 760, 1140, 1700]
params_list = [params_150, params_225, params_338, params_506, params_760, params_1140, params_1700]
tau1s = np.array([params[0] for params in params_list])
tau2s = np.array([params[1] for params in params_list])
tau8s = np.array([params[2] for params in params_list])
tau4s = np.array([params[3] for params in params_list])
y0s = np.array([params[7] for params in params_list])
q0s = np.array([params[8] for params in params_list])
print(np.mean(tau2s))
print(np.mean(y0s))
print(np.mean(q0s))

plt.figure()
plt.plot(powers, tau1s, '-o', label=r'$k_1$')
# plt.loglog(powers, tau2s)
plt.plot(powers, tau8s, '-o', label=r'$k_8$')
plt.plot(powers, tau4s, '-o', label=r'$k_4$')
plt.xlim(100, 2000)
# plt.ylim(0.01, 1)
plt.xlabel('Power (uW)')
plt.ylabel('Rate constant (s$^{-1}$)')
plt.legend()
plt.show()
