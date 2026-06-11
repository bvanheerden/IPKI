import numpy as np
import h5py
from matplotlib import pyplot as plt
import os
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

data_dir = '/home/bertus/Documents/Postdoc/Metings/Suurstofprojek'

timestep = 0.05  # time step of intensity trace in seconds
p0 = [1 / 4, 1 / 6, 1 / 10, 1 / 3, 0.5]

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '5900 uW.h5'), 'r')
# startind = 4
# onlen = 19

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '3900 uW.h5'), 'r')
# startind = 1
# onlen = 29

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '2780 uW.h5'), 'r')
# startind = 3
# onlen = 45

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '1120 uW.h5'), 'r')
# startind = 1
# onlen = 97

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '520 uW.h5'), 'r')
# startind = 2
# onlen = 206

# dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '224 uW.h5'), 'r')
# startind = 1
# onlen = 467

dataset = h5py.File(os.path.join(data_dir, 'Maart 2026', 'Power study new', '90 uW.h5'), 'r')
startind = 3
onlen = 690

onlyplot = True
offlen = 181


def kinetic(t, y, k1, k2, k3, k4):
    K = np.array([[0,  0,  k4,  k3],  # Bleached
                  [0, -k2, 0, k1],  # Quenced
                  [0, 0, -k4, 0],  # UnQuenched 2
                  [0,  k2, 0, -k1-k3]])  # Unquenched
    return K @ y


def modelfunc(t, k1, k2, k3, k4, q0, t_dark, t_light, t_dark2, t_light2, t_dark3):
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark+1]], [0, 0, q0, 1-q0], t_eval=t[0:t_dark+1],
                     args=[k1, k2, k3, k4])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[t_light+1]], sol1.y[:, -1], t_eval=t[t_dark:t_light+1],
                     args=[0, k2, 0, 0])
    sol3 = solve_ivp(kinetic, [t[t_light], t[t_dark2+1]], sol2.y[:, -1], t_eval=t[t_light:t_dark2+1],
                     args=[k1, k2, k3, k4])
    sol4 = solve_ivp(kinetic, [t[t_dark2], t[t_light2+1]], sol3.y[:, -1], t_eval=t[t_dark2:t_light2+1],
                     args=[0, k2, 0, 0])
    sol5 = solve_ivp(kinetic, [t[t_light2], t[t_dark3+1]], sol4.y[:, -1], t_eval=t[t_light2:t_dark3+1],
                     args=[k1, k2, k3, k4])
    sol6 = solve_ivp(kinetic, [t[t_dark3], t[-1]], sol5.y[:, -1], t_eval=t[t_dark3:-1],
                     args=[0, k2, 0, 0])
    return sol1, sol2, sol3, sol4, sol5, sol6


def onetrace(dataset, partnum):

    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])

    print(np.max(abstimes))
    difftime = np.diff(abstimes)
    boundary_photons = np.where(difftime > 20e6)[0]
    boundary_times = abstimes[boundary_photons]
    boundary_times_start = abstimes[boundary_photons + 1]
    boundary_times_start = np.insert(boundary_times_start, 0, abstimes[0])
    ms_pulse = (boundary_times - boundary_times_start[:-1]) / 1e6
    print(ms_pulse)

    pulsephotons = np.diff(boundary_photons)
    norm_pulsephotons = pulsephotons / ms_pulse[1:]
    print(norm_pulsephotons)
    norm_pulsephotons /= np.mean(norm_pulsephotons[startind])
    norm_pulsephotons = norm_pulsephotons[np.isfinite(norm_pulsephotons)]
    return norm_pulsephotons[:]


def avtrace(dataset, partnums):
    partnums_ = [onetrace(dataset, partnum) for partnum in partnums]
    minlength = np.min([len(partnum) for partnum in partnums_])
    partnums_ = [partnum[:minlength] for partnum in partnums_]
    return np.mean(partnums_, axis=0)


def fittrace(dataset, partnums):

    if onlyplot:
        norm_pulsephotons = avtrace(dataset, partnums)[:]
    else:
        norm_pulsephotons = avtrace(dataset, partnums)[startind:]
    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * timestep
    t = np.linspace(0, endpoint, datapoints)

    t_dark = onlen  # np.argmin(norm_pulsephotons[:len(norm_pulsephotons)//3])  # minimum of first third of trace
    t_light = t_dark + offlen
    t_dark2 = t_light + onlen  # (np.argmin(norm_pulsephotons[len(norm_pulsephotons) // 3:2 * len(norm_pulsephotons) // 3]) +
               # len(norm_pulsephotons) // 3)  # minimum of second third of trace
    t_light2 = t_dark2 + offlen
    t_dark3 = t_light2 + onlen  # np.argmin(norm_pulsephotons[2 * len(norm_pulsephotons) // 3:]) + 2 * len(norm_pulsephotons) // 3  # etc.
    print(t_dark, t_light, t_dark2, t_light2, t_dark3)


    def fitfunc(t, k1, k2, k3, k4, q0):
        sol1, sol2, sol3, sol4, sol5, sol6 = modelfunc(t, k1, k2, k3, k4, q0, t_dark,
                                                       t_light, t_dark2, t_light2, t_dark3)
        return np.concatenate((sol1.y[2]+sol1.y[3], sol2.y[2][1:]+sol2.y[3][1:], sol3.y[2][1:]+sol3.y[3][1:],
                               sol4.y[2][1:]+sol4.y[3][1:], sol5.y[2][1:]+sol5.y[3][1:], sol6.y[2][1:]+sol6.y[3][1:]))


    if onlyplot:
        popt = p0
    else:
        popt, pcov, *extra = curve_fit(fitfunc, t, norm_pulsephotons, p0=p0, bounds=([0, 0, 0, 0, 0],
                                       [10, 10, 10, 10, 1]), verbose=2)

    tau1 = 1 / popt[0]
    tau2 = 1 / popt[1]
    tau3 = 1 / popt[2]
    tau4 = 1 / popt[3]
    q0 = popt[4]

    print(f'Tau1 = {tau1:.2f} s')
    print(f'Tau2 = {tau2:.2f} s')
    print(f'Tau3 = {tau3:.2f} s')
    print(f'Tau4 = {tau4:.2f} s')
    print(f'Q0 = {q0:.2f} cps')

    t_plot = t
    if onlyplot:
        model = None
    else:
        model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4])

    return norm_pulsephotons, model, t_plot[:-1], tau1, tau2, tau3, tau4, q0


norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [1, 2, 3, 4, 5, 6, 7])#, 8, 9, 10])
# norm_pulsephotons_225, model_225, t_225, *params_225 = fittrace(dataset, [1, 4, 7, 8, 9, 10])
if not onlyplot:
    plt.plot(t_225, norm_pulsephotons_225[:], '--', color='gray')
    plt.plot(t_225, model_225[:], '-')
else:
    plt.plot(norm_pulsephotons_225[:], '--', color='gray')

# plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Normalized photon count')
plt.legend()
plt.tight_layout()
# plt.xlim(0, 140)
plt.show()

