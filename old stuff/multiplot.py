import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def kinetic(t, y, k1, k2, k3, k4, k5):
    K = np.array([[-k5,  k3,    k4],
                  [0, -k2-k3, k1],
                  [k5,  k2,   -k1-k4]])
    return K @ y


def modelfunc2(t, k1, k2, k3, k4, k5, y0, t_dark):
    global sol1, sol2, sol3
    sol1 = solve_ivp(kinetic, [t[0], t[t_dark]], [0, 0, y0], t_eval=t[0:t_dark], args=[k1, k2, k3, k4, k5])
    sol2 = solve_ivp(kinetic, [t[t_dark], t[-1]], sol1.y[:, -1], t_eval=t[t_dark:-1], args=[0, k2, k3, 0, k5])
    return np.concatenate((sol1.y[2], sol2.y[2]))


# dataset = h5py.File('blinking/Test 3000 mE 2.h5', 'r')
# dataset = h5py.File('blinking/Power study.h5', 'r')
# dataset = h5py.File('blinking/3000 mE.h5', 'r')
dataset_gco = h5py.File('blinking/4 Feb 2025/Low Oxygen New.h5', 'r')
dataset_aa = h5py.File('blinking/4 Feb 2025/Ascorbic.h5', 'r')
dataset_ph = h5py.File('blinking/4 Feb 2025/Low pH.h5', 'r')
dataset_az = h5py.File('blinking/4 Feb 2025/Azide.h5', 'r')
dataset = h5py.File('blinking/4 Feb 2025/3000 mE.h5', 'r')
# dataset = h5py.File('blinking/3000 mE.h5', 'r')
dataset_1500 = h5py.File('blinking/4 Feb 2025/1500 mE.h5', 'r')
# dataset_1500 = h5py.File('blinking/1500 mE.h5', 'r')
dataset_2380 = h5py.File('blinking/4 Feb 2025/2380 mE.h5', 'r')
dataset_700 = h5py.File('blinking/4 Feb 2025/700 mE.h5', 'r')

p0_control = [1 / 2, 1 / 13, 1 / 2, 1 / 100, 1 / 100, 1, 0]
p0_gco = [1 / 60, 1 / 20, 1 / 20, 1 / 100, 1 / 120, 1, 0]


def partint(dataset, partnum=1, p0=p0_control):
    particle_ = dataset[f'Particle {partnum}']
    abstimes = particle_['Absolute Times (ns)']
    # print(particle_.attrs['Description'])

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
    norm_pulsephotons = norm_pulsephotons[6:]

    datapoints = len(norm_pulsephotons) + 1
    endpoint = datapoints * 0.25
    t = np.linspace(0, endpoint, datapoints)
    t_plot = np.linspace(0, endpoint, datapoints * 10)

    norm_pulsephotons = np.interp(t_plot, t[:-1], norm_pulsephotons)[:-1]
    t_dark = np.argmin(norm_pulsephotons)
    norm_pulsephotons = norm_pulsephotons / norm_pulsephotons[0]

    def fitfunc(t, k1, k2, k3, k4, k5, y0, t_dark_offset):
        return modelfunc2(t, k1, k2, k3, k4, 1/120, y0, t_dark+int(t_dark_offset*1e9))

    # popt, pcov = curve_fit(fitfunc, t_plot, norm_pulsephotons,
    #                        p0=p0,
    #                        bounds=([0, 0, 0, 0, 0, 0, -40e-9],
    #                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 200e-9]))

    popt = [0.5, 1, 0.5, 0.01, 0.01, 1, 0]
    tau1 = 1 / popt[0]
    tau2 = 1 / popt[1]
    tau3 = 1 / popt[2]
    tau4 = 1 / popt[3]
    tau5 = 1 / popt[4]

    print(f'Tau1 = {tau1:.1f} s')
    print(f'Tau2 = {tau2:.1f} s')
    print(f'Tau3 = {tau3:.1f} s')
    print(f'Tau5 = {tau5:.1f} s')

    model = fitfunc(t_plot, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

    return norm_pulsephotons, model, t_plot[:-1]


# control, controlfit, t_control = partint(dataset, 1)
control, controlfit, t_control = partint(dataset, 1, p0=p0_gco)
# control2 = partint(dataset, 3)
# control3 = partint(dataset, 4)
c2380, c2380fit, t_c2380 = partint(dataset_2380, 1, p0=p0_gco)
c1500, c1500fit, t_c1500 = partint(dataset_1500, 1, p0=p0_gco)
c700, c700fit, t_c700 = partint(dataset_700, 1, p0=p0_gco)
gco, gcofit, t_gco = partint(dataset_gco, 4, p0=p0_gco)
# gco2 = partint(dataset_gco, 5)
# gco3 = partint(dataset_gco, 6)
aa, aafit, t_aa = partint(dataset_aa, 1, p0=p0_gco)
# aa2 = partint(dataset_aa, 4)
# aa3 = partint(dataset_aa, 5)
az, azfit, t_az = partint(dataset_az, 5, p0=p0_gco)
# az2 = partint(dataset_az, 7)
# az3 = partint(dataset_az, 8)
ph, phfit, t_ph = partint(dataset_ph, 4)
# ph2 = partint(dataset_ph, 4)
plt.plot(t_control, control, '-', label='Control')
# plt.plot(t_control, controlfit, color='C0', label='Control')
# plt.plot(control2)
# plt.plot(control3)

# plt.plot(t_c2380, c2380, '-', label='2380 mE')
# plt.plot(t_c1500, c1500, '-', label='1500 mE')
# plt.plot(t_c700, c700, '-', label='700 mE')

# plt.plot(t_c2380, c2380fit, color='C1', label='2380 mE')
# plt.plot(t_c1500, c1500fit, color='C2', label='1500 mE')

# plt.plot(t_c700, c700fit, color='C3', label='700 mE')
# plt.plot(t_gco, gcofit, color='C1', label='Low Oxygen')
# plt.plot(gco2)
# plt.plot(gco3)

plt.plot(t_gco[:-3000], gco[3000:], '-', label='Low Oxygen')
plt.plot(t_aa, aa, '-', label='Ascorbic Acid')
plt.plot(t_az, az, '-', label='Sodium Azide')
plt.plot(t_ph, ph, '-', label='Low pH')

# plt.plot(t_aa, aafit, color='C2', label='Ascorbic Acid')
# plt.plot(aa2)
# plt.plot(aa3)
# plt.plot(t_az, azfit, color='C3', label='Sodium Azide')
# plt.plot(az2)
# plt.plot(az3)
# plt.plot(t_ph, phfit, color='C4', label='Low pH')
# plt.plot(ph2)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Norm. Fluorescence')
plt.show()