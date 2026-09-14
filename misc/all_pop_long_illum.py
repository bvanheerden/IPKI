import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from kinetic_models import kinetic_model
from matplotlib import pyplot as plt
import utils
from brokenaxes import brokenaxes

utils.setup_plotting()

fontsize = 6
plt.rcParams.update({"font.size": fontsize,
                     'axes.titlesize': fontsize,
                     'axes.labelsize': fontsize,
                     'xtick.labelsize': fontsize,
                     'legend.fontsize': fontsize,
                     })
fig1, ax2 = plt.subplots(figsize=(85/25.4, 60/25.4))
fig2 = plt.figure(figsize=(80/25.4, 60/25.4))
ax_thy_aa = brokenaxes(ylims=((0, 0.25), (0.7, 0.85)), fig=fig2, hspace=0.05, d=0, despine=False)
# ax_thy_aa = fig2.add_subplot(111)

def fit_thylakoid(has_aa=False, lhcii=False, lowlight=False):
    # Parameters from Power_studies_June2026.py and typical thylakoid fits
    onlen_thy = 350
    offlen_thy = 50
    timestep = 0.1
    t = np.linspace(0, (1644-1)*timestep, 1644)

    t_dark = onlen_thy
    t_light = t_dark + offlen_thy
    t_dark2 = t_light + onlen_thy
    t_light2 = t_dark2 + offlen_thy
    t_dark3 = t_light2 + onlen_thy

    # Local fitfunc for 1-quencher thylakoid model
    def thy_fitfunc(t):
        if lhcii:
            sol1, sol2, sol3, sol4, sol5, sol6 = kinetic_model.modelfunc_2q(t, 0.449, 6.08, 0.110,
                                                                            0.05, 1.08, 0.86, 1, 0.2,
                                                                            t_dark, t_light, t_dark2, t_light2, t_dark3)
            pops = []
            for i in range(5):
                pop_i = np.concatenate((sol1.y[i], sol2.y[i][1:], sol3.y[i][1:],
                                        sol4.y[i][1:], sol5.y[i][1:], sol6.y[i][1:]))
                pops.append(pop_i)
            return np.array(pops)
        elif has_aa:
            if lowlight:
                k1, k2, k3, k4 = 0.096, 3.66, 0.0012, 0.083
            else:
                k1, k2, k3, k4 = 1.64, 3.66, 0.05, 0.933  # Thylakoid
                # k1, k2, k3, k4 = 0.1, 0.18, 0.03, 0.82  # LHCII 301 mE
                # k1, k2, k3, k4 = 0.1, 0, 0.03, 0.82  # LHCII 301 mE with no decay of quench
        else:
            k1, k2, k3, k4 = 0.232, 0.264, 0.17, 2.99
        res = kinetic_model.modelfunc(t, k1, k2, k3, k4, 1, 0.2, t_dark, t_light, t_dark2, t_light2,
                                      t_dark3, k2_light=None, return_populations=True)
        return res

    # Extract populations and model
    pops = thy_fitfunc(t)
    return t, pops

thy_pop_labels = ['B', 'Q', 'U$_2$', 'U$_1$']
colors = ['C1', 'C3', 'C2', 'C0']
lhcii_pop_labels = ['B', 'Q$_1$', 'Q$_2$', 'U$_2$', 'U$_1$']
lhcii_colors = ['C1', 'C3', 'C4', 'C2', 'C0']
t_lhcii, pops_lhcii = fit_thylakoid(lhcii=True)
# pops_lhcii[2] += pops_lhcii[1]
# pops_lhcii = np.delete(pops_lhcii, 1, axis=0)

for i in [4, 3, 2, 1, 0]:  # Order: U1, U2, Q, B
    ax2.plot(t_lhcii, pops_lhcii[i], label=lhcii_pop_labels[i], color=lhcii_colors[i])
    # ax2.text(4.2, 0.76, 'LHCII')

print("Fitting Thylakoid AA...")
t_aa, pops_aa = fit_thylakoid(has_aa=True, lowlight=True)

for i in [3, 2, 1, 0]: # Order: U1, U2, Q, B
    ax_thy_aa.plot(t_aa, pops_aa[i], label=thy_pop_labels[i], color=colors[i], lw=2)#3)
    # ax_thy_aa.text(3.0, 0.76, 'Thylakoids + AA')

for ax_curr in [ax2, ax_thy_aa]:
    ax_curr.set_ylabel('Population fraction')
    # ax_curr.set_ylim(top=0.8)
ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.01, 1.02))
ax_thy_aa.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=2)

ax2.set_xlim(0, 5)
ax_thy_aa.set_xlim(0, 30)
# ax_thy_aa.set_xlim(0, 10)
ax_thy_aa.set_xlabel('Time (s)')
ax2.set_xlabel('Time (s)')
fig1.tight_layout()
fig2.tight_layout()
ax_thy_aa.draw_diags(d=0.01)

plt.show()
