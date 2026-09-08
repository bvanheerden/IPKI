import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1


# ============================================================
# FRAP parameters
# ============================================================

D = 10.0                 # diffusion coefficient, µm^2/s
w = 75.0                 # bleach radius, µm

F_pre = 1.0             # fluorescence before bleaching
F_bleach = 0.4         # fluorescence immediately after bleaching

total_time = 30.0        # simulation duration, s
n_points = 1000


# ============================================================
# Time axis
# ============================================================

# Avoid t=0 because the analytical expression contains 1/t
t = np.linspace(1e-6, total_time, n_points)


# ============================================================
# Soumpasis FRAP model
# ============================================================

tau_D = w**2 / (4 * D)

x = tau_D / (2 * t)

recovery = np.exp(-x) * (i0(x) + i1(x))


# ============================================================
# Include incomplete bleaching
# ============================================================

F = F_bleach + (F_pre - F_bleach) * recovery


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(t, F, linewidth=2)

plt.axhline(F_pre, linestyle="--", alpha=0.5)
plt.axhline(F_bleach, linestyle="--", alpha=0.5)

plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.title("Analytical FRAP recovery")

# plt.ylim(0.7, 1.05)

plt.show()


# ============================================================
# Print characteristic diffusion time
# ============================================================

print(f"D = {D:.3f} µm²/s")
print(f"Bleach radius = {w:.3f} µm")
print(f"Characteristic diffusion time = {tau_D:.3f} s")