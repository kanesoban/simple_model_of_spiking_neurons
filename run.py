import numpy as np
import matplotlib.pyplot as plt

# milliseconds
SIM_TIME = 200
# milliampers
CONST_CURRENT = 7
MIN_POTENTIAL = -90
RESTING_POTENTIAL = -70
CLAMP_VOLTAGES = False
dt = 0.5
sim_length = int(SIM_TIME / dt)

input_current = np.ones(sim_length) * CONST_CURRENT
input_current[:200] = 0
input_current[-300:] = 0

a = 0.02
b = 0.2
c = -65
d = 8
SPIKING_THRESHOLD = 30

# Simulate single neuron

v = np.empty(sim_length)
u = np.empty(sim_length)
v[0] = RESTING_POTENTIAL
u[0] = b * v[0]

for t in range(1, sim_length):
    if v[t - 1] >= SPIKING_THRESHOLD:
        v[t] = c
        u[t] = u[t - 1] + d
    else:
        v[t] += 0.5 * (0.04 * v[t - 1] ** 2 + 5 * v[t - 1] + 140 - u[t - 1] + input_current[t - 1]) * dt
        if CLAMP_VOLTAGES:
            v[t] = min(SPIKING_THRESHOLD, v[t])
            v[t] = max(MIN_POTENTIAL, v[t])
        u[t] += a * (b * v[t - 1] - u[t - 1]) * dt
        if CLAMP_VOLTAGES:
            u[t] = max(u[t], -100)
            u[t] = min(u[t], 100)

fig, (ax1, ax2) = plt.subplots(2, 1)
time = list(range(SIM_TIME))
ax1.plot([v[int(i/dt)] for i in time])
ax1.set(ylabel='voltage (mV)',
        title='Membrane potential over time')
ax1.grid()

ax2.plot([input_current[int(i/dt)] for i in time])
ax2.set(xlabel='time (ms)', ylabel='input current (mA)')
ax2.grid()
fig.savefig("membrane_potential.png")
