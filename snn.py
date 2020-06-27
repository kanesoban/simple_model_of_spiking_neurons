import numpy as np


class Network:
    def __init__(self, a, b, c, d, resting_potentials, spiking_thresholds, dt, clamp_voltages=True):
        self.neurons = a.shape[0]
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.resting_potentials = resting_potentials
        self.spiking_thresholds = spiking_thresholds
        self.dt = dt
        self.clamp_voltages = clamp_voltages
        self.v = self.resting_potentials
        self.u = self.b * self.v

    def reset_network(self):
        self.v = self.resting_potentials
        self.u = self.b * self.v

    def step(self, input_currents):
        firing = np.where(self.v >= self.spiking_thresholds)
        self.v[firing] = self.c[firing]
        self.u[firing] = self.u[firing] + self.d[firing]

        not_firing = np.where(self.v < self.spiking_thresholds)
        self.v[not_firing] = self.v[not_firing] + 0.5 * (
                0.04 * self.v[not_firing] ** 2 + 5 * self.v[not_firing] + 140 - self.u[not_firing] + input_currents[
            not_firing]) * self.dt

        if self.clamp_voltages:
            clamps = np.where(self.v > self.spiking_thresholds)
            self.v[clamps] = self.spiking_thresholds[clamps]
            clamps = np.where(self.v < self.resting_potentials)
            self.v[clamps] = self.resting_potentials[clamps]

        self.u[not_firing] = self.u[not_firing] + self.a[not_firing] * (
                    self.b[not_firing] * self.v[not_firing] - self.u[not_firing]) * self.dt
