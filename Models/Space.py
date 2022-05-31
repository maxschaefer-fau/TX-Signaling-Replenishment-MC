from __future__ import annotations
import numpy as np
import scipy.signal as sig
from scipy.constants import Avogadro

class Space:

    def __init__(self, particles: dict, area=None, volume=None) -> None:
        # Requires particle names and concentration levels
        self.particles = particles
        self.area = area
        self.volume = volume

    def diffuse(self, other: Space, permeability, delta_t):
        for p_name, particle in self.particles.items():
            if particle.diffusive:
                self._diffuse_step(p_name, permeability, other, delta_t)
                other._diffuse_step(p_name, permeability, self, delta_t)
    
    def diffuse_step(self, p_name, flux, delta_t):
        if self.volume:
            conc_change = delta_t * flux * self.area / self.volume
            self.particles[p_name].set_conc(self.particles[p_name].concentration + conc_change)

    def _diffuse_step(self, p_name, permeability, other_space: Space, delta_t):
        this_conc = self.particles[p_name].concentration
        other_conc = other_space.particles[p_name].concentration
        if self.volume and other_space.volume:
            k1 = (this_conc * self.volume + other_conc * other_space.volume) / (self.volume + other_space.volume)
            k2 = (this_conc * other_space.volume - other_conc * other_space.volume) / (self.volume + other_space.volume)
            perm_coeff = permeability * self.area / self.volume * (1 + self.volume / other_space.volume)
        elif self.volume:
            k1 = other_conc
            k2 = this_conc - other_conc
            perm_coeff = permeability * self.area / self.volume
        elif other_space.volume:
            k1 = this_conc
            k2 = 0
            perm_coeff = permeability * self.area / other_space.volume
        else:
            raise ValueError('Volume must be defined in at least one of the spaces.')

        self.particles[p_name].set_conc(k2 * np.exp(-perm_coeff * delta_t) + k1)



            

class Particle():

    def __init__(self, count, diffusive, volume=0) -> None:
        self.count = count
        self.concentration = count / Avogadro / volume
        self.volume = volume
        self.diffusive = diffusive

    def set_count(self, new_count):
        self.count = new_count
        self.concentration = new_count / Avogadro / self.volume

    def set_conc(self, new_conc):
        self.concentration = new_conc
        self.count = new_conc * self.volume * Avogadro

class Receiver():

    def __init__(self, radius) -> None:
        self.r_rx = radius


    def hitting_prob(self, t, r_tx, D, dist, k_d = 0.0):
        rho = 0.25 / np.pi / r_tx / r_tx
        beta_1 = (r_tx + self.r_rx) * (r_tx + self.r_rx - 2 * dist) + dist * dist
        beta_1 /= 4 * D
        beta_1 = np.divide(beta_1, t, out=np.zeros_like(t), where=t!=0)
        beta_2 = (r_tx - self.r_rx) * (r_tx - self.r_rx + 2 * dist) + dist * dist
        beta_2 /= 4 * D
        beta_2 = np.divide(beta_2, t, out=np.zeros_like(t), where=t!=0)

        prob = 2 * rho * r_tx * self.r_rx / dist
        prob *= np.sqrt(np.divide(np.pi * D, t, out=np.zeros_like(t), where=t!=0))
        prob *= (np.exp(-beta_1-k_d*t) - np.exp(-beta_2-k_d*t))

        return prob

    def average_hits(self, t, N, r_tx, D, dist, k_d = 0.0):
        return sig.convolve(N, self.hitting_prob(t, r_tx, D, dist, k_d), mode='same')