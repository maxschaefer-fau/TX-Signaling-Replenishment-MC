from __future__ import annotations
import numpy as np
import scipy.signal as sig
from scipy.constants import Avogadro
from scipy.special import erfc, erf


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

    def __init__(self, count, diffusive, volume) -> None:
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


class AbsorbingReceiver():

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

        # print(prob, np.max(prob))

        return prob

    def hitting_prob_point(self, t, D, dist, k_d = 0.0):

        # Eq. 6 https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8742793
        # Concentration after diffusion at distance dist
        # beta1 = 1/np.sqrt((4 * np.pi * D)**3)
        # beta2 = (dist * dist)/(4 * D)
        # prob = beta1 * np.exp(-beta2)

        # Eq. 39 
        '''
        D -> Diffusion Constant of space D_space in config
        arx -> reciever radius config.r_rx
        d0 -> dist in config
        '''
        # dist = 6*dist
        beta1 = self.r_rx/dist
        beta2 = dist - self.r_rx
        beta3 = np.sqrt(4 * D * t)
        beta3 = np.divide(beta2, beta3, out=np.zeros_like(t), where=t!=0)
        prob = beta1 * erfc(beta3)
        prob[0] = 0
        print(t, prob)
        print(f'Probability point tx absorbing rx: {prob}')

        return prob

    def average_hits(self, t, N, r_tx, D, dist, k_d = 0.0, exp = None):
        if exp == 'point':
            return sig.convolve(N, self.hitting_prob_point(t, D, dist, k_d), mode='full')
        return sig.convolve(N, self.hitting_prob(t, r_tx, D, dist, k_d), mode='full')


class TransparentReceiver():
    def __init__(self, radius) -> None:
        self.r_rx = radius

    def hitting_prob_point(self, t, r_tx, D, dist, k_d = 0.0):

         # Eq. 35 for Passive Transparent Receiver 
         # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8742793

         # beta1 = (self.r_rx - dist)/np.sqrt(4 * D)
         # beta1 = np.divide(beta1, t, out=np.zeros_like(t), where=t!=0)

         # beta2 = (self.r_rx + dist)/np.sqrt(4 * D)
         # beta2 = np.divide(beta2, t, out=np.zeros_like(t), where=t!=0)

         # beta3 = np.sqrt(D * t)/(self.r_rx * np.sqrt(np.pi))

         # beta4 = (self.r_rx - dist)**2/(4 * D)
         # beta4 = np.divide(beta4, t, out=np.zeros_like(t), where=t!=0)

         # beta5 = (self.r_rx + dist)**2/(4 * D)
         # beta5 = np.divide(beta5, t, out=np.zeros_like(t), where=t!=0)

         # prob = 0.5 * (erf(beta1) + erf(beta2)) + beta3 * (np.exp(-beta4) + np.exp(-beta5))

        beta1 = np.sqrt((4 * np.pi * D * t)**3)
        # Safeguard against zero in beta1
        epsilon = 1e-10  # Small number to use in place of zero
        beta1 = np.where(beta1 == 0, epsilon, beta1)

        beta2 = (dist * dist)/(4 * D)
        beta2 = np.divide(beta2, t, out=np.zeros_like(t), where=t!=0) 

        vol_rec = 4/3 * np.pi * self.r_rx * self.r_rx * self.r_rx

        prob = (vol_rec * np.exp(-beta2))/beta1

        # Count NaN and Inf values
        # nan_count = np.sum(np.isnan(prob))
        # inf_count = np.sum(np.isinf(prob))

        # Replace NaNs and Infs with a specific value, for example, 0
        prob = np.nan_to_num(prob, nan=0, posinf=0, neginf=0)
        # print(prob, np.max(prob))

        return prob

    def average_hits(self, t, N, r_tx, D, dist, k_d = 0.0, exp=None):
        if exp == 'type':
            return sig.convolve(N, self.hitting_prob_point(t, r_tx, D, dist, k_d), mode='full')
        return sig.convolve(N, self.hitting_prob_point(t, r_tx, D, dist, k_d), mode='full')
