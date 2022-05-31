import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable

class Reaction(ABC):

    def __init__(self, k1, k_1, substrate_conc, product_conc, reaction_coeffs=None):
        self.k1 = k1    # Forward reaction rate
        self.k_1 = k_1  # Backward reaction rate

        self.substrate_conc = np.array(substrate_conc)
        self.product_conc = np.array(product_conc)

        len_coeffs = self.substrate_conc.size + self.product_conc.size
        if reaction_coeffs is None:
            self.reaction_coeffs = np.ones((len_coeffs,))
        else:
            assert len(reaction_coeffs) == len_coeffs
            self.reaction_coeffs = reaction_coeffs

        self.d_subs = np.zeros_like(substrate_conc)
        self.d_prod = np.zeros_like(product_conc)

    def update_conc(self, substrate_conc=None, product_conc=None):
        if substrate_conc is not None:
            # assert len(substrate_conc) == len(self.substrate_conc)
            self.substrate_conc = np.array(substrate_conc)

        if product_conc is not None:
            # assert len(product_conc) == len(self.product_conc)
            self.product_conc = np.array(product_conc)

    @abstractmethod
    def prepare_step(self):
        pass

class One2OneReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 1
        assert self.product_conc.size == 1

    def prepare_step(self):
        self.d_subs = (self.k_1 * self.product_conc -
                       self.k1 * self.substrate_conc)

        self.d_prod = -self.d_subs

        return self.d_subs, self.d_prod


class Two2OneReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 2
        assert self.product_conc.size == 1
        """
        while isinstance(self.product_conc, Iterable):
            self.product_conc = self.product_conc[0]
        """

    def prepare_step(self):
        self.d_subs = (self.k_1 * self.product_conc -
                      self.k1 * self.substrate_conc[0] * self.substrate_conc[1]) * \
                      np.ones_like(self.substrate_conc)

        self.d_prod = -self.d_subs[0]

        return self.d_subs, self.d_prod

class One2TwoReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 1
        assert self.product_conc.size == 2
        """
        while isinstance(self.substrate_conc, Iterable):
            self.substrate_conc = self.substrate_conc[0]
        """
    def prepare_step(self):
        self.d_subs = (self.k_1 * self.product_conc[0] * self.product_conc[1] -
                       self.k1 * self.substrate_conc)

        self.d_prod = -self.d_subs * np.ones_like(self.product_conc)

        return self.d_subs, self.d_prod


class Two2TwoReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 2
        assert self.product_conc.size == 2

    def prepare_step(self):
        self.d_subs = (self.k_1 * self.product_conc[0] * self.product_conc[1] -
                       self.k1 * self.substrate_conc[0] * self.substrate_conc[1]) * \
                      np.ones_like(self.substrate_conc)

        self.d_prod = -self.d_subs

        return self.d_subs, self.d_prod


def simplified_theoretical(time_vec, R_0, S_0, k1, k_1, E):
    """
    R and S concentrations can be found in closed form in the simplified reaction. They are defined as the following.

    R = c_R0 + c_R1 * exp(E * (k_-1 + k_1) * t)
    S = c_S0 + c_S1 * exp(E * (k_-1 + k_1) * t)

    Here, c's are the constants. E is the enzyme concentration, t is time, and k's are the reaction rates.

    :param time_vec: The time axis. The time inputs to calculate the concentration
    :param R_0: R concentration in the beginning
    :param S_0: S concentration in the beginning
    :param k1: Forward reaction rate (k 1)
    :param k_1: Backward reaction rate (k -1)
    :param E: Enzyme concentration
    :return: The R and S concentration vectors in a tuple
    """

    # Find the concentrations in the end
    R_inf = S_inf = (R_0 + S_0) / 2.0

    # Find the constants
    c_R0 = R_inf
    c_R1 = R_0 - c_R0
    c_S0 = S_inf
    c_S1 = S_0 - c_S0

    # Find the concentrations
    R = c_R0 + c_R1 * np.exp(-E * (k1 + k_1) * time_vec)
    S = c_S0 + c_S1 * np.exp(-E * (k1 + k_1) * time_vec)

    return R, S





