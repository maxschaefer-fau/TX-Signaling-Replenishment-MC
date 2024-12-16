import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable

class Reaction(ABC):

    def __init__(self, k1, k_1, substrate_conc, product_conc, reaction_coeffs=None):
        self.k1 = k1    # Forward reaction rate
        self.k_1 = k_1  # Backward reaction rate

        self.step_eps = 1E-15

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
    def step(self, t):
        pass

class One2OneReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 1
        assert self.product_conc.size == 1

    def step(self, t):
        self.d_subs = (self.k_1 * self.product_conc -
                       self.k1 * self.substrate_conc)
        if np.abs(self.d_subs) < self.step_eps:
            return
        subs_ss = (self.substrate_conc + self.product_conc) / (1 + self.k1 / self.k_1)
        subs_change = self.substrate_conc - subs_ss
        if np.abs(subs_change) >= self.step_eps:
            subs_change_degree = -self.d_subs / subs_change
        else:
            subs_change_degree = 0

        self.d_prod = -self.d_subs
        prod_ss = self.substrate_conc + self.product_conc - subs_ss
        prod_change = self.product_conc - prod_ss
        # prod_change_degree = -self.d_prod / prod_change

        new_subs = subs_ss + subs_change * np.exp(-subs_change_degree * t)
        self.product_conc += self.substrate_conc - new_subs
        self.substrate_conc = new_subs
        # self.product_conc = prod_ss + prod_change * np.exp(-prod_change_degree * t)


class Two2OneReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 2
        assert self.product_conc.size == 1
        """
        while isinstance(self.product_conc, Iterable):
            self.product_conc = self.product_conc[0]
        """

    def step(self, t):
        self.d_subs = (self.k_1 * self.product_conc -
                      self.k1 * self.substrate_conc[0] * self.substrate_conc[1]) * \
                      np.ones_like(self.substrate_conc)

        if (np.abs(self.d_subs) < self.step_eps).all():
            return
        self.d_prod = -self.d_subs[0]
        n_0 = self.substrate_conc[0] + self.product_conc
        n_1 = self.substrate_conc[1] + self.product_conc
        delta = self.k_1 * self.k_1 / self.k1 / self.k1 + (n_0 - n_1) * (n_0 - n_1) + 2 * self.k_1 / self.k1 * (n_0 + n_1)
        prod_ss = (self.k_1 / self.k1 + n_0 + n_1 - np.sqrt(delta)) / 2
        prod_change = self.product_conc - prod_ss
        # prod_change_degree = -self.d_prod / prod_change

        subs_ss = self.substrate_conc + self.product_conc - prod_ss
        subs_change = self.substrate_conc - subs_ss
        if (np.abs(subs_change) >= self.step_eps).all():
            subs_change_degree = -self.d_subs / subs_change
        else:
            subs_change_degree = 0

        new_subs = subs_ss + subs_change * np.exp(-subs_change_degree * t)
        self.product_conc += (self.substrate_conc - new_subs)[0]
        self.substrate_conc = new_subs
        # self.product_conc = prod_ss + prod_change * np.exp(-prod_change_degree * t)

class One2TwoReaction(Reaction):

    def __init__(self, k1, k_1, substrate_conc, product_conc):
        super().__init__(k1, k_1, substrate_conc, product_conc)
        assert self.substrate_conc.size == 1
        assert self.product_conc.size == 2
        """
        while isinstance(self.substrate_conc, Iterable):
            self.substrate_conc = self.substrate_conc[0]
        """
    def step(self, t):
        self.d_subs = (self.k_1 * self.product_conc[0] * self.product_conc[1] -
                       self.k1 * self.substrate_conc)
        
        if np.abs(self.d_subs) < self.step_eps:
            return
        n_0 = self.substrate_conc + self.product_conc[0]
        n_1 = self.substrate_conc + self.product_conc[1]
        delta = self.k1 * self.k1 / self.k_1 / self.k_1 + (n_0 - n_1) * (n_0 - n_1) + 2 * self.k1 / self.k_1 * (n_0 + n_1)
        subs_ss = (self.k1 / self.k_1 + n_0 + n_1 - np.sqrt(delta)) / 2
        subs_change = self.substrate_conc - subs_ss
        if np.abs(subs_change) >= self.step_eps:
            subs_change_degree = -self.d_subs / subs_change
        else:
            subs_change_degree = 0

        self.d_prod = -self.d_subs * np.ones_like(self.product_conc)
        prod_ss = self.substrate_conc + self.product_conc - subs_ss
        prod_change = self.product_conc - prod_ss
        # prod_change_degree = -self.d_prod / prod_change

        new_subs = subs_ss + subs_change * np.exp(-subs_change_degree * t)
        self.product_conc += np.ones((2,)) * (self.substrate_conc - new_subs)

        self.substrate_conc = new_subs
        # self.product_conc = prod_ss + prod_change * np.exp(-prod_change_degree * t)

