from abc import abstractmethod
from itertools import chain, islice, repeat


class PlanarDynamicalSystem:
    def __init__(self, n_dof):
        self.dof = n_dof

    def mass_matrix(self, q): raise NotImplementedError

    def gravity(self, q): raise NotImplementedError

    def cc_forces(self, q, dq): raise NotImplementedError

    def potential(self, q): raise NotImplementedError

    def kinetic_energy(self, q, dq): raise NotImplementedError

    def energy(self, q, dq): raise NotImplementedError

    def link_positions(self, q): raise NotImplementedError

    def fkin(self, q): raise NotImplementedError

    def jacobian(self, q): raise NotImplementedError

    def jacobi_metric(self, q, E): raise NotImplementedError

