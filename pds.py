from abc import abstractmethod
from itertools import chain, islice, repeat
import numpy as np


class PlanarDynamicalSystem:
    def __init__(self, n_dof):
        self.dof = n_dof
        self._torque_sources = []

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

    def add_torque_source(self, torque_source):
        self._torque_sources.append(torque_source)

    def _custom_torque(self, t, q, dq):
        if len(self._torque_sources) > 0:
            return np.sum([c(t, q, dq) for c in self._torque_sources], axis=0)
        else:
            return np.zeros(self.dof)
