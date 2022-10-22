import numpy as np
from functools import reduce
from . import DoublePendulum
from .equipotential_line import equipotential_line
from .pendulum import *

class ElasticDoublePendulum(DoublePendulum):
    def __init__(self,
                 K=None,
                 q_rest=None,
                 m1=0.4,
                 m2=0.4,
                 l1=1,
                 l2=1,
                 lc1=None,
                 lc2=None,
                 II1=0,
                 II2=0,
                 g=9.81
                 ):
        if lc1 is None:
            lc1 = l1
        if lc2 is None:
            lc2 = l2
        self._masses = np.array([m1, m2])
        self._K = np.diag([0, 0]) if K is None else K
        self._q_rest = np.array([0, np.pi / 2]) if q_rest is None else q_rest
        super().__init__(m1=m1, m2=m2, l1=l1, l2=l2, lc1=lc1, lc2=lc2, II1=II1, II2=II2, g=g, extra_params=dict(
            K=self._K,
            q_rest=self._q_rest,
        ))
        self.add_torque_source(self.elastic_torque_function())
        self._eq = None
        self._p0 = None

    @property
    def stiffness(self):
        return self._K

    @stiffness.setter
    def stiffness(self, K):
        self._K = K

    def elastic_torque_function(self):
        def f(_, q, dq):
            return -self._K @ (q - self._q_rest)

        return f

    def potential_energy(self, q, absolute=False):
        if len(q.shape) == 1:
            pot = 0.0
            t = np.eye(3)
            for i, (m, nt) in enumerate(zip(self._masses, self.link_trafos(q))):
                p1 = t[0:2, 2]
                t = t @ nt[0, :, :]
                p2 = t[0:2, 2]
                com = p1 + (p2 - p1) * (self.get_param(f"lc{i + 1}") / self.get_param(f"l{i + 1}"))
                pot += m * self.get_param('g') * com[1]
            pot += 0.5 * (q - self._q_rest).T @ self._K @ (q - self._q_rest)
            if absolute:
                return pot
            else:
                return pot - self.minimal_potential
        elif len(q.shape) == 2:
            p = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                p[i] = self.potential_energy(q[i, :])
            return p
        else:
            raise ValueError

    def equipotential_line(self, E, n, d0=.1):
        return equipotential_line(n, E, self.potential_energy, self.equilibrium, d0=d0)

    def compute_equilibrium(self, q0=None, entire_trajectory=False):
        def viscous_damping(_, q, dq):
            return -.5 * dq

        _, traj, _ = self.sim(
            q0=np.zeros(self.dof) if q0 is None else q0,
            dq0=np.zeros(self.dof),
            controllers=[viscous_damping],
            stop_integration_observer=convergence_checker(self),
            dt=1e-1,
        )
        if entire_trajectory:
            return traj
        else:
            return traj[-1, 0::2]

    @property
    def equilibrium(self):
        if self._eq is None:
            self._eq = self.compute_equilibrium()
        return self._eq

    @property
    def minimal_potential(self):
        if self._p0 is None:
            self._p0 = self.potential_energy(self.equilibrium, absolute=True)
        return self._p0

    def total_energy(self, q, dq):
        return self.kinetic_energy(q, dq) + self.potential_energy(q)

    def create_metric(self, E_max):
        def met(q, pd=False):
            if len(q.shape) == 1:
                if pd or (np.abs(self.get_param('g')) < 1e-12 and np.trace(self._K) < 1e-12):
                    scaling = 1
                else:
                    scaling = 2 * (E_max - self.potential_energy(q))
                return scaling * self.M(q)
            elif len(q.shape) == 2:
                m = np.zeros((q.shape[0], 2, 2))
                for i in range(q.shape[0]):
                    m[i, :, :] = met(q[i, :], pd=pd)
                    # assert np.all(np.linalg.eigvals(m[i, :, :]) > 1e-6), "Metric is singular, indefinite or negative definite"
                return m
            else:
                raise ValueError

        return met

    def convert_to_new_class(self):
        from ..planar_dynamical_system import DoublePendulum
        p = self.params
        return DoublePendulum(
            l=np.array([p['l1'], p['l2']]),
            m=np.array([p['m1'], p['m2']]),
            g=p['g'],
            k=np.diag(self._K),
            qr=self._q_rest,
        )


