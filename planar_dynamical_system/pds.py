from abc import abstractmethod
from itertools import chain, islice, repeat
import numpy as np
from ..planardynamics.pendulum import Pendulum
from scipy.integrate import solve_ivp


def hom2xyphi(hom):
    fkin = np.empty((hom.shape[0], 3))
    fkin[:, 0] = hom[:, 0, 2]
    fkin[:, 1] = hom[:, 1, 2]
    fkin[:, 2] = np.angle(np.exp(1j * (np.arctan2(hom[:, 1, 0], hom[:, 0, 0]))))
    return fkin


class PlanarDynamicalSystem:
    def __init__(self, n_dof, l, m, g, k, qr):
        self.dof = n_dof
        self._l = l
        self._m = m
        self._g = g
        self._k = k
        self._q_rest = qr
        self._p = l, m, g, k, qr

    @property
    def params(self):
        return self._p

    def mass_matrix(self, q):
        raise NotImplementedError

    def gravity(self, q):
        raise NotImplementedError

    def coriolis_centrifugal_forces(self, q, dq):
        raise NotImplementedError

    def potential(self, q):
        raise NotImplementedError

    def kinetic_energy(self, q, dq):
        raise NotImplementedError

    def energy(self, q, dq):
        raise NotImplementedError

    def _link_positions(self, q):
        raise NotImplementedError

    def _fkin(self, q):
        raise NotImplementedError

    def jacobian(self, q):
        raise NotImplementedError

    def jacobi_metric(self, q, E):
        raise NotImplementedError

    def endeffector_pose(self, q):
        raise NotImplementedError

    def forward_kinematics(self, q):
        if q.ndim == 1:
            return self._fkin(q)
        elif q.ndim == 2:
            return self._fkin(q.T).T.squeeze()

    def forward_kinematics_for_each_link(self, q):
        if q.ndim == 1:
            return hom2xyphi(self._link_positions(q).reshape((-1, 3, 3)))
        elif q.ndim == 2:
            out = np.empty((q.shape[0], self.dof, 3))
            for i in range(q.shape[0]):
                out[i, :, :] = hom2xyphi(self._link_positions(q[i]).reshape((-1, 3, 3)))
            return out
        else:
            raise ValueError

    def sim(self, q0, dq0, dt, t_max,
            controllers=None,
            events=None,
            verbose=False):
        n = self.dof
        if controllers is None:
            controllers = []
        if events is None:
            events = []

        def ode(t, y):
            q, dq = y[0:n], y[n:]
            M = self.mass_matrix(q)
            ddq = np.linalg.inv(M) @ (
                    sum((c(t, q, dq) for c in controllers), np.zeros(self.dof))
                    - self.coriolis_centrifugal_forces(q, dq)
                    - self.gravity(q)
            )
            return np.r_[dq, ddq]

        sol = solve_ivp(
            fun=ode,
            t_span=(0, t_max),
            y0=np.r_[q0, dq0],
            method='RK45',
            dense_output=True,
            events=events,
            t_eval=np.arange(0, t_max, dt),
            max_step=dt,
        )

        if verbose:
            print(f"Solve IVP finished with '{sol.message}'.")
            print(f"In total {len(sol.t)} time points were evaluated and the rhs was evaluated {sol.nfev} times.")

        traj = sol.y.T
        return sol.t, traj[:, 0:n], traj[:, n:]