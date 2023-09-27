from abc import abstractmethod
from itertools import chain, islice, repeat
import numpy as np
from ..planardynamics.pendulum import Pendulum
from scipy.integrate import solve_ivp
from functools import partial
from numdifftools import Jacobian


def hom2xyphi(hom):
    fkin = np.empty((hom.shape[0], 3))
    fkin[:, 0] = hom[:, 0, 2]
    fkin[:, 1] = hom[:, 1, 2]
    fkin[:, 2] = np.angle(np.exp(1j * (np.arctan2(hom[:, 1, 0], hom[:, 0, 0]))))
    return fkin


class PlanarDynamicalSystem:
    def __init__(self, n_dof, l, m, g, k, qr, invdyn):
        self.dof = n_dof
        self._l = l
        self._m = m
        self._g = g
        self._k = k
        self._q_rest = qr
        self._p = l, m, g, k, qr
        self._has_inverse_dynamics = invdyn
        self._eq = None
        self._U0 = None
        self._gravity_jacobian = None
        self._dc_dq = None
        self._dc_ddq = None

    @property
    def link_lengths(self):
        return np.array(self._l)

    @property
    def masses(self):
        return np.array(self._m)

    @property
    def stiffness(self):
        return np.array(self._k)

    def rest_position(self):
        return np.array(self._q_rest)

    @property
    def params(self):
        return self._p

    def mass_matrix(self, q):
        raise NotImplementedError

    def gravity(self, q):
        raise NotImplementedError

    def coriolis_centrifugal_forces(self, q, dq):
        raise NotImplementedError

    def _ddq(self, q, dq, tau_in):
        raise NotImplementedError

    def _potential(self, q):
        raise NotImplementedError

    def _kinetic_energy(self, q, dq):
        raise NotImplementedError

    def _energy(self, q, dq):
        raise NotImplementedError

    def _link_positions(self, q):
        raise NotImplementedError

    def _fkin(self, q):
        raise NotImplementedError

    @property
    def equilibrium(self):
        if self._eq is None:
            self._eq = self.compute_equilibrium()
        return self._eq

    @property
    def U0(self):
        if self._U0 is None:
            self._U0 = self._potential(self.equilibrium)
        return self._U0

    def jacobian(self, q):
        raise NotImplementedError

    def jacobi_metric(self, q, E):
        raise NotImplementedError

    def endeffector_pose(self, q):
        raise NotImplementedError

    def forward_kinematics(self, q):
        if q.ndim == 1:
            return self._fkin(q).flatten()  # Warning recently added .flatten() here
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

    def internal_forces(self, q, dq):
        return -self.coriolis_centrifugal_forces(q, dq) - self.gravity(q)

    def create_dynamics(self, controllers=None):
        n = self.dof
        if controllers is None:
            controllers = []

        def ode(t, y):
            q, dq = y[0:n], y[n:]
            M = self.mass_matrix(q)
            ddq = np.linalg.inv(M) @ (
                sum((c(t, q, dq) for c in controllers), np.zeros(self.dof))
                - self.coriolis_centrifugal_forces(q, dq)
                - self.gravity(q)
                - self.elastic_forces(q)
            )
            return np.r_[dq, ddq]

        def ode_precomputed(t, y):
            q, dq = y[0:n], y[n:]
            tau = sum((c(t, q, dq) for c in controllers), np.zeros(self.dof))
            ddq = self._ddq(q, dq, tau).flatten()
            return np.r_[dq, ddq]

        return ode_precomputed if self._has_inverse_dynamics else ode

    def linearized_stiffness(self, q, step=0.1):
        if self._gravity_jacobian is None:
            self._gravity_jacobian = Jacobian(self.gravity, step=step)
        return np.diag(self._k) + self._gravity_jacobian(q)

    def linearize_numeric(self, q, dq=None, input=None):
        from numdifftools import Jacobian

        if dq is None:
            dq = np.zeros(self.dof)
        if input is None:
            input = np.zeros(self.dof)
        eom = self.create_dynamics()
        A = Jacobian(lambda y: eom(0, y), step=0.1)(np.r_[q, dq])

        def f(x):
            return self.create_dynamics(controllers=[lambda t, q, dq: x])(
                0, np.r_[q, dq]
            )

        B = Jacobian(f)(input)
        return A, B

    def linearize(self, q, dq=None):
        n = self.dof
        A = np.zeros((2 * n, 2 * n))
        A[0:n, n:] = np.eye(n)
        minv = np.linalg.inv(self.mass_matrix(q))
        A[n:, 0:n] = -minv @ self.linearized_stiffness(q)
        B = np.zeros((2 * n, n))
        B[n:, :] = minv
        if dq is not None:
            A[n:, 0:n] -= minv @ self.coriolis_centrifugal_forces_dq(q, dq)
            A[n:, n:] = -minv @ self.coriolis_centrifugal_forces_ddq(q, dq)

        return A, B

    def sim(
        self,
        q0,
        dq0,
        t_max,
        controllers=None,
        dt=None,
        dense=True,
        return_sol=False,
        verbose=False,
        wrap=False,
        **kwargs,
    ):
        sol = solve_ivp(
            fun=self.create_dynamics(controllers),
            t_span=(0, t_max),
            y0=np.r_[q0, dq0],
            method="LSODA",
            dense_output=dense,
            t_eval=np.arange(0, t_max, dt) if dt and dense else None,
            max_step=dt if dt else np.inf,
            **kwargs,
        )

        if verbose:
            print(f"Solve IVP finished with '{sol.message}'.")
            print(
                f"In total {len(sol.t)} time points were evaluated and the rhs was evaluated {sol.nfev} times."
            )

        traj = sol.y.T
        n = self.dof
        q = (
            np.arctan2(np.sin(traj[:, 0:n]), np.cos(traj[:, 0:n]))
            if wrap
            else traj[:, 0:n]
        )
        if return_sol:
            return sol.t, q, traj[:, n:], sol
        else:
            return sol.t, q, traj[:, n:]

    def final_state(self, q0, dq0, t, controllers=None, speed_factor=1.0):
        t, q, dq = self.sim(q0, dq0, speed_factor * t, controllers=controllers)
        return q[-1], dq[-1]

    def create_metric(self, E_total):
        return partial(self.jacobi_metric, E=E_total)

    def potential_energy(self, q, absolute=False):
        if q.ndim == 1:
            return self._potential(q) - (0 if absolute else self.U0)
        elif q.ndim == 2:
            return self._potential(q.T) - (0 if absolute else self.U0)
        else:
            raise ValueError

    def equipotential_line(self, E, n, d0=0.1):
        from ..planardynamics.equipotential_line import equipotential_line

        return equipotential_line(n, E, self.potential_energy, self.equilibrium, d0=d0)

    def energy(self, q, dq, absolute=False):
        if q.ndim == 1:
            return self._energy(q, dq) - (0 if absolute else self.U0)
        elif q.ndim == 2:
            return self._energy(q.T, dq.T) - (0 if absolute else self.U0)
        else:
            raise ValueError

    def M(self, q):
        return self.mass_matrix(q)

    def kinetic_energy(self, q, dq):
        if q.ndim == dq.ndim == 1:
            return self._kinetic_energy(q, dq)
        elif q.ndim == dq.ndim == 2:
            return self._kinetic_energy(q.T, dq.T)
        else:
            raise ValueError

    def create_convergence_check(self, eps=1e-3, terminal=True):
        n = self.dof

        def convergence_check(t, y):
            q, dq = y[0:n], y[n:]
            if t < 1:
                return np.inf
            else:
                return np.linalg.norm(dq) - eps

        convergence_check.terminal = terminal
        return convergence_check

    def find_velocity_for_energy(self, q, tangent, energy):
        alpha = np.sqrt(
            2
            * (energy - self.potential_energy(q))
            / (tangent.T @ self.mass_matrix(q) @ tangent)
        )
        return alpha * tangent

    def compute_equilibrium(self, q0=None):
        if q0 is None:
            q0 = np.zeros(self.dof)

        def viscous_damping(t, q, dq):
            return -2 * self.mass_matrix(q) @ dq

        _, q, _, sol = self.sim(
            q0=q0,
            dq0=np.zeros(self.dof),
            dt=1e-2,
            controllers=[viscous_damping],
            events=self.create_convergence_check(),
            t_max=50.0,
            return_sol=True,
        )

        if sol.status == 1:  # Termination event occured
            return q[-1, :]
        else:
            raise ValueError("No equilibrium found.")

    def bad_invkin(self, cart, q0=None, K=1.0, tol=1e-3, max_steps=100):
        if q0 is not None:
            q = q0
        else:
            q = np.random.uniform(-np.pi, np.pi, self.dof)

        if cart.size == 3:
            f = lambda x: self.forward_kinematics(x)
            if self.dof == 3:
                A = lambda x: np.linalg.inv(self.jacobian(x))
            else:
                A = lambda x: np.linalg.pinv(self.jacobian(x))
        elif cart.size == 2:
            f = lambda x: self.forward_kinematics(x)[0:2]
            A = lambda x: np.linalg.pinv(self.jacobian(x)[0:2, :])
        elif cart.size == 1:
            f = lambda x: self.forward_kinematics(x)[0:1]
            A = lambda x: np.linalg.pinv(self.jacobian(x)[0:1, :])
        else:
            raise ValueError("Illegal length of desired task-space pose")

        for _ in range(max_steps):
            e = cart - f(q)
            if np.linalg.norm(e) < tol:
                break
            try:
                inc = A(q) @ np.squeeze(K * e)
            except ValueError:
                inc = (A(q) * np.squeeze(K * e))[:, 0]
            q += inc
        else:
            raise AssertionError(f"No invkin solution found for {cart}.")

        return np.arctan2(np.sin(q), np.cos(q))

    def coriolis_centrifugal_forces_dq(self, q, dq):
        raise NotIImplementedError

    def coriolis_centrifugal_forces_ddq(self, q, dq):
        raise NotIImplementedError

    def elastic_forces(self, q):
        raise NotIImplementedError
