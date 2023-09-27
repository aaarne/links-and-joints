import numpy as np
from .utils import damping_design, transform_mass, weighted_pinv
from numdifftools import Derivative


class JointPDPlus:
    def __init__(self, trajectory_cb, robot, K, Zeta):
        self._traj = trajectory_cb
        self._robot = robot
        self._K = K
        self._Zeta = Zeta

    def __call__(self, t, q, dq):
        qd, dqd, ddqd = self._traj(t, q, dq)
        tau_grav = self._robot.gravity(q)

        M = self._robot.mass(q)
        tau_cc = self._robot.coriolis_centrifugal_forces(q, dqd)
        D = damping_design(M, self._K, self._Zeta)

        return tau_grav \
            + tau_cc \
            + M @ ddqd \
            - self._K @ (q - qd) \
            - D @ (dq - dqd)


class TaskPDPlus:
    def __init__(self, robot, fkin_cb, jacobian_cb, K, Zeta, trajectory_cb=None,
                 secondary_task=None):
        self._traj = trajectory_cb
        self._robot = robot
        self._fkin = fkin_cb
        self._K = K
        self._Zeta = Zeta
        self._jacobian = jacobian_cb
        self._xd = None
        self._dxd = None
        self._ddxd = None
        self._second = secondary_task

    def set_target(self, xd, dxd, ddxd):
        if self._traj is not None:
            raise ValueError("Cannot set target when trajectory callback is set")
        self._xd = xd
        self._dxd = dxd
        self._ddxd = ddxd

    def __call__(self, t, q, dq):
        if self._traj is None:
            xd, dxd, ddxd = self._xd, self._dxd, self._ddxd
        else:
            xd, dxd, ddxd = self._traj(t, q, dq)

        x = self._fkin(q)

        J = self._jacobian(q)
        dx = J @ dq

        M = self._robot.mass_matrix(q)
        M_x = transform_mass(M, J)
        D_x = damping_design(M_x, self._K, self._Zeta)

        J_pinv = weighted_pinv(J, np.linalg.inv(M))
        J_dot = Derivative(lambda t: self._jacobian(q + t * dq))(0)

        f = M_x @ ddxd \
            - M_x @ J_dot @ J_pinv @ dxd \
            - self._K @ (x - xd) \
            - D_x @ (dx - dxd)

        tau_plus = self._robot.gravity(q) + self._robot.coriolis_centrifugal_forces(q, dq)

        if self._second is not None:
            P = np.eye(len(q)) - J_pinv @ J
            tau_second = P.T @ self._second(t, q, dq)
        else:
            tau_second = np.zeros_like(q)

        return tau_plus + J.T @ f + tau_second
