import numpy as np
from numpy import sin, cos
from .pendulum import Pendulum


class DoublePendulum(Pendulum):
    def __init__(self, m1=1, m2=1, l1=1, l2=1, lc1=.5, lc2=.5, II1=1, II2=1, g=0, extra_params=None, **kwargs):
        if extra_params is None:
            extra_params = {}
        self._alpha = II1 + II2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2)
        self._beta = m2 * l1 * lc2
        self._delta = II2 + m2 * lc2 ** 2

        super().__init__(
            lengths=(l1, l2),
            params={**dict(m1=m1, m2=m2, l1=l1, l2=l2, lc1=lc1, lc2=lc2, II1=II1, II2=II2, g=g), **extra_params},
        )

    def mass_matrix(self, q):
        alpha, beta, delta = self._alpha, self._beta, self._delta
        q1, q2 = q[0], q[1]

        return np.array([
            [alpha + 2 * beta * cos(q2), delta + beta * cos(q2)],
            [delta + beta * cos(q2), delta],
        ])

    def coriolis_centrifugal_forces(self, q, qd):
        q1, q2 = q[0], q[1]
        dq1, dq2 = qd[0], qd[1]
        m2 = self.get_param('m2')
        l1 = self.get_param('l1')
        lc2 = self.get_param('lc2')

        h = -m2 * l1 * lc2 * sin(q2)

        return np.array([
            [h * dq2, h * (dq1 + dq2)],
            [-h * dq1, 0],
        ]) @ qd

    def gravity(self, q):
        q1, q2 = q[0], q[1]
        m1 = self.get_param('m1')
        m2 = self.get_param('m2')
        l1 = self.get_param('l1')
        lc1 = self.get_param('lc1')
        lc2 = self.get_param('lc2')
        g = self.get_param('g')

        return np.array([
            (m1 * lc1 + m2 * l1) * g * cos(q1) + m2 * lc2 * g * cos(q1 + q2),
            m2 * lc2 * g * cos(q1 + q2),
        ])
