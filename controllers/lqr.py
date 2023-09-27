import numpy as np


def lqr(a, b, q, r):
    from scipy.linalg import solve_continuous_are

    s = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ s)


def ctrb(a, b):
    n = a.shape[0]
    return np.hstack([b] + [np.linalg.matrix_power(a, i) @ b for i in range(1, n)])


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def angular_difference(q1, q2):
    return wrap(q1 - q2)

def configuration_wrap(y):
    return np.r_[y[0], wrap(y[1:3])]

def configuration_distance(y1, y2):
    return np.r_[y1[0] - y2[0], angular_difference(y1[1:3], y2[1:3])]


class UnderactuatedLQR:
    def __init__(
        self, q_desired, actuation_map, system, state_cost, control_cost, verbose=False,
    ):
        a, b_big = system.linearize(q_desired, dq=np.zeros_like(q_desired))
        self._q_lin = q_desired
        self._system = system
        b = b_big[:, actuation_map]
        c = ctrb(a, b)
        if (rc := np.linalg.matrix_rank(c)) < 2 * len(q_desired):
            raise ValueError(
                f"System is not controllable. Is {rc} instead of {2 * len(q_desired)}."
            )

        self._q = np.diag(state_cost)
        self._r = np.diag(control_cost)

        self._y_desired = np.r_[q_desired, np.zeros_like(q_desired)]
        self._actuation_map = actuation_map
        self.update_feedback_gain(q_desired)

        self._verbose = verbose

        if verbose:
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print("Linearized System Matrix:")
                print(a)
                print()
                print(f"Rank of controllability matrix: {np.linalg.matrix_rank(c)}.")
                print(
                    f"Closed-loop eigenvalues: {np.linalg.eigvals(a - b @ self._k_f)}"
                )
                print(f"Feedback Gain:\n{self._k_f}")

    def update_feedback_gain(self, q, dq=None):
        self._q_lin = q
        a, b_big = self._system.linearize(q, dq=dq)
        b = b_big[:, self._actuation_map]
        if np.linalg.svd(ctrb(a, b), compute_uv=False)[-1] < 1e-3:
            raise ValueError("System is not controllable")
        self._k_f = lqr(a, b, self._q, self._r)

    def set_reference(self, q_desired, dq_desired=None):
        if dq_desired is None:
            dq_desired = np.zeros_like(q_desired)
        self._y_desired = np.r_[q_desired, dq_desired]

    def __call__(self, t, q, dq):
        y = np.r_[q, dq]
        delta_y = y - self._y_desired
        delta_y[:3] = configuration_distance(y[:3], self._y_desired[:3])
        # print(f"t={t:.3f}\t{delta_y}")
        tau = np.zeros_like(q)
        tau[self._actuation_map] = -self._k_f @ delta_y
        return tau
    

class NonlinearLQR(UnderactuatedLQR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, t, q, dq):
        self.update_feedback_gain(q)
        return super().__call__(t, q, dq)


class GainSchedulingUnderactuatedLQR(UnderactuatedLQR):
    def __init__(
        self,
        thresholds,
        **kwargs,
    ):
        self._thresholds = thresholds
        super().__init__(**kwargs)

    def __call__(self, t, q, dq):
        if np.any(np.abs(q - self._q_lin) > self._thresholds):
            if self._verbose:
                print(f"t={t:.3f}\tSwitching linearization point to {q}, {dq}")
            try:
                self.update_feedback_gain(q)
                self._q_lin = q
            except ValueError:
                pass
        return super().__call__(t, q, dq)
