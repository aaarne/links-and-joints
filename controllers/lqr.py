import numpy as np


def lqr(a, b, q, r):
    from scipy.linalg import solve_continuous_are
    s = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ s)


def ctrb(a, b):
    n = a.shape[0]
    return np.hstack(
        [b] + [np.linalg.matrix_power(a, i) @ b for i in range(1, n)]
    )


class UnderacuatedLQR:
    def __init__(self, q_desired, actuation_map, system, state_cost, control_cost, verbose=False):
        a, b_big = system.linearize(q_desired)
        b = b_big[:, actuation_map]
        c = ctrb(a, b)
        if (rc := np.linalg.matrix_rank(c)) < 2 * len(q_desired):
            raise ValueError(f"System is not controllable. Is {rc} instead of {2 * len(q_desired)}.")

        q = np.diag(state_cost)
        r = np.diag(control_cost)

        self._y_desired = np.r_[q_desired, np.zeros_like(q_desired)]
        self._k_f = lqr(a, b, q, r)
        self._actuation_map = actuation_map
        if verbose:
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print("Linearized System Matrix:")
                print(a)
                print()
                print(f"Rank of controllability matrix: {np.linalg.matrix_rank(c)}.")
                print(f"Closed-loop eigenvalues: {np.linalg.eigvals(a - b @ self._k_f)}")
                print(f"Feedback Gain:\n{self._k_f}")

    def __call__(self, t, q, dq):
        y = np.r_[q, dq]
        delta_y = y - self._y_desired
        tau = np.zeros_like(q)
        tau[self._actuation_map] = -self._k_f @ delta_y
        return tau