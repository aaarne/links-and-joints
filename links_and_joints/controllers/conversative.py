import qpsolvers
import numpy as np


def make_controller_conservative(controller):
    def conservative_controller(t, q, dq):
        if np.linalg.norm(dq) < 1e-3:
            return controller(t, q, dq)
        else:
            return qpsolvers.solve_qp(
                P=np.eye(len(q)),
                q=-2*controller(t, q, dq),
                A=dq,
                b=np.array([0.0]),
                solver='cvxopt',
            )

    return conservative_controller
