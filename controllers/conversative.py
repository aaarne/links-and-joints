import qpsolvers
import numpy as np


def make_controller_conservative(controller):
    def conservative_controller(t, q, dq):
        n = len(q)
        return qpsolvers.solve_qp(
            P=np.eye(n),
            q=-2*controller(t, q, dq).reshape((1, n)),
            A=dq.reshape((1, n)),
            b=np.array([0]),
        )

    return conservative_controller
