import qpsolvers
import numpy as np


def make_controller_conservative(controller):
    def conservative_controller(t, q, dq):
        n = len(q)
        return qpsolvers.solve_qp(
            P=np.eye(n),
            q=-2*controller(t, q, dq),
            A=dq,
            b=np.array([0]).reshape((1, 1)),
            solver='quadprog'
        )

    return conservative_controller
