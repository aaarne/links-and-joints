import numpy
from sympy import (
    Matrix,
    Symbol,
    Function,
    Array,
    atan2,
    diff,
    collect,
    Derivative,
    ImmutableDenseNDimArray,
    cos,
    sin,
    simplify,
)
from sympy.physics.mechanics import LagrangesMethod
from functools import reduce
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from allerlei.timer import Timer


def create_rot_trafo(q, l):
    return Matrix(
        [[cos(q), -sin(q), l * cos(q)], [sin(q), cos(q), l * sin(q)], [0, 0, 1]]
    )


def create_x_trafo(q, _):
    return Matrix([[1, 0, q], [0, 1, 0], [0, 0, 1]])


def create_y_trafo(q, _):
    return Matrix([[1, 0, 0], [0, 1, q], [0, 0, 1]])


trafo_funs = {
    "x": create_x_trafo,
    "y": create_y_trafo,
    "r": create_rot_trafo,
}


def generate_planar_model(
    spec,
    do_simplify=False,
    compute_metric=False,
    inverse_dynamics=False,
    cc_derivatives=False,
):
    """spec is a string x?y?r*
    example 'xr' is a single pendulum on a cart in x direction"""
    n = len(spec)
    t = Symbol("t")
    q = [Function(f"q{i}")(t) for i in range(n)]
    l = [Symbol(f"l[{i}]") for i in range(n)]
    m = [Symbol(f"m[{i}]") for i in range(n)]
    qr = [Symbol(f"qr[{i}]") for i in range(n)]
    k = [Symbol(f"k[{i}]") for i in range(n)]
    tau_in = [Symbol(f"tau_in[{i}]") for i in range(n)]
    g = Symbol("g")

    printer = SciPyPrinter()

    trafos = [trafo_funs[s](qi, li) for s, qi, li in zip(spec, q, l)]

    def com(i):
        return reduce(lambda x, y: x @ y, trafos[0:i])

    coms = Array([Array(com(i + 1)) for i in range(n)])

    tcp = coms[-1].tomatrix()

    fkin = Matrix(
        [
            tcp[0, 2],
            tcp[1, 2],
            atan2(tcp[1, 0], tcp[0, 0]),
        ]
    )

    def compute_kinetic_energy(i):
        vel_sqr = coms[i][0, 2].diff(t) ** 2 + coms[i][1, 2].diff(t) ** 2
        return 0.5 * m[i] * vel_sqr

    def compute_potential_energy(i):
        return m[i] * g * coms[i][1, 2] + 0.5 * k[i] * (q[i] - qr[i]) ** 2

    T = sum(compute_kinetic_energy(i) for i in range(n))
    V = sum(compute_potential_energy(i) for i in range(n))
    L = T - V

    with Timer("Generating EoM..."):
        lm = LagrangesMethod(L, q)
        lm.form_lagranges_equations()

    m = lm.mass_matrix
    forcing = lm.forcing

    g = Matrix([-collect(forcing[i], g).coeff(g) * g for i in range(n)])
    elastic_forces = simplify(Matrix([-collect(forcing[i], k[i]).coeff(k[i]) * k[i] for i in range(n)]))
    cc = simplify(-forcing - g - elastic_forces)

    def create_inverse_dynamics():
        taum = Matrix(tau_in)
        return m.inv() @ (taum + forcing)

    J = Matrix([diff(fkin, qi).T for qi in q]).T

    def cc_dq():
        return Matrix([diff(cc, qi).T for qi in q]).T

    def cc_ddq():
        return Matrix([diff(cc, qi.diff(t)).T for qi in q]).T

    def create_jacobi_metric():
        return 2 * (Symbol("E") - V) * m

    def expr_to_code(expr, key=None):
        with Timer(f"Generating {key}."):
            npq = [Symbol(f"q[{i}]") for i in range(n)]
            npdq = [Symbol(f"dq[{i}]") for i in range(n)]

            def replace_q(expr, i):
                if i == n:
                    return expr
                else:
                    return replace_q(expr.subs(q[i], npq[i]), i + 1)

            def replace_dq(expr, i):
                if i == n:
                    return expr
                else:
                    d = Derivative(q[i], t)
                    return replace_dq(expr.subs(d, npdq[i]), i + 1)

            if do_simplify:
                expr = simplify(expr)

            expr = replace_q(replace_dq(expr, 0), 0)

            if type(expr) is ImmutableDenseNDimArray:
                shape = expr.shape
                a, b = reduce(lambda x, y: x * y, shape[0:-1], 1), shape[-1]
                expr = expr.reshape(a, b).tomatrix()
                return f"return {printer.doprint(expr)}.reshape({shape})"
            else:
                return f"return {printer.doprint(expr)}"

    def create_code_optionally(option, gen_expr, key=None):
        if option:
            return f"{expr_to_code(gen_expr(), key)}"
        else:
            return f"raise NotImplementedError"

    print("Generating code...")
    return f"""import numpy
from ..pds import PlanarDynamicalSystem


class {spec.upper()}(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__({n}, l, m, g, k, qr, {inverse_dynamics})
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(m, 'Mass Matrix')}

    def gravity(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(g, 'Gravity Vector')}.flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        {expr_to_code(cc, 'Coriolis & Centrifugal Forces')}.flatten()

    def elastic_forces(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(elastic_forces, 'Elastic Forces')}.flatten()
        
    def _potential(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(V, 'Potential')}
        
    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr = self.params
        {create_code_optionally(inverse_dynamics, create_inverse_dynamics, 'Inverse Dynamics')} 
        
    def _kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        {expr_to_code(T, 'Kinetic Energy')}
        
    def _energy(self, q, dq):
        l, m, g, k, qr = self.params
        {expr_to_code(V + T, 'Total Energy')}
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(coms, 'CoM Positions')}
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(fkin, 'Forward Kinematic')}
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(tcp, 'TCP Pose')}

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        {expr_to_code(J, 'Jacobian')}
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        {create_code_optionally(compute_metric, create_jacobi_metric, 'Jacobi Metric')}

    def coriolis_centrifugal_forces_dq(self, q, dq):
        l, m, g, k, qr = self.params
        {create_code_optionally(cc_derivatives, cc_dq, 'Coriolis & Centrifugal Forces Jacobian dq')}

    def coriolis_centrifugal_forces_ddq(self, q, dq):
        l, m, g, k, qr = self.params
        {create_code_optionally(cc_derivatives, cc_ddq, 'Coriolis & Centrifugal Forces Jacobian ddq')}
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Planar Dynamical Model")
    parser.add_argument("spec", type=str, help="Specification. [xyr]*")
    parser.add_argument("-s", "--simplify", action="store_true")
    parser.add_argument("-m", "--metric", action="store_true")
    parser.add_argument("-i", "--inverse_dynamics", action="store_true")
    parser.add_argument("-c", "--cc_derivatives", action="store_true")
    parser.set_defaults(simplify=False, metric=False, inverse_dynamics=False)
    args = parser.parse_args()
    with open(f"planar_dynamical_system/generated/{args.spec}.py", "w") as f:
        f.write(
            generate_planar_model(
                args.spec,
                args.simplify,
                args.metric,
                args.inverse_dynamics,
                args.cc_derivatives,
            )
        )
