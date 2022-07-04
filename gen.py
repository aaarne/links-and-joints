from sympy import *
from sympy.physics.mechanics import LagrangesMethod
from functools import reduce
from itertools import islice
from sympy.printing.numpy import NumPyPrinter

do_simplify = True


def create_rot_trafo(q, l):
    return Matrix([
        [cos(q), -sin(q), l * cos(q)],
        [sin(q), cos(q), l * sin(q)],
        [0, 0, 1]
    ])


def create_x_trafo(q, l):
    return Matrix([
        [1, 0, q],
        [0, 1, 0],
        [0, 0, 1]
    ])


def create_y_trafo(q, l):
    return Matrix([
        [1, 0, 0],
        [0, 1, q],
        [0, 0, 1]
    ])


trafo_funs = {
    'x': create_x_trafo,
    'y': create_y_trafo,
    'r': create_rot_trafo,
}


def generate_planar_model(spec):
    """spec is a string x?y?r*
       example 'xr' is a single pendulum on a cart in x direction"""
    n = len(spec)
    t = Symbol('t')
    q = [Function(f'q{i}')(t) for i in range(n)]
    l = [Symbol(f'l[{i}]') for i in range(n)]
    m = [Symbol(f'm[{i}]') for i in range(n)]
    qr = [Symbol(f'qr[{i}]') for i in range(n)]
    k = [Symbol(f'k[{i}]') for i in range(n)]
    g = Symbol('g')

    trafos = [trafo_funs[s](qi, li) for s, qi, li in zip(spec, q, l)]

    def com(i):
        return reduce(lambda x, y: x @ y, trafos[0:i])

    coms = [com(i + 1) for i in range(n)]
    tcp = coms[-1]

    fkin = Matrix([
        tcp[0, 2],
        tcp[1, 2],
        atan2(tcp[1, 0], tcp[0, 0]),
    ])

    def compute_kinetic_energy(i):
        vel_sqr = coms[i][0, 2].diff(t) ** 2 + coms[i][1, 2].diff(t) ** 2
        return 0.5 * m[i] * vel_sqr

    def compute_potential_energy(i):
        return m[i] * g * coms[i][1, 2] + .5*k[i]*(q[i]-qr[i])**2

    T = sum(compute_kinetic_energy(i) for i in range(n))
    V = sum(compute_potential_energy(i) for i in range(n))
    L = T - V

    print("Generating EoM...")
    lm = LagrangesMethod(L, q)
    lm.form_lagranges_equations()
    print("done.\n")

    m = lm.mass_matrix
    forcing = lm.forcing

    g = Matrix([-collect(forcing[i], g).coeff(g) * g for i in range(n)])
    cc = -forcing - g

    J = Matrix([diff(fkin, qi).T for qi in q]).T

    jacobi_metric = 2*(Symbol('E') - V)*m

    def expr_to_code(expr, key=None):
        print(f"Generating {key}.")
        npq = [Symbol(f'q[{i}]') for i in range(n)]
        npdq = [Symbol(f'dq[{i}]') for i in range(n)]

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

        return NumPyPrinter().doprint(
            replace_q(
                replace_dq(
                    expr, 0), 0))

    print("Generating code...")
    return f"""import numpy
from ..pds import PlanarDynamicalSystem


class {spec.upper()}(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__({n}, l, m, g, k, qr)
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(m, 'Mass Matrix')}

    def gravity(self, q):
        l, m, g, k, qr = self.params
        expr = {expr_to_code(g, 'Gravity Vector')}
        return expr.flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        expr = {expr_to_code(cc, 'Coriolis & Centrifugal Forces')}
        return expr.flatten()
        
    def potential(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(V, 'Potential')}
        
    def kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        return {expr_to_code(T, 'Kinetic Energy')}
        
    def energy(self, q, dq):
        l, m, g, k, qr = self.params
        return {expr_to_code(V + T, 'Total Energy')}
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(Matrix(coms), 'CoM Positions')}
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(fkin, 'Forward Kinematic')}
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(tcp, 'TCP Pose')}

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        return {expr_to_code(J, 'Jacobian')}
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        return {expr_to_code(jacobi_metric, 'Jacobi Metric')}
"""


if __name__ == "__main__":
    import sys
    spec = sys.argv[1]
    with open(f"planar_dynamical_system/generated/{spec}.py", "w") as f:
        f.write(generate_planar_model(spec))

