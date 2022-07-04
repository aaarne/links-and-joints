from sympy import *
from functools import reduce

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

trafos = {
    'x': create_x_trafo,
    'y': create_y_trafo,
    'r': create_rot_trafo,
}


def generate_planar_model(spec):
    """spec is a string x?y?r*
       example 'xr' is a single pendulum on a cart in x direction"""
    pass
