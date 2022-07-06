import numpy
from ..pds import PlanarDynamicalSystem


class RY(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__(2, l, m, g, k, qr, False)
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[0]**2*m[0] + m[1]*(l[0]**2 + q[1]**2), 1.0*l[0]*m[1]], [1.0*l[0]*m[1], 1.0*m[1]]])

    def gravity(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[g*(l[0]*m[0]*numpy.cos(q[0]) + m[1]*(l[0]*numpy.cos(q[0]) - q[1]*numpy.sin(q[0])))], [g*m[1]*numpy.cos(q[0])]]).flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[2.0*dq[0]*dq[1]*m[1]*q[1] + 1.0*k[0]*q[0] - 1.0*k[0]*qr[0]], [-1.0*dq[0]**2*m[1]*q[1] + 1.0*k[1]*q[1] - 1.0*k[1]*qr[1]]]).flatten()
        
    def potential(self, q):
        l, m, g, k, qr = self.params
        return g*l[0]*m[0]*numpy.sin(q[0]) + g*m[1]*(l[0]*numpy.sin(q[0]) + q[1]*numpy.cos(q[0])) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2
        
    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr = self.params
        raise NotImplementedError 
        
    def kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*l[0]**2*m[0] + 0.5*m[1]*(dq[0]**2*l[0]**2 + dq[0]**2*q[1]**2 + 2*dq[0]*dq[1]*l[0] + dq[1]**2)
        
    def energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*l[0]**2*m[0] + g*l[0]*m[0]*numpy.sin(q[0]) + g*m[1]*(l[0]*numpy.sin(q[0]) + q[1]*numpy.cos(q[0])) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2 + 0.5*m[1]*(dq[0]**2*l[0]**2 + dq[0]**2*q[1]**2 + 2*dq[0]*dq[1]*l[0] + dq[1]**2)
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[0]), -numpy.sin(q[0]), l[0]*numpy.cos(q[0])], [numpy.sin(q[0]), numpy.cos(q[0]), l[0]*numpy.sin(q[0])], [0, 0, 1], [numpy.cos(q[0]), -numpy.sin(q[0]), l[0]*numpy.cos(q[0]) - q[1]*numpy.sin(q[0])], [numpy.sin(q[0]), numpy.cos(q[0]), l[0]*numpy.sin(q[0]) + q[1]*numpy.cos(q[0])], [0, 0, 1]]).reshape((2, 3, 3))
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[0]*numpy.cos(q[0]) - q[1]*numpy.sin(q[0])], [l[0]*numpy.sin(q[0]) + q[1]*numpy.cos(q[0])], [numpy.arctan2(numpy.sin(q[0]), numpy.cos(q[0]))]])
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[0]), -numpy.sin(q[0]), l[0]*numpy.cos(q[0]) - q[1]*numpy.sin(q[0])], [numpy.sin(q[0]), numpy.cos(q[0]), l[0]*numpy.sin(q[0]) + q[1]*numpy.cos(q[0])], [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[-l[0]*numpy.sin(q[0]) - q[1]*numpy.cos(q[0]), -numpy.sin(q[0])], [l[0]*numpy.cos(q[0]) - q[1]*numpy.sin(q[0]), numpy.cos(q[0])], [1, 0]])
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        raise NotImplementedError
