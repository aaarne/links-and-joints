import numpy
from ..pds import PlanarDynamicalSystem


class XR(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__(2, l, m, g, k, qr, False)
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1.0*m[0] + 1.0*m[1], -1.0*l[1]*m[1]*numpy.sin(q[1])], [-1.0*l[1]*m[1]*numpy.sin(q[1]), 1.0*l[1]**2*m[1]]])

    def gravity(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[0], [g*l[1]*m[1]*numpy.cos(q[1])]]).flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[-1.0*dq[1]**2*l[1]*m[1]*numpy.cos(q[1]) - 1.0*k[0]*(-q[0] + qr[0])], [1.0*k[1]*(q[1] - qr[1])]]).flatten()
        
    def potential(self, q):
        l, m, g, k, qr = self.params
        return g*l[1]*m[1]*numpy.sin(q[1]) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2
        
    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr = self.params
        raise NotImplementedError 
        
    def kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*m[0] + 0.5*m[1]*(dq[0]**2 - 2*dq[0]*dq[1]*l[1]*numpy.sin(q[1]) + dq[1]**2*l[1]**2)
        
    def energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*m[0] + g*l[1]*m[1]*numpy.sin(q[1]) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2 + 0.5*m[1]*(dq[0]**2 - 2*dq[0]*dq[1]*l[1]*numpy.sin(q[1]) + dq[1]**2*l[1]**2)
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1, 0, q[0]], [0, 1, 0], [0, 0, 1], [numpy.cos(q[1]), -numpy.sin(q[1]), l[1]*numpy.cos(q[1]) + q[0]], [numpy.sin(q[1]), numpy.cos(q[1]), l[1]*numpy.sin(q[1])], [0, 0, 1]]).reshape((2, 3, 3))
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[1]*numpy.cos(q[1]) + q[0]], [l[1]*numpy.sin(q[1])], [numpy.arctan2(numpy.sin(q[1]), numpy.cos(q[1]))]])
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[1]), -numpy.sin(q[1]), l[1]*numpy.cos(q[1]) + q[0]], [numpy.sin(q[1]), numpy.cos(q[1]), l[1]*numpy.sin(q[1])], [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1, -l[1]*numpy.sin(q[1])], [0, l[1]*numpy.cos(q[1])], [0, 1]])
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        raise NotImplementedError
