import numpy
from ..pds import PlanarDynamicalSystem


class RR(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__(2, l, m, g, k, qr, True)
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[0]**2*m[0] + m[1]*(l[0]**2 + 2*l[0]*l[1]*numpy.cos(q[1]) + l[1]**2), 1.0*l[1]*m[1]*(l[0]*numpy.cos(q[1]) + l[1])], [1.0*l[1]*m[1]*(l[0]*numpy.cos(q[1]) + l[1]), 1.0*l[1]**2*m[1]]])

    def gravity(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[g*(l[0]*m[0]*numpy.cos(q[0]) + m[1]*(l[0]*numpy.cos(q[0]) + l[1]*numpy.cos(q[0] + q[1])))], [g*l[1]*m[1]*numpy.cos(q[0] + q[1])]]).flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[-dq[1]*l[0]*l[1]*m[1]*(2*dq[0] + dq[1])*numpy.sin(q[1])], [dq[0]**2*l[0]*l[1]*m[1]*numpy.sin(q[1])]]).flatten()

    def elastic_forces(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1.0*k[0]*(q[0] - qr[0])], [1.0*k[1]*(q[1] - qr[1])]]).flatten()
        
    def _potential(self, q):
        l, m, g, k, qr = self.params
        return g*l[0]*m[0]*numpy.sin(q[0]) + g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2
        
    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr = self.params
        return numpy.array([[1.0*(0.5*dq[0]**2*l[0]**2*l[1]*m[1]*numpy.sin(2*q[1]) + 1.0*dq[0]**2*l[0]*l[1]**2*m[1]*numpy.sin(q[1]) + 2.0*dq[0]*dq[1]*l[0]*l[1]**2*m[1]*numpy.sin(q[1]) + 1.0*dq[1]**2*l[0]*l[1]**2*m[1]*numpy.sin(q[1]) - 1.0*g*l[0]*l[1]*m[0]*numpy.cos(q[0]) - 0.5*g*l[0]*l[1]*m[1]*numpy.cos(q[0]) + 0.5*g*l[0]*l[1]*m[1]*numpy.cos(q[0] + 2*q[1]) - 1.0*k[0]*l[1]*q[0] + 1.0*k[0]*l[1]*qr[0] + 1.0*k[1]*l[0]*q[1]*numpy.cos(q[1]) - 1.0*k[1]*l[0]*qr[1]*numpy.cos(q[1]) + 1.0*k[1]*l[1]*q[1] - 1.0*k[1]*l[1]*qr[1] - 1.0*l[0]*tau_in[1]*numpy.cos(q[1]) + 1.0*l[1]*tau_in[0] - 1.0*l[1]*tau_in[1])/(l[0]**2*l[1]*(m[0] + m[1]*numpy.sin(q[1])**2))], [2.0*(-1.0*dq[0]**2*l[0]**3*l[1]*m[0]*m[1]*numpy.sin(q[1]) - 1.0*dq[0]**2*l[0]**3*l[1]*m[1]**2*numpy.sin(q[1]) - 1.0*dq[0]**2*l[0]**2*l[1]**2*m[1]**2*numpy.sin(2*q[1]) - 1.0*dq[0]**2*l[0]*l[1]**3*m[1]**2*numpy.sin(q[1]) - 1.0*dq[0]*dq[1]*l[0]**2*l[1]**2*m[1]**2*numpy.sin(2*q[1]) - 2.0*dq[0]*dq[1]*l[0]*l[1]**3*m[1]**2*numpy.sin(q[1]) - 0.5*dq[1]**2*l[0]**2*l[1]**2*m[1]**2*numpy.sin(2*q[1]) - 1.0*dq[1]**2*l[0]*l[1]**3*m[1]**2*numpy.sin(q[1]) + 0.5*g*l[0]**2*l[1]*m[0]*m[1]*numpy.cos(q[0] - q[1]) - 0.5*g*l[0]**2*l[1]*m[0]*m[1]*numpy.cos(q[0] + q[1]) + 0.5*g*l[0]**2*l[1]*m[1]**2*numpy.cos(q[0] - q[1]) - 0.5*g*l[0]**2*l[1]*m[1]**2*numpy.cos(q[0] + q[1]) + 1.0*g*l[0]*l[1]**2*m[0]*m[1]*numpy.cos(q[0]) + 0.5*g*l[0]*l[1]**2*m[1]**2*numpy.cos(q[0]) - 0.5*g*l[0]*l[1]**2*m[1]**2*numpy.cos(q[0] + 2*q[1]) + 1.0*k[0]*l[0]*l[1]*m[1]*q[0]*numpy.cos(q[1]) - 1.0*k[0]*l[0]*l[1]*m[1]*qr[0]*numpy.cos(q[1]) + 1.0*k[0]*l[1]**2*m[1]*q[0] - 1.0*k[0]*l[1]**2*m[1]*qr[0] - 1.0*k[1]*l[0]**2*m[0]*q[1] + 1.0*k[1]*l[0]**2*m[0]*qr[1] - 1.0*k[1]*l[0]**2*m[1]*q[1] + 1.0*k[1]*l[0]**2*m[1]*qr[1] - 2.0*k[1]*l[0]*l[1]*m[1]*q[1]*numpy.cos(q[1]) + 2.0*k[1]*l[0]*l[1]*m[1]*qr[1]*numpy.cos(q[1]) - 1.0*k[1]*l[1]**2*m[1]*q[1] + 1.0*k[1]*l[1]**2*m[1]*qr[1] + 1.0*l[0]**2*m[0]*tau_in[1] + 1.0*l[0]**2*m[1]*tau_in[1] - 1.0*l[0]*l[1]*m[1]*tau_in[0]*numpy.cos(q[1]) + 2.0*l[0]*l[1]*m[1]*tau_in[1]*numpy.cos(q[1]) - 1.0*l[1]**2*m[1]*tau_in[0] + 1.0*l[1]**2*m[1]*tau_in[1])/(l[0]**2*l[1]**2*m[1]*(2*m[0] - m[1]*numpy.cos(2*q[1]) + m[1]))]]) 
        
    def _kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*l[0]**2*m[0] + 0.5*m[1]*(dq[0]**2*l[0]**2 + 2*dq[0]**2*l[0]*l[1]*numpy.cos(q[1]) + dq[0]**2*l[1]**2 + 2*dq[0]*dq[1]*l[0]*l[1]*numpy.cos(q[1]) + 2*dq[0]*dq[1]*l[1]**2 + dq[1]**2*l[1]**2)
        
    def _energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*l[0]**2*m[0] + g*l[0]*m[0]*numpy.sin(q[0]) + g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2 + 0.5*m[1]*(dq[0]**2*l[0]**2 + 2*dq[0]**2*l[0]*l[1]*numpy.cos(q[1]) + dq[0]**2*l[1]**2 + 2*dq[0]*dq[1]*l[0]*l[1]*numpy.cos(q[1]) + 2*dq[0]*dq[1]*l[1]**2 + dq[1]**2*l[1]**2)
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[0]), -numpy.sin(q[0]), l[0]*numpy.cos(q[0])], [numpy.sin(q[0]), numpy.cos(q[0]), l[0]*numpy.sin(q[0])], [0, 0, 1], [numpy.cos(q[0] + q[1]), -numpy.sin(q[0] + q[1]), l[0]*numpy.cos(q[0]) + l[1]*numpy.cos(q[0] + q[1])], [numpy.sin(q[0] + q[1]), numpy.cos(q[0] + q[1]), l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])], [0, 0, 1]]).reshape((2, 3, 3))
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[0]*numpy.cos(q[0]) + l[1]*numpy.cos(q[0] + q[1])], [l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])], [numpy.arctan2(numpy.sin(q[0] + q[1]), numpy.cos(q[0] + q[1]))]])
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[0] + q[1]), -numpy.sin(q[0] + q[1]), l[0]*numpy.cos(q[0]) + l[1]*numpy.cos(q[0] + q[1])], [numpy.sin(q[0] + q[1]), numpy.cos(q[0] + q[1]), l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])], [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[-l[0]*numpy.sin(q[0]) - l[1]*numpy.sin(q[0] + q[1]), -l[1]*numpy.sin(q[0] + q[1])], [l[0]*numpy.cos(q[0]) + l[1]*numpy.cos(q[0] + q[1]), l[1]*numpy.cos(q[0] + q[1])], [1, 1]])
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        return numpy.array([[-(l[0]**2*m[0] + m[1]*(l[0]**2 + 2*l[0]*l[1]*numpy.cos(q[1]) + l[1]**2))*(-2*E + 2*g*l[0]*m[0]*numpy.sin(q[0]) + 2*g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 1.0*k[0]*(-q[0] + qr[0])**2 + 1.0*k[1]*(-q[1] + qr[1])**2), -1.0*l[1]*m[1]*(l[0]*numpy.cos(q[1]) + l[1])*(-2*E + 2*g*l[0]*m[0]*numpy.sin(q[0]) + 2*g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 1.0*k[0]*(-q[0] + qr[0])**2 + 1.0*k[1]*(-q[1] + qr[1])**2)], [-1.0*l[1]*m[1]*(l[0]*numpy.cos(q[1]) + l[1])*(-2*E + 2*g*l[0]*m[0]*numpy.sin(q[0]) + 2*g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 1.0*k[0]*(-q[0] + qr[0])**2 + 1.0*k[1]*(-q[1] + qr[1])**2), -1.0*l[1]**2*m[1]*(-2*E + 2*g*l[0]*m[0]*numpy.sin(q[0]) + 2*g*m[1]*(l[0]*numpy.sin(q[0]) + l[1]*numpy.sin(q[0] + q[1])) + 1.0*k[0]*(-q[0] + qr[0])**2 + 1.0*k[1]*(-q[1] + qr[1])**2)]])

    def coriolis_centrifugal_forces_dq(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[0, -dq[1]*l[0]*l[1]*m[1]*(2*dq[0] + dq[1])*numpy.cos(q[1])], [0, dq[0]**2*l[0]*l[1]*m[1]*numpy.cos(q[1])]])

    def coriolis_centrifugal_forces_ddq(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[-2*dq[1]*l[0]*l[1]*m[1]*numpy.sin(q[1]), 2*l[0]*l[1]*m[1]*(-dq[0] - dq[1])*numpy.sin(q[1])], [2*dq[0]*l[0]*l[1]*m[1]*numpy.sin(q[1]), 0]])
