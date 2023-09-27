import numpy
from ..pds import PlanarDynamicalSystem


class XYRR(PlanarDynamicalSystem):
    def __init__(self, l, m, g, k, qr):
        super().__init__(4, l, m, g, k, qr, False)
        
    def mass_matrix(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1.0*m[0] + 1.0*m[1] + 1.0*m[2] + 1.0*m[3], 0, -1.0*l[2]*m[2]*numpy.sin(q[2]) - 1.0*l[2]*m[3]*numpy.sin(q[2]) - 1.0*l[3]*m[3]*numpy.sin(q[2] + q[3]), -1.0*l[3]*m[3]*numpy.sin(q[2] + q[3])], [0, 1.0*m[1] + 1.0*m[2] + 1.0*m[3], 1.0*l[2]*m[2]*numpy.cos(q[2]) + m[3]*(l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3])), 1.0*l[3]*m[3]*numpy.cos(q[2] + q[3])], [-1.0*l[2]*m[2]*numpy.sin(q[2]) - 1.0*l[2]*m[3]*numpy.sin(q[2]) - 1.0*l[3]*m[3]*numpy.sin(q[2] + q[3]), 1.0*l[2]*m[2]*numpy.cos(q[2]) + m[3]*(l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3])), l[2]**2*m[2] + m[3]*(l[2]**2 + 2*l[2]*l[3]*numpy.cos(q[3]) + l[3]**2), 1.0*l[3]*m[3]*(l[2]*numpy.cos(q[3]) + l[3])], [-1.0*l[3]*m[3]*numpy.sin(q[2] + q[3]), 1.0*l[3]*m[3]*numpy.cos(q[2] + q[3]), 1.0*l[3]*m[3]*(l[2]*numpy.cos(q[3]) + l[3]), 1.0*l[3]**2*m[3]]])

    def gravity(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[0], [g*(m[1] + m[2] + m[3])], [g*(l[2]*m[2]*numpy.cos(q[2]) + m[3]*(l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3])))], [g*l[3]*m[3]*numpy.cos(q[2] + q[3])]]).flatten()
        
    def coriolis_centrifugal_forces(self, q, dq):
        l, m, g, k, qr = self.params
        return numpy.array([[-1.0*dq[2]**2*l[2]*m[2]*numpy.cos(q[2]) - 1.0*k[0]*(-q[0] + qr[0]) - 1.0*m[3]*(dq[2]**2*l[2]*numpy.cos(q[2]) + dq[2]**2*l[3]*numpy.cos(q[2] + q[3]) + 2*dq[2]*dq[3]*l[3]*numpy.cos(q[2] + q[3]) + dq[3]**2*l[3]*numpy.cos(q[2] + q[3]))], [-1.0*dq[2]**2*l[2]*m[2]*numpy.sin(q[2]) - 1.0*dq[2]**2*l[2]*m[3]*numpy.sin(q[2]) - 1.0*dq[2]**2*l[3]*m[3]*numpy.sin(q[2] + q[3]) - 2.0*dq[2]*dq[3]*l[3]*m[3]*numpy.sin(q[2] + q[3]) - 1.0*dq[3]**2*l[3]*m[3]*numpy.sin(q[2] + q[3]) + 1.0*k[1]*q[1] - 1.0*k[1]*qr[1]], [-2.0*dq[2]*dq[3]*l[2]*l[3]*m[3]*numpy.sin(q[3]) - 1.0*dq[3]**2*l[2]*l[3]*m[3]*numpy.sin(q[3]) + 1.0*k[2]*q[2] - 1.0*k[2]*qr[2]], [1.0*dq[2]**2*l[2]*l[3]*m[3]*numpy.sin(q[3]) + 1.0*k[3]*q[3] - 1.0*k[3]*qr[3]]]).flatten()
        
    def _potential(self, q):
        l, m, g, k, qr = self.params
        return g*m[1]*q[1] + g*m[2]*(l[2]*numpy.sin(q[2]) + q[1]) + g*m[3]*(l[2]*numpy.sin(q[2]) + l[3]*numpy.sin(q[2] + q[3]) + q[1]) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2 + 0.5*k[2]*(-q[2] + qr[2])**2 + 0.5*k[3]*(-q[3] + qr[3])**2
        
    def _ddq(self, q, dq, tau_in):
        l, m, g, k, qr = self.params
        raise NotImplementedError 
        
    def _kinetic_energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*m[0] + 0.5*m[1]*(dq[0]**2 + dq[1]**2) + 0.5*m[2]*((dq[0] - dq[2]*l[2]*numpy.sin(q[2]))**2 + (dq[1] + dq[2]*l[2]*numpy.cos(q[2]))**2) + 0.5*m[3]*((-dq[0] + dq[2]*l[2]*numpy.sin(q[2]) + dq[2]*l[3]*numpy.sin(q[2] + q[3]) + dq[3]*l[3]*numpy.sin(q[2] + q[3]))**2 + (dq[1] + dq[2]*l[2]*numpy.cos(q[2]) + dq[2]*l[3]*numpy.cos(q[2] + q[3]) + dq[3]*l[3]*numpy.cos(q[2] + q[3]))**2)
        
    def _energy(self, q, dq):
        l, m, g, k, qr = self.params
        return 0.5*dq[0]**2*m[0] + g*m[1]*q[1] + g*m[2]*(l[2]*numpy.sin(q[2]) + q[1]) + g*m[3]*(l[2]*numpy.sin(q[2]) + l[3]*numpy.sin(q[2] + q[3]) + q[1]) + 0.5*k[0]*(-q[0] + qr[0])**2 + 0.5*k[1]*(-q[1] + qr[1])**2 + 0.5*k[2]*(-q[2] + qr[2])**2 + 0.5*k[3]*(-q[3] + qr[3])**2 + 0.5*m[1]*(dq[0]**2 + dq[1]**2) + 0.5*m[2]*((dq[0] - dq[2]*l[2]*numpy.sin(q[2]))**2 + (dq[1] + dq[2]*l[2]*numpy.cos(q[2]))**2) + 0.5*m[3]*((-dq[0] + dq[2]*l[2]*numpy.sin(q[2]) + dq[2]*l[3]*numpy.sin(q[2] + q[3]) + dq[3]*l[3]*numpy.sin(q[2] + q[3]))**2 + (dq[1] + dq[2]*l[2]*numpy.cos(q[2]) + dq[2]*l[3]*numpy.cos(q[2] + q[3]) + dq[3]*l[3]*numpy.cos(q[2] + q[3]))**2)
        
    def _link_positions(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1, 0, q[0]], [0, 1, 0], [0, 0, 1], [1, 0, q[0]], [0, 1, q[1]], [0, 0, 1], [numpy.cos(q[2]), -numpy.sin(q[2]), l[2]*numpy.cos(q[2]) + q[0]], [numpy.sin(q[2]), numpy.cos(q[2]), l[2]*numpy.sin(q[2]) + q[1]], [0, 0, 1], [numpy.cos(q[2] + q[3]), -numpy.sin(q[2] + q[3]), l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3]) + q[0]], [numpy.sin(q[2] + q[3]), numpy.cos(q[2] + q[3]), l[2]*numpy.sin(q[2]) + l[3]*numpy.sin(q[2] + q[3]) + q[1]], [0, 0, 1]]).reshape((4, 3, 3))
        
    def _fkin(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3]) + q[0]], [l[2]*numpy.sin(q[2]) + l[3]*numpy.sin(q[2] + q[3]) + q[1]], [numpy.arctan2(numpy.sin(q[2] + q[3]), numpy.cos(q[2] + q[3]))]])
        
    def endeffector_pose(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[numpy.cos(q[2] + q[3]), -numpy.sin(q[2] + q[3]), l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3]) + q[0]], [numpy.sin(q[2] + q[3]), numpy.cos(q[2] + q[3]), l[2]*numpy.sin(q[2]) + l[3]*numpy.sin(q[2] + q[3]) + q[1]], [0, 0, 1]])

    def jacobian(self, q):
        l, m, g, k, qr = self.params
        return numpy.array([[1, 0, -l[2]*numpy.sin(q[2]) - l[3]*numpy.sin(q[2] + q[3]), -l[3]*numpy.sin(q[2] + q[3])], [0, 1, l[2]*numpy.cos(q[2]) + l[3]*numpy.cos(q[2] + q[3]), l[3]*numpy.cos(q[2] + q[3])], [0, 0, 1, 1]])
        
    def jacobi_metric(self, q, E):
        l, m, g, k, qr = self.params
        raise NotImplementedError
