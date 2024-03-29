import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from scipy.linalg import sqrtm
from warnings import warn
from .utils import transform_mass, damping_design


class JointImpedanceController:
    def __init__(self, mass_fun, K, Zeta, desired_jpos_fun=None, desired_vel_fun=None):
        self._mass = mass_fun
        self._K = K
        self._Zeta = Zeta
        self._desired_jpos_fun = desired_jpos_fun
        self._desired_vel_fun = desired_vel_fun
        self._q_des = np.zeros(K.shape[0])
        self._q_dot_des = np.zeros(K.shape[0])

    def set_target(self, q_des, q_dot_des=None):
        if self._desired_jpos_fun:
            warn("Manually set desired position is ignored when desired_pos_fun is specified in constructor")
        self._q_des = q_des
        if q_dot_des is not None:
            self._q_dot_des = q_dot_des

    def __call__(self, t, q, dq):
        if self._desired_jpos_fun:
            q_des = self._desired_jpos_fun(t)
            self._q_des = q_des
        else:
            q_des = self._q_des

        if self._desired_vel_fun:
            qdot_des = self._desired_vel_fun(t)
            self._q_dot_des = qdot_des
        else:
            qdot_des = self._q_dot_des

        M = self._mass(q)
        D = damping_design(M, self._K, self._Zeta)

        tau = -self._K @ (q - q_des) - D @ (dq - qdot_des)
        return tau.flatten()


class ImpedanceController:
    def __init__(self, fkin_fun, jacobian_fun, mass_fun, K, Zeta, desired_pos_fun=None, desired_vel_fun=None):
        self._fkin = fkin_fun
        self._jacobian = jacobian_fun
        self._scalar = (type(K) is int) or (type(K) is float) or (K.size == 1)
        self._Zeta = Zeta
        self._K = K
        self._desired_pos_fun = desired_pos_fun
        self._desired_vel_fun = desired_vel_fun
        self._xdot_des = 0 if self._scalar else np.zeros(K.shape[0])
        self._x_des = 0 if self._scalar else np.zeros(K.shape[0])
        self._mass = mass_fun

    def fkin(self, q):
        return self._fkin(q)

    def jacobian(self, q):
        return self._jacobian(q)

    def set_target(self, x_des, x_dot_des=None):
        if self._desired_pos_fun:
            warn("Manually set desired position is ignored when desired_pos_fun is specified in constructor")
        self._x_des = x_des
        if x_dot_des is not None:
            self.set_target_velocity(x_dot_des)

    def set_target_velocity(self, x_dot_des):
        if self._desired_vel_fun:
            warn("Manually set desired velocity is ignored when desired_pos_fun is specified in constructor")
        self._xdot_des = x_dot_des

    def coordinate(self, q):
        return self._fkin(q)

    @property
    def xd(self):
        return self._x_des

    @property
    def xdotd(self):
        return self._xdot_des

    @xd.setter
    def xd(self, xd):
        self.set_target(xd)

    @xdotd.setter
    def xdotd(self, xdot_d):
        self.set_target_velocity(xdot_d)

    def __call__(self, t, q, dq):
        if self._desired_pos_fun:
            x_des = self._desired_pos_fun(t)
            self._x_des = x_des
        else:
            x_des = self._x_des

        if self._desired_vel_fun:
            xdot_des = self._desired_vel_fun(t)
            self._xdot_des = xdot_des
        else:
            xdot_des = self._xdot_des

        J = self._jacobian(q)
        x = self._fkin(q)
        xdot = J @ dq

        M_x = transform_mass(self._mass(q), J)
        D_x = damping_design(M_x, self._K, self._Zeta)

        if type(M_x) is np.ndarray:
            f_imp = -self._K @ (x - x_des) - D_x @ (xdot - xdot_des)
        else:
            f_imp = -self._K * (x - x_des) - D_x * (xdot - xdot_des)

        tau = J.T @ f_imp
        return tau.flatten()
