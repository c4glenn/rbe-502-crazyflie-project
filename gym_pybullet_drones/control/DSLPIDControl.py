import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Contributors: SiQi Zhou, James Xu, Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # You can initialize more parameters here

        self.K_p = np.array([0.2, 0.2, 1])
        self.K_d = np.array([0.15, 0.15, 0.4])
        self.K_p_att = np.diag([500000, 500000, 100000])  # Roll, pitch, yaw
        self.K_d_att = np.diag([100000, 100000, 50000])  # Roll, pitch, yaw rates
        self.max_tilt = np.pi / 4
        
        # Your code ends here
        
        ######################################################
        # Do not change these parameters below
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_acc = np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        mass = self._getURDFParameter('m')
        target_thrust, computed_target_rpy, pos_e, cur_rotation = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel,
                                                                         target_acc,
                                                                         mass=mass
                                                                         )
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )

        return rpm, pos_e, computed_target_rpy
    
    def _dslPIDPositionControl(self,
                               control_timestep: float,
                               cur_pos: np.ndarray,
                               cur_quat: np.ndarray,
                               cur_vel: np.ndarray,
                               target_pos: np.ndarray,
                               target_rpy: np.ndarray,
                               target_vel: np.ndarray,
                               target_acc: np.ndarray,
                               mass = 0.29
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray
            (3,1)-shaped array of floats containing the desired acceleration.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.
        ndarray
            (3,3)-shaped array of floats representing the current rotation matrix (from quaternion).
        """

        #Write your code here
        
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        R = Rotation.from_quat(cur_quat, scalar_first=False).as_matrix()
        
        _, _, target_yaw = target_rpy
        
        thrust_direction = self.K_p * pos_e + self.K_d * vel_e + mass * target_acc + np.array([0, 0, self.GRAVITY]) 
        xy_thrust = thrust_direction[:2]
        z_thrust = thrust_direction[2]  
        
        
        horiz_thrust_max = z_thrust * np.tan(self.max_tilt)
        thrust_ratio = horiz_thrust_max / np.linalg.norm(xy_thrust)
        if thrust_ratio < 1:
            xy_thrust *= thrust_ratio
        
        target_thrust = np.array([xy_thrust[0], xy_thrust[1], z_thrust])
        
        zdb = target_thrust / np.linalg.norm(target_thrust)
        xdc = np.asarray([np.cos(target_yaw), np.sin(target_yaw), 0])
        
        zcrossx = np.cross(zdb, xdc)
        
        ydb = zcrossx / np.linalg.norm(zcrossx)
        xdb = np.cross(ydb, zdb)
        
        Rd = np.column_stack([xdb, ydb, zdb])
        roll, pitch, yaw = Rotation.from_matrix(Rd).as_euler('xyz', degrees=False)
        target_rpy = np.array([roll, pitch, yaw])
        cur_rotation = R
        #Your code ends here

        return target_thrust, target_rpy, pos_e, cur_rotation

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """

        #Write your code here
        
        curr_R = Rotation.from_quat(cur_quat, scalar_first=False).as_matrix()
        R_d = Rotation.from_euler("xyz", target_euler).as_matrix()
                
        skew_sym = R_d.T @ curr_R - curr_R.T @ R_d
        e_R = 0.5 * np.array([skew_sym[2,1], skew_sym[0,2], skew_sym[1,0]])
                
        e_omega = target_rpy_rates - cur_ang_vel
                
        target_torques = -self.K_p_att @ e_R + self.K_d_att @ e_omega

        #Your code ends here

    ################################################################################

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
