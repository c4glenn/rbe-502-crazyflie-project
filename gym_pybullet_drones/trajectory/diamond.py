import numpy as np

from polynomial_traj import Trajectory

U = 1/np.sqrt(2)
UX = 1

def diamond(t, tfinal=8):
    """
    Generate the desired state of a drone following a diamond-shaped trajectory.

    The function computes the drone’s position, velocity, and acceleration at
    any given time 't' while following a diamond-shaped trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tfinal (float): Total trajectory duration.

    Returns:
        desired_state (dict):
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """    
    traj = Trajectory(3)\
            .add_point(0.0, np.zeros((3,3)))\
            .add_point(2.0, np.array([[0, U, U], [0, 0, 0], [0, 0, 0]]))\
            .add_point(4.0, np.array([[0, 0, 2*U], [0, 0, 0], [0, 0, 0]]))\
            .add_point(6.0, np.array([[0, -U, U], [0, 0, 0], [0, 0, 0]]))\
            .add_point(8.0, np.zeros((3,3))).build()
            
    x_traj = Trajectory(3).add_point(0.0, np.array([[0.0], [0.0], [0.0]])).add_point(8.0, np.array([[UX], [0.0], [0.0]])).build()

    val = traj.get_values(t)
    x_val = x_traj.get_values(t)
    
    desired_state = {
        'pos': [x_val[0][0], *val[0][1:]],
        'vel': [x_val[1][0], *val[1][1:]],
        'acc': [x_val[2][0], *val[2][1:]],
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
