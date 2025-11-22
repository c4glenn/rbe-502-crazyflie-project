import numpy as np

from polynomial_traj import Trajectory


RADIUS = 1
HOVER_HEIGHT = 1
V_MAX=2.5
START = np.array([[0, 0, 0.5], [0, 0, 0], [0, 0, 0]])
CIRCLE_POINT = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]])

start_to_circ = Trajectory(3).add_point(0.0, START).add_point(5.0, CIRCLE_POINT).build()
circ_to_start = Trajectory(3).add_point(10.0, CIRCLE_POINT).add_point(15.0, START).build()
theta_traj = Trajectory(3).add_point(0.0, np.array([[0.0], [0.0], [0.0]])).add_point(5.0, np.array([[2 * np.pi], [0.0], [0.0]])).build()

# for arr in omega_traj.traj.values():
#     for i in arr:
#         print(i[0])

def get_omega(t):
    return theta_traj.get_values(t)[0][0] * 1/t
def get_omega_dot(t):
    return theta_traj.get_values(t)[1][0]
def get_omega_ddot(t):
    return theta_traj.get_values(t)[2][0]

def get_circle_pos(t: float) -> np.ndarray:
    return np.array([RADIUS*np.cos(get_omega(t)*t), RADIUS*np.sin(get_omega(t)*t), HOVER_HEIGHT])

def get_circle_vel(t: float) -> np.ndarray:
    return np.array([-RADIUS*get_omega_dot(t)*np.sin(get_omega(t)*t),RADIUS*get_omega_dot(t)*np.cos(get_omega(t)*t),0])

def get_circle_acc(t:float) -> np.ndarray:
    return np.array([-RADIUS*(get_omega_ddot(t)*np.sin(get_omega(t)*t) + (get_omega_dot(t) ** 2) * np.cos(get_omega(t)*t)),RADIUS*(get_omega_ddot(t)*np.cos(get_omega(t)*t) - (get_omega_dot(t) ** 2)*np.sin(get_omega(t)*t)),0])


    
def circle(t, tf=8):
    """
    Generate the desired state of a drone following a circular trajectory.

    The function computes the drone’s position, velocity, and acceleration
    at a given time 't' while following a circular trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tf (float): Total trajectory duration.

    Returns:
        desired_state (dict):
        
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """
    if t <= 5.0:
        vals = start_to_circ.get_values(t)
        pos = vals[0]
        vel = vals[1]
        acc = vals[2]
    elif t <= 10.0:
        pos = get_circle_pos(t-5)
        vel = get_circle_vel(t-5)
        acc = get_circle_acc(t-5)
    else: # phase 3
        vals = circ_to_start.get_values(t)
        pos = vals[0]
        vel = vals[1]
        acc = vals[2]
    
    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state

