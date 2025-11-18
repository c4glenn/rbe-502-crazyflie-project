import numpy as np


RADIUS = 1
HOVER_HEIGHT = 1
OMEGA = 1
START_POS = np.array([0, 0, 0.5])
START_VEL = np.array([0, 0, 0])
START_ACC = np.array([0, 0, 0])


def get_circle_pos(t: float) -> np.ndarray:
    return np.array([RADIUS*np.cos(OMEGA*t), RADIUS*np.sin(OMEGA*t), HOVER_HEIGHT])
def get_circle_vel(t: float) -> np.ndarray:
    return np.array([-RADIUS*OMEGA*np.sin(OMEGA*t), RADIUS*OMEGA*np.cos(OMEGA*t), 0])
def get_circle_acc(t:float) -> np.ndarray:
    return np.array([-RADIUS*OMEGA*OMEGA*np.cos(OMEGA*t), -RADIUS*OMEGA*OMEGA*np.sin(OMEGA*t), 0])

def get_polynomial_pos(t:float, a:np.ndarray) -> np.ndarray:
    pos = a[0]
    for i, a_comp in enumerate(a[1:], 1):
        pos += a_comp*t**i
    return pos
def get_polynomial_vel(t:float, a:np.ndarray) -> np.ndarray:
    vel = a[1]
    for i, a_comp in enumerate(a[2:], 2):
        vel += i*a_comp*t**(i-1)
    return vel
def get_polynomial_acc(t:float, a:np.ndarray) -> np.ndarray:
    acc = 2 * a[2]
    for i, a_comp in enumerate(a[3:], 3):
        acc += i*(i-1)*a_comp*t**(i-2)
    return acc


def M_matrix(t: float) -> np.ndarray:
    return np.array([[1, t, t**2, t**3, t**4, t**5], 
                     [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], 
                     [0, 0, 2, 6*t, 12*t**2, 20*t**3]])

def create_polynomial_trajectory(start_pos: np.ndarray, start_vel: np.ndarray, start_acc: np.ndarray, start_time: float, end_pos: np.ndarray, end_vel: np.ndarray, end_acc: np.ndarray, end_time: float) -> np.ndarray:
    """Create a polynomial trajectory 

    Args:
        start_pos (np.ndarray, shape (3,)): starting position [x, y, z]
        start_vel (np.ndarray, shape (3,)): starting velocity [vx, vy, vz]
        start_acc (np.ndarray, shape (3,)): starting acceleration [ax, ay, az]
        start_time (float): time to start from
        end_pos (np.ndarray, shape (3,)): ending position [x, y, z]
        end_vel (np.ndarray, shape (3,)): ending velocity [vx, vy, vz]
        end_acc (np.ndarray, shape (3,)): ending acceleration [ax, ay, az]
        end_time (float): time to be done by
    
    Returns:
        a_coef (np.ndarray, shape (6, 3)): coeficents for the polonomials for x, y, and z respectively
    """
    A = np.vstack([M_matrix(start_time), M_matrix(end_time)])
    b = np.array([start_pos, start_vel, start_acc, end_pos, end_vel, end_acc])
    a = np.linalg.solve(A, b)
    return a
    
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
    start_to_circ = create_polynomial_trajectory(START_POS, START_VEL, START_ACC, 0.0, get_circle_pos(5.0), get_circle_vel(5.0), get_circle_acc(5.0), 5)
    circ_to_start = create_polynomial_trajectory(get_circle_pos(10.0), get_circle_vel(10.0), get_circle_acc(10.0), 10.0, START_POS, START_VEL, START_ACC, tf)
    if t <= 5.0:
        pos = get_polynomial_pos(t, start_to_circ)
        vel = get_polynomial_vel(t, start_to_circ)
        acc = get_polynomial_acc(t, start_to_circ)
    elif t <= 10.0:
        pos = get_circle_pos(t)
        vel = get_circle_vel(t)
        acc = get_circle_acc(t)
    else: # phase 3
        pos = get_polynomial_pos(t, circ_to_start)
        vel = get_polynomial_vel(t, circ_to_start)
        acc = get_polynomial_acc(t, circ_to_start)
    
    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state

