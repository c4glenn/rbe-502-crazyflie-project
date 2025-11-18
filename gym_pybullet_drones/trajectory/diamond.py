import numpy as np

U = 1/np.sqrt(2)
UX = 1


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
    return np.array([[1, t, t**2, t**3, t**4, t**5], [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4], [0, 0, 2, 6*t, 12*t**2, 20*t**3]])

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
    
    x_segment = create_polynomial_trajectory(np.array([0]), np.array([0]), np.array([0]), 0.0, np.array([UX]), np.array([0]), np.array([0]), tfinal)
    seg_0 = create_polynomial_trajectory(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, np.array([0, U, U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 2.0)
    seg_1 = create_polynomial_trajectory(np.array([0, U, U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 2.0, np.array([0, 0, 2*U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 4.0)
    seg_2 = create_polynomial_trajectory(np.array([0, 0, 2*U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 4.0, np.array([0, -U, U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 6.0)
    seg_3 = create_polynomial_trajectory(np.array([0, -U, U]), np.array([0, 0, 0]), np.array([0, 0, 0]), 6.0, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 8.0)
    
    seg = None
    if t < 2:
        seg = seg_0
    elif t < 4:
        seg = seg_1
    elif t < 6:
        seg = seg_2
    elif t <= tfinal:
        seg = seg_3   
        
    pos = get_polynomial_pos(t, seg)
    x_pos = get_polynomial_pos(t, x_segment)
    pos[0] = x_pos
    
    vel = get_polynomial_vel(t, seg)
    x_vel = get_polynomial_vel(t, x_segment)
    vel[0] = x_vel

    acc = get_polynomial_acc(t, seg)
    x_acc = get_polynomial_acc(t, x_segment)
    acc[0] = x_acc
    
    
    
    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state
