from typing import List, Dict

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec



from circle import circle
from diamond import diamond


def plot(trajectory: List[Dict[str, any]], traj_name:str, tf:int, dt:float=0.01):
    """plots the given trajectory according to assignemnt instructions

    Args:
        trajectory (List[Dict[str, any]]): List of desired_states (dict):
        
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
        
        dt (float, optional): what delta time to use when plotting trajectory. Defaults to 0.01.
    """
    if not trajectory:
        print("empty trajecotry")
        return
    
    n = len(trajectory)
    time = np.arange(0.0, tf, dt)
    
    pos = np.array([tj['pos'] for tj in trajectory])
    vel = np.array([tj['vel'] for tj in trajectory])
    acc = np.array([tj['acc'] for tj in trajectory])
    
    #because its an array of arrays in a weird way it makes it be (t, 1, 3) which isnt very helpful
    if pos.ndim == 3 and pos.shape[1] == 1:
        pos = pos.squeeze(axis=1)
    if vel.ndim == 3 and vel.shape[1] == 1:
        vel = vel.squeeze(axis=1)
    if acc.ndim == 3 and acc.shape[1] == 1:
        acc = acc.squeeze(axis=1)
        
    fig = plt.figure(figsize=(16,10))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title(f'3D {traj_name} Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    gs = GridSpec(3, 2, figure=fig)

    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, pos[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(time, pos[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(time, pos[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, vel[:, 0], 'r-', label='Vx', linewidth=2)
    ax3.plot(time, vel[:, 1], 'g-', label='Vy', linewidth=2)
    ax3.plot(time, vel[:, 2], 'b-', label='Vz', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(time, acc[:, 0], 'r-', label='Ax', linewidth=2)
    ax4.plot(time, acc[:, 1], 'g-', label='Ay', linewidth=2)
    ax4.plot(time, acc[:, 2], 'b-', label='Az', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('Acceleration vs Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()



def generate_circle(dt:float = 0.01) -> List[Dict[str, any]]:
    """Generates the trajectory for the circle

    Args:
        dt (float, optional): what delta time to use when generating trajectory. Defaults to 0.01.
    
    Returns:
        List of desired_states (dict):
        
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """
    traj = []
    for t in np.arange(0.0, 15.0, step=dt):
        traj.append(circle(t, 15))
    return traj

def generate_diamond(dt:float = 0.01) -> List[Dict[str, any]]:
    """_summary_

    Args:
        dt (float, optional): what delta time to use when generating trajectory. Defaults to 0.01.

    Returns:
        List of desired_states (dict):
        
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """
    traj = []
    for t in np.arange(0.0, 8, step=dt):
        traj.append(diamond(t, 8))
    return traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and plot trajectories for assignment")
    parser.add_argument("--circle", "-c", action="store_true")
    parser.add_argument("--diamond", "-d", action="store_true")
    parser.add_argument("--dt", type=float, default=0.01)
    
    args = parser.parse_args()
    
    assert args.circle or args.diamond, "Must turn on either circle or diamond"  
    
    if args.circle:
        plot(generate_circle(args.dt),"circle", 15, args.dt)
    if args.diamond:
        plot(generate_diamond(args.dt),"diamond", 8, args.dt)