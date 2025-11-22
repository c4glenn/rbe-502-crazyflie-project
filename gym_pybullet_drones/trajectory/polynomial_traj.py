import numpy as np
from math import factorial    
from typing import List, Tuple 
from itertools import product

def M_matrix(t: float, constraints: int, derivatives: int) -> np.ndarray:
    M = np.zeros((derivatives, constraints))
    for i in range(derivatives):
        for j in range(i, constraints):
            M[i, j] = factorial(j)/factorial(j-i) * (t ** (j-i))
    return M

def find_coefs(start_time:float, end_time:float, start_constraints:np.ndarray, end_constraints:np.ndarray, derivatives_to_calculate:int) -> np.ndarray:
    """_summary_

    Args:
        start_time (float): time this polynomial starts at
        end_time (float): time this polynomial ends at
        start_constraints (np.ndarray, shape (n,m)): n is number of derivatives and m is number of variables 
        end_constraints (np.ndarray, shape (n,m)): n is number of derivatives and m is number of variables

    Returns:
        np.ndarray: _description_
    """
    B = np.vstack((start_constraints, end_constraints))
    A = np.vstack([M_matrix(start_time, B.shape[0], derivatives_to_calculate), M_matrix(end_time, B.shape[0], derivatives_to_calculate)])
    coefs = np.linalg.solve(A, B)
    return coefs


def get_polynomial_result(time:float, coefs:np.ndarray, derivatives:int) -> np.ndarray:
    num_constraints = coefs.shape[0]
    M = M_matrix(time, num_constraints, derivatives)
    P = M@coefs
    return P

class Trajectory:
    def __init__(self, num_derivs:int):
        self.points = {}
        self.traj = {}
        self.num_derivs = num_derivs
    
    def add_point(self, time:float, constraint: np.ndarray) -> "Trajectory":
        self.points[time] = constraint
        return self
    
    def build(self, merge:bool=False) -> "Trajectory":
        sorted_time = sorted(self.points.keys())
        if not merge:
            for time_idx, time in enumerate(sorted_time[:-1]):
                next_time = sorted_time[time_idx + 1]
                self.traj[(time, next_time)] = find_coefs(time, next_time, self.points[time], self.points[next_time], self.num_derivs)
                self.points.pop(time)
        else:
            B = np.vstack([self.points[t] for t in sorted_time]) 
            A = np.vstack([M_matrix(t, B.shape[0], self.num_derivs) for t in sorted_time])
            print(A)
            self.traj[(sorted_time[0], sorted_time[-1])] = np.linalg.solve(A, B)
            for t in sorted_time[:-1]:
                self.points.pop(t)
        return self
    
    def get_values(self, time:float):
        for t1, t2 in self.traj.keys():
            if t1 <= time and time <= t2:
                
                return get_polynomial_result(time, self.traj[(t1, t2)], self.num_derivs)
