import numpy as np
from scipy.linalg import expm

A = np.array([[0, 1], [-1, 0]])
exp_A = expm(A)

print(exp_A)
theta = np.pi/2
print(exp_A*theta)

w = np.array([0, 0, -1])

S = np.array([])