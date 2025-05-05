# cinematica_inversa_general.py
# Generalización de Cinemática Inversa con casos de prueba
# Autor: [Tu Nombre]
# Fecha: Mayo 2025

import numpy as np
import sympy as sp
from class_robot_structure import Robot, cargar_robot_desde_yaml
from class_jacobian import MatrixExp6sp, Adjunta, calcular_jacobiana
import math

# Parámetros constantes
ROBOT_YAML = 'robot.yaml'
TOL_ORIENTATION = 1e-2  # rad
TOL_POSITION = 1e-3     # m
MAX_ITER = 20


def skew3(omega: np.ndarray) -> np.ndarray:
    """
    Convierte un vector de rotación 3D en una matriz antisimétrica 3x3 (so3).
    """
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])


def twist_to_se3(S: np.ndarray, theta: float) -> np.ndarray:
    """
    Calcula la exponencial de un twist (screw axis) en SE(3) para un ángulo theta.
    """
    w = S[:3]
    v = S[3:]
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-6:
        return np.block([[np.eye(3), v.reshape(3,1)*theta],
                         [np.zeros((1,3)), 1]])
    w_skew = skew3(w)
    R = np.eye(3) + np.sin(theta)*w_skew + (1-np.cos(theta))*(w_skew@w_skew)
    G = (np.eye(3)*theta + (1-np.cos(theta))*w_skew + (theta-np.sin(theta))*(w_skew@w_skew))
    p = G @ v
    return np.block([[R, p.reshape(3,1)], [np.zeros((1,3)), 1]])


def direct_kinematics(M: np.ndarray, S_list: list[np.ndarray], thetas: np.ndarray) -> np.ndarray:
    """
    Cinemática directa: exp([S1]θ1)...exp([Sn]θn) * M
    """
    T = np.eye(4)
    for S, theta in zip(S_list, thetas):
        T = T @ twist_to_se3(S, theta)
    return T @ M


def rpy_to_transform(rpy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Construye la matriz homogénea a partir de ángulos roll-pitch-yaw (grados en radianes)
    y vector traslación r.
    Rotación: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll, pitch, yaw = rpy
    # Matrices de rotación elementales
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0           ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    R = Rz @ Ry @ Rx
    return np.block([[R, r.reshape(3,1)], [np.zeros((1,3)), 1]])


def inverse_kinematics(robot: Robot,
                       target_T: np.ndarray,
                       thetas0: np.ndarray,
                       tol_orientation: float = TOL_ORIENTATION,
                       tol_position: float = TOL_POSITION,
                       max_iter: int = MAX_ITER) -> tuple[np.ndarray, int]:
    """
    Newton-Raphson para cinemática inversa.
    Devuelve thetas (rad) y número de iteraciones.
    """
    S_list = robot.ejes_helicoidales
    M = robot.get_home_matrix()
    thetas = thetas0.copy().astype(float)
    for i in range(max_iter):
        T_current = direct_kinematics(M, S_list, thetas)
        Vb_mat = np.dot(np.linalg.inv(T_current), target_T)
        Vb = MatrixExp6sp.logm6(Vb_mat)
        Vs = Adjunta(T_current) @ Vb
        if np.linalg.norm(Vs[:3]) < tol_orientation and np.linalg.norm(Vs[3:]) < tol_position:
            return thetas, i
        J_sym, theta_syms = calcular_jacobiana(robot)
        J_num = np.array(J_sym.subs({s: val for s, val in zip(theta_syms, thetas)})).astype(float)
        thetas += np.linalg.pinv(J_num) @ Vs
    return thetas, max_iter


def rpy_to_transform(rpy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Construye la matriz homogénea a partir de ángulos roll-pitch-yaw (rad) y vector posición r.
    Convención: roll alrededor de X, pitch alrededor de Y, yaw alrededor de Z.
    """
    roll, pitch, yaw = rpy
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll),  math.cos(roll)]])
    Ry = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return np.block([[R, r.reshape(3,1)], [np.zeros((1,3)), 1]])


def prueba_cinematica_inversa():
    """
    Casos de prueba predefinidos para validar en laboratorio.
    """
    robot = cargar_robot_desde_yaml(ROBOT_YAML)
    casos = [
        (np.zeros(len(robot.links)), np.array([0.1, 0.0, 0.1]), np.deg2rad([0, 0, 0])),
        (np.zeros(len(robot.links)), np.array([0.15, 0.05, 0.12]), np.deg2rad([45, 0, 0])),
        (np.zeros(len(robot.links)), np.array([0.05, -0.1, 0.08]), np.deg2rad([0, 30, 60])),
    ]
    for idx, (thetas0, xyz, rpy) in enumerate(casos, 1):
        target_T = rpy_to_transform(rpy, xyz)
        sol, iters = inverse_kinematics(robot, target_T, thetas0)
        print(f"Caso {idx}: objetivo xyz={xyz.tolist()}, rpy(deg)={np.rad2deg(rpy).tolist()} -> iter={iters}, solución=" + np.array2string(sol, precision=3))


def main():
    print("Ejecutando casos de prueba de cinemática inversa...")
    prueba_cinematica_inversa()

if __name__ == '__main__':
    main()
