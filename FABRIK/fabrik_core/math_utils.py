#!/usr/bin/env python3
"""
Utilidades matemáticas para el algoritmo FABRIK 3D.

Este módulo contiene funciones matemáticas puras utilizadas por el sistema FABRIK,
incluyendo operaciones vectoriales, rotaciones 3D, conversiones de coordenadas
y utilidades geométricas.

Todas las funciones son independientes y no tienen estado, lo que las hace
fáciles de testear y reutilizar en otros proyectos.
"""

import numpy as np


def wrap_angle(angle):
    """
    Envuelve el ángulo entre -PI y PI.
    
    Args:
        angle (float): Ángulo en radianes a normalizar
        
    Returns:
        float: Ángulo normalizado en el rango [-π, π]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_to_point(p1, p2):
    """
    Calcula el ángulo de p1 a p2.
    
    Args:
        p1 (np.ndarray): Punto de origen [x, y]
        p2 (np.ndarray): Punto de destino [x, y]
        
    Returns:
        float: Ángulo en radianes desde p1 hacia p2
    """
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


def distance_squared(p1, p2):
    """
    Calcula la distancia al cuadrado entre dos puntos.
    
    Más eficiente que calcular la distancia completa cuando solo
    se necesita comparar distancias.
    
    Args:
        p1 (np.ndarray): Primer punto [x, y] o [x, y, z]
        p2 (np.ndarray): Segundo punto [x, y] o [x, y, z]
        
    Returns:
        float: Distancia al cuadrado entre los puntos
    """
    return np.sum((p1 - p2)**2)


def distance(p1, p2):
    """
    Calcula la distancia entre dos puntos.
    
    Args:
        p1 (np.ndarray): Primer punto [x, y] o [x, y, z]
        p2 (np.ndarray): Segundo punto [x, y] o [x, y, z]
        
    Returns:
        float: Distancia euclidiana entre los puntos
    """
    return np.linalg.norm(p1 - p2)


def normalized(v):
    """
    Normaliza un vector 3D.
    
    Args:
        v (np.ndarray): Vector 3D a normalizar [x, y, z]
        
    Returns:
        np.ndarray: Vector normalizado (magnitud = 1) o vector original si magnitud = 0
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def rotated_3d(v, axis, angle):
    """
    Rota un vector 3D alrededor de un eje dado usando la fórmula de Rodrigues.
    
    Args:
        v (np.ndarray): Vector 3D a rotar [x, y, z]
        axis (np.ndarray): Eje de rotación normalizado [x, y, z]
        angle (float): Ángulo de rotación en radianes
        
    Returns:
        np.ndarray: Vector rotado en 3D
    """
    axis = normalized(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Fórmula de Rodrigues: v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    cross_product = np.cross(axis, v)
    dot_product = np.dot(axis, v)
    
    return (v * cos_angle + 
            cross_product * sin_angle + 
            axis * dot_product * (1 - cos_angle))


def spherical_to_cartesian(r, theta, phi):
    """
    Convierte coordenadas esféricas a cartesianas.
    
    Args:
        r (float): Radio/distancia desde el origen
        theta (float): Ángulo polar desde el eje Z (0 a π)
        phi (float): Ángulo azimutal desde el eje X (0 a 2π)
        
    Returns:
        np.ndarray: Vector cartesiano [x, y, z]
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cartesian_to_spherical(v):
    """
    Convierte coordenadas cartesianas a esféricas.
    
    Args:
        v (np.ndarray): Vector cartesiano [x, y, z]
        
    Returns:
        tuple: (r, theta, phi) - radio, ángulo polar, ángulo azimutal
    """
    r = np.linalg.norm(v)
    if r == 0:
        return 0, 0, 0
    
    theta = np.arccos(v[2] / r)  # Ángulo polar
    phi = np.arctan2(v[1], v[0])  # Ángulo azimutal
    return r, theta, phi


def rotated_2d(v, angle):
    """
    Rota un vector 2D por un ángulo dado (para visualización).
    
    Args:
        v (np.ndarray): Vector 2D a rotar [x, y]
        angle (float): Ángulo de rotación en radianes
        
    Returns:
        np.ndarray: Vector rotado en 2D
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        v[0] * cos_a - v[1] * sin_a,
        v[0] * sin_a + v[1] * cos_a
    ])


# Alias para compatibilidad con código existente
def rotated(v, angle):
    """Alias para rotated_2d para compatibilidad."""
    return rotated_2d(v, angle)
