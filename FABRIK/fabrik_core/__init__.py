#!/usr/bin/env python3
"""
FABRIK - Forward And Backward Reaching Inverse Kinematics

Este paquete implementa el algoritmo FABRIK para cinemática inversa en 2D y 3D,
junto con utilidades matemáticas y sistemas de visualización.

Módulos principales:
- math_utils: Funciones matemáticas para operaciones vectoriales y geométricas
- visualization: Sistema de grabación y visualización
- Implementaciones FABRIK en 2D y 3D
"""

from .math_utils import (
    # Operaciones básicas de vectores
    distance,
    distance_squared,
    normalized,
    
    # Rotaciones
    rotated_2d,
    rotated_3d,
    rotated,  # alias para rotated_2d
    
    # Conversiones de coordenadas
    spherical_to_cartesian,
    cartesian_to_spherical,
    
    # Utilidades angulares
    wrap_angle,
    angle_to_point,
)

# Metadatos del paquete
__version__ = "1.0.0"
__author__ = "FABRIK Physics II Project"
__description__ = "Forward And Backward Reaching Inverse Kinematics implementation"

# Exportaciones públicas
__all__ = [
    # Funciones matemáticas básicas
    'distance',
    'distance_squared',
    'normalized',
    
    # Funciones de rotación
    'rotated_2d',
    'rotated_3d', 
    'rotated',
    
    # Conversiones de coordenadas
    'spherical_to_cartesian',
    'cartesian_to_spherical',
    
    # Utilidades angulares
    'wrap_angle',
    'angle_to_point',
    
    # Metadatos
    '__version__',
    '__author__',
    '__description__',
]
