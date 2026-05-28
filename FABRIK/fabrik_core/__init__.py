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
    # Operaciones basicas de vectores
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

from .quaternion_utils import (
    quat_identity,
    quat_normalize,
    quat_conjugate,
    quat_inverse,
    quat_multiply,
    quat_from_axis_angle,
    quat_from_two_vectors,
    quat_rotation_rotor,
    quat_rotation_angle,
    quat_clamp_rotation,
    quat_rotate_vector,
    quat_to_rotation_matrix,
    quat_from_rotation_matrix,
)

from .fabrik_serial_solver import (
    FabrikSerialSolver,
    JointDescriptor,
    JointType,
    SolverResult,
)

# Metadatos del paquete
__version__ = "1.0.0"
__author__ = "FABRIK Physics II Project"
__description__ = "Forward And Backward Reaching Inverse Kinematics implementation"

# Exportaciones publicas
__all__ = [
    # Funciones matematicas basicas
    'distance',
    'distance_squared',
    'normalized',

    # Funciones de rotacion
    'rotated_2d',
    'rotated_3d',
    'rotated',

    # Conversiones de coordenadas
    'spherical_to_cartesian',
    'cartesian_to_spherical',

    # Utilidades angulares
    'wrap_angle',
    'angle_to_point',

    # Quaterniones
    'quat_identity',
    'quat_normalize',
    'quat_conjugate',
    'quat_inverse',
    'quat_multiply',
    'quat_from_axis_angle',
    'quat_from_two_vectors',
    'quat_rotation_rotor',
    'quat_rotation_angle',
    'quat_clamp_rotation',
    'quat_rotate_vector',
    'quat_to_rotation_matrix',
    'quat_from_rotation_matrix',

    # Solver serial
    'FabrikSerialSolver',
    'JointDescriptor',
    'JointType',
    'SolverResult',

    # Metadatos
    '__version__',
    '__author__',
    '__description__',
]
