"""
Compatibilidad de importación para versiones previas de FABRIK.

Este módulo expone `FabrikSerialSolver` como alias de `FABRIKRSolver`.
El código actual usa `fabrik_core.fabrik_r_solver` como implementación principal.
"""

from __future__ import annotations

from .fabrik_r_solver import (
    FABRIKRSolver,
    RevoluteJoint,
    SolverResult,
    project_to_axis_plane,
    clamp_angle_rodrigues,
    rotate_vector_rodrigues,
    extract_joint_angle,
    _safe_normalize,
    EPS,
)

FabrikSerialSolver = FABRIKRSolver

__all__ = [
    "FabrikSerialSolver",
    "FABRIKRSolver",
    "RevoluteJoint",
    "SolverResult",
    "project_to_axis_plane",
    "clamp_angle_rodrigues",
    "rotate_vector_rodrigues",
    "extract_joint_angle",
    "_safe_normalize",
    "EPS",
]
