"""
quaternion_utils.py
===================
Operaciones con cuaterniones para rotaciones 3D en el contexto del algoritmo FABRIK.

Un cuaternion de rotacion se representa como un array numpy de 4 elementos [w, x, y, z],
donde w es la parte escalar y (x, y, z) es el vector imaginario.

Convenio: q = w + xi + yj + zk, con ||q|| = 1 para cuaterniones de rotacion puros.

Referencia: Aristidou & Lasenby (2011) usan quaterniones en los Algorithms 4 y 6
para rastrear la orientacion de cada articulacion durante el proceso FABRIK.
Las referencias FABRIK_chain_3D y FABRIK_Full_Body implementan la misma convencion
[w, x, y, z] para calcular el rotor entre orientaciones consecutivas.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Operaciones fundamentales
# ---------------------------------------------------------------------------

def quat_identity():
    """Retorna el cuaternion identidad [1, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_norm(q):
    """Norma euclidiana del cuaternion."""
    return np.linalg.norm(q)


def quat_normalize(q):
    """
    Normaliza un cuaternion a norma unitaria.

    Args:
        q (np.ndarray): Cuaternion [w, x, y, z].

    Returns:
        np.ndarray: Cuaternion normalizado.

    Raises:
        ValueError: Si el cuaternion tiene norma cero.
    """
    n = quat_norm(q)
    if n < 1e-10:
        raise ValueError("No se puede normalizar un cuaternion de norma cero.")
    return q / n


def quat_conjugate(q):
    """
    Conjugado del cuaternion: [w, -x, -y, -z].

    Args:
        q (np.ndarray): Cuaternion [w, x, y, z].

    Returns:
        np.ndarray: Conjugado.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q):
    """
    Inverso del cuaternion: conjugado dividido por la norma al cuadrado.
    Para cuaterniones unitarios, el inverso es igual al conjugado.

    Args:
        q (np.ndarray): Cuaternion [w, x, y, z].

    Returns:
        np.ndarray: Cuaternion inverso.
    """
    norm_sq = np.dot(q, q)
    if norm_sq < 1e-10:
        raise ValueError("No se puede invertir un cuaternion de norma cero.")
    return quat_conjugate(q) / norm_sq


def quat_multiply(q1, q2):
    """
    Producto de Hamilton de dos cuaterniones: q1 * q2.

    q1 = [a, b, c, d]  (w, x, y, z)
    q2 = [e, f, g, h]

    Producto:
        w = ae - bf - cg - dh
        x = be + af + ch - dg
        y = ce - bf + ag + df   -> ae + cg formula standard
        z = de + af - bg + ch   -> (Cayley-Dickson)

    Args:
        q1 (np.ndarray): Cuaternion izquierdo [w, x, y, z].
        q2 (np.ndarray): Cuaternion derecho [w, x, y, z].

    Returns:
        np.ndarray: Producto [w, x, y, z].
    """
    a, b, c, d = q1
    e, f, g, h = q2
    return np.array([
        a * e - b * f - c * g - d * h,
        b * e + a * f + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ])


# ---------------------------------------------------------------------------
# Construccion de cuaterniones de rotacion
# ---------------------------------------------------------------------------

def quat_from_axis_angle(axis, angle_rad):
    """
    Construye un cuaternion de rotacion a partir de un eje y un angulo.

    q = [cos(theta/2), sin(theta/2)*axis]

    Args:
        axis (np.ndarray): Eje de rotacion (no necesariamente unitario) [x, y, z].
        angle_rad (float): Angulo de rotacion en radianes.

    Returns:
        np.ndarray: Cuaternion unitario [w, x, y, z].

    Raises:
        ValueError: Si el eje tiene norma cero.
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-10:
        raise ValueError("El eje de rotacion no puede ser el vector cero.")
    axis = axis / n
    half = angle_rad * 0.5
    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half),
    ])


def quat_from_two_vectors(v1, v2):
    """
    Construye el cuaternion de rotacion minima que lleva v1 a v2.

    Usa la formula: q = normalize([dot + |v1||v2|, cross(v1, v2)])

    Args:
        v1 (np.ndarray): Vector origen [x, y, z] (no tiene que ser unitario).
        v2 (np.ndarray): Vector destino [x, y, z] (no tiene que ser unitario).

    Returns:
        np.ndarray: Cuaternion unitario [w, x, y, z].
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-10 or n2 < 1e-10:
        return quat_identity()

    v1 = v1 / n1
    v2 = v2 / n2

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

    # Vectores antiparalelos: rotacion de 180 grados alrededor de cualquier eje perp.
    if dot < -1.0 + 1e-8:
        perp = _perpendicular_vector(v1)
        return quat_from_axis_angle(perp, np.pi)

    cross = np.cross(v1, v2)
    w = dot + 1.0  # = 1 + cos(theta)
    q = np.array([w, cross[0], cross[1], cross[2]])
    return quat_normalize(q)


def quat_rotation_rotor(q_outer, q_inner):
    """
    Calcula el rotor (cuaternion de rotacion relativa) entre dos orientaciones.

    Segun la convencion de FABRIK_chain_3D y FABRIK_Full_Body:
        rotor = q_inner * inverse(q_outer)

    Esto representa la rotacion que lleva q_outer a q_inner.

    Args:
        q_outer (np.ndarray): Orientacion de la articulacion exterior [w, x, y, z].
        q_inner (np.ndarray): Orientacion de la articulacion interior [w, x, y, z].

    Returns:
        np.ndarray: Cuaternion rotor normalizado [w, x, y, z].
    """
    q_out_inv = quat_inverse(quat_normalize(q_outer))
    rotor = quat_multiply(q_inner, q_out_inv)
    return quat_normalize(rotor)


def quat_rotation_angle(q):
    """
    Extrae el angulo de rotacion de un cuaternion unitario.

    theta = 2 * arccos(w)   con w = q[0] clampado a [-1, 1]

    Args:
        q (np.ndarray): Cuaternion unitario [w, x, y, z].

    Returns:
        float: Angulo de rotacion en radianes en [0, pi].
    """
    w = np.clip(q[0], -1.0, 1.0)
    return 2.0 * np.arccos(abs(w))


def quat_clamp_rotation(q, max_angle_rad):
    """
    Restringe la rotacion de un cuaternion a un angulo maximo.

    Si la rotacion supera max_angle_rad, escala el angulo al maximo
    manteniendo el mismo eje de rotacion.

    Usado para implementar el bone_twist_limit de FABRIK_chain_3D y el
    bone_orientation_limit de FABRIK_Full_Body.

    Args:
        q (np.ndarray): Cuaternion de rotacion [w, x, y, z].
        max_angle_rad (float): Angulo maximo permitido en radianes.

    Returns:
        np.ndarray: Cuaternion con rotacion restringida [w, x, y, z].
    """
    w = np.clip(q[0], -1.0, 1.0)
    current_angle = 2.0 * np.arccos(abs(w))

    if current_angle <= max_angle_rad:
        return q

    # Extraer eje de rotacion
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-10:
        # Cuaternion casi identidad: usar eje arbitrario
        return quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), max_angle_rad)

    axis = q[1:4] / sin_half
    return quat_from_axis_angle(axis, max_angle_rad)


# ---------------------------------------------------------------------------
# Aplicacion de rotacion a vectores
# ---------------------------------------------------------------------------

def quat_rotate_vector(q, v):
    """
    Rota el vector v usando el cuaternion q mediante la formula sandwich:
        v' = q * [0, v] * q^-1

    Para cuaterniones unitarios, q^-1 = q* (conjugado).

    Args:
        q (np.ndarray): Cuaternion unitario [w, x, y, z].
        v (np.ndarray): Vector 3D [x, y, z].

    Returns:
        np.ndarray: Vector rotado [x, y, z].
    """
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return result[1:4]


def quat_to_rotation_matrix(q):
    """
    Convierte un cuaternion unitario a una matriz de rotacion 3x3.

    Args:
        q (np.ndarray): Cuaternion unitario [w, x, y, z].

    Returns:
        np.ndarray: Matriz de rotacion 3x3.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_from_rotation_matrix(R):
    """
    Convierte una matriz de rotacion 3x3 a cuaternion unitario.

    Usa el algoritmo de Shepperd para estabilidad numerica.

    Args:
        R (np.ndarray): Matriz de rotacion 3x3.

    Returns:
        np.ndarray: Cuaternion unitario [w, x, y, z].
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z]))


# ---------------------------------------------------------------------------
# Utilitarios internos
# ---------------------------------------------------------------------------

def _perpendicular_vector(v):
    """
    Devuelve un vector unitario perpendicular a v.

    Args:
        v (np.ndarray): Vector unitario [x, y, z].

    Returns:
        np.ndarray: Vector unitario perpendicular.
    """
    if abs(v[0]) < 0.9:
        candidate = np.array([1.0, 0.0, 0.0])
    else:
        candidate = np.array([0.0, 1.0, 0.0])
    perp = np.cross(v, candidate)
    return perp / np.linalg.norm(perp)
