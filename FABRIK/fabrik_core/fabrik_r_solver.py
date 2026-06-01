"""
fabrik_r_solver.py
==================
Implementacion de FABRIK-R para cadenas cinematicas seriales compuestas
exclusivamente de articulaciones revolutas de 1-DOF.

Fuentes normativas (orden de prioridad):
  - SANTOS21: Santos et al., "FABRIK-R: An Extension Developed Based on FABRIK
    for Robotics Manipulators." IEEE Access, vol. 9, 2021.
    DOI: 10.1109/ACCESS.2021.3070693
  - SANTOS22: Santos et al., "Inverse kinematics of a subsea constrained
    manipulator based on FABRIK-R." OCEANS 2022.
    DOI: 10.1109/OCEANS47191.2022.9977290

Diferencias criticas frente al FABRIK clasico (AL11 / ACL16):
  - El plano de rotacion de cada articulacion se determina de forma ENDOGENA
    usando el eje local z_i de la bisagra, NO el vector al target global.
  - Tras cada proyeccion planar, la longitud del eslabon se re-normaliza
    de forma estricta al valor nominal l_i.
  - El clamping angular se realiza via formula de Rodrigues en el plano
    de la bisagra (SANTOS21 Sec. III, SANTOS22 Sec. II-B).

Flujo del solver (SANTOS21, Algorithm 1 + Algorithm 2):
  Iteracion hasta convergencia o max_iterations:
    1. Pasada Backward (efector -> base): proyeccion planar local en cada joint.
    2. Restablece la base en su posicion original.
    3. Pasada Forward (base -> efector): proyeccion planar local en cada joint.
    4. Aplica clamping angular [theta_min, theta_max] via Rodrigues.
    5. Actualiza los ejes de rotacion locales propagando cuaterniones.

Trampa critica (AGENTS.md, Sec. 4):
  Si ||v_proj|| < EPS tras la proyeccion, el vector del eslabon coincide con
  el eje de la bisagra -> singularidad. Se debe usar el vector del paso
  anterior en lugar de dividir por cero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

EPS = 1e-9  # umbral para detectar vectores casi-nulos (singularidad)


# ---------------------------------------------------------------------------
# Descriptor de articulacion revoluta de 1-DOF
# ---------------------------------------------------------------------------

@dataclass
class RevoluteJoint:
    """
    Describe una articulacion revoluta de 1-DOF con su restriccion angular.

    Attributes:
        length      : Longitud del eslabon que conecta ESTE joint con el SIGUIENTE
                      (metros). Para el ultimo joint, puede ser 0 o no usarse.
        axis_local  : Eje de rotacion de la bisagra en coordenadas LOCALES del joint
                      (frame del segmento padre). DEBE ser unitario.
                      Referencia: AGENTS.md Sec. 3 "No Suposiciones Globales".
        theta_min   : Limite inferior del angulo de junta (radianes). Negativo = CW.
        theta_max   : Limite superior del angulo de junta (radianes). Positivo = ACW.
        ref_axis_local: Vector de referencia para theta=0, en el plano ortogonal a
                        axis_local. DEBE ser ortogonal a axis_local y unitario.
                        Referencia: SANTOS21 Sec. V, ec. (3); SANTOS22 Sec. II-B.
        segment_direction: Direccion inicial del segmento asociado al joint.
                   Solo se usa para la base cuando es una junta twist
                   pura (por ejemplo, el yaw de Niryo One).
        offset          : Desplazamiento de cero mecanico (radianes).
                          theta_hw = theta_calculado - offset.
                          Compensa la diferencia entre el cero cinematico y el cero
                          del encoder/hardware del actuador. AGENTS.md Sec. 5.
    """
    length:           float      = 1.0
    axis_local:       np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    theta_min:        float      = -math.pi
    theta_max:        float      =  math.pi
    ref_axis_local:   np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    segment_direction: Optional[np.ndarray] = None
    offset:           float      = 0.0

    def __post_init__(self) -> None:
        self.axis_local     = np.asarray(self.axis_local,   dtype=float)
        self.ref_axis_local = np.asarray(self.ref_axis_local, dtype=float)
        if self.segment_direction is not None:
            self.segment_direction = np.asarray(self.segment_direction, dtype=float)
        # Garantizar ejes unitarios
        self.axis_local     = _safe_normalize(self.axis_local)
        self.ref_axis_local = _safe_normalize(self.ref_axis_local)
        if self.segment_direction is not None:
            self.segment_direction = _safe_normalize(self.segment_direction)
        # Asegurar ortogonalidad ref_axis_local ⊥ axis_local (Gram-Schmidt)
        self.ref_axis_local = _safe_normalize(
            self.ref_axis_local - np.dot(self.ref_axis_local, self.axis_local) * self.axis_local
        )


# ---------------------------------------------------------------------------
# Resultado del solver
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Resultado de una llamada a FABRIKRSolver.solve().

    Attributes:
        joint_positions : Lista de n+1 posiciones 3D (base en [0], EE en [-1]).
        end_effector    : Posicion final del efector final (= joint_positions[-1]).
        iterations      : Iteraciones realizadas.
        converged       : True si ||EE - target|| < tolerance al terminar.
        final_error     : Distancia euclidiana EE - target al terminar.
        exit_reason     : Motivo de parada del bucle. Posibles valores:
                          'converged'      - error < tolerance (exito normal).
                          'stable'         - movimiento < stability_tolerance
                                             (punto fijo, posiblemente limitado).
                          'unreachable'    - target fuera del alcance total.
                          'converged_soft' - mejor postura dentro de soft_tolerance;
                                             robot bloqueado por limites pero
                                             suficientemente cerca del objetivo.
                                             converged=True.
                          'best_state_fallback' - se devuelve la mejor postura
                                             observada tras estancamiento;
                                             error > soft_tolerance.
                          'max_iterations' - se agotaron las iteraciones.
                          Referencia: AGENTS.md Sec. 5 (Fase 6 Optimizacion).
    """
    joint_positions: List[np.ndarray]
    end_effector:    np.ndarray
    iterations:      int
    converged:       bool
    final_error:     float
    exit_reason:     str = "unknown"


# ---------------------------------------------------------------------------
# Utilidades de algebra vectorial
# ---------------------------------------------------------------------------

def _safe_normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normaliza el vector v. Si ||v|| < EPS (singularidad), devuelve `fallback`
    (o [0,0,1] si no se provee).

    Trampa critica AGENTS.md Sec. 4: evita division por cero cuando el vector
    del eslabon propuesto es paralelo al eje de la bisagra.
    """
    n = np.linalg.norm(v)
    if n < EPS:
        if fallback is not None:
            return np.asarray(fallback, dtype=float).copy()
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _find_perpendicular(u: np.ndarray) -> np.ndarray:
    """
    Devuelve un vector unitario perpendicular a u (estable numericamente).

    Elige el eje canonico de menor alineacion con u para maximizar la
    estabilidad numerica del producto vectorial.
    Usado en el caso antiparalelo de _rotation_matrix_from_two_vectors.
    """
    abs_u = np.abs(u)
    if abs_u[0] <= abs_u[1] and abs_u[0] <= abs_u[2]:
        candidate = np.array([1.0, 0.0, 0.0])
    elif abs_u[1] <= abs_u[2]:
        candidate = np.array([0.0, 1.0, 0.0])
    else:
        candidate = np.array([0.0, 0.0, 1.0])
    return _safe_normalize(np.cross(u, candidate))


def _rotation_matrix_from_two_vectors(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Calcula la matriz de rotacion 3x3 minima que lleva el vector unitario u
    al vector unitario v (rotacion de angulo minimo, eje = normalize(u x v)).

    Usa la formula de Rodrigues en forma matricial:
        R = I + sin(a)*K + (1-cos(a))*K^2
    donde K es la matriz antisimetrica de k = normalize(u x v)
    y a = arccos(u . v).

    Casos especiales (AGENTS.md Sec. 4):
      - u ≈  v : devuelve identidad (sin rotacion).
      - u ≈ -v : rotacion de 180° alrededor de un eje perp. elegido de forma
                 numericamente estable via _find_perpendicular.

    Equivalente matematico a los cuaterniones usados en SANTOS21, ec. (3).

    Args:
        u : Vector unitario origen.
        v : Vector unitario destino.

    Returns:
        R : Matriz de rotacion 3x3.
    """
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))

    if dot >= 1.0 - EPS:
        return np.eye(3)

    if dot <= -1.0 + EPS:
        # Caso antiparalelo: rotar 180° alrededor de un eje perpendicular.
        # R = -I + 2 * perp * perp^T  (reflexion respecto al plano perp)
        perp = _find_perpendicular(u)
        return -np.eye(3) + 2.0 * np.outer(perp, perp)

    k     = _safe_normalize(np.cross(u, v))  # eje de rotacion
    sin_a = math.sqrt(max(0.0, 1.0 - dot * dot))  # sin(acos(dot)), numericamente estable
    cos_a = dot

    # Matriz antisimetrica de k (operador producto vectorial: K @ w = k x w)
    K = np.array([
        [    0.0, -k[2],  k[1]],
        [ k[2],    0.0,  -k[0]],
        [-k[1],  k[0],    0.0],
    ])

    # Formula de Rodrigues matricial: R = I + sin(a)*K + (1-cos(a))*K^2
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def project_to_axis_plane(
    link_vector: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """
    Proyecta `link_vector` sobre el plano ortogonal a `axis`.

    Esto es el paso central de FABRIK-R: fuerza al vector del eslabon a yacer
    en el plano de rotacion permitido por la bisagra de eje `axis`.

    Matematica (SANTOS21, Sec. V, ec. antes de ec. (1)):
        v_proj = link_vector - (link_vector . axis) * axis

    Args:
        link_vector : Vector 3D del eslabon propuesto (NO necesita ser unitario).
        axis        : Eje de rotacion de la bisagra (unitario).

    Returns:
        v_proj : Componente de `link_vector` en el plano ortogonal a `axis`.
                 Si ||v_proj|| < EPS (singularidad), se devuelve vector nulo.
                 El llamador DEBE manejar ese caso via `_safe_normalize`.
    """
    return link_vector - np.dot(link_vector, axis) * axis


def compute_link_vector_constrained(
    p_from: np.ndarray,
    p_to: np.ndarray,
    axis_global: np.ndarray,
    length: float,
    prev_link_direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calcula la nueva posicion del joint extremo al mover desde `p_from`
    con longitud `length`, restringido al plano ortogonal a `axis_global`.

    Pasos (SANTOS21, Sec. V, pasada Backward ec. antes de Algorithm 1):
      1. v_raw  = p_to - p_from
      2. v_proj = project_to_axis_plane(v_raw, axis_global)
      3. Si ||v_proj|| < EPS: usar `prev_link_direction` como fallback.
      4. p_new  = p_from + length * normalize(v_proj)

    Args:
        p_from              : Punto de origen (joint fijo en este paso).
        p_to                : Punto objetivo provisional (sin restriccion).
        axis_global         : Eje de la bisagra en el frame global actual.
        length              : Longitud exacta del eslabon.
        prev_link_direction : Direccion del eslabon en la iteracion anterior
                              (fallback ante singularidad). Unitario.

    Returns:
        p_new : Nueva posicion del joint a distancia `length` de `p_from`,
                en el plano de rotacion de la bisagra.
    """
    v_raw  = p_to - p_from
    v_proj = project_to_axis_plane(v_raw, axis_global)
    v_unit = _safe_normalize(v_proj, fallback=prev_link_direction)
    return p_from + length * v_unit


def clamp_angle_rodrigues(
    v_proj: np.ndarray,
    ref_axis:  np.ndarray,
    rot_axis:  np.ndarray,
    theta_min: float,
    theta_max: float,
    soft_margin: float = 0.0,
) -> np.ndarray:
    """
    Aplica clamping angular al vector `v_proj` dentro del plano de la bisagra.

    El angulo actual se mide desde `ref_axis` usando atan2. Si esta fuera de
    [theta_min, theta_max], se fuerza al limite mas cercano via rotacion de
    Rodrigues.

    Matematica (SANTOS21, Sec. V, ec. (3); SANTOS22, Sec. II-B):
        theta_actual = atan2( (v_proj x ref_axis) . rot_axis,  v_proj . ref_axis )
        theta_clamped = clamp(theta_actual, theta_min, theta_max)
        v_final = cos(theta_clamped)*ref_axis + sin(theta_clamped)*(rot_axis x ref_axis)

    Args:
        v_proj      : Vector unitario ya proyectado en el plano ortogonal a rot_axis.
        ref_axis    : Vector de referencia para theta=0 (unitario, en el plano).
        rot_axis    : Eje de rotacion de la bisagra (unitario), normal al plano.
        theta_min   : Limite angular inferior (radianes).
        theta_max   : Limite angular superior (radianes).
        soft_margin : Margen de amortiguacion (radianes). Cuando > 0, el clamping
                      efectivo usa [theta_min + soft_margin, theta_max - soft_margin].
                      Evita que el joint quede exactamente en el limite duro,
                      reduciendo oscilaciones de bordes ('boundary chattering').
                      Si el rango efectivo resulta invalido (2*soft_margin > rango),
                      se ignora el margen y se usan los limites duros sin modificar.
                      Referencia: SANTOS22 Sec. II-B; AGENTS.md Sec. 5 Fase 6.

    Returns:
        v_clamped : Vector unitario resultante tras el clamping, en el plano.
    """
    # Eje perpendicular a ref_axis dentro del plano: rot_axis x ref_axis
    perp = np.cross(rot_axis, ref_axis)

    # Angulo actual respecto a ref_axis (SANTOS21 ec. (2) / SANTOS22 Sec. II-B)
    theta_actual = math.atan2(np.dot(v_proj, perp), np.dot(v_proj, ref_axis))

    # Limites efectivos con margen de amortiguacion (Fase 6, clamping suave)
    # Si el margen es demasiado grande para el rango, caer al limite duro.
    eff_min = theta_min + soft_margin
    eff_max = theta_max - soft_margin
    if eff_min >= eff_max:
        eff_min, eff_max = theta_min, theta_max

    # Clamping al rango efectivo
    theta_clamped = max(eff_min, min(eff_max, theta_actual))

    # Reconstruccion via Rodrigues en el plano (SANTOS21 ec. (3) simplificada)
    v_clamped = math.cos(theta_clamped) * ref_axis + math.sin(theta_clamped) * perp
    return _safe_normalize(v_clamped)


def rotate_vector_rodrigues(
    v: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rota el vector `v` un angulo `angle` (radianes) alrededor de `axis`
    usando la formula de Rodrigues.

    Matematica (SANTOS21, Sec. V, ec. (3)):
        v_rot = cos(angle)*v + (1-cos(angle))*(axis.v)*axis + sin(angle)*(axis x v)

    Usado para propagar los ejes locales de las bisagras a lo largo de la cadena
    tras cada iteracion (AGENTS.md Sec. 4, "Giro del Eje del Frame Anterior").

    Args:
        v     : Vector a rotar (no necesita ser unitario).
        axis  : Eje de rotacion (unitario).
        angle : Angulo de rotacion en radianes.

    Returns:
        v_rot : Vector rotado.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (cos_a * v
            + (1.0 - cos_a) * np.dot(axis, v) * axis
            + sin_a * np.cross(axis, v))


def extract_joint_angle(
    link_vector: np.ndarray,
    ref_axis: np.ndarray,
    rot_axis: np.ndarray,
) -> float:
    """
    Extrae el angulo de junta theta_i desde el vector del eslabon resultante.

    Matematica (SANTOS21, Sec. V; SANTOS22, Sec. II-B):
        theta = atan2( (link_vector x ref_axis) . rot_axis,
                        link_vector . ref_axis )

    Args:
        link_vector : Direccion normalizada del eslabon (en el plano de la bisagra).
        ref_axis    : Referencia de cero (unitario, en el plano).
        rot_axis    : Eje de rotacion (unitario).

    Returns:
        theta : Angulo de junta en radianes en [-pi, pi].
    """
    perp = np.cross(rot_axis, ref_axis)
    return math.atan2(np.dot(link_vector, perp), np.dot(link_vector, ref_axis))


# ---------------------------------------------------------------------------
# Solver principal FABRIK-R
# ---------------------------------------------------------------------------

class FABRIKRSolver:
    """
    Solver de cinematica inversa FABRIK-R para cadenas seriales de 1-DOF.

    Referencia central: SANTOS21 (Algorithm 1 + Algorithm 2) y SANTOS22.

    Uso basico:
        joints = [RevoluteJoint(length=0.1, ...), ...]
        initial_positions = [np.array([0,0,0]), np.array([0,0,0.1]), ...]
        solver = FABRIKRSolver(joints, initial_positions)
        result = solver.solve(target=np.array([0.2, 0.1, 0.15]))

    Nota: `initial_positions` debe tener len(joints) + 1 puntos.
          El punto [0] es la base (fija). El punto [-1] es el efector final.
    """

    def __init__(
        self,
        joints: List[RevoluteJoint],
        initial_positions: List[np.ndarray],
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        angle_soft_margin: float = 0.01,
        stability_tolerance: float = 1e-6,
        soft_tolerance: float = 0.0,
    ) -> None:
        """
        Args:
            joints               : Descriptores de las articulaciones (n joints).
            initial_positions    : Posiciones iniciales de n+1 puntos (base + joints).
            max_iterations       : Limite de iteraciones del bucle FABRIK-R.
            tolerance            : Distancia al target para convergencia (m).
            angle_soft_margin    : Margen de amortiguacion para clamping angular (rad).
                                   El clamping efectivo usa
                                   [theta_min + margin, theta_max - margin].
                                   Reduce 'boundary chattering' en joints limitados.
                                   Valor por defecto ~0.57 deg. Use 0.0 para
                                   clamping duro sin margen.
                                   Referencia: AGENTS.md Sec. 5 Fase 6.
            stability_tolerance  : Umbral de movimiento para early exit (m).
                                   Si el desplazamiento maximo de cualquier punto
                                   entre dos iteraciones consecutivas es menor que
                                   este valor, el solver para (punto fijo).
                                   Referencia: AGENTS.md Sec. 5 Fase 6.
            soft_tolerance       : Umbral de convergencia practica (m).
                                   Si best_error < soft_tolerance al salir por
                                   best_state_fallback, se clasifica como
                                   'converged_soft' (converged=True). Util cuando
                                   los limites articulares impiden llegar a
                                   tolerance pero el error fisico es aceptable.
                                   Cuando es 0.0 (defecto) se usa tolerance*5.0.
        """
        assert len(initial_positions) == len(joints) + 1, (
            f"Se esperan {len(joints)+1} posiciones para {len(joints)} joints."
        )

        self.joints     = joints
        self.n          = len(joints)
        self.base       = np.array(initial_positions[0], dtype=float)
        self._init_pos  = [np.array(p, dtype=float) for p in initial_positions]
        self.positions  = [np.array(p, dtype=float) for p in initial_positions]

        self.max_iterations     = max_iterations
        self.tolerance          = tolerance
        self.angle_soft_margin  = angle_soft_margin
        self.stability_tolerance = stability_tolerance
        self.soft_tolerance      = soft_tolerance if soft_tolerance > 0.0 else tolerance * 5.0
        self._last_theta0       = 0.0
        # RNG local: no contamina el estado global de numpy (vital para tests
        # con semilla fija que generan targets despues de crear el solver).
        self._rng = np.random.default_rng()

        # Longitudes nominales de eslabones (fijas durante toda la resolucion)
        # AGENTS.md Sec. 3: "Preservacion de Longitudes"
        self.lengths: List[float] = [j.length for j in joints]

        # Ejes de bisagra en frame global, inicialmente = axis_local
        # Se actualizan al propagar rotaciones entre iteraciones.
        # AGENTS.md Sec. 4: "Giro del Eje del Frame Anterior (Twist)"
        self._axes_global: List[np.ndarray] = [
            np.array(j.axis_local, dtype=float) for j in joints
        ]
        # Ejes de referencia (theta=0) en frame global
        self._refs_global: List[np.ndarray] = [
            np.array(j.ref_axis_local, dtype=float) for j in joints
        ]

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_robot(
        cls,
        robot,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        angle_soft_margin: float = 0.01,
        stability_tolerance: float = 1e-6,
        soft_tolerance: float = 0.0,
    ) -> "FABRIKRSolver":
        """
        Crea un FABRIKRSolver a partir de un objeto Robot cargado desde YAML.

        Construye automaticamente la lista de RevoluteJoint y las posiciones
        iniciales a partir de los eslabones del robot (link.joint_coords
        acumulados desde el origen).

        La longitud de cada eslabon es la distancia euclidiana entre puntos
        consecutivos en la configuracion cero (calcula a partir de joint_coords),
        asegurando la invariante de FABRIK: ||p[i+1]-p[i]|| = length_i.

        La referencia angular (ref_axis_local[i]) se inicializa a la direccion
        del eslabon i en la configuracion cero. Si esa direccion coincide con
        el eje de bisagra (singularidad de Gram-Schmidt), se usa un eje
        perpendicular calculado de forma numericamente estable.

        Referencia: AGENTS.md Sec. 5; SANTOS21 Sec. II (definicion de la cadena).

        Args:
            robot          : Objeto Robot con .links (lista de Link).
            max_iterations : Limite de iteraciones del bucle FABRIK-R.
            tolerance      : Tolerancia de convergencia (metros).

        Returns:
            FABRIKRSolver configurado con los joints del robot.
        """
        # ------ Posiciones iniciales (acumulacion de joint_coords) ------
        # p[0] = base (origen del mundo); p[i+1] = p[i] + link_i.joint_coords
        # Coincide con el calculo de robot.M en cargar_robot_desde_yaml.
        p: List[np.ndarray] = [np.zeros(3)]
        for link in robot.links:
            p.append(p[-1] + np.asarray(link.joint_coords, dtype=float))

        # ------ Descriptores de articulacion ----------------------------
        joints_list: List[RevoluteJoint] = []
        for i, link in enumerate(robot.links):
            axis = _safe_normalize(np.asarray(link.joint_axis, dtype=float))

            # Longitud = distancia real entre puntos consecutivos en config cero
            # (puede diferir de link.length para eslabones con desplazamiento lateral)
            # AGENTS.md Sec. 3 "Preservacion de Longitudes"
            length = float(np.linalg.norm(p[i + 1] - p[i]))
            if length < EPS:
                raise ValueError(
                    f"Eslabon {link.id} tiene longitud cero "
                    f"(joint_coords={link.joint_coords}). No es valido para FABRIK-R."
                )

            # ref_axis_local: direccion inicial del eslabon i en su frame local
            # Si es paralela al eje de bisagra, usar una perpendicular estable
            link_dir = _safe_normalize(p[i + 1] - p[i])
            if abs(float(np.dot(axis, link_dir))) > 1.0 - 1e-3:
                # Caso especial: eslabon alineado con eje (ej. Base de Niryo)
                # __post_init__ haria Gram-Schmidt = vector nulo -> usar perpendicular
                ref = _find_perpendicular(axis)
            else:
                # __post_init__ ortogonalizara este vector via Gram-Schmidt
                ref = link_dir

            # Limites articulares desde el YAML; sin limites = libre
            if link.joint_limits is not None:
                lo, hi = float(link.joint_limits[0]), float(link.joint_limits[1])
            else:
                lo, hi = -math.pi, math.pi

            joints_list.append(RevoluteJoint(
                length=length,
                axis_local=axis,
                ref_axis_local=ref,
                segment_direction=link_dir if i == 0 and abs(float(np.dot(axis, link_dir))) > 1.0 - 1e-3 else None,
                theta_min=lo,
                theta_max=hi,
                offset=0.0,  # sin compensacion por defecto; ajustar segun hardware
            ))

        return cls(
            joints_list, p,
            max_iterations=max_iterations,
            tolerance=tolerance,
            angle_soft_margin=angle_soft_margin,
            stability_tolerance=stability_tolerance,
            soft_tolerance=soft_tolerance,
        )

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def reset_to_initial(self) -> None:
        """Reinicia la cadena a la configuracion inicial y los ejes locales."""
        self.positions   = [p.copy() for p in self._init_pos]
        self._axes_global = [np.array(j.axis_local, dtype=float) for j in self.joints]
        self._refs_global = [np.array(j.ref_axis_local, dtype=float) for j in self.joints]
        self._last_theta0 = 0.0

    def _yaw_preprocess(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Reexpresa el target en el plano del brazo cuando J0 es una junta twist.

        La base de robots como Niryo One gira alrededor de un eje paralelo al
        primer segmento. Esa rotacion no cambia posiciones, asi que el angulo
        de yaw debe fijarse con una convencion geometrica adicional para que no
        quede libre.
        """
        desc0 = self.joints[0]
        if desc0.segment_direction is None:
            return target, 0.0

        yaw_axis = _safe_normalize(desc0.segment_direction)
        base = self.base
        t_rel = target - base

        t_perp = project_to_axis_plane(t_rel, yaw_axis)
        if np.linalg.norm(t_perp) < EPS:
            return target, 0.0

        x_ref = project_to_axis_plane(np.array([1.0, 0.0, 0.0]), yaw_axis)
        if np.linalg.norm(x_ref) < EPS:
            x_ref = project_to_axis_plane(np.array([0.0, 1.0, 0.0]), yaw_axis)
        x_ref = _safe_normalize(x_ref)

        theta0 = extract_joint_angle(
            link_vector=_safe_normalize(t_perp),
            ref_axis=x_ref,
            rot_axis=yaw_axis,
        )
        target_local = base + rotate_vector_rodrigues(t_rel, yaw_axis, -theta0)
        return target_local, theta0

    def _yaw_postprocess(self, positions: List[np.ndarray], theta0: float) -> List[np.ndarray]:
        """Rota la solucion completa alrededor del eje de yaw de J0."""
        if abs(theta0) < EPS:
            return positions

        desc0 = self.joints[0]
        if desc0.segment_direction is None:
            return positions

        yaw_axis = _safe_normalize(desc0.segment_direction)
        base = self.base
        rotated = [positions[0].copy()]
        for point in positions[1:]:
            rel = point - base
            rotated.append(base + rotate_vector_rodrigues(rel, yaw_axis, theta0))
        return rotated

    def _finalize_yaw_solution(self, positions: List[np.ndarray], theta0: float) -> List[np.ndarray]:
        """Aplica la rotacion de yaw y sincroniza los ejes globales finales."""
        final_positions = self._yaw_postprocess(positions, theta0)
        self._update_global_axes(final_positions)
        self._last_theta0 = theta0
        return final_positions

    def solve(self, target: np.ndarray) -> SolverResult:
        """
        Ejecuta el bucle FABRIK-R hasta convergencia o max_iterations.

        Args:
            target : Posicion objetivo 3D para el efector final.

        Returns:
            SolverResult con posiciones finales, iteraciones y error.
        """
        target = np.asarray(target, dtype=float)

        solve_target = target
        yaw_theta0 = 0.0
        if self.joints and self.joints[0].segment_direction is not None:
            solve_target, yaw_theta0 = self._yaw_preprocess(target)

        # Verificar alcanzabilidad: longitud total de la cadena
        total_length = sum(self.lengths)
        dist_to_target = np.linalg.norm(solve_target - self.base)
        if dist_to_target > total_length + EPS:
            # Target inalcanzable: estirar la cadena hacia el target
            result = self._solve_unreachable(solve_target)
            result.joint_positions = self._finalize_yaw_solution(result.joint_positions, yaw_theta0)
            result.end_effector = result.joint_positions[-1].copy()
            result.final_error = float(np.linalg.norm(result.end_effector - target))
            self.positions = result.joint_positions
            return result

        p = [pos.copy() for pos in self.positions]
        if self.joints and self.joints[0].segment_direction is not None:
            # Para J0 twist, resolver en el frame local del brazo evita mezclar
            # un target de-yawed con una cadena que venia en yaw global acumulado.
            # Referencia: SANTOS21 Sec. V (frames locales consistentes).
            p = self._yaw_postprocess(p, -yaw_theta0)
            p[0] = self.base.copy()
            p[1] = p[0] + self.lengths[0] * self.joints[0].segment_direction

        best_positions = [pos.copy() for pos in p]
        best_error = float(np.linalg.norm(p[-1] - solve_target))
        max_escapes = 3
        escapes_used = 0
        patience_limit = 4
        patience_counter = 0
        min_escape_iter = max(20, int(0.45 * self.max_iterations))
        improvement_tol = max(self.stability_tolerance, self.tolerance * 1e-6)
        fallback_error_threshold = max(1e-3, self.tolerance)
        limit_violation_eps = max(1e-6, self.angle_soft_margin * 0.2)
        limit_stuck_counter = 0
        limit_stuck_required = 2

        p_prev2: Optional[List[np.ndarray]] = None
        prev_error = float("inf")

        for iteration in range(1, self.max_iterations + 1):

            # Item 3 - Early Exit: instantanea de posiciones al inicio de la
            # iteracion para calcular el movimiento total de la cadena.
            # AGENTS.md Sec. 5 Fase 6 (Optimizacion de Convergencia).
            p_prev = [pos.copy() for pos in p]

            # ---- Actualizar ejes globales (OBLIGATORIO antes de Backward) ---
            # Los ejes se sincronizan con la configuracion de la iteracion
            # anterior (o con la inicial en la primera iteracion) usando FK
            # acumulativa. SANTOS21, Sec. V; AGENTS.md Sec. 4.
            self._update_global_axes(p)

            # ---- Pasada Backward (efector -> base) ----------------------
            # SANTOS21, Sec. V, pasada Backward modificada para 1-DOF.
            # README Sec. 2B:
            #   p*_i = p_{i+1} + l_i*(p_i - p_{i+1})/||p_i - p_{i+1}||
            #   v_proj = (p*_i - p_{i+1}) - [(p*_i-p_{i+1})·z_i]*z_i
            #   p_final_i = p_{i+1} + l_i * v_proj/||v_proj||
            p[-1] = solve_target.copy()
            for i in range(self.n - 1, 0, -1):
                # Eje global del joint i (pre-calculado; no se actualiza mid-pass)
                axis_g   = self._axes_global[i]
                prev_dir = _safe_normalize(p[i] - p[i + 1])
                p[i] = compute_link_vector_constrained(
                    p_from=p[i + 1],
                    p_to=p[i],
                    axis_global=axis_g,
                    length=self.lengths[i],
                    prev_link_direction=prev_dir,
                )
                # Clamping angular via Rodrigues (SANTOS21 Sec. V, ec. (3))
                p[i] = self._apply_angle_clamp(i, p, from_idx=i + 1)

            # ---- Restaurar base (AL11, Sec. 3; SANTOS21, Sec. V) ---------
            p[0] = self.base.copy()

            # ---- Pasada Forward (base -> efector) -----------------------
            # SANTOS21, Sec. V, pasada Forward modificada para 1-DOF.
            # README Sec. 2C:
            #   p*_i = p_{i-1} + l_{i-1}*(p_i - p_{i-1})/||p_i - p_{i-1}||
            #   v_proj = (p*_i - p_{i-1}) - [(p*_i-p_{i-1})·z_{i-1}]*z_{i-1}
            #   p_final_i = p_{i-1} + l_{i-1} * v_proj/||v_proj||
            for i in range(0, self.n):
                if i == 0 and self.joints[0].segment_direction is not None:
                    # Junta twist en base: el primer segmento no debe inclinarse.
                    # La base solo rota sobre su propio eje; no traslada p[1].
                    p[1] = p[0] + self.lengths[0] * self.joints[0].segment_direction
                    continue

                # Mismo set de ejes del inicio de la iteracion (coherente con SANTOS21)
                axis_g   = self._axes_global[i]
                prev_dir = _safe_normalize(p[i + 1] - p[i])
                p[i + 1] = compute_link_vector_constrained(
                    p_from=p[i],
                    p_to=p[i + 1],
                    axis_global=axis_g,
                    length=self.lengths[i],
                    prev_link_direction=prev_dir,
                )
                # Clamping angular via Rodrigues (SANTOS21 Sec. V, ec. (3))
                p[i + 1] = self._apply_angle_clamp(i, p, from_idx=i)

            # ---- Early Exit por estabilidad (Fase 6) -------------------
            # Si el maximo desplazamiento de cualquier punto en esta iteracion
            # es menor que stability_tolerance, el solver alcanzo un punto fijo.
            # Esto ocurre cuando los joints estan bloqueados por sus limites y
            # no pueden acercarse mas al target. Salir antes de max_iterations
            # evita iteraciones innecesarias con movimiento nulo.
            # Referencia: AGENTS.md Sec. 5 Fase 6; SANTOS21 Sec. V.
            chain_movement = max(
                float(np.linalg.norm(p[k] - p_prev[k]))
                for k in range(self.n + 1)
            )

            # Detecta oscilacion de borde tipo 2-ciclo (A-B-A-B), comun cuando
            # el target empuja contra limites angulares y el clamping alterna
            # entre dos configuraciones cercanas.
            chain_movement_2cycle = None
            if p_prev2 is not None:
                chain_movement_2cycle = max(
                    float(np.linalg.norm(p[k] - p_prev2[k]))
                    for k in range(self.n + 1)
                )

            # ---- Test de convergencia -----------------------------------
            error = float(np.linalg.norm(p[-1] - solve_target))

            if error < best_error:
                best_error = error
                best_positions = [pos.copy() for pos in p]

            if error < self.tolerance:
                p = self._finalize_yaw_solution(p, yaw_theta0)
                self.positions = p
                return SolverResult(
                    joint_positions=p,
                    end_effector=p[-1].copy(),
                    iterations=iteration,
                    converged=True,
                    final_error=float(np.linalg.norm(p[-1] - target)),
                    exit_reason="converged",
                )

            small_motion = (
                chain_movement < self.stability_tolerance
                or (
                    chain_movement_2cycle is not None
                    and chain_movement_2cycle < self.stability_tolerance
                )
            )
            rel_tol = max(improvement_tol, abs(prev_error) * 1e-3)
            improved = (prev_error - error) > rel_tol

            # Medir cuan forzada esta cada junta respecto a sus limites.
            # Si no hay saturacion real, evitar reseeds para no canibalizar
            # convergencia natural en problemas recuperables.
            thetas_now = self.extract_joint_angles(p)
            violation_scores = self._joint_limit_violation_scores(thetas_now)
            has_limit_violations = any(score > limit_violation_eps for score in violation_scores)
            if has_limit_violations and small_motion and (not improved):
                limit_stuck_counter += 1
            else:
                limit_stuck_counter = 0

            # Paciencia adaptativa: solo cuenta cuando la cadena YA esta casi fija
            # y ademas no hay mejora significativa de error.
            if small_motion and (not improved):
                patience_counter += 1
            else:
                patience_counter = 0

            # Dos caminos de escape:
            # 1) Violaciones activas + persistentes: escape agresivo temprano.
            # 2) Sin violaciones pero sin mejora prolongada: escape uniforme.
            stuck_by_limits = (
                has_limit_violations
                and limit_stuck_counter >= limit_stuck_required
                and patience_counter >= patience_limit
            )
            stuck_by_patience = patience_counter >= patience_limit * 2
            stagnated = (
                iteration >= min_escape_iter
                and (stuck_by_limits or stuck_by_patience)
            )
            if stagnated and error > self.tolerance:
                if escapes_used < max_escapes:
                    p_reseed = self._apply_controlled_reseed(
                        p,
                        escape_idx=escapes_used,
                        violation_scores=violation_scores,
                    )
                    escapes_used += 1  # siempre contar el intento
                    if p_reseed is not None:
                        # Reseed exitoso: aplicar nueva configuracion y reiniciar
                        # contadores para dar al solver un arranque fresco.
                        p = p_reseed
                        reseed_error = float(np.linalg.norm(p[-1] - solve_target))
                        if reseed_error < best_error:
                            best_error = reseed_error
                            best_positions = [pos.copy() for pos in p]
                        patience_counter = 0
                        limit_stuck_counter = 0
                        prev_error = float("inf")
                        p_prev2 = None
                        continue
                    # Reseed fallido (None): solo contar el intento; NO reiniciar
                    # paciencia ni prev_error para que el solver siga convergiendo
                    # normalmente sin agotarse prematuramente.
                else:
                    # Escapes agotados y solver aun atascado: devolver mejor estado.
                    p = [pos.copy() for pos in best_positions]
                    p = self._finalize_yaw_solution(p, yaw_theta0)
                    self.positions = p
                    final_err_fb = float(np.linalg.norm(p[-1] - target))
                    # Convergencia practica: cerca del objetivo dadas las
                    # restricciones articulares -> converged_soft.
                    soft_ok = best_error < self.soft_tolerance
                    return SolverResult(
                        joint_positions=p,
                        end_effector=p[-1].copy(),
                        iterations=iteration,
                        converged=soft_ok,
                        final_error=final_err_fb,
                        exit_reason="converged_soft" if soft_ok else "best_state_fallback",
                    )

            prev_error = error

            p_prev2 = p_prev

        # max_iterations alcanzado: restaurar el mejor estado absoluto observado.
        # Solo etiquetar fallback si realmente se agotaron escapes y el error
        # sigue alto; en otro caso, reportar max_iterations.

            prev_error = error

            p_prev2 = p_prev

        # max_iterations alcanzado: restaurar el mejor estado absoluto observado.
        # Solo etiquetar fallback si realmente se agotaron escapes y el error
        # sigue alto; en otro caso, reportar max_iterations.
        p = [pos.copy() for pos in best_positions]
        p = self._finalize_yaw_solution(p, yaw_theta0)
        self.positions = p
        exit_reason = "max_iterations"
        if escapes_used >= max_escapes and best_error > fallback_error_threshold:
            soft_ok = best_error < self.soft_tolerance
            exit_reason = "converged_soft" if soft_ok else "best_state_fallback"
        return SolverResult(
            joint_positions=p,
            end_effector=p[-1].copy(),
            iterations=self.max_iterations,
            converged=(
                float(np.linalg.norm(p[-1] - target)) < self.tolerance
                or exit_reason == "converged_soft"
            ),
            final_error=float(np.linalg.norm(p[-1] - target)),
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Metodos internos
    # ------------------------------------------------------------------

    def _apply_angle_clamp(
        self, joint_idx: int, p: List[np.ndarray], from_idx: int
    ) -> np.ndarray:
        """
        Aplica el clamping angular al punto p[from_idx +/- 1] usando la
        formula de Rodrigues sobre el plano de la bisagra.

        Referencia: SANTOS21 Sec. V ec. (3); SANTOS22 Sec. II-B.

        Args:
            joint_idx : Indice del joint cuya restriccion se aplica.
            p         : Lista de posiciones actuales.
            from_idx  : Indice del punto que actua como origen del eslabon.

        Returns:
            Posicion corregida del punto distal del eslabon.
        """
        j = self.joints[joint_idx]
        # Indice del punto distal
        to_idx = from_idx + 1 if from_idx == joint_idx else from_idx - 1

        link_vec = p[to_idx] - p[from_idx]
        length   = self.lengths[joint_idx]
        axis_g   = self._axes_global[joint_idx]
        ref_g    = self._refs_global[joint_idx]

        # Proyectar sobre el plano de la bisagra
        v_proj   = project_to_axis_plane(link_vec, axis_g)
        v_unit   = _safe_normalize(v_proj, fallback=ref_g)

        # Clamping angular via Rodrigues con soft_margin (SANTOS21 Sec. V, ec. (3))
        # El soft_margin reduce 'boundary chattering': en lugar de clampear al
        # limite exacto, deja un pequeno margen interior. Fase 6 Optimizacion.
        v_clamped = clamp_angle_rodrigues(
            v_proj=v_unit,
            ref_axis=ref_g,
            rot_axis=axis_g,
            theta_min=j.theta_min,
            theta_max=j.theta_max,
            soft_margin=self.angle_soft_margin,
        )

        return p[from_idx] + length * v_clamped

    def _compute_chain_frames(self, p: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calcula ejes y referencias globales de toda la cadena para una configuracion.

        Este metodo centraliza la FK acumulativa de frames para evitar desfasajes
        entre la logica usada por el clamping interno y la extraccion de angulos
        hacia hardware. Los ejes/referencias se devuelven en el frame del PADRE
        de cada joint (R_cumul antes de actualizar con el eslabon i).

        Referencia: SANTOS21 Sec. V (composicion de rotaciones por eslabon).
        """
        axes_global: List[np.ndarray] = [np.zeros(3) for _ in range(self.n)]
        refs_global: List[np.ndarray] = [np.zeros(3) for _ in range(self.n)]

        R_cumul = np.eye(3)
        for i in range(self.n):
            # Frame del padre de i (antes de incluir la rotacion del eslabon i).
            axes_global[i] = _safe_normalize(R_cumul @ self.joints[i].axis_local)
            refs_global[i] = _safe_normalize(R_cumul @ self.joints[i].ref_axis_local)

            d_init = _safe_normalize(self._init_pos[i + 1] - self._init_pos[i])
            d_curr = _safe_normalize(p[i + 1] - p[i])
            d_init_world = R_cumul @ d_init
            R_link = _rotation_matrix_from_two_vectors(d_init_world, d_curr)
            R_cumul = R_link @ R_cumul

        return axes_global, refs_global

    def _update_global_axes(self, p: List[np.ndarray]) -> None:
        """
        Actualiza los ejes de bisagra en frame global mediante FK acumulativa.

        Para cada joint i, el frame global se obtiene componiendo las rotaciones
        de los eslabones 0..i en orden, desde la base hacia el efector. Esto
        garantiza que el eje de cada bisagra rota solidariamente con todos los
        eslabones previos (AGENTS.md Sec. 3 'No Suposiciones Globales').

        Algoritmo (SANTOS21, Sec. V; AGENTS.md Sec. 4 'Twist del Frame'):
          R_cumul = I
          Para cada eslabon k de 0 a n-1:
            d_init_k_world = R_cumul @ d_init_k        (dir. inicial ya rotada)
            R_k = rot_matrix(d_init_k_world, d_curr_k) (rot. adicional del eslabon k)
            R_cumul = R_k @ R_cumul                    (composicion en frame global)
            axis_global[k]  = normalize(R_cumul @ axis_local[k])
            ref_global[k]   = normalize(R_cumul @ ref_axis_local[k])

        La composicion R_k @ R_cumul es equivalente a los cuaterniones acumulados
        descritos en SANTOS21 ec. (3). Se implementa con matrices de rotacion
        (via _rotation_matrix_from_two_vectors) para evitar dependencias externas.

        Singularidades gestionadas por _rotation_matrix_from_two_vectors:
          - Eslabon paralelo a posicion inicial  -> R = I (sin cambio).
          - Eslabon antiparalelo                 -> rotacion 180° numericamente estable.
        """
        self._axes_global, self._refs_global = self._compute_chain_frames(p)

    def _solve_unreachable(self, target: np.ndarray) -> SolverResult:
        """
        Caso target inalcanzable (SANTOS21, Sec. III): estirar la cadena en
        la direccion del target desde la base.

        Referencia: AL11, Sec. 3 "When the target is unreachable".
        """
        p = [self.base.copy()]

        if self.joints and self.joints[0].segment_direction is not None:
            # Base twist (ej. Niryo J0): mantener fijo el primer segmento sobre su eje.
            p.append(p[0] + self.lengths[0] * self.joints[0].segment_direction)
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, self.n):
            direction = _safe_normalize(target - p[i])
            p.append(p[i] + self.lengths[i] * direction)

        self.positions = p
        return SolverResult(
            joint_positions=p,
            end_effector=p[-1].copy(),
            iterations=1,
            converged=False,
            final_error=float(np.linalg.norm(p[-1] - target)),
            exit_reason="unreachable",
        )

    def _positions_from_joint_angles(self, thetas: List[float]) -> List[np.ndarray]:
        """
        Reconstruye posiciones de joints desde angulos, preservando longitudes.

        Usa el mismo convenio de frames que _compute_chain_frames para mantener
        coherencia exacta entre clamping y extraccion de angulos.
        """
        p: List[np.ndarray] = [self.base.copy()]
        R_cumul = np.eye(3)

        for i in range(self.n):
            j = self.joints[i]
            if i == 0 and j.segment_direction is not None:
                link_dir = _safe_normalize(j.segment_direction)
            else:
                axis_g = _safe_normalize(R_cumul @ j.axis_local)
                ref_g = _safe_normalize(R_cumul @ j.ref_axis_local)
                perp = _safe_normalize(np.cross(axis_g, ref_g), fallback=_find_perpendicular(axis_g))
                link_dir = _safe_normalize(
                    math.cos(thetas[i]) * ref_g + math.sin(thetas[i]) * perp,
                    fallback=ref_g,
                )

            p.append(p[i] + self.lengths[i] * link_dir)

            d_init = _safe_normalize(self._init_pos[i + 1] - self._init_pos[i])
            d_init_world = R_cumul @ d_init
            R_link = _rotation_matrix_from_two_vectors(d_init_world, link_dir)
            R_cumul = R_link @ R_cumul

        return p

    def _joint_limit_violation_scores(self, thetas: List[float]) -> List[float]:
        """
        Calcula una puntuacion de saturacion por junta.

        score = violacion_fuera_de_limites + proximidad_al_borde.
        Se usa para activar reseed solo cuando hay bloqueo real por limites
        y para priorizar perturbaciones en las juntas mas comprometidas.
        """
        scores: List[float] = []
        edge_band = max(1e-4, self.angle_soft_margin * 2.0)

        for i, theta in enumerate(thetas):
            lo = self.joints[i].theta_min
            hi = self.joints[i].theta_max

            outside_low = max(0.0, lo - theta)
            outside_high = max(0.0, theta - hi)
            outside = outside_low + outside_high

            dist_to_edge = min(abs(theta - lo), abs(hi - theta))
            edge_pressure = max(0.0, edge_band - dist_to_edge)

            scores.append(float(outside + edge_pressure))

        return scores

    def _apply_controlled_reseed(
        self,
        p: List[np.ndarray],
        escape_idx: int,
        violation_scores: List[float],
    ) -> Optional[List[np.ndarray]]:
        """
        Escape focalizado: perturba 1-2 joints mas saturados (2°..5°) dentro
        de limites y reconstruye la cadena via FK.
        """
        if self.n < 2:
            return None

        thetas = self.extract_joint_angles(p)

        has_twist_base = (
            self.joints
            and self.joints[0].segment_direction is not None
        )
        start_idx = 1 if has_twist_base else 0
        candidates = [i for i in range(start_idx, self.n)]
        if not candidates:
            return None

        weighted = []
        for i in candidates:
            score = max(0.0, float(violation_scores[i]))
            weighted.append((score, i))
        weighted.sort(reverse=True)

        # Sin saturacion detectable: reseed aleatorio uniforme.
        # No devolver None porque el solver puede estar en un minimo local
        # sin violaciones de limites (chattering puro, no por saturacion).
        if weighted[0][0] <= 0.0:
            idxs = candidates
            ws = np.ones(len(candidates), dtype=float) / len(candidates)
        else:
            top = weighted[: min(3, len(weighted))]
            idxs = [j for _, j in top]
            ws = np.array([max(1e-6, s) for s, _ in top], dtype=float)
            ws = ws / np.sum(ws)

        primary = int(self._rng.choice(idxs, p=ws))
        # Reseed escalonado (Fase 7): primeros 2 intentos perturban 1 joint
        # (menor ruido, favorece precision); 3er intento en adelante activa
        # 2 joints para romper deadlocks profundos donde 1 solo no basta.
        secondary = None
        if escape_idx >= 2 and len(idxs) > 1:
            remain = [j for j in idxs if j != primary]
            if remain:
                secondary = remain[escape_idx % len(remain)]

        delta_deg = [2.0, 3.0, 4.0, 5.0][escape_idx % 4]
        delta = math.radians(delta_deg)

        def _clip_theta(idx: int, value: float) -> float:
            lo, hi = self.joints[idx].theta_min, self.joints[idx].theta_max
            return float(np.clip(value, lo, hi))

        def _sign_to_center(idx: int, theta_value: float) -> float:
            lo, hi = self.joints[idx].theta_min, self.joints[idx].theta_max
            center = 0.5 * (lo + hi)
            s = 1.0 if theta_value < center else -1.0
            # Pequena aleatoriedad para salir de ciclos limite-limite.
            if self._rng.random() < 0.15:
                s *= -1.0
            return s

        sign_primary = _sign_to_center(primary, thetas[primary])
        thetas[primary] = _clip_theta(primary, thetas[primary] + sign_primary * delta)
        if secondary is not None and secondary != primary:
            sign_secondary = _sign_to_center(secondary, thetas[secondary])
            thetas[secondary] = _clip_theta(secondary, thetas[secondary] + sign_secondary * delta)

        return self._positions_from_joint_angles(thetas)

    def extract_joint_angles(self, p: List[np.ndarray]) -> List[float]:
        """
        Calcula el angulo de junta theta_i para cada articulacion a partir de
        la configuracion de posiciones `p`.

        Para la articulacion i, theta_i se mide desde la direccion de referencia
        (theta=0) expresada en el frame del eslabon PADRE (link i-1). El frame
        del padre se obtiene acumulando las rotaciones de los eslabones 0..(i-1)
        via FK directa con matrices de Rodrigues.

        Algoritmo (SANTOS21 Sec. V; AGENTS.md Sec. 5):
          R_cumul = I  (frame del mundo)
          Para cada joint i de 0 a n-1:
            ref_global[i]  = R_cumul @ ref_axis_local[i]   <- frame del PADRE
            axis_global[i] = R_cumul @ axis_local[i]       <- eje en frame global
            v_proj = project_to_axis_plane(link_dir_i, axis_global[i])
            theta_i = atan2(v_proj . perp_i, v_proj . ref_global[i])
              donde perp_i = axis_global[i] x ref_global[i]
            # AHORA actualizar R_cumul incluyendo la rotacion del eslabon i
            R_cumul = R(d_init_i -> d_curr_i) @ R_cumul

        Diferencia critica con _update_global_axes:
          _update_global_axes usa R_cumul DESPUES de actualizar (para las
          proyecciones incrementales del bucle de resolucion).
          extract_joint_angles usa R_cumul ANTES de actualizar, lo que da
          el angulo absoluto desde el cero mecanico de cada joint.

        Args:
            p : Lista de n+1 posiciones 3D. Puede ser self.positions o
                cualquier configuracion valida de la cadena.

        Returns:
            Lista de n floats con los angulos de junta en radianes [-pi, pi].
        """
        axes_global, refs_global = self._compute_chain_frames(p)
        angles: List[float] = []

        for i in range(self.n):
            axis_g = axes_global[i]
            ref_g  = refs_global[i]
            link_dir = _safe_normalize(p[i + 1] - p[i])
            v_proj = project_to_axis_plane(link_dir, axis_g)
            v_unit = _safe_normalize(v_proj, fallback=ref_g)

            theta = extract_joint_angle(
                link_vector=v_unit,
                ref_axis=ref_g,
                rot_axis=axis_g,
            )
            angles.append(theta)

        return angles

    def get_joint_angles(self) -> List[float]:
        """
        Extrae los angulos de junta desde la configuracion actual (self.positions).

        Atajo sobre extract_joint_angles(self.positions).
        Referencia: SANTOS21 Sec. V; AGENTS.md Sec. 5.

        Returns:
            Lista de n angulos en radianes.
        """
        angles = self.extract_joint_angles(self.positions)
        if self.joints and self.joints[0].segment_direction is not None:
            angles[0] = self._last_theta0
        return angles

    def get_hardware_command(self) -> List[float]:
        """
        Devuelve los angulos de junta en radianes listos para enviar al hardware.

        Aplica la compensacion de 'Zero Offset' de cada articulacion:
            theta_hw_i = theta_calculado_i - joint_i.offset

        El offset compensa el desfase entre el cero cinematico (definido por
        ref_axis_local en la configuracion de reposo) y el cero del encoder
        fisico del motor. Definir offset=0 si no hay compensacion necesaria.

        Referencia: AGENTS.md Sec. 5 (Extraccion de Angulos para Hardware).

        Returns:
            Lista de n floats (radianes) para enviar a los controladores.
        """
        raw = self.get_joint_angles()
        return [theta - self.joints[i].offset for i, theta in enumerate(raw)]

    def get_debug_state(self) -> dict:
        """
        Imprime y devuelve el estado interno de los frames globales.

        Util para verificar que la propagacion del twist (_update_global_axes)
        es correcta antes de ejecutar la bateria de pruebas unitarias (Fase 6).
        Comprueba ortogonalidad axis/ref y que las longitudes de eslabon no se
        hayan distorsionado (AGENTS.md Sec. 3 'Preservacion de Longitudes').

        Returns:
            dict con:
              'axes_global'  : lista de n arrays 3D (ejes bisagra en frame global)
              'refs_global'  : lista de n arrays 3D (referencias theta=0 en global)
              'joint_angles' : lista de n floats en radianes
              'link_lengths' : lista de n floats (longitudes actuales)
              'ortho_errors' : lista de n floats (|axis . ref|, debe ser ~0)
        """
        # Asegurar que los ejes estan sincronizados con la configuracion actual
        self._update_global_axes(self.positions)

        angles     = self.get_joint_angles()
        ortho_errs = []
        link_lens  = []

        print()
        print("=" * 80)
        print(f"  FABRIKRSolver debug state  ({self.n} joints)")
        print("=" * 80)
        header = (
            f"  {'J':>2}  "
            f"{'axis_global [x, y, z]':>30}  "
            f"{'ref_global  [x, y, z]':>30}  "
            f"{'theta(deg)':>10}  "
            f"{'|ax·ref|':>9}  "
            f"{'lenErr':>8}"
        )
        print(header)
        print("-" * 98)

        for i in range(self.n):
            ax  = self._axes_global[i]
            rf  = self._refs_global[i]
            ortho     = float(abs(np.dot(ax, rf)))
            actual_l  = float(np.linalg.norm(self.positions[i + 1] - self.positions[i]))
            len_err   = abs(actual_l - self.lengths[i])

            ortho_errs.append(ortho)
            link_lens.append(actual_l)

            o_flag = "OK" if ortho   < 1e-4 else "WARN"
            l_flag = "OK" if len_err < 1e-6 else "WARN"
            print(
                f"  J{i}  "
                f"[{ax[0]:+.4f} {ax[1]:+.4f} {ax[2]:+.4f}]  "
                f"[{rf[0]:+.4f} {rf[1]:+.4f} {rf[2]:+.4f}]  "
                f"{math.degrees(angles[i]):>+10.3f}  "
                f"{ortho:>9.2e} {o_flag:>4}  "
                f"{len_err:>8.2e} {l_flag:>4}"
            )

        print("-" * 98)
        max_len_err = max(abs(link_lens[i] - self.lengths[i]) for i in range(self.n))
        max_ortho   = max(ortho_errs)
        print(f"  Max length error : {max_len_err:.2e}  {'OK' if max_len_err < 1e-6 else 'WARN'}")
        print(f"  Max ortho  error : {max_ortho:.2e}   {'OK' if max_ortho   < 1e-4 else 'WARN'}")
        print("=" * 80)

        return {
            "axes_global":  [ax.copy() for ax in self._axes_global],
            "refs_global":  [rf.copy() for rf in self._refs_global],
            "joint_angles": angles,
            "link_lengths": link_lens,
            "ortho_errors": ortho_errs,
        }
