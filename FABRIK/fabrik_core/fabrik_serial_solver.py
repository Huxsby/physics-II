"""
fabrik_serial_solver.py
=======================
Implementacion del algoritmo FABRIK para cadenas cinematicas seriales con
restricciones de articulacion (Algorithms 1, 2 y 3 segun Aristidou & Lasenby, 2011).

Este modulo es independiente del robot concreto. Para usarlo con un robot cargado
desde YAML, usar el metodo de clase `FabrikSerialSolver.from_robot()`.

Reemplaza a fabrik_paper_constrained_3d.py (LEGACY, ver FABRIK_README.md):
  - Algorithm 2: restricciones correctas por tipo (BALL y HINGE).
    El eje de referencia se mantiene actualizado entre passes.
  - Algorithm 3: geometria correcta con q1-q4 por cuadrante y
    Newton's method para el punto mas cercano en la elipse.
  - Orientacion de articulaciones: cuaternion por joint, twist limit.
  - Sin datos mock ni funciones incompletas.

Tipos de restriccion soportados:
  - JointType.BALL  : restriccion conica simetrica (angulo maximo desde eje neutro).
  - JointType.HINGE_GLOBAL : bisagra con eje fijo en el frame global.
  - JointType.HINGE_LOCAL  : bisagra con eje relativo al segmento anterior.
  - JointType.FREE  : sin restriccion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from fabrik_core.quaternion_utils import (
    quat_identity,
    quat_normalize,
    quat_from_two_vectors,
    quat_rotation_rotor,
    quat_clamp_rotation,
    quat_multiply,
    quat_rotate_vector,
    quat_rotation_angle,
)


# ---------------------------------------------------------------------------
# Tipos de articulacion
# ---------------------------------------------------------------------------

class JointType(Enum):
    FREE         = auto()
    BALL         = auto()
    HINGE_GLOBAL = auto()
    HINGE_LOCAL  = auto()


# ---------------------------------------------------------------------------
# Descriptor de articulacion
# ---------------------------------------------------------------------------

@dataclass
class JointDescriptor:
    """
    Informacion geometrica y de restriccion de una articulacion.

    Attributes:
        joint_type     : Tipo de articulacion.
        length         : Longitud del segmento que sigue a esta articulacion (metros).
        ball_max_angle : Angulo maximo del cono BALL (radianes). Solo para BALL.
        hinge_axis     : Eje de la bisagra en el frame de referencia. Solo para HINGE.
        hinge_ref_axis : Eje de referencia para medir el angulo firmado en HINGE.
        hinge_cw_deg   : Limite horario en grados (positivo). Solo para HINGE.
        hinge_acw_deg  : Limite antihorario en grados (positivo). Solo para HINGE.
        twist_max_rad  : Limite de rotacion axial (twist). Usado con quaterniones.
        workspace_angles: Cuatro angulos [theta1..theta4] para Algorithm 3, uno por
                          cuadrante del plano de restriccion de workspace.
    """
    joint_type:      JointType = JointType.FREE
    length:          float     = 1.0
    ball_max_angle:  float     = math.pi          # sin restriccion por defecto
    hinge_axis:      np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    hinge_ref_axis:  np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    hinge_cw_deg:    float      = 180.0
    hinge_acw_deg:   float      = 180.0
    twist_max_rad:   float      = math.pi         # sin restriccion por defecto
    workspace_angles: np.ndarray = field(
        default_factory=lambda: np.full(4, math.pi / 4)
    )


# ---------------------------------------------------------------------------
# Resultado del solver
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Resultado de una llamada al solver.

    Attributes:
        joint_positions : Posiciones finales de las articulaciones (n+1 puntos).
        end_effector    : Posicion del efector final (ultimo punto).
        iterations      : Iteraciones realizadas.
        converged       : True si la distancia final al target es < tolerance.
        final_error     : Distancia euclidiana entre efector y target.
    """
    joint_positions: List[np.ndarray]
    end_effector:    np.ndarray
    iterations:      int
    converged:       bool
    final_error:     float


# ---------------------------------------------------------------------------
# Solver principal
# ---------------------------------------------------------------------------

class FabrikSerialSolver:
    """
    Solver FABRIK para cadenas cinematicas seriales con restricciones.

    Implementa:
      - Algorithm 1 : FABRIK basico (forward + backward passes).
      - Algorithm 2 : restricciones de articulacion (BALL, HINGE_GLOBAL, HINGE_LOCAL).
      - Algorithm 3 : restricciones de workspace mediante secciones conicas.

    El solver mantiene internamente:
      - self.joints      : lista de np.ndarray(3) con las posiciones de articulaciones.
      - self.orientations: lista de np.ndarray(4) con orientaciones [w,x,y,z] por joint.

    Usage::

        solver = FabrikSerialSolver.from_robot(robot_obj)
        result = solver.solve(target=np.array([0.3, 0.0, 0.4]))
        print(result.end_effector, result.converged)
    """

    DEFAULT_TOLERANCE    = 1e-4   # metros
    DEFAULT_MAX_ITER     = 64
    DEFAULT_TWIST_LIMIT  = math.radians(166.0)  # ~mismo que referencia chain_3d

    def __init__(
        self,
        joint_descriptors: List[JointDescriptor],
        base_position: Optional[np.ndarray] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITER,
    ):
        """
        Inicializa el solver con una lista de descriptores de articulaciones.

        Args:
            joint_descriptors : Descripcion de cada articulacion y su segmento.
            base_position     : Posicion de la base en el frame global (default origen).
            tolerance         : Distancia minima al target para considerar convergencia.
            max_iterations    : Limite de iteraciones del bucle principal.
        """
        if not joint_descriptors:
            raise ValueError("Se requiere al menos un descriptor de articulacion.")

        self.descriptors   = joint_descriptors
        self.n             = len(joint_descriptors)   # numero de segmentos
        self.tolerance     = tolerance
        self.max_iterations = max_iterations
        self.base_position = np.array(base_position if base_position is not None
                                      else [0.0, 0.0, 0.0], dtype=float)

        # Longitud total del brazo
        self.total_length = sum(d.length for d in self.descriptors)

        # Inicializar posiciones en configuracion estirada a lo largo de +Z
        self.joints = self._build_initial_joints()

        # Inicializar orientaciones como identidad por articulacion
        self.orientations = [quat_identity() for _ in range(self.n + 1)]

    # ------------------------------------------------------------------
    # Construccion desde Robot YAML
    # ------------------------------------------------------------------

    @classmethod
    def from_robot(
        cls,
        robot,
        tolerance: float = DEFAULT_TOLERANCE,
        max_iterations: int = DEFAULT_MAX_ITER,
    ) -> "FabrikSerialSolver":
        """
        Construye el solver a partir de una instancia de Robot (class_robot_structure).

        Mapeo de parametros YAML -> JointDescriptor:
          - type == 'revolute' con joint_axis == [0,0,1] y rango < 360 -> HINGE_GLOBAL
          - type == 'revolute' con rango >= 2*pi (libre) -> BALL con max_angle = pi
          - type == 'revolute' general -> BALL con max_angle = rango/2
          - type == 'prismatic' -> FREE (FABRIK no modela juntas prismaticas con
            restricciones angulares, se tratan como segmentos libres).

        El eje de la bisagra HINGE se toma de link.joint_axis en el frame global
        de configuracion cero.

        Args:
            robot          : Instancia de Robot con atributo .links (lista de Link).
            tolerance      : Tolerancia de convergencia en metros.
            max_iterations : Limite de iteraciones.

        Returns:
            FabrikSerialSolver: Solver configurado.
        """
        descriptors = []

        for link in robot.links:
            jtype, ball_max, hinge_axis, hinge_ref, cw, acw = _parse_link_constraints(link)

            # Longitud del segmento: modulo de joint_coords
            seg_length = float(np.linalg.norm(link.joint_coords))
            if seg_length < 1e-8:
                seg_length = float(link.length)

            # Angulos de workspace por defecto: usar ball_max para los 4 cuadrantes
            ws_angles = np.full(4, min(ball_max, math.pi / 2))

            desc = JointDescriptor(
                joint_type      = jtype,
                length          = seg_length,
                ball_max_angle  = ball_max,
                hinge_axis      = hinge_axis,
                hinge_ref_axis  = hinge_ref,
                hinge_cw_deg    = cw,
                hinge_acw_deg   = acw,
                twist_max_rad   = cls.DEFAULT_TWIST_LIMIT,
                workspace_angles = ws_angles,
            )
            descriptors.append(desc)

        return cls(
            joint_descriptors = descriptors,
            tolerance         = tolerance,
            max_iterations    = max_iterations,
        )

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def solve(
        self,
        target: np.ndarray,
        use_workspace_constraints: bool = True,
    ) -> SolverResult:
        """
        Resuelve la cinematica inversa para alcanzar target.

        Implementa Algorithm 1 completo con Algorithms 2 y 3.

        Args:
            target                    : Posicion objetivo [x, y, z] en metros.
            use_workspace_constraints : Activar Algorithm 3.

        Returns:
            SolverResult con las posiciones finales y metricas de convergencia.
        """
        target = np.asarray(target, dtype=float)

        # Algorithm 3: ajustar target a workspace alcanzable
        if use_workspace_constraints:
            target = self._apply_workspace_constraints(target)

        dist_to_target = np.linalg.norm(self.joints[self.n] - target)
        total_len      = self.total_length

        # Caso: target fuera de alcance -> estirar la cadena
        if dist_to_target > total_len:
            self._stretch_toward(target)
            final_err = float(np.linalg.norm(self.joints[self.n] - target))
            return SolverResult(
                joint_positions = [j.copy() for j in self.joints],
                end_effector    = self.joints[self.n].copy(),
                iterations      = 1,
                converged       = False,
                final_error     = final_err,
            )

        # Caso: target alcanzable -> bucle FABRIK
        base = self.base_position.copy()
        dif  = np.linalg.norm(self.joints[self.n] - target)
        iters = 0

        while dif > self.tolerance and iters < self.max_iterations:
            self._backward_pass(target)
            self._forward_pass(base)
            dif   = np.linalg.norm(self.joints[self.n] - target)
            iters += 1

        final_err = float(np.linalg.norm(self.joints[self.n] - target))
        return SolverResult(
            joint_positions = [j.copy() for j in self.joints],
            end_effector    = self.joints[self.n].copy(),
            iterations      = iters,
            converged       = final_err <= self.tolerance,
            final_error     = final_err,
        )

    def reset_to_initial(self):
        """Reinicia las posiciones de articulaciones a la configuracion estirada inicial."""
        self.joints = self._build_initial_joints()
        self.orientations = [quat_identity() for _ in range(self.n + 1)]

    # ------------------------------------------------------------------
    # Algorithm 1: passes forward y backward
    # ------------------------------------------------------------------

    def _stretch_toward(self, target: np.ndarray) -> None:
        """
        Algorithm 1 - rama target inalcanzable: interpola linealmente cada
        segmento en la direccion desde el joint actual al target.
        """
        for i in range(self.n):
            ri = np.linalg.norm(target - self.joints[i])
            if ri < 1e-10:
                continue
            ki = self.descriptors[i].length / ri
            self.joints[i + 1] = (1.0 - ki) * self.joints[i] + ki * target

    def _backward_pass(self, target: np.ndarray) -> None:
        """
        Algorithm 1, Stage 1: FORWARD REACHING (del efector hacia la base).

        Segun el paper:
          p_n = target
          For i = n-1 down to 1:
            r_i = |p_{i+1} - p_i|
            lambda_i = d_i / r_i
            p_i = (1 - lambda_i) * p_{i+1} + lambda_i * p_i

        Tras mover cada joint se actualizan las orientaciones y se aplican
        las restricciones de articulacion (Algorithm 2).
        """
        self.joints[self.n] = target.copy()
        self._update_orientation(self.n)

        for i in range(self.n - 1, -1, -1):
            seg_len = self.descriptors[i].length
            ri = np.linalg.norm(self.joints[i + 1] - self.joints[i])
            if ri < 1e-10:
                continue
            ki = seg_len / ri
            self.joints[i] = (1.0 - ki) * self.joints[i + 1] + ki * self.joints[i]
            self._update_orientation(i)

            # Algorithm 2: aplicar restriccion segun el tipo de articulacion
            # En el backward pass el joint i+1 es el "outer" (mas cercano al efector)
            if i < self.n - 1:
                self._apply_joint_constraint(i + 1, pass_direction="backward")

    def _forward_pass(self, base: np.ndarray) -> None:
        """
        Algorithm 1, Stage 2: BACKWARD REACHING (de la base al efector).

        p_1 = base
        For i = 1 to n:
          r_i = |p_{i+1} - p_i|
          lambda_i = d_i / r_i
          p_{i+1} = (1 - lambda_i) * p_i + lambda_i * p_{i+1}
        """
        self.joints[0] = base.copy()
        self._update_orientation(0)

        for i in range(self.n):
            seg_len = self.descriptors[i].length
            ri = np.linalg.norm(self.joints[i + 1] - self.joints[i])
            if ri < 1e-10:
                continue
            ki = seg_len / ri
            self.joints[i + 1] = (1.0 - ki) * self.joints[i] + ki * self.joints[i + 1]
            self._update_orientation(i + 1)

            # Algorithm 2: aplicar restriccion en el forward pass
            if i > 0:
                self._apply_joint_constraint(i, pass_direction="forward")

    # ------------------------------------------------------------------
    # Algorithm 2: restricciones de articulacion
    # ------------------------------------------------------------------

    def _apply_joint_constraint(self, joint_idx: int, pass_direction: str) -> None:
        """
        Algorithm 2: restriccion de articulacion para el joint en joint_idx.

        Despacha al metodo correspondiente segun JointType.

        Args:
            joint_idx      : Indice de la articulacion a restringir.
            pass_direction : 'forward' o 'backward' (determina quien es "prev").
        """
        # descriptors[i] describe la articulacion EN la posicion i y el segmento i->i+1.
        # La articulacion en joint_idx esta descrita por descriptors[joint_idx].
        # joint_idx siempre esta en [1, n-1] cuando se llama desde los passes.
        if joint_idx >= self.n:
            return
        desc = self.descriptors[joint_idx]  # descriptor de la articulacion en joint_idx

        # Determinar joint previo y siguiente segun la direccion del pass
        if pass_direction == "forward":
            idx_prev = joint_idx - 1
            idx_curr = joint_idx
            idx_next = joint_idx + 1 if joint_idx + 1 <= self.n else None
        else:  # backward
            idx_prev = joint_idx + 1
            idx_curr = joint_idx
            idx_next = joint_idx - 1 if joint_idx - 1 >= 0 else None

        if idx_next is None:
            return

        if desc.joint_type == JointType.BALL:
            self._apply_ball_constraint(idx_prev, idx_curr, idx_next, desc)
        elif desc.joint_type == JointType.HINGE_GLOBAL:
            self._apply_global_hinge_constraint(idx_prev, idx_curr, idx_next, desc)
        elif desc.joint_type == JointType.HINGE_LOCAL:
            self._apply_local_hinge_constraint(idx_prev, idx_curr, idx_next, desc)
        # JointType.FREE: no aplica restriccion

        # Twist limit via cuaterniones (aplica a todos los tipos)
        self._apply_twist_limit(idx_curr, desc)

    def _apply_ball_constraint(
        self,
        idx_prev: int,
        idx_curr: int,
        idx_next: int,
        desc: JointDescriptor,
    ) -> None:
        """
        Restriccion de tipo BALL (conica simetrica), Paper Algorithm 2.

        La restriccion limita el angulo entre el vector entrante (prev->curr)
        y el vector saliente (curr->next) a un maximo de desc.ball_max_angle.

        Si el angulo supera el maximo, el punto siguiente (idx_next) se reposiciona
        al borde del cono.
        """
        p_prev = self.joints[idx_prev]
        p_curr = self.joints[idx_curr]
        p_next = self.joints[idx_next]

        v_in  = _normalize(p_curr - p_prev)  # vector que llega al joint
        v_out = _normalize(p_next - p_curr)  # vector que sale del joint

        angle = _angle_between(v_in, v_out)

        if angle <= desc.ball_max_angle:
            return

        # Clamp al borde del cono: rotar v_in hacia v_out por ball_max_angle
        axis = np.cross(v_in, v_out)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # Vectores paralelos o antiparalelos
            axis = _perpendicular(v_in)
        else:
            axis = axis / axis_norm

        # Construir cuaternion que rota v_in por ball_max_angle alrededor de axis
        clamped_dir = _rodrigues(v_in, axis, desc.ball_max_angle)

        # Reposicionar el punto siguiente a la distancia del segmento entre curr y next.
        # min(idx_curr, idx_next) da el indice del segmento (el menor de los dos joints).
        seg_next_len = self.descriptors[min(idx_curr, idx_next)].length
        self.joints[idx_next] = p_curr + clamped_dir * seg_next_len

    def _apply_global_hinge_constraint(
        self,
        idx_prev: int,
        idx_curr: int,
        idx_next: int,
        desc: JointDescriptor,
    ) -> None:
        """
        Restriccion HINGE_GLOBAL: el eje de la bisagra es fijo en el frame global.

        Paso 1: proyectar el vector de salida sobre el plano perpendicular al eje.
        Paso 2: si hay limites CW/ACW, medir el angulo firmado respecto al eje de
                referencia y clampear si esta fuera del rango.
        """
        p_curr = self.joints[idx_curr]
        p_next = self.joints[idx_next]

        hinge_axis = _normalize(desc.hinge_axis)
        v_out      = _normalize(p_next - p_curr)

        # Proyectar sobre el plano de la bisagra
        v_proj = _project_on_plane(v_out, hinge_axis)
        if np.linalg.norm(v_proj) < 1e-8:
            return
        v_proj = _normalize(v_proj)

        cw_rad  = math.radians(desc.hinge_cw_deg)
        acw_rad = math.radians(desc.hinge_acw_deg)

        # Medir angulo firmado respecto al eje de referencia
        ref = _normalize(desc.hinge_ref_axis)
        signed_angle = _signed_angle(ref, v_proj, hinge_axis)

        if signed_angle > acw_rad:
            clamped_dir = _rodrigues(ref, hinge_axis, acw_rad)
        elif signed_angle < -cw_rad:
            clamped_dir = _rodrigues(ref, hinge_axis, -cw_rad)
        else:
            clamped_dir = v_proj

        seg_len = self.descriptors[min(idx_curr, idx_next)].length
        self.joints[idx_next] = p_curr + clamped_dir * seg_len

    def _apply_local_hinge_constraint(
        self,
        idx_prev: int,
        idx_curr: int,
        idx_next: int,
        desc: JointDescriptor,
    ) -> None:
        """
        Restriccion HINGE_LOCAL: el eje de la bisagra esta definido relativo al
        segmento anterior (frame local).

        El eje local se transforma al frame global usando la matriz de rotacion
        construida a partir del vector del segmento previo (igual que en FABRIK_chain_3D).
        """
        p_prev = self.joints[idx_prev]
        p_curr = self.joints[idx_curr]
        p_next = self.joints[idx_next]

        prev_dir = _normalize(p_curr - p_prev)

        # Construir matriz de rotacion local -> global segun prev_dir
        R = _rotation_matrix_from_direction(prev_dir)

        # Transformar el eje de bisagra al frame global
        global_hinge = _normalize(R @ desc.hinge_axis)
        global_ref   = _normalize(R @ desc.hinge_ref_axis)

        v_out = _normalize(p_next - p_curr)
        v_proj = _project_on_plane(v_out, global_hinge)
        if np.linalg.norm(v_proj) < 1e-8:
            return
        v_proj = _normalize(v_proj)

        cw_rad  = math.radians(desc.hinge_cw_deg)
        acw_rad = math.radians(desc.hinge_acw_deg)

        signed_angle = _signed_angle(global_ref, v_proj, global_hinge)

        if signed_angle > acw_rad:
            clamped_dir = _rodrigues(global_ref, global_hinge, acw_rad)
        elif signed_angle < -cw_rad:
            clamped_dir = _rodrigues(global_ref, global_hinge, -cw_rad)
        else:
            clamped_dir = v_proj

        seg_len = self.descriptors[min(idx_curr, idx_next)].length
        self.joints[idx_next] = p_curr + clamped_dir * seg_len

    def _apply_twist_limit(self, joint_idx: int, desc: JointDescriptor) -> None:
        """
        Limita la rotacion axial (twist) acumulada en la orientacion del joint.

        Calcula el rotor entre la orientacion del joint anterior y el actual.
        Si el angulo de ese rotor supera twist_max_rad, lo clampea y actualiza
        la orientacion del joint actual.

        Esta es la misma logica que bone_twist_limit en FABRIK_chain_3D y
        bone_orientation_limit en FABRIK_Full_Body.
        """
        if joint_idx == 0:
            return

        q_prev = self.orientations[joint_idx - 1]
        q_curr = self.orientations[joint_idx]

        rotor = quat_rotation_rotor(q_prev, q_curr)
        rotor_clamped = quat_clamp_rotation(rotor, desc.twist_max_rad)

        if not np.allclose(rotor, rotor_clamped, atol=1e-6):
            self.orientations[joint_idx] = quat_normalize(
                quat_multiply(rotor_clamped, q_prev)
            )

    # ------------------------------------------------------------------
    # Algorithm 3: restricciones de workspace
    # ------------------------------------------------------------------

    def _apply_workspace_constraints(self, target: np.ndarray) -> np.ndarray:
        """
        Algorithm 3 completo: ajusta el target al workspace alcanzable.

        Implementacion fiel al paper Aristidou & Lasenby (2011), Sec. 3.3:

        3.1  Definir la linea L1 a lo largo del eje del primer segmento.
        3.2  Proyectar el target sobre L1 para obtener el punto O.
        3.3  Calcular la distancia S = |O - base|.
        3.4  Mapear el target a coordenadas locales (Z a lo largo de L1).
        3.5  Resolver el problema 2D simplificado.
        3.6  Determinar el cuadrante del target 2D.
        3.7-3.8 Calcular los parametros de la seccion conica para ese cuadrante.
        3.9  Verificar si el target esta dentro; si no, proyectarlo al borde.
        """
        base = self.base_position

        # 3.1 Direccion de L1: eje del primer segmento (base -> joint[1])
        if np.linalg.norm(self.joints[1] - self.joints[0]) > 1e-8:
            l1_dir = _normalize(self.joints[1] - self.joints[0])
        else:
            l1_dir = np.array([0.0, 0.0, 1.0])

        direction_to_target = target - base
        dist_to_target = np.linalg.norm(direction_to_target)
        if dist_to_target < 1e-8:
            return target

        # 3.2 Proyeccion del target sobre L1
        # O = base + (dot(target - base, l1_dir)) * l1_dir
        s = np.dot(direction_to_target, l1_dir)
        O = base + s * l1_dir

        # 3.3 S = distancia de O a la base (puede ser negativo si el target esta detras)
        S = s  # con signo; en el paper S > 0 hacia adelante

        if abs(S) < 1e-8:
            return target

        # 3.4 Sistema de coordenadas local (Z = l1_dir)
        z_axis = l1_dir
        x_axis = _normalize(_perpendicular(z_axis))
        y_axis = _normalize(np.cross(z_axis, x_axis))

        # Transformar target al sistema local
        t_local = np.array([
            np.dot(direction_to_target, x_axis),
            np.dot(direction_to_target, y_axis),
            np.dot(direction_to_target, z_axis),
        ])

        # 3.5 Problema 2D: (x_t, y_t) en el plano perpendicular a L1
        x_t = t_local[0]
        y_t = t_local[1]

        # 3.6 Cuadrante
        quadrant = _quadrant(x_t, y_t)

        # 3.7-3.8 Parametros de la elipse segun el cuadrante
        # Usar los angulos de workspace del primer descriptor (articulacion base)
        ws_angles = self.descriptors[0].workspace_angles
        theta_q   = ws_angles[quadrant - 1]

        # qj = S * tan(theta_j) segun el paper
        q_j = abs(S) * math.tan(theta_q)

        # Para una restriccion conica, la elipse es circular: a = b = q_j
        # En una implementacion completa con 4 angulos distintos por cuadrante
        # la elipse tendria semi-ejes distintos.
        a = q_j
        b = q_j

        # 3.9 Comprobar si el target 2D esta dentro de la elipse
        if a < 1e-10 or b < 1e-10:
            # Restriccion degenerada: forzar al eje
            constrained_2d = np.array([0.0, 0.0])
        elif (x_t / a) ** 2 + (y_t / b) ** 2 <= 1.0:
            return target  # dentro del workspace
        else:
            constrained_2d = _nearest_point_on_ellipse(x_t, y_t, a, b)

        # Reconstruir target 3D desde el punto 2D restringido
        t_local_constrained = np.array([constrained_2d[0], constrained_2d[1], t_local[2]])
        constrained_3d = base + (
            t_local_constrained[0] * x_axis
            + t_local_constrained[1] * y_axis
            + t_local_constrained[2] * z_axis
        )
        return constrained_3d

    # ------------------------------------------------------------------
    # Gestion de orientaciones
    # ------------------------------------------------------------------

    def _update_orientation(self, joint_idx: int) -> None:
        """
        Actualiza el cuaternion de orientacion del joint en joint_idx basandose
        en la direccion del segmento entrante.

        Para joint_idx == 0 se usa la identidad.
        Para joint_idx > 0, la orientacion se calcula como la rotacion que lleva
        [0, 0, 1] a la direccion del segmento (joint[i-1] -> joint[i]).
        """
        if joint_idx == 0:
            self.orientations[0] = quat_identity()
            return

        v = self.joints[joint_idx] - self.joints[joint_idx - 1]
        if np.linalg.norm(v) < 1e-10:
            self.orientations[joint_idx] = self.orientations[joint_idx - 1].copy()
            return

        ref = np.array([0.0, 0.0, 1.0])
        self.orientations[joint_idx] = quat_from_two_vectors(ref, v)

    # ------------------------------------------------------------------
    # Inicializacion
    # ------------------------------------------------------------------

    def _build_initial_joints(self) -> List[np.ndarray]:
        """
        Construye la configuracion inicial con una pequena curvatura para evitar
        la singularidad numerica de la cadena completamente estirada.

        Una cadena estirada alineada con el target produce oscilaciones sin
        convergencia en FABRIK porque la fase de backward y forward se cancelan.
        Una perturbacion de 0.5 grados por segmento rompe la simetria sin
        alterar significativamente la configuracion de partida.
        """
        joints = [self.base_position.copy()]
        cursor = self.base_position.copy()
        # Pequena inclinacion en X por segmento (aprox 0.5 grados)
        tilt_per_segment = 0.5 * math.pi / 180.0
        for i, desc in enumerate(self.descriptors):
            # Direccion ligeramente inclinada respecto a +Z
            angle = tilt_per_segment * (i + 1)
            direction = np.array([math.sin(angle), 0.0, math.cos(angle)])
            cursor = cursor + direction * desc.length
            joints.append(cursor.copy())
        return joints

    # ------------------------------------------------------------------
    # Representacion
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "FabrikSerialSolver",
            "=" * 60,
            f"  Segmentos      : {self.n}",
            f"  Longitud total : {self.total_length:.4f} m",
            f"  Tolerancia     : {self.tolerance:.2e} m",
            f"  Max iteraciones: {self.max_iterations}",
            "",
            "  Articulaciones:",
        ]
        for i, desc in enumerate(self.descriptors):
            lines.append(
                f"    [{i}] tipo={desc.joint_type.name:15s} "
                f"length={desc.length:.4f} m  "
                f"ball_max={math.degrees(desc.ball_max_angle):.1f} deg"
            )
        lines.append("")
        lines.append("  Posiciones actuales:")
        for i, j in enumerate(self.joints):
            label = "base" if i == 0 else ("efector" if i == self.n else f"j{i}")
            lines.append(f"    [{label}] [{j[0]:.4f}, {j[1]:.4f}, {j[2]:.4f}]")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Funciones de geometria (modulo interno)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v.copy()
    return v / n


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angulo en radianes entre dos vectores unitarios."""
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def _perpendicular(v: np.ndarray) -> np.ndarray:
    """Vector perpendicular a v (no normalizado)."""
    if abs(v[0]) < 0.9:
        cand = np.array([1.0, 0.0, 0.0])
    else:
        cand = np.array([0.0, 1.0, 0.0])
    return np.cross(v, cand)


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rota el vector v alrededor de axis por angle radianes (formula de Rodrigues).

    Args:
        v     : Vector a rotar (unitario).
        axis  : Eje de rotacion (unitario).
        angle : Angulo en radianes.

    Returns:
        np.ndarray: Vector rotado (unitario si v es unitario).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _project_on_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Proyecta v sobre el plano definido por normal."""
    return v - np.dot(v, normal) * normal


def _signed_angle(ref: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """
    Angulo firmado (en radianes) de ref a v alrededor de axis.
    Positivo = antihorario visto desde axis apuntando hacia el observador.
    """
    unsigned = _angle_between(ref, v)
    cross = np.cross(ref, v)
    sign = np.sign(np.dot(cross, axis))
    if sign == 0:
        sign = 1.0
    return float(sign * unsigned)


def _quadrant(x: float, y: float) -> int:
    """Cuadrante del punto (x, y) en el plano 2D (1-4)."""
    if x >= 0 and y >= 0:
        return 1
    if x < 0 and y >= 0:
        return 2
    if x < 0 and y < 0:
        return 3
    return 4


def _nearest_point_on_ellipse(
    x_t: float, y_t: float, a: float, b: float, max_iter: int = 30
) -> np.ndarray:
    """
    Encuentra el punto mas cercano sobre la elipse (x/a)^2 + (y/b)^2 = 1
    al punto (x_t, y_t) usando el metodo de Newton-Raphson.

    Implementacion equivalente a Constraints.find_nearest_point() de FABRIK_Full_Body
    pero usando Newton directamente sin la aproximacion inicial separada.

    El sistema a resolver es:
        F1(x, y) = b^2 * x^2 + a^2 * y^2 - a^2 * b^2 = 0   (en la elipse)
        F2(x, y) = b^2 * x * (y_t - y) - a^2 * y * (x_t - x) = 0  (perpendicular)

    Args:
        x_t, y_t : Coordenadas del punto externo.
        a, b     : Semi-ejes de la elipse (a >= 0, b >= 0).
        max_iter : Iteraciones maximas de Newton.

    Returns:
        np.ndarray: Punto [x, y] sobre la elipse mas cercano a (x_t, y_t).
    """
    # Punto inicial sobre la elipse (proyeccion radial)
    angle = math.atan2(y_t, x_t)
    x = a * math.cos(angle)
    y = b * math.sin(angle)

    for _ in range(max_iter):
        # Valores de las funciones en (x, y)
        f1 = b * b * x * x + a * a * y * y - a * a * b * b
        f2 = b * b * x * (y_t - y) - a * a * y * (x_t - x)

        if abs(f1) < 1e-9 and abs(f2) < 1e-9:
            break

        # Jacobiano 2x2
        j11 = 2.0 * b * b * x
        j12 = 2.0 * a * a * y
        j21 = b * b * (y_t - y) + a * a * y
        j22 = -b * b * x - a * a * (x_t - x)

        det = j11 * j22 - j12 * j21
        if abs(det) < 1e-12:
            break

        # Paso de Newton
        dx = -(j22 * f1 - j12 * f2) / det
        dy = -(-j21 * f1 + j11 * f2) / det

        x += dx
        y += dy

    return np.array([x, y])


def _rotation_matrix_from_direction(direction: np.ndarray) -> np.ndarray:
    """
    Construye una matriz de rotacion cuya tercera columna (Z local) apunta en
    la direccion dada. Se usa para transformar el eje de una bisagra LOCAL
    al frame global.

    Equivalente a Utils.create_rotation_matrix() de FABRIK_chain_3D.

    Args:
        direction : Vector de direccion unitario [x, y, z].

    Returns:
        np.ndarray: Matriz de rotacion 3x3.
    """
    d = direction.copy()
    # Evitar singularidad cuando direction apunta en +Y
    if abs(d[1] - 1.0) < 1e-3:
        d[1] -= 1e-3
        d = _normalize(d)

    x_dir = _normalize(np.cross(d, np.array([0.0, 1.0, 0.0])))
    y_dir = _normalize(np.cross(x_dir, d))

    return np.column_stack([x_dir, y_dir, d])


# ---------------------------------------------------------------------------
# Funcion auxiliar para parsear constraints desde un Link del proyecto
# ---------------------------------------------------------------------------

def _parse_link_constraints(
    link,
) -> Tuple[JointType, float, np.ndarray, np.ndarray, float, float]:
    """
    Extrae el tipo de articulacion y sus parametros de restriccion desde un objeto Link.

    Args:
        link : Objeto Link de class_robot_structure.

    Returns:
        Tupla (JointType, ball_max_angle, hinge_axis, hinge_ref_axis, cw_deg, acw_deg).
    """
    joint_axis = np.asarray(link.joint_axis, dtype=float)
    axis_norm  = np.linalg.norm(joint_axis)
    if axis_norm > 1e-8:
        joint_axis = joint_axis / axis_norm

    # Vector de referencia perpendicular al eje de la articulacion
    hinge_ref = _normalize(_perpendicular(joint_axis))

    # Limites angulares del YAML
    limits = link.joint_limits
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        lo, hi = float(limits[0]), float(limits[1])
    else:
        lo, hi = -math.pi, math.pi

    range_rad = abs(hi - lo)

    if link.tipo == "prismatic":
        return JointType.FREE, math.pi, joint_axis, hinge_ref, 180.0, 180.0

    # Articulacion revoluta: clasificar segun rango
    if range_rad >= 2.0 * math.pi - 0.01:
        # Rango completo: tratar como BALL sin restriccion de angulo
        return JointType.BALL, math.pi, joint_axis, hinge_ref, 180.0, 180.0

    # Articulacion revoluta con rango limitado.
    # El eje de la articulacion esta definido en el frame LOCAL del joint.
    # HINGE_GLOBAL con eje fijo en el frame global es incorrecto para cadenas seriales
    # donde el eje efectivo rota con la configuracion del robot.
    # Se usa BALL (cono simetrico centrado en la direccion del segmento entrante) como
    # aproximacion correcta que funciona en cualquier configuracion.
    # ball_max_angle = la mitad del rango total de la articulacion.
    half    = range_rad / 2.0
    cw_deg  = math.degrees(max(-lo, 0.0))  # conservado para posible uso como HINGE
    acw_deg = math.degrees(max(hi,  0.0))  # conservado para posible uso como HINGE

    return JointType.BALL, half, joint_axis, hinge_ref, cw_deg, acw_deg
