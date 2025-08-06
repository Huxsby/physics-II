import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from animation import guardar_animacion
from core import Robot, Link, cargar_robot_desde_yaml
from calculations.class_helicoidales import CinematicaDirecta, calcular_posiciones_articulaciones
from visualization import RecordingSystem
from fabrik_core.math_utils import (
    wrap_angle, angle_to_point, distance_squared, distance, 
    normalized, rotated_3d, spherical_to_cartesian, cartesian_to_spherical, rotated
)

# ===============================================================================
# ALGORITMO IMPLEMENTADO: FABRIK 3D según Aristidou & Lasenby (2011)
# ===============================================================================
# 
# Algorithm 1: FABRIK (Forward And Backward Reaching Inverse Kinematics) 3D
# - Implementación fiel al algoritmo básico FABRIK con fases forward/backward
# - Expansión completa a espacio 3D con vectores y restricciones espaciales
# - Manejo de targets fuera del alcance con stretching automático
# - Convergencia iterativa con tolerancia exacta del paper
# 
# Algorithm 2: Joint Constraint Application (for restricted joints) 3D
# - Aplicación de restricciones angulares esféricas post-procesamiento
# - Restricciones aplicadas después de cada iteración completa
# - Soporte para restricciones cónicas en articulaciones esféricas
# 
# Algorithm 3: Target Constraint Application (workspace limits) 3D NUEVO
# - Implementación de restricciones de workspace usando secciones cónicas
# - Mapeo de problemas 3D a 2D para cálculos eficientes
# - Aplicación de límites de workspace en tiempo real
# 
# FIEL AL PAPER ORIGINAL + EXTENSIÓN 3D:
# ===============================================================================
# 
# IMPLEMENTACIÓN EXACTA:
# - [X] Usa distancia simple (no al cuadrado) como en el algoritmo original
# - [X] Convergencia basada en tolerancia difA > tol del paper
# - [X] Direcciones unitarias simples: direction = (pi+1 - pi) / ri en 3D
# - [X] Post-procesamiento de restricciones (Algorithm 2) para articulaciones esféricas
# - [X] Stretching lineal para targets fuera de alcance en espacio 3D
# - [X] Preservación exacta del punto base fijo
# - [ ] Workspace constraints usando secciones cónicas (Algorithm 3)
# 
#  ALGORITMOS PENDIENTES DE IMPLEMENTAR:
# ===============================================================================
# [ ] Algorithm 4: Position to Joint Angles Conversion (3D con quaternions)
# [ ] Algorithm 5: Multi-Target FABRIK (multiple end effectors en 3D)
# [ ] Algorithm 6: FABRIK with Orientation Control (3D con quaternions)
# ===============================================================================

class Fabrik_3D:
    """
    Implementación del algoritmo FABRIK 3D (Forward And Backward Reaching Inverse Kinematics)
    para cinemática inversa con restricciones angulares en 3D.
    
    Este algoritmo utiliza un enfoque iterativo de dos fases:
    1. Fase hacia atrás (backward pass): desde el objetivo hacia la base
    2. Fase hacia adelante (forward pass): desde la base hacia el objetivo
    
    Incluye implementación del Algorithm 3 para restricciones de workspace.
    
    Adaptado y traducido de GDScript a Python con Matplotlib para visualización 3D.
    """
    
    # CONSTRUCTOR Y CONFIGURACIÓN INICIAL
    
    def __init__(self):
        """
        Inicializa el sistema de cinemática inversa FABRIK 3D.
        
        Configura las constantes, parámetros del algoritmo y estructura inicial
        de las articulaciones del robot en espacio 3D.
        """
        
        self.name = "FABRIK_3D" # Nombre del sistema FABRIK 3D por defecto
        # Índices para el almacenamiento de las extremidades: Longitud ; Ángulo mínimo ; Ángulo máximo.
        self.LIMB_LEN = 0 ; self.LIMB_MIN = 1 ; self.LIMB_MAX = 2 # (restricción esférica)
        
        # Usado para calcular cuánto falla el IK en alcanzar el objetivo
        self.BIAS = 3.0         # Tolerancia de error para considerar el objetivo alcanzado
        self.MAX_ITERATIONS = 48    # Número máximo de iteraciones del algoritmo
        self.iteration = 0      # Contador de iteraciones para el algoritmo FABRIK
        self.base_point = np.array([0.0, 0.0, 0.0]) # Punto base fijo del robot en 3D
        
        # Longitudes de cada extremidad y restricciones angulares esféricas
        # [longitud, ángulo_mín_cono, ángulo_máx_cono] en radianes
        self.limbs = [
            [80, np.deg2rad(30), np.deg2rad(120)],  # Articulación base - mayor rango
            [60, np.deg2rad(20), np.deg2rad(90)],   # Articulación media
            [80, np.deg2rad(25), np.deg2rad(135)],  # Articulación final
        ]
        
        self.info_joint_constraints = []
        self.joints = [] # Los puntos con los que trabajamos (articulaciones del robot) en 3D
        self.limbs_size = len(self.limbs) # Para no llamar a len(limbs) cada vez
        self.limbs_len = 0.0 # Usado para calcular el sobrepaso del objetivo desde el rango posible
        self.lerp_amount = 0.5 # Cantidad de interpolación (lerp) de las articulaciones antiguas a las nuevas
        self.target = np.array([0.0, 0.0, 0.0]) # Posición objetivo del efector final en 3D

        # Parámetros para Algorithm 3 - Target Constraint Application
        self.workspace_constraints_enabled = True
        self.workspace_angles = [
            np.deg2rad(45),  # θ1 - ángulo de restricción cuadrante 1
            np.deg2rad(60),  # θ2 - ángulo de restricción cuadrante 2
            np.deg2rad(45),  # θ3 - ángulo de restricción cuadrante 3
            np.deg2rad(30),  # θ4 - ángulo de restricción cuadrante 4
        ]

        # Sistema de grabación avanzado (delegado al módulo de grabación)
        self.recorder = RecordingSystem(fps=20, dpi=300, max_buffer_seconds=10)

        self._ready()

    # MÉTODOS DE REPRESENTACIÓN Y DEPURACIÓN

    def __str__(self):
        """
        Representación en cadena de texto completa del sistema FABRIK 3D.
        
        Incluye toda la información relevante de la clase: configuración del algoritmo,
        estructura del robot, estado actual, restricciones y parámetros de workspace.
        
        Returns:
            str: Descripción detallada del sistema FABRIK 3D
        """
        info = []
        info.append("=" * 60)
        info.append("FABRIK 3D - Forward And Backward Reaching Inverse Kinematics")
        info.append("=" * 60)
        
        info.append("\nCONFIGURACIÓN DEL ALGORITMO:")
        info.append(f" - Máximo de iteraciones: {self.MAX_ITERATIONS}")
        info.append(f" - Tolerancia (BIAS): {self.BIAS}")
        info.append(f" - Factor de interpolación (lerp): {self.lerp_amount}")
        
        info.append("\nESTADO DEL TARGET:")
        info.append(f" - Posición objetivo: [{self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}]")
        info.append(f" - Paso de movimiento: {self.target_step:.3f}")
        
        info.append("\nESTRUCTURA DEL ROBOT:")
        info.append(f" - Punto base: [{self.base_point[0]:.3f}, {self.base_point[1]:.3f}, {self.base_point[2]:.3f}]")
        info.append(f" - Número de eslabones: {self.limbs_size}")
        info.append(f" - Longitud total: {self.limbs_len:.3f}")
        
        info.append("\nESLABONES Y RESTRICCIONES:")

        info += self.info_joint_constraints

        # Posiciones actuales de las articulaciones
        info.append("\nPOSICIONES ACTUALES DE ARTICULACIONES:")
        for i, joint in enumerate(self.joints):
            if i == 0:
                info.append(f" - Base: [{joint[0]:.3f}, {joint[1]:.3f}, {joint[2]:.3f}]")
            elif i == len(self.joints) - 1:
                info.append(f" - Efector final: [{joint[0]:.3f}, {joint[1]:.3f}, {joint[2]:.3f}]")
            else:
                info.append(f" - Articulación {i}: [{joint[0]:.3f}, {joint[1]:.3f}, {joint[2]:.3f}]")
        
        # Restricciones de workspace
        info.append("\nRESTRICCIONES DE WORKSPACE:")
        info.append(f" - Restricciones habilitadas: {'SÍ' if self.workspace_constraints_enabled else 'NO'}")
        
        # Métricas actuales
        info.append("\nMÉTRICAS ACTUALES:")
        if len(self.joints) > 0:
            end_effector = self.joints[-1]
            distance_to_target = np.linalg.norm(end_effector - self.target)
            distance_from_base = np.linalg.norm(end_effector - self.base_point)
            reachability = (distance_to_target / self.limbs_len) * 100 if self.limbs_len > 0 else 0
            
            info.append(f" - Distancia al objetivo: {distance_to_target:.6f}")
            info.append(f" - Distancia desde la base: {distance_from_base:.3f}")
            info.append(f" - Alcanzabilidad del objetivo: {reachability:.1f}% del alcance máximo")
            info.append(f" - ¿Objetivo alcanzable?: {'SÍ' if distance_to_target <= self.limbs_len else 'NO'}")
        
        info.append("\n" + "=" * 80)
        
        return "\n".join(info)

    # ALGORITMOS PRINCIPALES DE FABRIK 3D

    def update(self, target: np.ndarray) -> None:
        """
        Función principal de actualización del sistema FABRIK.
        
        Maneja casos especiales para mejorar la estabilidad:
        1. Interpolación suave de articulaciones (lerp_amount)
        2. Fallback para targets demasiado cercanos al punto base
        3. Manejo de casos donde las restricciones impiden alcanzar el objetivo
        
        LÓGICA DE FALLBACK:
        Si dist > BIAS²: El IK con restricciones falló → usar configuración proyectada
        Calcula un target alternativo en la dirección correcta pero a distancia alcanzable
        
        Args:
            target (np.ndarray): Posición objetivo [x, y] para el efector final
            
        Returns:
            None
        """
        # Almacenando las articulaciones antiguas antes de calcular las nuevas
        joints_p = list(self.joints)
        # Actualizando el IK usando el algoritmo FABRIK
        dist = self.update_ik(target)

        # Comprueba si estamos fuera del objetivo - eso sucede cuando el IK está restringido
        # y el objetivo está demasiado cerca del base_point
        # Como recibimos la distancia al cuadrado - comparamos con BIAS al cuadrado
        if dist > self.BIAS * self.BIAS:
            # Restablece las articulaciones a las antiguas
            self.joints = joints_p
            # Llama a update_ik de nuevo con el objetivo como objetivo,
            # pero alejado a la distancia de la última articulación, para que sea
            # posible de alcanzar, y esté posiblemente cerca del objetivo deseado
            self.update_ik(
                self.base_point + normalized(target - self.base_point) *
                distance(self.base_point, self.joints[self.limbs_size])
            )
            return
            
        # Si la interpolación no es 1.0 (que es lo mismo que no hacer nada) -
        # interpola entre las articulaciones antiguas y las nuevas
        if self.lerp_amount != 1.0:
            for i in range(self.limbs_size + 1):
                self.joints[i] = joints_p[i] + (self.joints[i] - joints_p[i]) * self.lerp_amount

    def update_ik(self, target: np.ndarray) -> float:
        """
        Ejecuta el algoritmo FABRIK principal según el paper original (Algorithm 1) con Algorithm 3.
        
        ALGORITMOS IMPLEMENTADOS:
        - Algorithm 1 (FABRIK básico) según Aristidou & Lasenby en 3D
        - Algorithm 3 (Target Constraint Application) para restricciones de workspace
        
        Implementa la lógica exacta del paper:
        1. Aplica Algorithm 3 para restricciones de workspace (NUEVO)
        2. Verifica si el target está dentro del alcance (dist_to_target vs total_length)
        3. Si está fuera de alcance: stretching con interpolación lineal
        4. Si está en alcance: iteraciones alternadas backward/forward hasta convergencia
        5. Usa direcciones unitarias simples como en el paper original (en 3D)
        
        FIEL AL PAPER ORIGINAL + Algorithm 3:
        - Usa distancia simple (no al cuadrado) como en el algoritmo
        - Convergencia basada en tolerancia difA > tol
        - Direcciones unitarias simples: direction = (pi+1 - pi) / ri en 3D
        - Aplicación de restricciones de workspace usando secciones cónicas
        
        Args:
            target (np.ndarray): Posición objetivo [x, y, z] para el efector final
            
        Returns:
            float: Distancia al cuadrado entre el efector final y el objetivo
        """
        # NUEVO: Algorithm 3 - Apply target constraints first
        constrained_target = self.apply_target_constraints(target)
        
        # Check if the target is within reachable distance (según Algorithm 1)
        dist_to_target = distance(self.joints[self.limbs_size], constrained_target)
        total_length = self.limbs_len
        
        if dist_to_target > total_length:
            # The target is unreachable; stretch the chain towards the target
            # Implementación exacta del paper: interpolación lineal en 3D
            for i in range(self.limbs_size):
                # Find the distance ri between the target t and the joint position pi
                ri = distance(constrained_target, self.joints[i])
                if ri > 1e-8:  # Evitar división por cero
                    # Find the scaling factor ki to maintain link length
                    ki = self.limbs[i][self.LIMB_LEN] / ri
                    # Find the new joint positions pi+1 using linear interpolation
                    self.joints[i + 1] = (1 - ki) * self.joints[i] + ki * constrained_target
            return 0.0
        else:
            # The target is reachable; implementar bucle principal del Algorithm 1
            # Set as b the initial position of the joint p1
            b = self.base_point.copy()
            
            # Check whether the distance between the end effector pn and target t is greater than tolerance
            difA = distance(self.joints[self.limbs_size], constrained_target)
            self.iteration = 0
            tol = 1e-3  # Tolerancia según el paper

            while difA > tol and self.iteration < self.MAX_ITERATIONS:
                # STAGE 1: FORWARD REACHING (from end effector to base)
                self._backward_pass(constrained_target)
                
                # STAGE 2: BACKWARD REACHING (from base to end effector)  
                self._forward_pass(b)
                
                # Update the distance to target for convergence check
                difA = distance(self.joints[self.limbs_size], constrained_target)
                self.iteration += 1
            
            return distance_squared(self.joints[self.limbs_size], constrained_target)

    def _forward_pass(self, base_position: np.ndarray) -> None:
        """
        STAGE 2: BACKWARD REACHING (from base to end effector) - Según Algorithm 1 del paper.
        
        Implementación exacta del Algorithm 1, Stage 2 del paper original:
        1. Set the root p1 to its initial position b
        2. For i = 1, 2, ..., n-1 do:
           - Calculate ri = |pi+1 - pi|
           - direction = (pi+1 - pi) / ri
           - pi+1 = pi + di * direction
        
        FIEL AL PAPER:
        - Usa direcciones unitarias simples sin ángulos complejos
        - No aplica restricciones inline (se aplicarían como post-procesamiento)
        - Mantiene longitudes de eslabón exactas
        
        Args:
            base_position (np.ndarray): Posición base original a restaurar
            
        Returns:
            None
        """
        # Set the root p1 to its initial position b
        self.joints[0] = base_position
        
        # For i = 1, 2, ..., n-1 do
        for i in range(self.limbs_size):
            # Calculate the distance between consecutive joints
            ri = distance(self.joints[i + 1], self.joints[i])
            
            if ri > 1e-8:  # Evitar división por cero
                # Calculate the unit direction vector from pi to pi+1
                direction = (self.joints[i + 1] - self.joints[i]) / ri
                # Place pi+1 at distance di from pi along the direction vector
                self.joints[i + 1] = self.joints[i] + self.limbs[i][self.LIMB_LEN] * direction
        
        # Aplicar restricciones como post-procesamiento (Algorithm 2)
        self._apply_joint_constraints()

    def _backward_pass(self, target: np.ndarray) -> None:
        """
        STAGE 1: FORWARD REACHING (from end effector to base) - Según Algorithm 1 del paper.
        
        Implementación exacta del Algorithm 1, Stage 1 del paper original:
        1. Set the end effector pn as target t
        2. For i = n-1, n-2, ..., 1 do:
           - Calculate ri = |pi+1 - pi|
           - direction = (pi - pi+1) / ri
           - pi = pi+1 + di * direction
        
        FIEL AL PAPER:
        - Usa direcciones unitarias simples sin ángulos complejos
        - No aplica restricciones inline (se aplicarían como post-procesamiento)
        - Procesa desde el efector final hacia la base
        
        Args:
            target (np.ndarray): Posición objetivo para el efector final
            
        Returns:
            None
        """
        # Set the end effector pn as target t
        self.joints[self.limbs_size] = target
        
        # For i = n-1, n-2, ..., 1 do
        for i in range(self.limbs_size, 0, -1):
            # Calculate the distance between consecutive joints
            ri = distance(self.joints[i], self.joints[i - 1])
            
            if ri > 1e-8:  # Evitar división por cero
                # Calculate the unit direction vector from pi+1 to pi
                direction = (self.joints[i - 1] - self.joints[i]) / ri
                # Place pi at distance di from pi+1 along the direction vector
                self.joints[i - 1] = self.joints[i] + self.limbs[i - 1][self.LIMB_LEN] * direction
        
        # Aplicar restricciones como post-procesamiento (Algorithm 2)
        self._apply_joint_constraints()

    # ALGORITMOS DE RESTRICCIONES Y WORKSPACE

    def _apply_joint_constraints(self) -> None:
        """
        Algorithm 2: Joint Constraint Application (for restricted joints) - Post-procesamiento 3D.
        
        Aplica las restricciones de articulación esféricas después de cada paso de FABRIK,
        según el Algorithm 2 del paper original adaptado para 3D. Este método implementa 
        restricciones cónicas para articulaciones esféricas.
        
        SEGÚN EL PAPER (adaptado a 3D):
        1. Check whether the rotor R is within the motion range bounds (cónico)
        2. If outside bounds: clamp to nearest boundary on cone surface
        3. Reorient the joint pi-1 to respect spherical constraints
        
        IMPLEMENTACIÓN 3D:
        - Usa restricciones cónicas en lugar de angulares planas
        - Aplica clipping esférico para mantener dentro del cono permitido
        - Recalcula posiciones basadas en direcciones restringidas
        
        Returns:
            None
        """
        # Comenzar desde la base y aplicar restricciones secuencialmente
        for i in range(self.limbs_size):
            # NUEVA IMPLEMENTACIÓN: Reorientación basada en Algorithm 3
            # En lugar de usar la dirección del eslabón anterior como referencia,
            # calcular la dirección neutral real basada en joint_limits del robot
            
            if i == 0:
                # Para el primer segmento, usar la dirección de extensión por defecto
                default_direction = np.array([1.0, 0.0, 0.0])
            else:
                # Para segmentos subsecuentes, la dirección por defecto es la extensión lineal
                prev_segment = self.joints[i] - self.joints[i - 1]
                default_direction = normalized(prev_segment)
            
            # CLAVE: Verificar si tenemos información de orientación neutral del robot real
            limb = self.limbs[i]
            if len(limb) > 3 and isinstance(limb[3], dict):
                # Si tenemos metadata del robot (neutral_angle, range_type)
                constraint_metadata = limb[3]
                if 'neutral_angle' in constraint_metadata and 'range_type' in constraint_metadata:
                    # Calcular la dirección neutral real basada en el ángulo del robot
                    neutral_angle = constraint_metadata['neutral_angle']
                    
                    # Crear matriz de rotación para orientar hacia la posición neutral
                    # (esto simula la orientación real de la articulación del robot)
                    if i > 0:
                        # Rotar la dirección por defecto según el ángulo neutral real
                        axis = np.array([0.0, 0.0, 1.0])  # Asumir rotación en Z por defecto
                        cos_theta = np.cos(neutral_angle)
                        sin_theta = np.sin(neutral_angle)
                        
                        # Aplicar rotación de Rodrigues para reorientar hacia neutral
                        K = np.array([[0, -axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
                        
                        R_neutral = np.eye(3) + sin_theta * K + (1 - cos_theta) * K @ K
                        reference_direction = R_neutral @ default_direction
                    else:
                        reference_direction = default_direction
                else:
                    # Fallback al método original
                    reference_direction = default_direction
            else:
                # Método original: usar dirección del eslabón anterior
                reference_direction = default_direction
            
            # Dirección actual del segmento
            current_segment = self.joints[i + 1] - self.joints[i]
            current_direction = normalized(current_segment)
            
            # Calcular el ángulo entre la dirección de referencia y la actual
            dot_product = np.clip(np.dot(reference_direction, current_direction), -1.0, 1.0)
            current_angle = np.arccos(dot_product)
            
            # Aplicar restricciones cónicas (Algorithm 2 para 3D)
            limb = self.limbs[i]
            min_cone_angle = limb[self.LIMB_MIN]
            max_cone_angle = limb[self.LIMB_MAX]
            
            # Verificar si está dentro del cono permitido
            if current_angle < min_cone_angle or current_angle > max_cone_angle:
                # Clamp al ángulo más cercano permitido
                if current_angle < min_cone_angle:
                    constrained_angle = min_cone_angle
                else:
                    constrained_angle = max_cone_angle
                
                # Crear nueva dirección con el ángulo restringido
                # Usar el eje de rotación perpendicular a ambas direcciones
                rotation_axis = np.cross(reference_direction, current_direction)
                rotation_axis_norm = np.linalg.norm(rotation_axis)
                
                if rotation_axis_norm > 1e-8:
                    rotation_axis = rotation_axis / rotation_axis_norm
                    
                    # Rotar la dirección de referencia por el ángulo restringido
                    constrained_direction = rotated_3d(reference_direction, rotation_axis, constrained_angle)
                else:
                    # Direcciones paralelas, usar la dirección de referencia
                    constrained_direction = reference_direction
                
                # Recalcular la posición del punto final del segmento
                self.joints[i + 1] = self.joints[i] + constrained_direction * limb[self.LIMB_LEN]

    def apply_target_constraints(self, target: np.ndarray) -> np.ndarray:
        """
        Algorithm 3: Target Constraint Application (for constrained targets) - 3D.
        
        Implementa el Algorithm 3 del paper original para aplicar restricciones
        de workspace usando secciones cónicas. Modifica el target para que esté
        dentro del workspace alcanzable considerando las restricciones de las articulaciones.
        
        SEGÚN EL PAPER (Algorithm 3):
        3.1 Find the line equation L1
        3.2 Find the projection O of the target t on line L1
        3.3 Find the distance between the point O and the joint position
        3.4 Map the target (rotate and translate) to standard coordinate system
        3.5 Solve the 2D simplified problem
        3.6 Find in which quadrant the target belongs
        3.7 Find what conic section describes the allowed range of motion
        3.8 Find the conic section parameters for the quadrant
        3.9 Check whether the target is within the conic section or not
        
        Args:
            target (np.ndarray): Posición objetivo [x, y, z] para el efector final
            
        Returns:
            np.ndarray: Nuevo target que respeta las restricciones de workspace
        """
        if not self.workspace_constraints_enabled:
            return target
        
        # 3.1 Find the line equation L1
        base_position = self.base_point
        direction_vector = target - base_position
        direction_norm = np.linalg.norm(direction_vector)
        
        if direction_norm < 1e-8:
            return target  # Target demasiado cerca de la base
        
        direction_vector = direction_vector / direction_norm
        
        # 3.2 Find the projection O of the target t on line L1
        # (En este caso, O es simplemente la proyección del target sobre la línea)
        distance_to_target = direction_norm
        O = base_position + direction_vector * distance_to_target
        
        # 3.3 Find the distance between the point O and the joint position
        dist_O = np.linalg.norm(O - base_position)
        
        # 3.4 Map the target (rotate and translate) to standard coordinate system
        # Crear un sistema de coordenadas local donde Z apunta hacia el target
        z_axis = direction_vector
        
        # Crear un eje X arbitrario perpendicular a Z
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = normalized(x_axis)
        
        # Y es perpendicular a ambos
        y_axis = np.cross(z_axis, x_axis)
        y_axis = normalized(y_axis)
        
        # Matriz de transformación a coordenadas locales
        transform_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Transformar target a coordenadas locales
        local_target = transform_matrix.T @ (target - base_position)
        
        # 3.5 Solve the 2D simplified problem
        # Trabajar en el plano XY local
        local_2d = local_target[:2]  # [x, y] en coordenadas locales
        local_distance = dist_O
        
        # 3.6 Find in which quadrant the target belongs
        quadrant = self._determine_quadrant(local_2d[0], local_2d[1])
        
        # 3.7 & 3.8 Find what conic section describes the allowed range of motion
        conic_params = self._calculate_conic_parameters(quadrant, local_distance)
        
        # 3.9 Check whether the target is within the conic section or not
        if self._is_point_inside_conic(local_2d, conic_params):
            # Target está dentro del workspace permitido
            return target
        else:
            # Find the nearest point on that conic section from the target
            nearest_2d = self._find_nearest_point_on_conic(local_2d, conic_params)
            
            # Convertir de vuelta a 3D
            nearest_local_3d = np.array([nearest_2d[0], nearest_2d[1], local_target[2]])
            
            # Map back to original coordinate system
            constrained_target = base_position + transform_matrix @ nearest_local_3d
            
            return constrained_target

    def _determine_quadrant(self, x: float, y: float) -> int:
        """
        Determina en qué cuadrante se encuentra un punto 2D.
        
        Args:
            x (float): Coordenada X
            y (float): Coordenada Y
            
        Returns:
            int: Número de cuadrante (1-4)
        """
        if x >= 0 and y >= 0:
            return 1
        elif x < 0 and y >= 0:
            return 2
        elif x < 0 and y < 0:
            return 3
        else:  # x >= 0 and y < 0
            return 4

    def _calculate_conic_parameters(self, quadrant: int, distance: float) -> dict:
        """
        Calcula los parámetros de la sección cónica para un cuadrante dado.
        
        Implementa los pasos 3.7 y 3.8 del Algorithm 3:
        - Determina el tipo de sección cónica
        - Calcula qj = S*tan(θj) donde S es el factor de escala
        
        Args:
            quadrant (int): Cuadrante (1-4)
            distance (float): Distancia desde la base
            
        Returns:
            dict: Parámetros de la sección cónica
        """
        # Factor de escala basado en las longitudes de los eslabones
        S = self.limbs_len * 0.5  # Factor de escala configurable
        
        # Calcular qj = S*tan(θj) para el cuadrante
        theta_index = quadrant - 1
        theta = self.workspace_angles[theta_index]
        q = S * np.tan(theta)
        
        # Para simplicidad, usamos elipses como secciones cónicas
        # En una implementación completa, se determinaría el tipo basado en las restricciones
        conic_params = {
            'type': 'ellipse',
            'center': np.array([0.0, 0.0]),
            'semi_major': q,
            'semi_minor': q * 0.7,  # Elipse con ratio 0.7
            'rotation': 0.0
        }
        
        return conic_params

    def _is_point_inside_conic(self, point: np.ndarray, conic_params: dict) -> bool:
        """
        Verifica si un punto está dentro de la sección cónica definida.
        
        Args:
            point (np.ndarray): Punto 2D [x, y]
            conic_params (dict): Parámetros de la sección cónica
            
        Returns:
            bool: True si el punto está dentro de la sección cónica
        """
        if conic_params['type'] == 'ellipse':
            center = conic_params['center']
            a = conic_params['semi_major']
            b = conic_params['semi_minor']
            
            # Trasladar punto al centro de la elipse
            p = point - center
            
            # Ecuación de la elipse: (x/a)² + (y/b)² <= 1
            ellipse_eq = (p[0]/a)**2 + (p[1]/b)**2
            
            return ellipse_eq <= 1.0
        
        # Implementar otros tipos de secciones cónicas según sea necesario
        return True

    def _find_nearest_point_on_conic(self, point: np.ndarray, conic_params: dict) -> np.ndarray:
        """
        Encuentra el punto más cercano sobre la sección cónica desde un punto dado.
        
        Args:
            point (np.ndarray): Punto 2D [x, y] fuera de la cónica
            conic_params (dict): Parámetros de la sección cónica
            
        Returns:
            np.ndarray: Punto más cercano sobre la sección cónica
        """
        if conic_params['type'] == 'ellipse':
            center = conic_params['center']
            a = conic_params['semi_major']
            b = conic_params['semi_minor']
            
            # Algoritmo iterativo para encontrar el punto más cercano en una elipse
            p = point - center
            
            # Aproximación simple: proyectar radialmente sobre la elipse
            angle = np.arctan2(p[1], p[0])
            
            # Punto sobre la elipse en esa dirección
            ellipse_point = np.array([
                a * np.cos(angle),
                b * np.sin(angle)
            ])
            
            return ellipse_point + center
        
        # Para otros tipos de cónicas, implementar algoritmos específicos
        return point

    # INICIALIZACIÓN Y CONFIGURACIÓN

    def _ready(self):
        """
        Inicializa la estructura de articulaciones del robot en 3D.
        
        Configura las posiciones iniciales de las articulaciones en una línea recta
        desde el punto base, calcula la longitud total del robot y prepara
        los arrays de datos para el procesamiento en espacio 3D.
        
        Returns:
            None
        """
        # El tamaño de las articulaciones es el tamaño de las extremidades + 1
        self.joints = [np.zeros(3) for _ in range(self.limbs_size + 1)]
        
        # Configuración inicial: línea recta desde el base_point en dirección X positiva
        self.joints[0] = self.base_point.copy()
        for i in range(self.limbs_size):
            self.limbs_len += self.limbs[i][self.LIMB_LEN]
            # Inicializar en línea recta a lo largo del eje X
            self.joints[i + 1] = self.joints[i] + np.array([self.limbs[i][self.LIMB_LEN], 0, 0])
        
        # Convertir a arrays de float para operaciones numéricas
        self.joints = [np.array(j, dtype=float) for j in self.joints]

    # SISTEMA DE GRABACIÓN Y ANIMACIÓN (delegado al módulo de grabación)

    def start_recording(self):
        """Inicia la grabación de la animación."""
        self.recorder.start_recording()
    
    def pause_recording(self):
        """Pausa o reanuda la grabación."""
        self.recorder.pause_recording()
    
    def stop_recording(self):
        """Detiene la grabación y guarda el archivo."""
        # Configurar límites de plot antes de guardar
        if hasattr(self, 'ax'):
            ax_limits = {
                'x': self.ax.get_xlim(),
                'y': self.ax.get_ylim(), 
                'z': self.ax.get_zlim()
            }
        else:
            ax_limits = None
        
        # Temporal monkey patch para pasar contexto
        original_save_recorded = self.recorder._save_recorded_frames
        original_save_buffer = self.recorder._save_buffer_frames
        
        def patched_save_recorded(prefix, robot_name, base_point=None, limits=None):
            return original_save_recorded(prefix, robot_name, base_point or self.base_point, limits or ax_limits)
        
        def patched_save_buffer(prefix, robot_name, base_point=None, limits=None):
            return original_save_buffer(prefix, robot_name, base_point or self.base_point, limits or ax_limits)
        
        self.recorder._save_recorded_frames = patched_save_recorded
        self.recorder._save_buffer_frames = patched_save_buffer
        
        self.recorder.stop_recording(self.name)
        
        # Restaurar métodos originales
        self.recorder._save_recorded_frames = original_save_recorded
        self.recorder._save_buffer_frames = original_save_buffer
    
    def capture_recap(self):
        """Captura los últimos 10 segundos del buffer."""
        # Configurar límites de plot antes de guardar
        if hasattr(self, 'ax'):
            ax_limits = {
                'x': self.ax.get_xlim(),
                'y': self.ax.get_ylim(),
                'z': self.ax.get_zlim()
            }
        else:
            ax_limits = None
        
        # Temporal monkey patch para pasar contexto
        original_save_recorded = self.recorder._save_recorded_frames
        original_save_buffer = self.recorder._save_buffer_frames
        
        def patched_save_recorded(prefix, robot_name, base_point=None, limits=None):
            return original_save_recorded(prefix, robot_name, base_point or self.base_point, limits or ax_limits)
        
        def patched_save_buffer(prefix, robot_name, base_point=None, limits=None):
            return original_save_buffer(prefix, robot_name, base_point or self.base_point, limits or ax_limits)
        
        self.recorder._save_recorded_frames = patched_save_recorded
        self.recorder._save_buffer_frames = patched_save_buffer
        
        self.recorder.capture_recap(self.name)
        
        # Restaurar métodos originales
        self.recorder._save_recorded_frames = original_save_recorded
        self.recorder._save_buffer_frames = original_save_buffer

    # INTERFAZ GRÁFICA Y VISUALIZACIÓN

    def setup_plot(self):
        """
        Configura la visualización 3D de Matplotlib para el robot.
        
        Inicializa la ventana de graficación 3D, establece los límites del plot,
        configura los elementos visuales y conecta los eventos de teclado
        para la interacción en tiempo real.
        
        Usa rc_context para deshabilitar keymaps reservados de matplotlib.
        
        Returns:
            None
        """
        # Deshabilitar keymaps reservados de matplotlib para usar Q, S y otras teclas
        with mpl.rc_context({
            'keymap.save': [],        # Deshabilitar 's' para save
            'keymap.quit': [],        # Deshabilitar 'q' para quit
            'keymap.pan': [],         # Deshabilitar 'p' para pan
            'keymap.zoom': [],        # Deshabilitar 'o' para zoom
            'keymap.home': [],        # Deshabilitar 'h' y 'r' para home/reset vista
            'keymap.back': [],        # Deshabilitar navegación
            'keymap.forward': [],     # Deshabilitar navegación
          # 'keymap.fullscreen': [],  # Deshabilitar 'f' para fullscreen
            'keymap.grid': [],        # Deshabilitar 'g' para grid
            'keymap.yscale': [],      # Deshabilitar 'l' para log scale
            'keymap.xscale': [],      # Deshabilitar 'k' para log scale
        }):
            self.fig = plt.figure(figsize=(12, 9))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Establecer límites basados en la longitud total de las extremidades
            plot_range = self.limbs_len * 1.2
            self.ax.set_xlim(self.base_point[0] - plot_range, self.base_point[0] + plot_range)
            self.ax.set_ylim(self.base_point[1] - plot_range, self.base_point[1] + plot_range)
            self.ax.set_zlim(self.base_point[2] - plot_range, self.base_point[2] + plot_range)
            
            # Configurar etiquetas de los ejes
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            
            # Configurar título
            self.ax.set_title('FABRIK 3D - Cinemática Inversa en Tiempo Real\n[WASD: XY, QE: Z, R: Reset]')
            
            # Inicializar líneas del robot en 3D
            self.line, = self.ax.plot([], [], [], 'o-', color='blue', lw=2, markersize=6, markerfacecolor='red')
            
            # Lista para líneas de restricciones
            self.constraint_lines = []
            
            # Target indicator
            self.target_point, = self.ax.plot([], [], [], 'o', color='green', markersize=10, alpha=0.7)
            
            # Base point indicator
            self.base_indicator, = self.ax.plot([self.base_point[0]], [self.base_point[1]], [self.base_point[2]], 
                                              'o', color='black', markersize=8, markerfacecolor='yellow')
            
            # Inicializar texto de información (para evitar superposiciones)
            self.info_text = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes, 
                                          fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
            # Conectar eventos del teclado
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            
            # Intentar establecer el foco en la ventana para capturar eventos de teclado
            try:
                # Para Tkinter backend (Windows/Linux)
                if hasattr(self.fig.canvas.manager, 'window'):
                    self.fig.canvas.manager.window.wm_attributes('-topmost', 1)
                    self.fig.canvas.manager.window.wm_attributes('-topmost', 0)
                    self.fig.canvas.manager.window.focus_set()
            except AttributeError:
                # Para otros backends, simplemente continuar
                pass
            
            # Hacer que la figura sea focuseable
            self.fig.canvas.manager.set_window_title("FABRIK 3D - Presiona H para ayuda")
            
            # Inicializar target en una posición visible proporcional al robot
            self.target = np.array([self.limbs_len * 0.6, self.limbs_len * 0.3, self.limbs_len * 0.2])
            
            # Variables para controlar la velocidad de movimiento (proporcional al robot)
            self.target_step = self.limbs_len * 0.05  # 5% de la longitud total del robot
            
            self.anim = FuncAnimation(self.fig, self.animate, interval=50, blit=False, cache_frame_data=False)

            plt.show()

    def on_key_press(self, event):
        """Manejador de eventos para las teclas presionadas."""
        if not event.key:
            return
            
        key = event.key.lower()
        
        match key:
            # === MOVIMIENTO DEL TARGET ===
            case 'up' | 'w':     self.target[1] += self.target_step # Y+
            case 'down' | 's':   self.target[1] -= self.target_step # Y-
            case 'left' | 'a':   self.target[0] -= self.target_step # X-
            case 'right' | 'd':  self.target[0] += self.target_step # X+
            case 'u' | 'q':      self.target[2] += self.target_step # Z+
            case 'j' | 'e':      self.target[2] -= self.target_step # Z-
            # === CONTROLES ESPECIALES ===
            case 'r':       self.target = np.array([self.limbs_len * 0.6, self.limbs_len * 0.3, self.limbs_len * 0.2]) # Reset posición
            case '+' | '=': self._adjust_speed(1.5)     # Aumentar velocidad
            case '-':       self._adjust_speed(1/1.5)   # Disminuir velocidad  
            case 'h':       self.print_help()           # Ayuda
            # === SISTEMA DE GRABACIÓN ===
            case 'g': self.start_recording()    # Iniciar
            case 'p': self.pause_recording()    # Pausar/Reanudar
            case 'x': self.stop_recording()     # Parar y guardar
            case 'c': self.capture_recap()      # Recap
            
        # Limitar el target dentro del área visible
        plot_range = self.limbs_len * 1.2
        self.target = np.clip(self.target, -plot_range, plot_range)

    def _adjust_speed(self, factor: float):
        """Ajusta la velocidad de movimiento."""
        self.target_step = np.clip(
            self.target_step * factor,
            self.limbs_len * 0.01,
            self.limbs_len * 0.2
        )
        # Sobrescribir la línea anterior en lugar de crear nueva
        print(f"\rVelocidad {'aumentada:' if factor > 1 else 'reducida: '} {self.target_step:6.1f}", end='', flush=True)

    def print_help(self):
        """
        Imprime la ayuda de controles en la consola con formato mejorado.
        """
        print("\n" + "="*60)
        print("CONTROLES DE LA SIMULACIÓN - FABRIK 3D")
        print("="*60)

        print("┌──────────────────────────────────────┐")
        print("│         MOVIMIENTO DEL TARGET        │")
        print("├──────────────────────────────────────┤")
        print("│  WASD   │ Movimiento primario XY     │")
        print("│  Q/E    │ Eje Z (arriba/abajo)       │")
        print("│ Flechas │ Movimiento alternativo XY  │")
        print("│  U/J    │ Eje Z alternativo          │")
        print("│  Mouse  │ Rotar vista (matplotlib)   │")
        print("│   R     │ Reset a posición inicial   │")
        print("│  +/-    │ Velocidad de movimiento    │")
        print("├──────────────────────────────────────┤")
        print("│         SISTEMA DE GRABACIÓN         │")
        print("├──────────────────────────────────────┤")
        print("│   G     │ Iniciar grabación          │")
        print("│   P     │ Pausar/Reanudar            │")
        print("│   X     │ Parar y guardar            │")
        print("│   C     │ Recap últimos 10s          │")
        print("├──────────────────────────────────────┤")
        print("│   H     │ Ayuda completa             │")
        print("└──────────────────────────────────────┘")
        
        # Mostrar estado actual de grabación usando el nuevo sistema
        status = self.recorder.get_recording_status()
        if status['state'] != 'stopped':
            estado_indicador = "[REC]" if status['state'] == 'recording' else "[PAUSADO]"
            print(f"\n{estado_indicador} Estado: {status['state'].upper()}")
            print(f"Frames grabados: {status['frames_recorded']} ({status['duration']:.1f}s)")

    def animate(self, i):
        """
        Función de animación para la actualización en tiempo real en 3D.
        
        Se ejecuta continuamente para actualizar la visualización del robot,
        incluyendo las articulaciones, enlaces y restricciones angulares en 3D.
        También maneja el sistema de grabación y buffer de frames.
        
        Args:
            i (int): Número de frame de la animación (no utilizado)
            
        Returns:
            list: Lista de elementos gráficos actualizados
        """
        self.update(self.target)
        
        # Capturar frame actual para el sistema de grabación
        self.recorder.capture_frame(self.joints, self.target, i)
        
        # Actualizar las articulaciones del robot en 3D
        points = np.array(self.joints)
        self.line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
        
        # Actualizar la posición del target
        self.target_point.set_data_3d([self.target[0]], [self.target[1]], [self.target[2]])
        
        # Limpiar líneas de restricción antiguas
        for l in self.constraint_lines:
            l.remove()
        self.constraint_lines.clear()

        # Dibujar restricciones cónicas en 3D (simplificado)
        for i in range(self.limbs_size):
            if i > 0:
                # Base de la articulación
                p_base = self.joints[i]
                
                # Dirección del segmento anterior para establecer referencia
                if i > 1:
                    prev_direction = normalized(self.joints[i] - self.joints[i-1])
                else:
                    prev_direction = np.array([1.0, 0.0, 0.0])  # Eje X como referencia
                
                # Dibujar cono de restricción simplificado
                # Crear círculo en el plano perpendicular a la dirección anterior
                cone_radius = self.limbs[i-1][self.LIMB_LEN] * 0.3  # Radio proporcional al link
                min_angle = self.limbs[i-1][self.LIMB_MIN]
                max_angle = self.limbs[i-1][self.LIMB_MAX]
                
                # Crear puntos del cono para visualización
                theta_points = np.linspace(0, 2*np.pi, 12)
                
                # Cono mínimo
                min_cone_points = []
                max_cone_points = []
                
                for theta in theta_points:
                    # Vector perpendicular para crear el círculo
                    if abs(prev_direction[2]) < 0.9:
                        perp1 = np.cross(prev_direction, np.array([0, 0, 1]))
                    else:
                        perp1 = np.cross(prev_direction, np.array([1, 0, 0]))
                    perp1 = normalized(perp1)
                    perp2 = np.cross(prev_direction, perp1)
                    perp2 = normalized(perp2)
                    
                    # Puntos en el círculo
                    circle_point = perp1 * np.cos(theta) + perp2 * np.sin(theta)
                    
                    # Rotar para crear el cono
                    min_direction = rotated_3d(prev_direction, circle_point, min_angle)
                    max_direction = rotated_3d(prev_direction, circle_point, max_angle)
                    
                    min_cone_points.append(p_base + min_direction * cone_radius)
                    max_cone_points.append(p_base + max_direction * cone_radius)
                
                # Dibujar líneas del cono (solo algunas para no saturar)
                for j in range(0, len(min_cone_points), 3):
                    min_line_data = np.array([p_base, min_cone_points[j]])
                    max_line_data = np.array([p_base, max_cone_points[j]])
                    
                    l_min, = self.ax.plot(min_line_data[:,0], min_line_data[:,1], min_line_data[:,2], 
                                        color='orange', lw=0.5, alpha=0.6)
                    l_max, = self.ax.plot(max_line_data[:,0], max_line_data[:,1], max_line_data[:,2], 
                                        color='red', lw=0.5, alpha=0.6)
                    self.constraint_lines.extend([l_min, l_max])

        # Actualizar información del target y estado de grabación
        target_info = f"Target: [{self.target[0]:.1f}, {self.target[1]:.1f}, {self.target[2]:.1f}] | Stress: {self.iteration/self.MAX_ITERATIONS*100:5.2f}%"
        
        # Agregar indicador de estado de grabación usando el nuevo sistema
        status = self.recorder.get_recording_status()
        if status['state'] == 'recording':
            target_info += f" | [REC] ({status['frames_recorded']} frames)"
        elif status['state'] == 'paused':
            target_info += f" | [PAUSADO] ({status['frames_recorded']} frames)"
        
        self.info_text.set_text(target_info)

        return [self.line, self.target_point, self.base_indicator, self.info_text] + self.constraint_lines

    # MÉTODOS DE CARGA Y CONFIGURACIÓN DESDE YAML

    @classmethod
    def from_robot_yaml(cls, robot: Robot, **kwargs):
        """
        Carga un robot desde un archivo YAML y convierte al formato FABRIK.
        
        Este método actúa como puente entre el TAD Robot/Link del proyecto
        y el sistema FABRIK, convirtiendo coordenadas relativas a absolutas
        y transformando restricciones de articulaciones a formato FABRIK.
        
        Args:
            yaml_file_path (str): Ruta al archivo YAML del robot
            **kwargs: Argumentos adicionales para el constructor de Fabrik_3D
                     (max_iterations, tolerance, etc.)
        
        Returns:
            Fabrik_3D: Instancia configurada desde el robot YAML
            
        Raises:
            ImportError: Si no se pueden importar las clases Robot/Link
            FileNotFoundError: Si el archivo YAML no existe
            ValueError: Si el robot YAML no tiene estructura válida
        
        Ejemplo:
            robot_fabrik = Fabrik_3D.from_robot_yaml(robot)
            resultado, exito = robot_fabrik.solve_ik([0.3, 0.2, 0.4])
        """
        if robot is None:
            raise ImportError("No se pueden cargar las clases Robot/Link. Verifica la instalación del módulo core.class_robot_structure")
        
        # Convertir coordenadas relativas del TAD a absolutas para FABRIK
        joints_pos = []
        current_pos = np.array([0.0, 0.0, 0.0])  # Base en origen
        joints_pos.append(current_pos.copy())
        
        # Calcular posiciones absolutas de las articulaciones
        for i, link in enumerate(robot.links):
            if hasattr(link, 'joint_coords') and link.joint_coords is not None:
                # Las coordenadas del joint están en el sistema local del link
                joint_offset = np.array(link.joint_coords)
            else:
                # Si no hay coordenadas específicas, usar la longitud del link
                if hasattr(link, 'length') and link.length is not None:
                    joint_offset = np.array([link.length, 0.0, 0.0])
                else:
                    # Longitud por defecto
                    joint_offset = np.array([1.0, 0.0, 0.0])
            
            current_pos = current_pos + joint_offset
            joints_pos.append(current_pos.copy())
        
        # Convertir restricciones de articulaciones del TAD a restricciones FABRIK
        joint_constraints = []
        for i, link in enumerate(robot.links):
            constraint_info = {
                'type': 'conic',
                'center_direction': [1.0, 0.0, 0.0],  # Dirección por defecto
                'max_angle': np.pi/4  # 45 grados por defecto
            }
            
            if hasattr(link, 'joint_limits') and link.joint_limits is not None:
                limits = link.joint_limits
                if isinstance(limits, (tuple, list)) and len(limits) == 2:
                    # Límites como tupla/lista (min, max) del YAML
                    min_limit, max_limit = limits
                    
                    # NUEVA ESTRATEGIA: Inspirada en Algorithm 3 (secciones cónicas)
                    # En lugar de heurística arbitraria, usar matemática de restricciones cónicas
                    range_radians = abs(max_limit - min_limit)
                    center_angle = (max_limit + min_limit) / 2.0  # Posición neutral real
                    
                    # El radio del cono es la mitad del rango total (máxima desviación desde neutral)
                    # Esto es matemáticamente correcto: si el rango es [-175°, +175°], 
                    # entonces podemos desviarnos ±175° desde neutral (0°)
                    constraint_info['max_angle'] = min(range_radians / 2.0, np.pi)  # Máximo 180°
                    constraint_info['min_angle'] = 0.0  # Sin restricción hacia el centro del cono
                    
                    # CLAVE: Guardar la orientación neutral para reorientar la cónica
                    constraint_info['neutral_angle'] = center_angle
                    constraint_info['range_type'] = 'symmetric_from_neutral'
                    constraint_info['original_limits'] = (min_limit, max_limit)  # Preservar originales
                elif isinstance(limits, dict):
                    # Extraer límites en diferentes ejes y convertir a restricción cónica
                    angles = []
                    for key in ['min_theta', 'max_theta', 'min_phi', 'max_phi', 'min_angle', 'max_angle']:
                        if key in limits:
                            angles.append(abs(limits[key]))
                    
                    if angles:
                        max_angle = max(angles)
                        constraint_info['max_angle'] = min(max_angle, np.pi/2)  # Limitar a 90 grados máximo
                elif isinstance(limits, (int, float)):
                    # Límite escalar
                    constraint_info['max_angle'] = min(abs(limits), np.pi/2)
            
            # Calcular dirección del link para la restricción cónica
            if i > 0 and len(joints_pos) > i:
                direction = joints_pos[i] - joints_pos[i-1]
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    constraint_info['center_direction'] = direction.tolist()
            
            joint_constraints.append(constraint_info)
            
        print(f"\033[92mRobot cargado desde YAML: {len(joints_pos)} articulaciones, {len(joint_constraints)} restricciones\033[0m")
        
        # Crear instancia FABRIK básica
        instance = cls()
        instance.name = robot.name
        info = instance.info_joint_constraints
        
        # Construir información de restricciones de articulaciones para la representación
        for i, constraint in enumerate(joint_constraints):
            center_direction = constraint.get('center_direction', [1.0, 0.0, 0.0])
            min_angle = constraint.get('min_angle', 0.0)
            max_angle = constraint.get('max_angle', np.pi/4)
            neutral_angle = constraint.get('neutral_angle', 0.0)
            range_type = constraint.get('range_type', 'default')
            original_limits = constraint.get('original_limits', (None, None))
            
            info.append(f"Articulación ({i+1}):")
            info.append(f"  ├─ Tipo: {constraint.get('type', 'conic')}")
            info.append(f"  ├─ Dirección central: [{center_direction[0]:.3f}, {center_direction[1]:.3f}, {center_direction[2]:.3f}]")
            info.append(f"  ├─ Ángulo (mínimo, máximo): ({np.rad2deg(min_angle):.1f}°, {np.rad2deg(max_angle):.1f}°)")
            info.append(f"  ├─ Ángulo neutral: {np.rad2deg(neutral_angle):.1f}°")
            info.append(f"  ├─ Tipo de rango: {range_type}")
            if original_limits[0] is not None and original_limits[1] is not None:
                info.append(f"  └─ Límites originales: ({np.rad2deg(original_limits[0]):.1f}°, {np.rad2deg(original_limits[1]):.1f}°)")
            else:
                info.append(f"  └─ Límites originales: No especificados")

        # Configurar las posiciones de articulaciones desde el YAML
        instance.joints = [np.array(pos, dtype=float) for pos in joints_pos]
        instance.limbs_size = len(joints_pos) - 1
        
        # Configurar las restricciones desde el YAML
        instance.limbs = []
        for i, constraint in enumerate(joint_constraints):
            if i < len(joints_pos) - 1:
                # Calcular longitud del link en metros, convertir a unidades de visualización
                link_length_meters = np.linalg.norm(joints_pos[i+1] - joints_pos[i])
                link_length_viz = link_length_meters * 1000  # Convertir metros a milímetros para visualización
                max_angle = constraint.get('max_angle', np.pi/4)
                min_angle = constraint.get('min_angle', 0.0)  # Usar el ángulo mínimo real
                
                # NUEVA IMPLEMENTACIÓN: Incluir metadatos de restricción del robot
                limb_entry = [
                    link_length_viz,  # [0] Longitud escalada
                    min_angle,        # [1] Ángulo mínimo real
                    max_angle         # [2] Ángulo máximo real
                ]
                
                # Si tenemos información de orientación neutral, agregarla como metadatos
                if 'neutral_angle' in constraint and 'range_type' in constraint:
                    robot_metadata = {
                        'neutral_angle': constraint['neutral_angle'],
                        'range_type': constraint['range_type'],
                        'original_limits': constraint.get('original_limits', None)
                    }
                    limb_entry.append(robot_metadata)  # [3] Metadatos del robot
                
                instance.limbs.append(limb_entry)
        
        # Recalcular longitud total
        instance.limbs_len = sum(limb[instance.LIMB_LEN] for limb in instance.limbs)
        
        # Inicializar propiedades de target y visualización
        instance.target = np.array([instance.limbs_len * 0.6, instance.limbs_len * 0.3, instance.limbs_len * 0.2])
        instance.target_step = instance.limbs_len * 0.05  # 5% de la longitud total del robot
        
        # Reconfigurar las articulaciones con las longitudes correctas para visualización
        instance.joints = [np.zeros(3) for _ in range(len(joints_pos))]
        instance.joints[0] = instance.base_point.copy()
        
        for i in range(len(instance.limbs)):
            # Usar las direcciones originales pero con longitudes escaladas
            if i < len(joints_pos) - 1:
                direction = joints_pos[i+1] - joints_pos[i]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    # Aplicar la longitud escalada
                    instance.joints[i + 1] = instance.joints[i] + direction * instance.limbs[i][instance.LIMB_LEN]
        
        return instance
    
    # MÉTODOS DE UTILIDAD Y API PÚBLICA
    
    def solve_ik(self, target_pos, verbose=False):
        """
        Resuelve la cinemática inversa para alcanzar la posición objetivo.
        
        Método simplificado que encapsula la lógica del algoritmo FABRIK
        para uso con robots cargados desde YAML.
        
        Args:
            target_pos (list/array): Posición objetivo [x, y, z]
            verbose (bool): Si imprimir información de convergencia
            
        Returns:
            tuple: (joints_positions, converged)
                - joints_positions: Lista de posiciones finales de articulaciones
                - converged: True si alcanzó la tolerancia, False si no
        """
        if not hasattr(self, 'joints') or len(self.joints) == 0:
            raise ValueError("La cadena cinemática debe ser inicializada antes de resolver")
        
        target_np = np.array(target_pos)
        
        # Aplicar el algoritmo FABRIK principal usando update_ik
        residual = self.update_ik(target_np)
        
        # Comprobar convergencia con tolerancia más permisiva
        final_distance = np.linalg.norm(self.joints[-1] - target_np)
        converged = final_distance < 0.02  # Tolerancia más permisiva: 2cm
        
        if verbose:
            print(f"Solución FABRIK:")
            print(f"  Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            print(f"  End effector final: [{self.joints[-1][0]:.3f}, {self.joints[-1][1]:.3f}, {self.joints[-1][2]:.3f}]")
            print(f"  Distancia final: {final_distance:.6f}")
            print(f"  Convergencia: {'SÍ' if converged else 'NO'}")
        
        # Convertir posiciones a lista de listas para compatibilidad
        joints_list = [joint.tolist() for joint in self.joints]
        
        return joints_list, converged

# Demostración de uso del sistema FABRIK 3D
if __name__ == '__main__':
    print("FABRIK 3D Implementation - Visualización 3D Real")
    ik_system = Fabrik_3D()
    
    # Intentar cargar robot desde YAML
    try:
        robot = cargar_robot_desde_yaml('config/robot-niryo.yaml')
        ik_system = ik_system.from_robot_yaml(robot)
    except Exception as e:
        print(f"Error cargando robot: {e}")
        print("\tUsando configuración por defecto")

    print(ik_system)
    ik_system.setup_plot()
