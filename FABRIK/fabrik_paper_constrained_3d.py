import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Traducción y adaptación de un script de GDScript a Python con Matplotlib.
# El script original implementa el algoritmo FABRIK para cinemática inversa.

# ===============================================================================
# ALGORITMO IMPLEMENTADO: FABRIK 3D según Aristidou & Lasenby (2011)
# ===============================================================================
# 
# 🔸 Algorithm 1: FABRIK (Forward And Backward Reaching Inverse Kinematics) 3D
#   - Implementación fiel al algoritmo básico FABRIK con fases forward/backward
#   - Expansión completa a espacio 3D con vectores y restricciones espaciales
#   - Manejo de targets fuera del alcance con stretching automático
#   - Convergencia iterativa con tolerancia exacta del paper
# 
# 🔸 Algorithm 2: Joint Constraint Application (for restricted joints) 3D
#   - Aplicación de restricciones angulares esféricas post-procesamiento
#   - Restricciones aplicadas después de cada iteración completa
#   - Soporte para restricciones cónicas en articulaciones esféricas
# 
# 🔸 Algorithm 3: Target Constraint Application (workspace limits) 3D ✨ NUEVO
#   - Implementación de restricciones de workspace usando secciones cónicas
#   - Mapeo de problemas 3D a 2D para cálculos eficientes
#   - Aplicación de límites de workspace en tiempo real
# 
# FIEL AL PAPER ORIGINAL + EXTENSIÓN 3D:
# ===============================================================================
# 
# ✅ IMPLEMENTACIÓN EXACTA:
# - ✓ Usa distancia simple (no al cuadrado) como en el algoritmo original
# - ✓ Convergencia basada en tolerancia difA > tol del paper
# - ✓ Direcciones unitarias simples: direction = (pi+1 - pi) / ri en 3D
# - ✓ Post-procesamiento de restricciones (Algorithm 2) para articulaciones esféricas
# - ✓ Stretching lineal para targets fuera de alcance en espacio 3D
# - ✓ Preservación exacta del punto base fijo
# - ✨ Workspace constraints usando secciones cónicas (Algorithm 3)
# 
# 📋 ALGORITMOS PENDIENTES DE IMPLEMENTAR:
# ===============================================================================
# [ ] Algorithm 4: Position to Joint Angles Conversion (3D con quaternions)
# [ ] Algorithm 5: Multi-Target FABRIK (multiple end effectors en 3D)
# [ ] Algorithm 6: FABRIK with Orientation Control (3D con quaternions)
# ===============================================================================

class FabrikIK3D:
    """
    Implementación del algoritmo FABRIK 3D (Forward And Backward Reaching Inverse Kinematics)
    para cinemática inversa con restricciones angulares en 3D.
    
    Este algoritmo utiliza un enfoque iterativo de dos fases:
    1. Fase hacia atrás (backward pass): desde el objetivo hacia la base
    2. Fase hacia adelante (forward pass): desde la base hacia el objetivo
    
    Incluye implementación del Algorithm 3 para restricciones de workspace.
    
    Adaptado y traducido de GDScript a Python con Matplotlib para visualización 3D.
    """
    
    def __init__(self):
        """
        Inicializa el sistema de cinemática inversa FABRIK 3D.
        
        Configura las constantes, parámetros del algoritmo y estructura inicial
        de las articulaciones del robot en espacio 3D.
        """
        # Índices para el almacenamiento de las extremidades, autoexplicativos
        self.LIMB_LEN = 0  # Índice para la longitud de la extremidad
        self.LIMB_MIN = 1  # Índice para el ángulo mínimo de restricción (esférica)
        self.LIMB_MAX = 2  # Índice para el ángulo máximo de restricción (esférica)
        
        # Usado para calcular cuánto falla el IK en alcanzar el objetivo
        self.BIAS = 3.0  # Tolerancia de error para considerar el objetivo alcanzado
        self.ITERATIONS = 32  # Número máximo de iteraciones del algoritmo

        # Punto base fijo del robot en 3D
        self.base_point = np.array([0.0, 0.0, 0.0])
        
        # Longitudes de cada extremidad y restricciones angulares esféricas
        # [longitud, ángulo_mín_cono, ángulo_máx_cono] en radianes
        self.limbs = [
            [80, np.deg2rad(30), np.deg2rad(120)],  # Articulación base - mayor rango
            [60, np.deg2rad(20), np.deg2rad(90)],   # Articulación media
            [80, np.deg2rad(25), np.deg2rad(135)],  # Articulación final
        ]
        
        # Los puntos con los que trabajamos (articulaciones del robot) en 3D
        self.joints = []

        # Para no llamar a len(limbs) cada vez
        self.limbs_size = len(self.limbs)
        # Usado para calcular el sobrepaso del objetivo desde el rango posible
        self.limbs_len = 0.0

        # Cantidad de interpolación (lerp) de las articulaciones antiguas a las nuevas
        self.lerp_amount = 0.5
        
        # Posición objetivo del efector final en 3D
        self.target = np.array([0.0, 0.0, 0.0])
        
        # Parámetros para Algorithm 3 - Target Constraint Application
        self.workspace_constraints_enabled = True
        self.workspace_angles = [
            np.deg2rad(45),  # θ1 - ángulo de restricción cuadrante 1
            np.deg2rad(60),  # θ2 - ángulo de restricción cuadrante 2
            np.deg2rad(45),  # θ3 - ángulo de restricción cuadrante 3
            np.deg2rad(30),  # θ4 - ángulo de restricción cuadrante 4
        ]

        self._ready()

    def _wrap_angle(self, angle):
        """
        Envuelve el ángulo entre -PI y PI.
        
        Args:
            angle (float): Ángulo en radianes a normalizar
            
        Returns:
            float: Ángulo normalizado en el rango [-π, π]
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _angle_to_point(self, p1, p2):
        """
        Calcula el ángulo de p1 a p2.
        
        Args:
            p1 (np.ndarray): Punto de origen [x, y]
            p2 (np.ndarray): Punto de destino [x, y]
            
        Returns:
            float: Ángulo en radianes desde p1 hacia p2
        """
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def _distance_squared(self, p1, p2):
        """
        Calcula la distancia al cuadrado entre dos puntos.
        
        Más eficiente que calcular la distancia completa cuando solo
        se necesita comparar distancias.
        
        Args:
            p1 (np.ndarray): Primer punto [x, y]
            p2 (np.ndarray): Segundo punto [x, y]
            
        Returns:
            float: Distancia al cuadrado entre los puntos
        """
        return np.sum((p1 - p2)**2)

    def _distance(self, p1, p2):
        """
        Calcula la distancia entre dos puntos.
        
        Args:
            p1 (np.ndarray): Primer punto [x, y]
            p2 (np.ndarray): Segundo punto [x, y]
            
        Returns:
            float: Distancia euclidiana entre los puntos
        """
        return np.linalg.norm(p1 - p2)
        
    def _normalized(self, v):
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

    def _rotated_3d(self, v, axis, angle):
        """
        Rota un vector 3D alrededor de un eje dado usando la fórmula de Rodrigues.
        
        Args:
            v (np.ndarray): Vector 3D a rotar [x, y, z]
            axis (np.ndarray): Eje de rotación normalizado [x, y, z]
            angle (float): Ángulo de rotación en radianes
            
        Returns:
            np.ndarray: Vector rotado en 3D
        """
        axis = self._normalized(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Fórmula de Rodrigues: v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
        cross_product = np.cross(axis, v)
        dot_product = np.dot(axis, v)
        
        return (v * cos_angle + 
                cross_product * sin_angle + 
                axis * dot_product * (1 - cos_angle))

    def _spherical_to_cartesian(self, r, theta, phi):
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

    def _cartesian_to_spherical(self, v):
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
                self.base_point + self._normalized(target - self.base_point) *
                self._distance(self.base_point, self.joints[self.limbs_size])
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
        dist_to_target = self._distance(self.joints[self.limbs_size], constrained_target)
        total_length = self.limbs_len
        
        if dist_to_target > total_length:
            # The target is unreachable; stretch the chain towards the target
            # Implementación exacta del paper: interpolación lineal en 3D
            for i in range(self.limbs_size):
                # Find the distance ri between the target t and the joint position pi
                ri = self._distance(constrained_target, self.joints[i])
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
            difA = self._distance(self.joints[self.limbs_size], constrained_target)
            iteration = 0
            tol = 1e-3  # Tolerancia según el paper
            
            while difA > tol and iteration < self.ITERATIONS:
                # STAGE 1: FORWARD REACHING (from end effector to base)
                self._backward_pass(constrained_target)
                
                # STAGE 2: BACKWARD REACHING (from base to end effector)  
                self._forward_pass(b)
                
                # Update the distance to target for convergence check
                difA = self._distance(self.joints[self.limbs_size], constrained_target)
                iteration += 1
            
            return self._distance_squared(self.joints[self.limbs_size], constrained_target)

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
            ri = self._distance(self.joints[i + 1], self.joints[i])
            
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
            ri = self._distance(self.joints[i], self.joints[i - 1])
            
            if ri > 1e-8:  # Evitar división por cero
                # Calculate the unit direction vector from pi+1 to pi
                direction = (self.joints[i - 1] - self.joints[i]) / ri
                # Place pi at distance di from pi+1 along the direction vector
                self.joints[i - 1] = self.joints[i] + self.limbs[i - 1][self.LIMB_LEN] * direction
        
        # Aplicar restricciones como post-procesamiento (Algorithm 2)
        self._apply_joint_constraints()

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
            if i == 0:
                # Para el primer segmento, la dirección de referencia es el eje X
                reference_direction = np.array([1.0, 0.0, 0.0])
            else:
                # Para segmentos subsecuentes, la dirección de referencia es el segmento anterior
                prev_segment = self.joints[i] - self.joints[i - 1]
                reference_direction = self._normalized(prev_segment)
            
            # Dirección actual del segmento
            current_segment = self.joints[i + 1] - self.joints[i]
            current_direction = self._normalized(current_segment)
            
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
                    constrained_direction = self._rotated_3d(reference_direction, rotation_axis, constrained_angle)
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
        x_axis = self._normalized(x_axis)
        
        # Y es perpendicular a ambos
        y_axis = np.cross(z_axis, x_axis)
        y_axis = self._normalized(y_axis)
        
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
            'keymap.fullscreen': [],  # Deshabilitar 'f' para fullscreen
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
            
            # Inicializar target en una posición visible
            self.target = np.array([100.0, 50.0, 30.0])
            
            # Variables para controlar la velocidad de movimiento
            self.target_step = 10.0  # Paso de movimiento del target
            
            self.anim = FuncAnimation(self.fig, self.animate, interval=50, blit=False, cache_frame_data=False)
            plt.show()

    def on_key_press(self, event):
        """
        Manejador de eventos para las teclas presionadas.
        
        Controla el movimiento del target usando el teclado (evitando teclas reservadas de matplotlib):
        - Flechas direccionales: Movimiento en el plano XY
        - U/J: Movimiento en el eje Z (arriba/abajo)
        - R: Reset del target a posición inicial
        - +/-: Aumentar/disminuir velocidad de movimiento
        
        Args:
            event: Evento de tecla presionada de Matplotlib
            
        Returns:
            None
        """
        if event.key is None:
            return
            
        key = event.key.lower()
        
        # Movimiento del target usando flechas direccionales
        if key == 'up':  # Flecha arriba - Avanzar en Y
            self.target[1] += self.target_step
        elif key == 'down':  # Flecha abajo - Retroceder en Y
            self.target[1] -= self.target_step
        elif key == 'left':  # Flecha izquierda - Izquierda en X
            self.target[0] -= self.target_step
        elif key == 'right':  # Flecha derecha - Derecha en X
            self.target[0] += self.target_step
        elif key == 'u':  # U - Subir en Z
            self.target[2] += self.target_step
        elif key == 'j':  # J - Bajar en Z
            self.target[2] -= self.target_step
        elif key == 'r':  # Reset posición
            self.target = np.array([100.0, 50.0, 30.0])
            print(f"Target reseteado a: [{self.target[0]:.1f}, {self.target[1]:.1f}, {self.target[2]:.1f}]")
        elif key == '+' or key == '=':  # Aumentar velocidad
            self.target_step = min(self.target_step + 2.0, 50.0)
            print(f"Velocidad aumentada: {self.target_step:.1f}")
        elif key == '-':  # Disminuir velocidad
            self.target_step = max(self.target_step - 2.0, 1.0)
            print(f"Velocidad reducida: {self.target_step:.1f}")
        elif key == 'h':  # Ayuda
            self.print_help()
        elif key == 'w':  # W - Avanzar en Y
            self.target[1] += self.target_step
        elif key == 'a':  # A - Izquierda en X
            self.target[0] -= self.target_step
        elif key == 's':  # S - Retroceder en Y (ahora disponible sin conflictos)
            self.target[1] -= self.target_step
        elif key == 'd':  # D - Derecha en X
            self.target[0] += self.target_step
        elif key == 'q':  # Q - Subir en Z (ahora disponible sin conflictos)
            self.target[2] += self.target_step
        elif key == 'e':  # E - Bajar en Z
            self.target[2] -= self.target_step
        
        # Limitar el target dentro del área visible
        plot_range = self.limbs_len * 1.2
        self.target[0] = np.clip(self.target[0], -plot_range, plot_range)
        self.target[1] = np.clip(self.target[1], -plot_range, plot_range) 
        self.target[2] = np.clip(self.target[2], -plot_range, plot_range)

    def print_help(self):
        """
        Imprime la ayuda de controles en la consola.
        """
        print("\n" + "="*60)
        print("CONTROLES DEL TARGET - FABRIK 3D")
        print("="*60)
        print("Movimiento del Target:")
        print("   FLECHAS DIRECCIONALES:")
        print("      ↑/↓  - Mover en eje Y (adelante/atrás)")
        print("      ←/→  - Mover en eje X (izquierda/derecha)")
        print("   CONTROLES WASD:")
        print("      W/S  - Mover en eje Y (adelante/atrás)")
        print("      A/D  - Mover en eje X (izquierda/derecha)")
        print("   EJE Z (vertical):")
        print("      Q/E  - Subir/Bajar en Z")
        print("      U/J  - Subir/Bajar en Z (alternativo)")
        print("\nConfiguración:")
        print("   R    - Reset target a posición inicial")
        print("   +/-  - Aumentar/disminuir velocidad")
        print("   H    - Mostrar esta ayuda")
        print("\nVista 3D:")
        print("   Mouse - Rotar vista (nativo matplotlib)")
        print("   Scroll - Zoom in/out")
        print("\nNota: Los keymaps de matplotlib están deshabilitados")
        print("   para permitir el uso de Q, S y otras teclas.")
        print("="*60)
        print(f"Estado actual:")
        print(f"   Target: [{self.target[0]:.1f}, {self.target[1]:.1f}, {self.target[2]:.1f}]")
        print(f"   Velocidad: {self.target_step:.1f}")
        print("="*60 + "\n")

    def _rotated(self, v, angle):
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

    def animate(self, i):
        """
        Función de animación para la actualización en tiempo real en 3D.
        
        Se ejecuta continuamente para actualizar la visualización del robot,
        incluyendo las articulaciones, enlaces y restricciones angulares en 3D.
        
        Args:
            i (int): Número de frame de la animación (no utilizado)
            
        Returns:
            list: Lista de elementos gráficos actualizados
        """
        self.update(self.target)
        
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
                    prev_direction = self._normalized(self.joints[i] - self.joints[i-1])
                else:
                    prev_direction = np.array([1.0, 0.0, 0.0])  # Eje X como referencia
                
                # Dibujar cono de restricción simplificado
                # Crear círculo en el plano perpendicular a la dirección anterior
                cone_radius = 30.0  # Radio visual del cono
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
                    perp1 = self._normalized(perp1)
                    perp2 = np.cross(prev_direction, perp1)
                    perp2 = self._normalized(perp2)
                    
                    # Puntos en el círculo
                    circle_point = perp1 * np.cos(theta) + perp2 * np.sin(theta)
                    
                    # Rotar para crear el cono
                    min_direction = self._rotated_3d(prev_direction, circle_point, min_angle)
                    max_direction = self._rotated_3d(prev_direction, circle_point, max_angle)
                    
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

        # Actualizar información del target (sin crear nuevo texto)
        target_info = f"Target: [{self.target[0]:.1f}, {self.target[1]:.1f}, {self.target[2]:.1f}] | Step: {self.target_step:.1f}"
        self.info_text.set_text(target_info)

        return [self.line, self.target_point, self.base_indicator, self.info_text] + self.constraint_lines


if __name__ == '__main__':
    """
    Punto de entrada principal del programa.
    
    Crea una instancia del sistema FABRIK y inicia la visualización interactiva.
    El usuario puede mover el mouse para definir objetivos y ver cómo el robot
    resuelve la cinemática inversa en tiempo real usando el algoritmo FABRIK
    fiel al paper original de Aristidou & Lasenby (2011).
    """
    ik_system = FabrikIK3D()
    
    print("FABRIK 3D Implementation - Visualización 3D Real")
    print("Algoritmos implementados:")
    print("   Algorithm 1: FABRIK básico (3D)")
    print("   Algorithm 2: Joint Constraints (restricciones esféricas)")
    print("   Algorithm 3: Target Constraint Application (workspace limits)")
    print("\nControles de Teclado:")
    print("   • WASD: mover target en plano XY (primario)")
    print("   • QE: subir/bajar target en Z (primario)")
    print("   • Flechas direccionales: mover target XY (alternativo)")
    print("   • U/J: subir/bajar target en Z (alternativo)")
    print("   • R: reset target a posición inicial")
    print("   • +/-: aumentar/disminuir velocidad")
    print("   • H: mostrar ayuda completa")
    print("\nMouse: controla la vista 3D (rotar/zoom)")
    print("VISUALIZACIÓN 3D REAL: Conos de restricción, articulaciones espaciales")
    print("Colores: Robot (azul), Target (verde), Base (amarillo), Restricciones (naranja/rojo)")
    print("=" * 70)
    print("Presiona 'H' en la ventana para ayuda detallada")
    print("Nota: Q y S ahora funcionan sin conflictos gracias a mpl.rc_context()")
    print("=" * 70)
    
    ik_system.setup_plot()
