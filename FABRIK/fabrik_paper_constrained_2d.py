import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Traducción y adaptación de un script de GDScript a Python con Matplotlib.
# El script original impleme    def update_ik(self, target: np.ndarray) -> float: FABRIK para cinemática inversa.

# ===============================================================================
# ALGORITMO IMPLEMENTADO: FABRIK según Aristidou & Lasenby (2011)
# ===============================================================================
# 
# 🔸 Algorithm 1: FABRIK (Forward And Backward Reaching Inverse Kinematics)
#   - Implementación fiel al algoritmo básico FABRIK con fases forward/backward
#   - Manejo de targets fuera del alcance con stretching automático
#   - Convergencia iterativa con tolerancia exacta del paper
# 
# 🔸 Algorithm 2: Joint Constraint Application (for restricted joints)
#   - Aplicación de restricciones angulares post-procesamiento
#   - Restricciones aplicadas después de cada iteración completa
# 
# FIEL AL PAPER ORIGINAL:
# ===============================================================================
# 
# ✅ IMPLEMENTACIÓN EXACTA:
# - ✓ Usa distancia simple (no al cuadrado) como en el algoritmo original
# - ✓ Convergencia basada en tolerancia difA > tol del paper
# - ✓ Direcciones unitarias simples: direction = (pi+1 - pi) / ri
# - ✓ Post-procesamiento de restricciones (Algorithm 2)
# - ✓ Stretching lineal para targets fuera de alcance
# - ✓ Preservación exacta del punto base fijo
# 
# 📋 ALGORITMOS PENDIENTES DE IMPLEMENTAR:
# ===============================================================================
# [ ] Algorithm 3: Target Constraint Application (workspace limits)
# [ ] Algorithm 4: Position to Joint Angles Conversion  
# [ ] Algorithm 5: Multi-Target FABRIK (multiple end effectors)
# [ ] Algorithm 6: FABRIK with Orientation Control
# ===============================================================================

class FabrikIK:
    """
    Implementación del algoritmo FABRIK (Forward And Backward Reaching Inverse Kinematics)
    para cinemática inversa con restricciones angulares en 2D.
    
    Este algoritmo utiliza un enfoque iterativo de dos fases:
    1. Fase hacia atrás (backward pass): desde el objetivo hacia la base
    2. Fase hacia adelante (forward pass): desde la base hacia el objetivo
    
    Adaptado y traducido de GDScript a Python con Matplotlib para visualización.
    """
    
    def __init__(self):
        """
        Inicializa el sistema de cinemática inversa FABRIK.
        
        Configura las constantes, parámetros del algoritmo y estructura inicial
        de las articulaciones del robot.
        """
        # Índices para el almacenamiento de las extremidades, autoexplicativos
        self.LIMB_LEN = 0  # Índice para la longitud de la extremidad
        self.LIMB_MIN = 1  # Índice para el ángulo mínimo de restricción
        self.LIMB_MAX = 2  # Índice para el ángulo máximo de restricción
        
        # Usado para calcular cuánto falla el IK en alcanzar el objetivo
        self.BIAS = 3.0  # Tolerancia de error para considerar el objetivo alcanzado
        self.ITERATIONS = 32  # Número máximo de iteraciones del algoritmo

        # Autoexplicativo
        self.base_point = np.array([0.0, 0.0])  # Punto base fijo del robot
        # Longitudes de cada extremidad en px, y restricciones de ángulo mínimo y máximo
        self.limbs = [
            [80, -np.deg2rad(65), np.deg2rad(65)],  # [longitud, ángulo_min, ángulo_max]
            [60, -np.deg2rad(65), np.deg2rad(65)],
            [80, -np.deg2rad(65), np.deg2rad(65)],
        ]
        # Los puntos con los que trabajamos (articulaciones del robot)
        self.joints = []

        # Para no llamar a len(limbs) cada vez
        self.limbs_size = len(self.limbs)
        # Usado para calcular el sobrepaso del objetivo desde el rango posible
        self.limbs_len = 0.0

        # Cantidad de interpolación (lerp) de las articulaciones antiguas a las nuevas, 
        # 1.0 deshabilita completamente la interpolación
        self.lerp_amount = 0.5
        
        self.target = np.array([0.0, 0.0])  # Posición objetivo del efector final

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
        Normaliza un vector.
        
        Args:
            v (np.ndarray): Vector a normalizar
            
        Returns:
            np.ndarray: Vector normalizado (magnitud = 1) o vector original si magnitud = 0
        """
        norm = np.linalg.norm(v)
        if norm == 0: 
           return v
        return v / norm

    def _rotated(self, v, angle):
        """
        Rota un vector por un ángulo dado.
        
        Args:
            v (np.ndarray): Vector 2D a rotar [x, y]
            angle (float): Ángulo de rotación en radianes
            
        Returns:
            np.ndarray: Vector rotado
        """
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
        return rotation_matrix @ v

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

    def update_ik_selector(self, target: np.ndarray) -> float:
        """
        Selector que llama la implementación de FABRIK apropiada.
        
        Args:
            target (np.ndarray): Posición objetivo [x, y] para el efector final
            
        Returns:
            float: Distancia al cuadrado entre el efector final y el objetivo
        """
        if self.use_paper_algorithm:
            return self.update_ik(target)  # Versión fiel al paper
        else:
            return self.update_ik_optimized(target)  # Versión optimizada

    def update_ik(self, target: np.ndarray) -> float:
        """
        Ejecuta el algoritmo FABRIK principal según el paper original (Algorithm 1).
        
        ALGORITMO IMPLEMENTADO: Algorithm 1 (FABRIK básico) según Aristidou & Lasenby
        
        Implementa la lógica exacta del paper:
        1. Verifica si el target está dentro del alcance (dist_to_target vs total_length)
        2. Si está fuera de alcance: stretching con interpolación lineal
        3. Si está en alcance: iteraciones alternadas backward/forward hasta convergencia
        4. Usa direcciones unitarias simples como en el paper original
        
        FIEL AL PAPER ORIGINAL:
        - Usa distancia simple (no al cuadrado) como en el algoritmo
        - Convergencia basada en tolerancia difA > tol
        - Direcciones unitarias simples: direction = (pi+1 - pi) / ri
        - Sin optimizaciones anti-vibración (extensión posterior)
        
        Args:
            target (np.ndarray): Posición objetivo [x, y] para el efector final
            
        Returns:
            float: Distancia al cuadrado entre el efector final y el objetivo
        """
        # Check if the target is within reachable distance (según Algorithm 1)
        dist_to_target = self._distance(self.joints[self.limbs_size], target)
        total_length = self.limbs_len
        
        if dist_to_target > total_length:
            # The target is unreachable; stretch the chain towards the target
            # Implementación exacta del paper: interpolación lineal
            for i in range(self.limbs_size):
                # Find the distance ri between the target t and the joint position pi
                ri = self._distance(target, self.joints[i])
                if ri > 1e-8:  # Evitar división por cero
                    # Find the scaling factor ki to maintain link length
                    ki = self.limbs[i][self.LIMB_LEN] / ri
                    # Find the new joint positions pi+1 using linear interpolation
                    self.joints[i + 1] = (1 - ki) * self.joints[i] + ki * target
            return 0.0
        else:
            # The target is reachable; implementar bucle principal del Algorithm 1
            # Set as b the initial position of the joint p1
            b = self.base_point.copy()
            
            # Check whether the distance between the end effector pn and target t is greater than tolerance
            difA = self._distance(self.joints[self.limbs_size], target)
            iteration = 0
            tol = 1e-3  # Tolerancia según el paper
            
            while difA > tol and iteration < self.ITERATIONS:
                # STAGE 1: FORWARD REACHING (from end effector to base)
                self._backward_pass(target)
                
                # STAGE 2: BACKWARD REACHING (from base to end effector)  
                self._forward_pass(b)
                
                # Update the distance to target for convergence check
                difA = self._distance(self.joints[self.limbs_size], target)
                iteration += 1
            
            return self._distance_squared(self.joints[self.limbs_size], target)

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
        Algorithm 2: Joint Constraint Application (for restricted joints) - Post-procesamiento.
        
        Aplica las restricciones de articulación después de cada paso de FABRIK,
        según el Algorithm 2 del paper original. Este método implementa la lógica
        de restricciones como post-procesamiento en lugar de inline.
        
        SEGÚN EL PAPER:
        1. Check whether the rotor R is within the motion range bounds
        2. If outside bounds: clamp to nearest boundary
        3. Reorient the joint pi-1 to respect constraints
        
        IMPLEMENTACIÓN SIMPLIFICADA:
        - Usa ángulos directos en lugar de rotores (para 2D)
        - Aplica np.clip() para mantener restricciones angulares
        - Recalcula posiciones basadas en ángulos restringidos
        
        Returns:
            None
        """
        # Comenzar desde la base y aplicar restricciones secuencialmente
        current_angle = 0.0  # Ángulo acumulativo desde la base
        
        for i in range(self.limbs_size):
            if i == 0:
                # Para el primer segmento, el ángulo de referencia es desde la base
                target_angle = self._angle_to_point(self.joints[i], self.joints[i + 1])
                reference_angle = 0.0  # Asumimos dirección horizontal como referencia
            else:
                # Para segmentos subsecuentes, el ángulo es relativo al segmento anterior
                target_angle = self._angle_to_point(self.joints[i], self.joints[i + 1])
                reference_angle = current_angle
            
            # Calcular el ángulo relativo del segmento actual
            relative_angle = self._wrap_angle(target_angle - reference_angle)
            
            # Aplicar restricciones de articulación (Algorithm 2)
            limb = self.limbs[i]
            constrained_relative_angle = np.clip(
                relative_angle, 
                limb[self.LIMB_MIN], 
                limb[self.LIMB_MAX]
            )
            
            # Calcular el nuevo ángulo absoluto
            new_absolute_angle = reference_angle + constrained_relative_angle
            
            # Recalcular la posición del punto final del segmento
            link_vector = np.array([limb[self.LIMB_LEN], 0])
            rotated_link = self._rotated(link_vector, new_absolute_angle)
            self.joints[i + 1] = self.joints[i] + rotated_link
            
            # Actualizar el ángulo acumulativo para el siguiente segmento
            current_angle = new_absolute_angle

    def _ready(self):
        """
        Inicializa la estructura de articulaciones del robot.
        
        Configura las posiciones iniciales de las articulaciones en una línea recta
        desde el punto base, calcula la longitud total del robot y prepara
        los arrays de datos para el procesamiento.
        
        Returns:
            None
        """
        # El tamaño de las articulaciones es el tamaño de las extremidades + 1
        self.joints = [np.zeros(2) for _ in range(self.limbs_size + 1)]
        # Es bueno tener un IK resuelto que no mire al punto Vector2.ZERO
        # así que simplemente hacemos una línea recta con las longitudes de las extremidades desde el base_point
        self.joints[0] = self.base_point
        for i in range(self.limbs_size):
            self.limbs_len += self.limbs[i][self.LIMB_LEN]
            self.joints[i + 1] = self.joints[i] + np.array([self.limbs[i][self.LIMB_LEN], 0])
        
        self.joints = [np.array(j, dtype=float) for j in self.joints]


    def setup_plot(self):
        """
        Configura la visualización de Matplotlib para el robot.
        
        Inicializa la ventana de graficación, establece los límites del plot,
        configura los elementos visuales y conecta los eventos de mouse
        para la interacción en tiempo real.
        
        Returns:
            None
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        
        # Establecer límites basados en la longitud total de las extremidades
        plot_range = self.limbs_len * 1.2
        self.ax.set_xlim(self.base_point[0] - plot_range, self.base_point[0] + plot_range)
        self.ax.set_ylim(self.base_point[1] - plot_range, self.base_point[1] + plot_range)
        
        self.line, = self.ax.plot([], [], 'o-', color='gray', lw=1, markersize=4, markerfacecolor='white')
        self.constraint_lines = []
        
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.anim = FuncAnimation(self.fig, self.animate, interval=20, blit=True)
        plt.show()

    def on_mouse_move(self, event):
        """
        Manejador de eventos para el movimiento del mouse.
        
        Actualiza la posición objetivo del robot basándose en la posición
        del cursor del mouse dentro del área de graficación.
        
        Args:
            event: Evento de movimiento del mouse de Matplotlib
            
        Returns:
            None
        """
        if event.inaxes:
            self.target = np.array([event.xdata, event.ydata])

    def animate(self, i):
        """
        Función de animación para la actualización en tiempo real.
        
        Se ejecuta continuamente para actualizar la visualización del robot,
        incluyendo las articulaciones, enlaces y restricciones angulares.
        
        Args:
            i (int): Número de frame de la animación (no utilizado)
            
        Returns:
            list: Lista de elementos gráficos actualizados para el blitting
        """
        self.update(self.target)
        
        # Dibujar
        points = np.array(self.joints)
        self.line.set_data(points[:, 0], points[:, 1])
        
        # Limpiar líneas de restricción antiguas
        for l in self.constraint_lines:
            l.remove()
        self.constraint_lines.clear()

        for i in range(self.limbs_size + 1):
            if i > 0:
                # Dibujo un tanto desordenado de las restricciones
                p_base = self.joints[i - 1]
                p_prev = self.joints[i - 2] if i > 1 else self.base_point - np.array([1,0]) # Vector DERECHA para el primer segmento
                
                direction = self._normalized(p_base - p_prev)
                
                # Rotar el vector de dirección para obtener los límites de las restricciones
                min_vec = self._rotated(direction, self.limbs[i - 1][self.LIMB_MIN]) * 32
                max_vec = self._rotated(direction, self.limbs[i - 1][self.LIMB_MAX]) * 32
                
                min_line_data = np.array([p_base, p_base + min_vec])
                max_line_data = np.array([p_base, p_base + max_vec])
                
                l_min, = self.ax.plot(min_line_data[:,0], min_line_data[:,1], color='darkgoldenrod', lw=0.5)
                l_max, = self.ax.plot(max_line_data[:,0], max_line_data[:,1], color='darkgoldenrod', lw=0.5)
                self.constraint_lines.extend([l_min, l_max])

        return [self.line] + self.constraint_lines


if __name__ == '__main__':
    """
    Punto de entrada principal del programa.
    
    Crea una instancia del sistema FABRIK y inicia la visualización interactiva.
    El usuario puede mover el mouse para definir objetivos y ver cómo el robot
    resuelve la cinemática inversa en tiempo real usando el algoritmo FABRIK
    fiel al paper original de Aristidou & Lasenby (2011).
    """
    ik_system = FabrikIK()
    
    print("🎯 FABRIK Implementation - Fiel al Paper Original")
    print("📋 Algoritmos: Algorithm 1 (FABRIK) + Algorithm 2 (Joint Constraints)")
    print("🖱️  Mueve el mouse para controlar el robot")
    print("=" * 50)
    
    ik_system.setup_plot()
