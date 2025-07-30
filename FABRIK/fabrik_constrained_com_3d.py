import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Traducción y adaptación de un script de GDScript a Python con Matplotlib.
# El script original implementa el algoritmo FABRIK para cinemática inversa.

# ===============================================================================
# ALGORITMOS IMPLEMENTADOS (basados en Aristidou & Lasenby, 2011):
# ===============================================================================
# 
# ALGORITMO PRINCIPAL: Híbrido Algorithm 1 + Algorithm 2
# 
# 🔸 Algorithm 1: FABRIK (Forward And Backward Reaching Inverse Kinematics)
#   - Implementación del algoritmo básico FABRIK con fases forward/backward
#   - Optimización de anti-vibración para evitar oscilaciones cerca del objetivo
#   - Manejo de targets fuera del alcance con stretching automático
# 
# 🔸 Algorithm 2: Joint Constraint Application (for restricted joints)
#   - Aplicación de restricciones angulares en cada articulación
#   - Clipping de ángulos usando np.clip() en lugar de rotores
#   - Restricciones aplicadas en cada paso de forward/backward pass
# 
# DIFERENCIAS RESPECTO AL PAPER ORIGINAL:
# ===============================================================================
# 
# ✅ IMPLEMENTADO SEGÚN EL PAPER:
# - ✓ Fases Forward y Backward alternadas
# - ✓ Aplicación de restricciones de articulación (Algorithm 2)
# - ✓ Manejo de targets fuera de alcance
# - ✓ Convergencia iterativa con tolerancia
# - ✓ Preservación del punto base fijo
# 
# 🔄 OPTIMIZACIONES AÑADIDAS:
# - 🔸 Anti-vibración: Almacena configuración de distancia mínima
# - 🔸 Interpolación suave (lerp_amount) para movimientos más naturales
# - 🔸 Fallback inteligente para targets demasiado cercanos al base
# - 🔸 Cálculo de distancia al cuadrado para eficiencia computacional
# 
# ⚠️ DIFERENCIAS TÉCNICAS:
# - 📐 Usa ángulos directos en lugar de rotores/quaterniones del paper
# - 🎯 Implementa restricciones durante cada paso vs. post-procesamiento
# - 🔄 Método de restricción simplificado (clip vs. projection en cónicas)
# - 📊 Optimización específica para 2D (no extensible directamente a 3D)
# 
# 📋 LISTA DE TAREAS PENDIENTES:
# ===============================================================================
# [ ] Implementar Algorithm 3: Target Constraint Application (workspace limits)
# [ ] Implementar Algorithm 4: Position to Joint Angles Conversion  
# [ ] Implementar Algorithm 5: Multi-Target FABRIK (multiple end effectors)
# [ ] Implementar Algorithm 6: FABRIK with Orientation Control
# [ ] Extensión a 3D con restricciones esféricas
# [ ] Optimización con matrices de rotación para mejor precision
# [ ] Implementación de restricciones de workspace usando secciones cónicas
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

    # Nueva función de actualización general introducida para manejar la interpolación y
    # el caso en que el objetivo está demasiado cerca del punto base para el IK con restricciones
    def update(self, target: np.ndarray) -> None:
        """
        Función principal de actualización del sistema FABRIK.
        
        EXTENSIÓN NO DOCUMENTADA EN EL PAPER ORIGINAL:
        Esta función implementa optimizaciones adicionales no presentes en los 
        algoritmos 1-6 del paper de Aristidou & Lasenby.
        
        Maneja casos especiales y optimizaciones:
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
        # Actualizando el IK, y obteniendo la distancia al cuadrado (más rápido de calcular)
        # entre la última articulación y el objetivo
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
        Ejecuta el algoritmo FABRIK principal con fases hacia atrás y hacia adelante.
        
        ALGORITMO IMPLEMENTADO: Híbrido Algorithm 1 (FABRIK básico) + Algorithm 2 (restricciones)
        
        Implementa la lógica central del algoritmo FABRIK según el paper de Aristidou & Lasenby:
        1. Verifica si el target está dentro del alcance (Algorithm 1, paso inicial)
        2. Si está fuera de alcance: una sola iteración con stretching
        3. Si está en alcance: iteraciones alternadas backward/forward con restricciones
        4. Optimización anti-vibración: almacena configuración de distancia mínima
        
        DIFERENCIAS DEL PAPER:
        - Usa distancia al cuadrado en lugar de distancia simple (optimización)
        - Implementa anti-vibración no mencionada en el paper original
        - Las restricciones se aplican durante cada paso, no como post-procesamiento
        
        Args:
            target (np.ndarray): Posición objetivo [x, y] para el efector final
            
        Returns:
            float: Distancia al cuadrado entre el efector final y el objetivo
        """
        # Almacenamiento de la distancia entre la última articulación y el objetivo,
        # la primera inicialización se usa para el cálculo del sobrepaso del objetivo desde el rango posible
        dist = self._distance_squared(self.base_point, target)
        iterations = 0
        
        # Si el objetivo está lejos del rango, podemos usar solo una iteración
        if dist > self.limbs_len * self.limbs_len:
            self._backward_pass(target)
            self._forward_pass()
            return 0.0
        
        # Almacenamiento de la distancia mínima entre la última articulación y el objetivo
        min_dist = float('inf')
        min_joints = []
        while iterations < self.ITERATIONS:
            self._backward_pass(target)
            self._forward_pass()
            iterations += 1
            # Distancia entre la última articulación y el objetivo
            dist = self._distance_squared(self.joints[self.limbs_size], target)
            # Si la última articulación está cerca del punto objetivo, puede empezar a "vibrar"
            # así que si la distancia actual es mayor que la mínima -
            # sabemos que estamos bastante cerca del objetivo y podemos
            # romper el bucle de forma segura
            if min_dist > dist:
                min_dist = dist
                # Almacena todos los estados de las articulaciones mínimas,
                # para que podamos restaurarlos cuando haya sobrepaso
                min_joints = list(self.joints)
            else:
                break
        
        # Restaurando desde las articulaciones sobrepasadas a las últimas mínimas
        if dist > min_dist:
            self.joints = min_joints
        
        return self._distance_squared(self.joints[self.limbs_size], target)

    def _forward_pass(self) -> None:
        """
        Ejecuta la fase hacia adelante del algoritmo FABRIK (Algorithm 1, Stage 2).
        
        IMPLEMENTACIÓN: Algorithm 1 (FABRIK) + Algorithm 2 (Joint Constraints)
        
        Procesa desde el punto base hacia el efector final, aplicando las restricciones
        angulares de cada articulación según Algorithm 2. Esta es la "Stage 2: BACKWARD 
        REACHING" del paper (confusamente llamada pero va desde base hacia efector).
        
        ALGORITMO SEGÚN PAPER:
        1. Fijar p1 = base_point (preservar punto base)
        2. Para cada i de 1 a n-1:
           - Calcular ri = |pi+1 - pi|
           - direction = (pi+1 - pi) / ri  
           - pi+1 = pi + di * direction
        3. Aplicar restricciones de articulación (Algorithm 2)
        
        DIFERENCIAS DE IMPLEMENTACIÓN:
        - Usa ángulos y rotaciones en lugar de direcciones unitarias simples
        - Aplica restricciones inline con np.clip() vs. post-procesamiento
        - Maneja root_angle acumulativo para restricciones relativas
        
        Returns:
            None
        """
        # Define la variable root_angle fuera del bucle, ya que podemos
        # no calcularla y simplemente establecerla al final de la iteración
        # al ángulo
        root_angle = 0.0
        # Forzar el primer punto al base_point,
        # debido a que el paso hacia atrás muy probablemente no lo colocó allí
        self.joints[0] = self.base_point
        # Para cada extremidad, así como para cada par de articulaciones de esa extremidad (i e i + 1)
        for i in range(self.limbs_size):
            limb = self.limbs[i]
            a = self.joints[i]
            b = self.joints[i + 1]
            # Calculando la diferencia entre el par de puntos actual
            # restando el ángulo base del par anterior
            # y envolviéndolo a -PI, PI para bien
            # (evitar esa cosa de matemáticas de matrices a toda costa)
            diff_angle_raw = self._wrap_angle(self._angle_to_point(a, b) - root_angle)
            # Sujeta ese ángulo de diferencia al mínimo/máximo de la extremidad actual
            diff_angle = np.clip(
                    diff_angle_raw, limb[self.LIMB_MIN], limb[self.LIMB_MAX]
            )
            # El ángulo ahora es la suma del ángulo raíz del par anterior
            # y el ángulo de diferencia sujetado actual
            angle = root_angle + diff_angle
            
            # Establece la articulación final de la extremidad en inicio + longitud rotada de esa extremidad
            self.joints[i + 1] = a + self._rotated(np.array([limb[self.LIMB_LEN], 0]), angle)
            # Establece el ángulo raíz para la siguiente iteración
            root_angle = angle

    def _backward_pass(self, target: np.ndarray) -> None:
        """
        Ejecuta la fase hacia atrás del algoritmo FABRIK (Algorithm 1, Stage 1).
        
        IMPLEMENTACIÓN: Algorithm 1 (FABRIK) + Algorithm 2 (Joint Constraints)
        
        Procesa desde el objetivo hacia el punto base, aplicando las restricciones
        angulares. Esta es la "Stage 1: FORWARD REACHING" del paper (confusamente 
        llamada pero va desde el efector hacia la base).
        
        ALGORITMO SEGÚN PAPER:
        1. Fijar pn = target (efector final en objetivo)
        2. Para cada i de n-1, n-2, ..., 1:
           - Calcular ri = |pi+1 - pi|
           - direction = (pi - pi+1) / ri
           - pi = pi+1 + di * direction
        3. Aplicar restricciones de articulación (Algorithm 2)
        
        DIFERENCIAS DE IMPLEMENTACIÓN:
        - Calcula ángulos root_angle para restricciones relativas
        - Aplica restricciones usando np.clip() inline
        - Maneja la rotación de π radianes para dirección inversa
        - Usa c = joints[i-2] para calcular ángulos de referencia
        
        Args:
            target (np.ndarray): Posición objetivo [x, y] para el efector final
            
        Returns:
            None
        """
        # Forzar el último punto al punto objetivo,
        # para que podamos ir hacia atrás hasta aproximadamente el base_point
        # Nota que el tamaño de las articulaciones es el tamaño de las extremidades + 1
        self.joints[self.limbs_size] = target
        # Para cada extremidad, así como para cada par de articulaciones
        # de esa extremidad (i e i - 1) hacia atrás
        for i in range(self.limbs_size, 0, -1):
            limb = self.limbs[i - 1]
            # Este es el primer punto desde el FINAL
            a = self.joints[i]
            # Este es el segundo punto desde el FINAL
            b = self.joints[i - 1]
            # Este es el tercer punto desde el FINAL
            # o el punto base, si estamos al principio de las articulaciones
            # se necesita para el cálculo de root_angle desde atrás
            c = self.joints[i - 2] if i > 1 else self.base_point
            root_angle = self._angle_to_point(c, b)
            # Calculando la diferencia entre el par de puntos actual
            # restando el ángulo base del... siguiente par, ya que vamos
            # hacia atrás
            diff_angle_raw = self._angle_to_point(b, a) - root_angle
            # Sujeta ese ángulo de diferencia al mínimo/máximo de la extremidad actual
            diff_angle = np.clip(
                    diff_angle_raw, limb[self.LIMB_MIN], limb[self.LIMB_MAX]
            )
            # El ángulo ahora es la suma del ángulo raíz del siguiente par
            # y el ángulo de diferencia sujetado actual
            angle = root_angle + diff_angle
            
            # Establece la articulación inicial de la extremidad en final + longitud rotada de esa extremidad + PI
            # ya que estamos calculando ángulos desde el final
            self.joints[i - 1] = a + self._rotated(np.array([limb[self.LIMB_LEN], 0]), angle + np.pi)

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
    resuelve la cinemática inversa en tiempo real.
    """
    ik_system = FabrikIK()
    ik_system.setup_plot()
