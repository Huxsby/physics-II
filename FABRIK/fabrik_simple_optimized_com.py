import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FabrikIK:
    def __init__(self, limbs, base_point):
        # Longitudes de cada miembro en px
        self.limbs = np.array(limbs)
        # Punto base autoexplicativo
        self.base_point = np.array(base_point, dtype=float)
        # Número de miembros
        self.limbs_size = len(self.limbs)
        # Longitud total de los miembros, usada para calcular el desbordamiento del objetivo desde el rango posible
        self.limbs_len = np.sum(self.limbs)
        # Los puntos con los que trabajamos
        self.joints = np.zeros((self.limbs_size + 1, 2))
        
        # Inicializar la posición de las articulaciones en una línea recta
        self.joints[0] = self.base_point
        for i in range(self.limbs_size):
            self.joints[i + 1] = self.joints[i] + np.array([self.limbs[i], 0])

        # Constante para las iteraciones
        self.ITERATIONS = 32
        
        # Para el benchmark de iteraciones, puede ser eliminado
        self.all_iterations = 0
        self.iterations_count = 0
        
        # Almacenamiento del objetivo (posición del ratón)
        self.target = np.array([0, 0], dtype=float)

    def _forward_pass(self):
        # Forzar el primer punto al punto base,
        # ya que el paso hacia atrás probablemente no lo dejó allí
        self.joints[0] = self.base_point
        # Para cada miembro, así como para cada par de articulaciones de ese miembro (i y i + 1)
        for i in range(self.limbs_size):
            a = self.joints[i]
            b = self.joints[i + 1]
            # Vector de a a b
            vec = b - a
            # Distancia
            dist = np.linalg.norm(vec)
            # Normalizar y escalar por la longitud del miembro
            direction = vec / dist if dist > 0 else np.array([1, 0])
            
            # Establecer la articulación final del miembro en inicio + longitud rotada de ese miembro
            self.joints[i + 1] = a + direction * self.limbs[i]

    def _backward_pass(self):
        # Forzar el último punto al punto objetivo,
        # para que podamos retroceder aproximadamente al punto base
        # Nota que el tamaño de las articulaciones es el tamaño de los miembros + 1
        self.joints[self.limbs_size] = self.target
        # Para cada miembro, así como para cada par de articulaciones
        # de ese miembro (i y i - 1) hacia atrás
        for i in range(self.limbs_size, 0, -1):
            a = self.joints[i]
            b = self.joints[i - 1]
            # Vector de a a b
            vec = b - a
            # Distancia
            dist = np.linalg.norm(vec)
            # Normalizar y escalar por la longitud del miembro
            direction = vec / dist if dist > 0 else np.array([1, 0])
            
            # Establecer la articulación inicial del miembro en final + longitud rotada de ese miembro
            self.joints[i - 1] = a + direction * self.limbs[i - 1]

    def update_ik(self):
        # Almacenamiento de la distancia entre la última articulación y el objetivo,
        # la primera inicialización se usa para el cálculo del desbordamiento del objetivo desde el rango posible
        dist_sq = np.sum((self.base_point - self.target)**2)
        iterations = 0
        
        # Si el objetivo está lejos del alcance, podemos usar solo una iteración
        if dist_sq > self.limbs_len**2:
            self._backward_pass()
            self._forward_pass()
            # Para fines de benchmark de iteraciones, puede ser eliminado
            iterations = 1
        # Sin el benchmark, todo el bloque de abajo
        # puede ser aplanado un nivel de tabulación en lugar de la rama else
        else:
            # Almacenamiento de la distancia mínima entre la última articulación y el objetivo
            min_dist_sq = np.inf
            min_joints = None
            while iterations < self.ITERATIONS:
                self._backward_pass()
                self._forward_pass()
                iterations += 1
                # Distancia entre la última articulación y el objetivo
                dist_sq_to_target = np.sum((self.joints[self.limbs_size] - self.target)**2)
                # Si la última articulación está cerca del punto objetivo, puede empezar a "vibrar"
                # así que si la distancia actual es mayor que la mínima -
                # sabemos que estamos bastante cerca del objetivo y podemos
                # salir del bucle de forma segura
                if min_dist_sq > dist_sq_to_target:
                    min_dist_sq = dist_sq_to_target
                    # Almacenar todos los estados mínimos de las articulaciones,
                    # para que podamos restaurarlos en caso de desbordamiento
                    min_joints = self.joints.copy()
                else:
                    break
            
            # Restaurar desde las articulaciones desbordadas a las últimas mínimas
            if min_joints is not None and dist_sq_to_target > min_dist_sq:
                self.joints = min_joints
        
        # Benchmark de iteraciones, puede ser eliminado
        self.all_iterations += iterations
        self.iterations_count += 1
        if self.iterations_count > 0:
            avg_iter = self.all_iterations / self.iterations_count
            # print(f"Avg iterations: {avg_iter:.2f}") # Descomentar para ver el benchmark

# --- Configuración de Matplotlib ---
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)

# Instancia de la clase IK
fabrik = FabrikIK(limbs=[80, 60, 80], base_point=[250, 250])

# Elementos del gráfico a actualizar
line, = ax.plot([], [], 'o-', lw=2, color='gray')
joints_dots, = ax.plot([], [], 'o', color='white', markersize=4)
info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def on_motion(event):
    # Actualizar el objetivo con la posición del ratón si está dentro de los ejes
    if event.inaxes:
        fabrik.target[0] = event.xdata
        fabrik.target[1] = event.ydata

fig.canvas.mpl_connect('motion_notify_event', on_motion)

def init():
    line.set_data([], [])
    joints_dots.set_data([], [])
    info_text.set_text('')
    return line, joints_dots, info_text

def animate(i):
    # Actualizar la cinemática inversa
    fabrik.update_ik()
    
    # Actualizar los datos del gráfico
    line.set_data(fabrik.joints[:, 0], fabrik.joints[:, 1])
    joints_dots.set_data(fabrik.joints[:, 0], fabrik.joints[:, 1])
    
    # Actualizar texto de benchmark
    if fabrik.iterations_count > 0:
        avg_iter = fabrik.all_iterations / fabrik.iterations_count
        info_text.set_text(f'Iteraciones promedio: {avg_iter:.2f}')
        
    return line, joints_dots, info_text

# Crear y ejecutar la animación
ani = FuncAnimation(fig, animate, init_func=init, blit=True, interval=20)

plt.title("FABRIK IK con Python")
plt.grid(True)
plt.show()
