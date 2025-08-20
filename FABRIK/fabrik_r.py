import numpy as np
from enum import Enum
import math
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

# Importaciones locales - ajustadas para estructura del proyecto
from core import cargar_robot_desde_yaml, Robot
from visualization import RecordingSystem

# ===============================================================================
# 0. CLASES AUXILIARES PARA MEJORAR EL DEBUGGING Y ESTADO
# ===============================================================================

class JointState:
    """Estado detallado de una articulación durante FABRIK-R."""
    def __init__(self, joint_idx, joint_type, axis, limits, length):
        self.idx = joint_idx
        self.type = joint_type
        self.axis = np.array(axis, dtype=float)
        self.axis_normalized = self.normalize_vector(self.axis)
        self.min_limit = limits[0]
        self.max_limit = limits[1] 
        self.length = length
        
        # Estado durante algoritmo
        self.current_angle = 0.0
        self.constraint_plane_normal = None
        self.phi_angle = 0.0
        self.last_rotation_quaternion = None
        
    def normalize_vector(self, v):
        """Normaliza un vector de forma segura."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0, 0, 1])  # Default Z axis
        return v / norm
        
    def is_angle_within_limits(self, angle):
        """Verifica si un ángulo está dentro de los límites de la articulación."""
        return self.min_limit <= angle <= self.max_limit
        
    def clamp_angle(self, angle):
        """Restringe un ángulo a los límites de la articulación."""
        return np.clip(angle, self.min_limit, self.max_limit)
        
    def update_state(self, new_angle, constraint_normal=None, phi=0.0):
        """Actualiza el estado de la articulación."""
        self.current_angle = self.clamp_angle(new_angle)
        if constraint_normal is not None:
            self.constraint_plane_normal = self.normalize_vector(constraint_normal)
        self.phi_angle = phi
        
    def get_debug_info(self):
        """Retorna información de debugging de la articulación."""
        return {
            'idx': self.idx,
            'type': self.type.value if hasattr(self.type, 'value') else str(self.type),
            'axis': self.axis_normalized,
            'angle': self.current_angle,
            'angle_deg': np.degrees(self.current_angle),
            'limits_deg': (np.degrees(self.min_limit), np.degrees(self.max_limit)),
            'phi_deg': np.degrees(self.phi_angle),
            'length': self.length
        }

# ===============================================================================
# 1. ESTRUCTURAS DE DATOS Y DEFINICIONES (Simplificadas)
# ===============================================================================

class JointType(Enum):
    """Define los tipos de articulaciones soportadas."""
    REVOLUTE = "revolute"
    PIVOTE = "pivote"
    PRISMATIC = "prismatic"

class Joint:
    """Representa una articulación del robot con estado mejorado."""
    def __init__(self, joint_type, axis=np.array([0, 0, 1]), limits=(-np.pi, np.pi), length=0.0):
        self.type = joint_type
        self.axis = np.array(axis, dtype=float)
        self.axis_normalized = self._normalize_safe(self.axis)
        self.min_limit = limits[0]
        self.max_limit = limits[1]
        self.length = length
        
        # Estado de la articulación
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.constraint_satisfied = True
        
    def _normalize_safe(self, v):
        """Normalización segura de vectores."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.array([0, 0, 1])  # Eje Z por defecto
        return v / norm
        
    def get_rotation_matrix(self, angle=None):
        """Obtiene la matriz de rotación para el ángulo dado."""
        if angle is None:
            angle = self.current_angle
            
        # Usar scipy para rotaciones más precisas
        rotation = R.from_rotvec(angle * self.axis_normalized)
        return rotation.as_matrix()
        
    def apply_angle_limits(self, angle):
        """Aplica los límites de la articulación al ángulo."""
        return np.clip(angle, self.min_limit, self.max_limit)
        
    def get_debug_info(self):
        """Información de debugging de la articulación."""
        return {
            'type': self.type.value if hasattr(self.type, 'value') else str(self.type),
            'axis': self.axis_normalized.tolist(),
            'limits_deg': [np.degrees(self.min_limit), np.degrees(self.max_limit)],
            'current_angle_deg': np.degrees(self.current_angle),
            'length': self.length,
            'constraint_ok': self.constraint_satisfied
        }

class SerialChain:
    """Representa la cadena cinemática del robot con estado mejorado."""
    def __init__(self, source):
        if isinstance(source, Robot):
            # Inicializar desde un objeto Robot
            self.joints = self._from_robot_structure(source)
            self.name = source.name if hasattr(source, 'name') else "robot_from_yaml"
        elif isinstance(source, list):
            # Inicializar desde una lista de Joints
            self.joints = source
            self.name = "unnamed_robot"
        else:
            raise TypeError("La fuente para SerialChain debe ser una lista de Joints o un objeto Robot.")

        self.n_joints = len(self.joints)
        # Inicializar posiciones (n_joints + 1 para incluir efector final)
        self.positions = [np.zeros(3) for _ in range(self.n_joints + 1)]
        self.orientations = [np.identity(3) for _ in range(self.n_joints)]
        
        # Estado adicional para debugging
        self.joint_states = [JointState(i, joint.type, joint.axis_normalized, 
                                       (joint.min_limit, joint.max_limit), joint.length) 
                            for i, joint in enumerate(self.joints)]
        
        self.initialize_positions()

    def _from_robot_structure(self, robot_structure):
        """Convierte los Links de Robot a Joints para FABRIK, con mejor manejo de ejes."""
        new_joints = []
        print(f"Convirtiendo robot con {len(robot_structure.links)} enlaces...")
        
        for i, link in enumerate(robot_structure.links):
            if link.tipo.lower() == 'revolute':
                j_type = JointType.REVOLUTE
            elif link.tipo.lower() == 'prismatic':
                j_type = JointType.PRISMATIC
            else:
                j_type = JointType.REVOLUTE
            
            # Cargar ejes - mejor manejo de diferentes formatos
            if hasattr(link, 'axis') and link.axis is not None:
                if isinstance(link.axis, (list, tuple)):
                    axis = np.array(link.axis, dtype=float)
                elif isinstance(link.axis, np.ndarray):
                    axis = link.axis.astype(float)
                else:
                    axis = np.array([0, 0, 1])  # Default
            else:
                # Ejes por defecto basados en convenciones robóticas
                if i == 0:  # Primera articulación (base) - rotación Z
                    axis = np.array([0, 0, 1])
                elif i % 2 == 1:  # Articulaciones impares - rotación Y
                    axis = np.array([0, 1, 0])
                else:  # Articulaciones pares - rotación Y también
                    axis = np.array([0, 1, 0])
            
            # Normalizar eje
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axis = axis / norm
            else:
                axis = np.array([0, 0, 1])
            
            # Los límites pueden estar como 'joint_limits' (tupla) o 'limits' (dict)
            if hasattr(link, 'joint_limits'):
                # Formato del YAML: joint_limits: (-3.054, 3.054)
                limits = link.joint_limits
            elif hasattr(link, 'limits') and isinstance(link.limits, dict):
                # Formato dict: limits: {'min': -3.054, 'max': 3.054}
                limits = (link.limits['min'], link.limits['max'])
            else:
                # Valores por defecto más realistas
                if j_type == JointType.REVOLUTE:
                    limits = (-np.pi, np.pi)  # ±180°
                else:
                    limits = (-0.5, 0.5)  # ±50cm para prismáticas
            
            print(f"  Joint {i}: {j_type.value}, axis={axis}, limits={limits}, length={link.length}")
            new_joints.append(Joint(j_type, axis=axis, limits=limits, length=link.length))
        return new_joints

    def initialize_positions(self):
        """Coloca las articulaciones en una configuración inicial más natural."""
        # La base está en el origen
        self.positions[0] = np.zeros(3)
        
        # Configuración inicial: ligeramente curvada hacia arriba usando transformaciones
        cumulative_transform = np.identity(4)  # Matriz de transformación acumulada
        
        for i in range(self.n_joints):
            joint = self.joints[i]
            
            # Configuración inicial de ángulos para una postura natural
            if i == 0:  # Base
                initial_angle = 0.0
            elif i == 1:  # Segundo joint - levantar un poco
                initial_angle = np.pi/6  # 30°
            elif i == 2:  # Tercer joint - continuar curvatura
                initial_angle = np.pi/4  # 45°
            else:  # Joints restantes - curvatura gradual
                initial_angle = np.pi/8 * (1 - i/self.n_joints)  # Decreciente
            
            # Aplicar límites
            initial_angle = joint.apply_angle_limits(initial_angle)
            joint.current_angle = initial_angle
            
            # Calcular transformación de este joint
            rotation_matrix = joint.get_rotation_matrix(initial_angle)
            translation = np.array([0, 0, joint.length])  # Longitud en Z local
            
            # Crear matriz de transformación homogénea
            transform = np.identity(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = translation
            
            # Acumular transformación
            cumulative_transform = cumulative_transform @ transform
            
            # Extraer posición del siguiente punto
            self.positions[i + 1] = cumulative_transform[:3, 3]
            
            # Guardar orientación
            self.orientations[i] = cumulative_transform[:3, :3]

    def get_total_length(self):
        """Calcula la longitud total de la cadena."""
        return sum(j.length for j in self.joints)
        
    def get_debug_state(self):
        """Retorna el estado completo para debugging."""
        return {
            'name': self.name,
            'n_joints': self.n_joints,
            'total_length': self.get_total_length(),
            'positions': [pos.tolist() for pos in self.positions],
            'joint_states': [state.get_debug_info() for state in self.joint_states],
            'end_effector_pos': self.positions[-1].tolist()
        }

class Target:
    """Representa el objetivo que el robot debe alcanzar."""
    def __init__(self, position, orientation=None):
        self.position = np.array(position)
        self.orientation = np.array(orientation) if orientation is not None else np.identity(3)
        self.position_only = (orientation is None)

# ===============================================================================
# 2. IMPLEMENTACIÓN EXACTA DEL ALGORITMO FABRIK-R SEGÚN EL PAPER
# ===============================================================================

def normalize(v):
    """Normaliza un vector.""" 
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_position_error(p_final, p_target):
    """Calcula el error de posición."""
    return np.linalg.norm(p_final - p_target)

def find_concurrent(chain, joint_idx, v_init):
    """
    FIND_CONCURRENT mejorado del Algoritmo 2.
    Encuentra la siguiente articulación con dirección de actuación diferente.
    """
    if joint_idx >= len(chain.joints):
        return len(chain.joints) - 1
        
    current_axis = normalize(chain.joints[joint_idx].axis_normalized)
    
    for j in range(joint_idx + 1, len(chain.joints)):
        other_axis = normalize(chain.joints[j].axis_normalized)
        # Si los ejes no son paralelos (producto punto != ±1)
        dot_product = abs(np.dot(current_axis, other_axis))
        if dot_product < 0.95:  # Menos estricto para mejor detección
            return j
    
    # Si no se encuentra, retornar el último joint disponible
    return len(chain.joints) - 1

def generate_random(v_prev, v_j, constraint_plane_normal):
    """
    GENERATE_RANDOM mejorado del Algoritmo 2.
    Genera un vector aleatorio en el plano perpendicular al eje de la articulación.
    """
    # Asegurar que el eje esté normalizado
    axis_norm = normalize(constraint_plane_normal)
    
    # Usar una semilla basada en los vectores para reproducibilidad
    np.random.seed(int(1000 * (abs(np.sum(v_prev)) + abs(np.sum(axis_norm)))) % 2147483647)
    
    # Generar un vector aleatorio más controlado
    if abs(axis_norm[2]) < 0.9:  # No es principalmente Z
        base_vec = np.array([0, 0, 1])
    else:  # Es principalmente Z, usar X
        base_vec = np.array([1, 0, 0])
    
    # Proyectar en el plano perpendicular al eje
    random_proj = base_vec - np.dot(base_vec, axis_norm) * axis_norm
    
    # Añadir pequeña componente aleatoria
    noise = 0.1 * np.random.randn(3)
    random_proj += noise - np.dot(noise, axis_norm) * axis_norm
    
    # Normalizar el resultado
    return normalize(random_proj)

def rot_quaternions(l_vec, v_vec, angle):
    """
    ROT_QUATERNIONS mejorado del Algoritmo 1 con mejor manejo de casos límite.
    Rota v_vec alrededor del eje l_vec por el ángulo θ usando quaterniones.
    """
    if np.linalg.norm(l_vec) < 1e-8 or abs(angle) < 1e-8:
        return v_vec.copy()
    
    # Asegurar que los vectores son arrays numpy
    l_vec = np.array(l_vec, dtype=float)
    v_vec = np.array(v_vec, dtype=float)
    
    axis_norm = normalize(l_vec)
    
    # Usar scipy para rotación más robusta
    try:
        rotation = R.from_rotvec(angle * axis_norm)
        return rotation.apply(v_vec)
    except Exception as e:
        print(f"Warning: Rotation failed: {e}, returning original vector")
        return v_vec.copy()

def create_new_pi(phi_prev, p_prev_new, current_pos, link_length):
    """
    CREATE_NEW_Pi mejorado del Algoritmo 1.
    Crea una nueva posición para la articulación pi respetando las restricciones previas.
    """
    # Vector desde la nueva posición anterior hasta la posición actual
    link_vector = current_pos - p_prev_new
    link_distance = np.linalg.norm(link_vector)
    
    # Manejar el caso donde las posiciones son muy cercanas
    if link_distance < 1e-8:
        # Usar dirección por defecto hacia arriba
        direction = np.array([0, 0, 1])
    else:
        direction = link_vector / link_distance
    
    # Calcular nueva posición manteniendo la longitud del eslabón
    p_hat_i = p_prev_new + direction * link_length
    
    # El vector de dirección de la articulación
    v_hat_i = direction
    
    return p_hat_i, v_hat_i

def define_phi_i(chain, joint_idx, v_prev, p_prev_new):
    """
    DEFINE_Φi mejorado del Algoritmo 2 del paper.
    Define el plano de restricción Φi para la articulación con mejor manejo de casos límite.
    """
    if joint_idx >= len(chain.joints):
        return 0.0
    
    joint = chain.joints[joint_idx]
    
    # Paso 1: j = FIND_CONCURRENT(i, vinit)
    j = find_concurrent(chain, joint_idx, v_prev)
    
    # Paso 2: (α, β, γ) = pprev - pj
    if j < len(chain.positions):
        pos_diff = p_prev_new - chain.positions[j]
    else:
        pos_diff = p_prev_new - chain.positions[-1]
    
    alpha, beta, gamma = pos_diff
    
    # Verificar si la diferencia de posición es significativa
    if np.linalg.norm(pos_diff) < 1e-6:
        return 0.0  # No hay restricción significativa
    
    # Paso 3: v⃗ = GENERATE_RANDOM(vprev, vj)
    if j < len(chain.joints):
        v_j = chain.joints[j].axis_normalized
    else:
        v_j = np.array([0, 0, 1])
    
    v_random = generate_random(v_prev, v_j, joint.axis_normalized)
    
    # Paso 4: t⃗ = l⃗ × v⃗
    l_vec = normalize(joint.axis_normalized)
    t_vec = np.cross(l_vec, v_random)
    
    # Verificar que el producto cruz no sea cero
    if np.linalg.norm(t_vec) < 1e-6:
        return 0.0  # Vectores paralelos, no hay restricción
    
    t_vec = normalize(t_vec)
    
    # Pasos 5-7: Calcular constantes K con mejor precisión numérica
    v1, v2, v3 = v_random
    l1, l2, l3 = l_vec
    t1, t2, t3 = t_vec
    
    K1 = alpha * v1 + beta * v2 + gamma * v3
    K2 = np.dot(l_vec, v_random) * (alpha * l1 + beta * l2 + gamma * l3)  
    K3 = alpha * t1 + beta * t2 + gamma * t3
    
    # Paso 8: Resolver ecuación trigonométrica con mejor estabilidad numérica
    # cos(2θ)(K1 - K2) + K2 + sin(2θ)K3 = 0
    # Forma estándar: A*cos(2θ) + B*sin(2θ) + C = 0
    A = K1 - K2
    B = K3  
    C = K2
    
    # Verificar si los coeficientes son significativos
    if abs(A) < 1e-8 and abs(B) < 1e-8:
        return 0.0  # Ecuación trivial
    
    # Resolver la ecuación trigonométrica con mejor robustez
    if abs(B) > 1e-8:
        # Usar la fórmula estándar para A*cos(x) + B*sin(x) + C = 0
        R_mag = np.sqrt(A*A + B*B)
        if R_mag > abs(C) + 1e-8:  # Margen para errores numéricos
            # Existe solución real
            phi = np.arctan2(B, A)
            cos_arg = np.clip(-C / R_mag, -1.0, 1.0)  # Asegurar rango válido
            
            two_theta_candidates = [
                np.arccos(cos_arg) - phi,
                -np.arccos(cos_arg) - phi
            ]
            
            # Elegir la solución que minimiza el ángulo y respeta límites de articulación
            best_theta = 0.0
            min_cost = float('inf')
            
            for two_theta in two_theta_candidates:
                theta = two_theta / 2
                
                # Normalizar ángulo a [-π, π]
                theta = np.arctan2(np.sin(theta), np.cos(theta))
                
                # Verificar límites de articulación
                if joint.min_limit <= theta <= joint.max_limit:
                    cost = abs(theta)  # Preferir ángulos menores
                    if cost < min_cost:
                        min_cost = cost
                        best_theta = theta
            
            theta = best_theta
        else:
            # No hay solución real, usar aproximación
            theta = 0.0
    else:
        # Si B ≈ 0, la ecuación se simplifica a A*cos(2θ) + C = 0
        if abs(A) > 1e-8:
            cos_2theta = np.clip(-C / A, -1.0, 1.0)
            theta = np.arccos(cos_2theta) / 2
            
            # Normalizar y verificar límites
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            theta = np.clip(theta, joint.min_limit, joint.max_limit)
        else:
            theta = 0.0
    
    # Actualizar estado de la articulación
    if joint_idx < len(chain.joint_states):
        chain.joint_states[joint_idx].update_state(theta, constraint_normal=l_vec, phi=theta)
    
    return theta

def fabrik_r_single_joint(chain, joint_idx, p_prev_new, phi_prev):
    """
    Implementa el Algoritmo 1: FABRIK-R para una sola articulación.
    
    Args:
        chain: Cadena cinemática
        joint_idx: Índice de la articulación a procesar
        p_prev_new: Nueva posición de la articulación anterior
        phi_prev: Plano de restricción de la articulación anterior
    
    Returns:
        p_next_new: Nueva posición de la articulación siguiente
    """
    if joint_idx >= len(chain.positions) or joint_idx >= len(chain.joints):
        return chain.positions[joint_idx] if joint_idx < len(chain.positions) else chain.positions[-1]
    
    # Paso 1: DEFINE_Φprev() - ya tenemos phi_prev como parámetro
    
    # Paso 2: [p̂i, v̂i] = CREATE_NEW_Pi(Φprev, p'prev)
    if joint_idx > 0:
        link_length = chain.joints[joint_idx - 1].length
    else:
        link_length = 1.0  # Valor por defecto para la base
    
    p_hat_i, v_hat_i = create_new_pi(phi_prev, p_prev_new, chain.positions[joint_idx], link_length)
    
    # Paso 3: θ = DEFINE_Φi(vprev, p'prev)
    theta = define_phi_i(chain, joint_idx, v_hat_i, p_prev_new)
    
    # Paso 4: p'next = ROT_QUATERNIONS(vprev, p̂i, v̂i, θ)
    if joint_idx < len(chain.joints):
        joint_axis = normalize(chain.joints[joint_idx].axis)
    else:
        joint_axis = np.array([0, 0, 1])
    
    # Rotar p̂i alrededor del eje de la articulación
    rotated_position = rot_quaternions(joint_axis, p_hat_i - p_prev_new, theta)
    p_next_new = p_prev_new + rotated_position
    
    return p_next_new


# ===============================================================================
# 3. ALGORITMO FABRIK-R PRINCIPAL
# ===============================================================================

def fabrik_r(chain, target, tolerance=1e-3, max_iterations=100, debug=True):
    """
    Implementación mejorada y simplificada del algoritmo FABRIK-R.
    
    Esta versión se enfoca en la funcionalidad core del algoritmo con mejor debugging
    y manejo de articulaciones individuales.
    """
    
    if debug:
        print(f"\n=== FABRIK-R INICIANDO ===")
        print(f"Robot: {chain.name}")
        print(f"Articulaciones: {chain.n_joints}")
        print(f"Longitud total: {chain.get_total_length():.3f}m")
        print(f"Objetivo: {target.position}")
    
    # 1. Verificar alcanzabilidad del objetivo
    base_position = chain.positions[0].copy()
    dist_to_target = np.linalg.norm(target.position - base_position)
    total_length = chain.get_total_length()

    if dist_to_target > total_length * 0.95:  # 95% del alcance máximo
        if debug:
            print(f"Objetivo muy lejano ({dist_to_target:.3f}m > {total_length*0.95:.3f}m)")
            print("Estirando el robot hacia el objetivo...")
        
        # Estirar la cadena hacia el objetivo manteniendo proporciones
        direction_to_target = normalize(target.position - base_position)
        cumulative_length = 0
        
        for i in range(len(chain.joints)):
            cumulative_length += chain.joints[i].length
            chain.positions[i+1] = base_position + direction_to_target * cumulative_length
            
        return chain.positions

    # 2. Algoritmo FABRIK-R simplificado pero robusto
    iterations = 0
    error = calculate_position_error(chain.positions[-1], target.position)
    initial_error = error
    
    if debug:
        print(f"Error inicial: {error:.4f}m")
        print("Iniciando iteraciones...")

    while error > tolerance and iterations < max_iterations:
        
        # FASE 1: FORWARD REACHING (Efector final → Base)
        # Colocar el efector final en el objetivo
        chain.positions[-1] = target.position.copy()
        
        # Procesar cada articulación hacia atrás
        for i in range(len(chain.positions) - 2, 0, -1):
            if i-1 >= 0 and i-1 < len(chain.joints):
                # Calcular nueva posición manteniendo longitud del eslabón
                link_length = chain.joints[i-1].length
                link_vector = chain.positions[i] - chain.positions[i+1]
                link_direction = normalize(link_vector)
                
                # Aplicar restricciones de la articulación
                joint = chain.joints[i-1]
                
                # Calcular ángulo requerido
                if i > 1:
                    prev_link = normalize(chain.positions[i-1] - chain.positions[i])
                    required_angle = np.arccos(np.clip(np.dot(prev_link, link_direction), -1, 1))
                    
                    # Verificar si el ángulo está dentro de los límites
                    if not (joint.min_limit <= required_angle <= joint.max_limit):
                        # Ajustar ángulo a los límites
                        clamped_angle = np.clip(required_angle, joint.min_limit, joint.max_limit)
                        
                        # Calcular nueva dirección usando el eje de rotación
                        axis = joint.axis_normalized
                        rotation = R.from_rotvec(clamped_angle * axis)
                        link_direction = rotation.apply(prev_link)
                
                # Actualizar posición
                chain.positions[i] = chain.positions[i+1] + link_direction * link_length
                
                # Actualizar estado de la articulación
                if i-1 < len(chain.joint_states):
                    chain.joint_states[i-1].current_angle = required_angle if 'required_angle' in locals() else 0.0

        # FASE 2: BACKWARD REACHING (Base → Efector final)
        # Fijar la base en su posición original
        chain.positions[0] = base_position.copy()
        
        # Procesar cada articulación hacia adelante
        for i in range(1, len(chain.positions)):
            if i-1 < len(chain.joints):
                # Calcular nueva posición manteniendo longitud del eslabón
                link_length = chain.joints[i-1].length
                link_vector = chain.positions[i] - chain.positions[i-1]
                link_direction = normalize(link_vector)
                
                # Aplicar restricciones de la articulación
                joint = chain.joints[i-1]
                
                # Calcular ángulo actual
                if i > 1:
                    prev_link = normalize(chain.positions[i-1] - chain.positions[i-2]) if i > 1 else np.array([0, 0, 1])
                    current_angle = np.arccos(np.clip(np.dot(prev_link, link_direction), -1, 1))
                    
                    # Verificar límites y ajustar si es necesario
                    if not (joint.min_limit <= current_angle <= joint.max_limit):
                        clamped_angle = np.clip(current_angle, joint.min_limit, joint.max_limit)
                        
                        # Calcular nueva dirección
                        axis = joint.axis_normalized
                        rotation = R.from_rotvec(clamped_angle * axis)
                        link_direction = rotation.apply(prev_link)
                
                # Actualizar posición
                chain.positions[i] = chain.positions[i-1] + link_direction * link_length
                
                # Actualizar estado de la articulación
                if i-1 < len(chain.joint_states):
                    chain.joint_states[i-1].current_angle = current_angle if 'current_angle' in locals() else 0.0

        # Calcular nuevo error
        error = calculate_position_error(chain.positions[-1], target.position)
        iterations += 1
        
        if debug and iterations % 10 == 0:
            print(f"Iteración {iterations}: Error = {error:.6f}m")
            
        # Verificar convergencia por cambio mínimo
        if iterations > 5 and error > initial_error * 0.99:
            if debug:
                print("Convergencia estancada, terminando...")
            break

    if debug:
        print(f"\nFABRIK-R completado:")
        print(f"  Iteraciones: {iterations}")
        print(f"  Error final: {error:.6f}m")
        print(f"  Mejora: {((initial_error - error) / initial_error * 100):.1f}%")
        
        # Estado de las articulaciones
        print(f"  Estado de articulaciones:")
        for i, state in enumerate(chain.joint_states):
            info = state.get_debug_info()
            print(f"    Joint {i}: {info['angle_deg']:.1f}° (límites: {info['limits_deg'][0]:.1f}° to {info['limits_deg'][1]:.1f}°)")
    
    return chain.positions


# ===============================================================================
# 4. VISUALIZADOR INTERACTIVO 3D CON GRABACIÓN
# ===============================================================================

class FABRIK3DVisualizer:
    """
    Visualizador interactivo 3D para el algoritmo FABRIK con capacidades de grabación.
    
    Proporciona una interfaz visual en tiempo real donde el usuario puede mover el objetivo
    del robot usando el teclado y ver cómo el algoritmo FABRIK resuelve la cinemática inversa.
    También incluye un sistema de grabación completo para capturar animaciones.
    """
    
    def __init__(self, robot_chain, robot_name="robot", initial_target=None):
        """
        Inicializa el visualizador.
        
        Args:
            robot_chain (SerialChain): La cadena cinemática del robot
            robot_name (str): Nombre del robot para identificación
            initial_target (np.ndarray, optional): Posición inicial del objetivo
        """
        self.chain = robot_chain
        self.name = robot_name
        self.base_point = self.chain.positions[0].copy()
        
        # Configuración del objetivo
        if initial_target is not None:
            self.target = np.array(initial_target)
        else:
            # Objetivo por defecto: cerca del final del robot
            total_length = self.chain.get_total_length()
            self.target = np.array([total_length * 0.7, total_length * 0.3, total_length * 0.5])
        
        self.initial_target = self.target.copy()
        
        # Configuración de movimiento
        self.limbs_len = self.chain.get_total_length()
        self.target_step = self.limbs_len * 0.05  # 5% de la longitud total
        
        # Sistema de grabación (si está disponible) - DEBE ir antes de setup_plot()
        if RecordingSystem is not None:
            self.recorder = RecordingSystem(fps=20, dpi=300, max_buffer_seconds=10)
        else:
            self.recorder = None
        
        # Configuración de visualización - DESPUÉS de inicializar recorder
        self.setup_plot()
        
        # Contador de frames para la grabación
        self.frame_count = 0
        
    def setup_plot(self):
        """Configura la ventana de matplotlib y los elementos visuales."""
        # Configuración obligatoria de matplotlib para deshabilitar keymaps
        mpl.rcParams['keymap.save'] = []        # Deshabilitar 's' para save
        mpl.rcParams['keymap.quit'] = []        # Deshabilitar 'q' para quit
        mpl.rcParams['keymap.pan'] = []         # Deshabilitar 'p' para pan
        mpl.rcParams['keymap.zoom'] = []        # Deshabilitar 'o' para zoom
        mpl.rcParams['keymap.home'] = []        # Deshabilitar 'h' y 'r' para home/reset vista
        mpl.rcParams['keymap.back'] = []        # Deshabilitar navegación
        mpl.rcParams['keymap.forward'] = []     # Deshabilitar navegación
        mpl.rcParams['keymap.grid'] = []        # Deshabilitar 'g' para grid
        mpl.rcParams['keymap.yscale'] = []      # Deshabilitar 'l' para log scale
        mpl.rcParams['keymap.xscale'] = []      # Deshabilitar 'k' para log scale
        
        # Crear figura y eje 3D
        self.fig, self.ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        self.fig.suptitle(f'FABRIK-R Interactivo - {self.name}', fontsize=16)
        
        # Configurar límites del plot
        plot_range = self.limbs_len * 1.2
        self.ax.set_xlim([-plot_range, plot_range])
        self.ax.set_ylim([-plot_range, plot_range])
        self.ax.set_zlim([0, plot_range * 1.5])
        
        # Etiquetas y configuración
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.grid(True, alpha=0.3)
        
        self.fig.canvas.mpl_connect('key_press_event', 
                                    lambda event: self.recorder.on_key_press(event, self) if self.recorder else None)


        # Elementos gráficos (se inicializarán en el primer frame)
        self.robot_line = None
        self.joints_scatter = None
        self.target_scatter = None
        self.base_scatter = None
        self.text_info = None
    
    def update_robot(self):
        """Actualiza la posición del robot usando FABRIK-R."""
        target_obj = Target(position=self.target)
        
        # Ejecutar FABRIK-R simplificado para tiempo real
        base_position = self.chain.positions[0].copy()
        dist_to_target = np.linalg.norm(target_obj.position - base_position)
        total_length = self.chain.get_total_length()

        if dist_to_target > total_length:
            # Estirar la cadena hacia el objetivo
            direction_to_target = normalize(target_obj.position - base_position)
            cumulative_length = 0
            
            for i in range(len(self.chain.positions) - 1):
                cumulative_length += self.chain.joints[i].length
                self.chain.positions[i+1] = base_position + direction_to_target * cumulative_length
        else:
            # Bucle FABRIK-R simplificado para tiempo real
            tolerance = 1e-3
            max_iterations = 5  # Reducido para tiempo real
            iterations = 0
            error = calculate_position_error(self.chain.positions[-1], target_obj.position)

            while error > tolerance and iterations < max_iterations:
                # Fase 1: Forward reaching (hacia atrás)
                self.chain.positions[-1] = target_obj.position.copy()
                
                for i in range(len(self.chain.positions) - 2, 0, -1):
                    # Aplicar FABRIK-R básico
                    if i < len(self.chain.joints):
                        phi_next = self.chain.joints[i].axis if i < len(self.chain.joints) else np.array([0, 0, 1])
                    else:
                        phi_next = np.array([0, 0, 1])
                    
                    # Calcular nueva posición manteniendo longitud
                    if i > 0 and i-1 < len(self.chain.joints):
                        link_length = self.chain.joints[i-1].length
                        direction = normalize(self.chain.positions[i] - self.chain.positions[i+1])
                        self.chain.positions[i] = self.chain.positions[i+1] + direction * link_length

                # Fase 2: Backward reaching (hacia adelante)
                self.chain.positions[0] = base_position
                
                for i in range(1, len(self.chain.positions)):
                    # Aplicar FABRIK-R básico
                    if i > 0 and i-1 < len(self.chain.joints):
                        link_length = self.chain.joints[i-1].length
                        direction = normalize(self.chain.positions[i] - self.chain.positions[i-1])
                        self.chain.positions[i] = self.chain.positions[i-1] + direction * link_length

                error = calculate_position_error(self.chain.positions[-1], target_obj.position)
                iterations += 1
    
    def animate(self, frame):
        """
        Función de animación llamada por matplotlib.
        
        Args:
            frame: Número del frame actual
        """
        # Actualizar robot
        self.update_robot()
        
        # Capturar frame para grabación si está disponible
        if self.recorder:
            self.recorder.capture_frame(
                joints=self.chain.positions,
                target=self.target,
                timestamp=frame
            )
        
        # Limpiar el eje y redibujar
        self.ax.clear()
        
        # Reconfigurar límites y etiquetas
        plot_range = self.limbs_len * 1.2
        self.ax.set_xlim([-plot_range, plot_range])
        self.ax.set_ylim([-plot_range, plot_range])
        self.ax.set_zlim([0, plot_range * 1.5])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.grid(True, alpha=0.3)
        
        # Dibujar robot
        points = np.array(self.chain.positions)
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o-', 
                    color='blue', lw=3, markersize=8, markerfacecolor='red', 
                    markeredgecolor='darkred', markeredgewidth=1, alpha=0.8)
        
        # Dibujar objetivo
        self.ax.plot([self.target[0]], [self.target[1]], [self.target[2]], 
                    'o', color='green', markersize=12, alpha=0.8, 
                    markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Dibujar base
        self.ax.plot([self.base_point[0]], [self.base_point[1]], [self.base_point[2]], 
                    'o', color='black', markersize=10, markerfacecolor='yellow',
                    markeredgecolor='black', markeredgewidth=2)
        
        # Información del estado
        error = np.linalg.norm(self.chain.positions[-1] - self.target)
        info_text = f'Error: {error:.4f}m\nTarget: [{self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f}]'
        
        # Agregar información de las articulaciones y sus límites
        info_text += '\n\nArticulaciones:'
        for i, joint in enumerate(self.chain.joints):
            joint_name = f'J{i+1}'
            min_deg = np.degrees(joint.min_limit)
            max_deg = np.degrees(joint.max_limit)
            
            # Calcular ángulo actual de la articulación
            if i < len(self.chain.positions) - 1:
                # Vector del eslabón anterior
                if i > 0:
                    v_prev = self.chain.positions[i] - self.chain.positions[i-1]
                else:
                    v_prev = np.array([0, 0, 1])  # Referencia para la primera articulación
            
                # Vector del eslabón actual
                if i < len(self.chain.positions) - 1:
                    v_curr = self.chain.positions[i+1] - self.chain.positions[i]
                    
                    # Calcular ángulo entre vectores
                    v_prev_norm = normalize(v_prev)
                    v_curr_norm = normalize(v_curr)
                    dot_product = np.clip(np.dot(v_prev_norm, v_curr_norm), -1.0, 1.0)
                    current_angle_rad = np.arccos(dot_product)
                    current_angle_deg = np.degrees(current_angle_rad)
                else:
                    current_angle_deg = 0.0
            else:
                current_angle_deg = 0.0

            info_text += f'\n{joint_name}: {current_angle_deg:4.1f}° ({min_deg:4.0f}°, {max_deg:4.0f}°)'

        # Agregar información de grabación si está disponible
        if self.recorder:
            status = self.recorder.get_recording_status()
            if status['state'] != 'stopped':
                estado_rec = "🔴 REC" if status['state'] == 'recording' else "⏸️ PAUSA"
                info_text += f'\n{estado_rec} {status["duration"]:.1f}s'
            info_text += f'\nBuffer: {status["buffer_duration"]:.1f}s'
        
        # Mostrar información
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Mostrar controles
        if frame == 0 or frame % 100 == 0:  # Mostrar cada 100 frames
            self.ax.text2D(0.98, 0.02, 'Presiona H para ayuda', transform=self.ax.transAxes, 
                          fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.frame_count += 1
        return []
    
    def start_visualization(self):
        """Inicia la visualización interactiva."""
        print(f"\n{'='*60}")
        print(f"S-FABRIK 3D INTERACTIVO - {self.name}")
        if self.recorder:
            self.recorder.print_help()
        else:
            print("Sistema de grabación no disponible.")
        
        # Configurar animación
        self.anim = FuncAnimation(
            self.fig, 
            self.animate, 
            interval=50,  # ~20 FPS
            blit=False, 
            cache_frame_data=False,
            repeat=True
        )
        
        # Mostrar ventana
        plt.show()
        
        return self.anim

# ===============================================================================
# 5. EJEMPLO DE USO
# ===============================================================================

def run_static_demo(robot_chain, robot_name):
    """Ejecuta la demostración estática con debugging mejorado."""
    print("=== MODO DEMOSTRACIÓN ESTÁTICA MEJORADA ===")
    
    # Mostrar estado inicial del robot
    print(f"\nRobot: {robot_name}")
    print(f"Articulaciones: {robot_chain.n_joints}")
    print(f"Longitud total: {robot_chain.get_total_length():.3f}m")
    
    print("\nEstado inicial de las articulaciones:")
    for i, pos in enumerate(robot_chain.positions):
        print(f"  Posición {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\nConfiguración de articulaciones:")
    for i, joint in enumerate(robot_chain.joints):
        info = joint.get_debug_info()
        print(f"  Joint {i}: {info['type']}, eje={info['axis']}, límites={info['limits_deg']}°")

    # Definir varios objetivos para probar
    total_length = robot_chain.get_total_length()
    test_targets = [
        [total_length * 0.4, 0.0, total_length * 0.3],           # Adelante
        [0.0, total_length * 0.4, total_length * 0.3],           # A la derecha
        [total_length * 0.3, total_length * 0.3, total_length * 0.5], # Diagonal
        [total_length * 0.6, 0.1, 0.1],                          # Cerca del suelo
    ]
    
    for test_idx, target_pos in enumerate(test_targets):
        print(f"\n{'='*60}")
        print(f"PRUEBA {test_idx + 1}: Objetivo en {target_pos}")
        print("="*60)
        
        robot_target = Target(position=target_pos)
        
        # Ejecutar FABRIK-R con debugging
        final_positions = fabrik_r(robot_chain, robot_target, debug=True)

        # Verificar la preservación de longitudes de eslabones
        print(f"\nVerificación de integridad:")
        max_length_error = 0.0
        for i in range(len(final_positions) - 1):
            actual_length = np.linalg.norm(final_positions[i+1] - final_positions[i])
            expected_length = robot_chain.joints[i].length
            length_error = abs(actual_length - expected_length)
            max_length_error = max(max_length_error, length_error)
            
            if length_error > 0.001:  # 1mm tolerancia
                print(f"  ⚠️  Eslabón {i}: {actual_length:.4f}m (esperado: {expected_length:.4f}m)")
            else:
                print(f"  ✓  Eslabón {i}: {actual_length:.4f}m")
        
        # Verificar error final
        final_error = np.linalg.norm(final_positions[-1] - robot_target.position)
        print(f"\nResultado final:")
        print(f"  Error de posición: {final_error:.4f}m")
        print(f"  Max error de eslabón: {max_length_error:.4f}m")
        
        if final_error < 0.01:  # 1cm tolerancia
            print(f"  ✓ Objetivo alcanzado exitosamente")
        else:
            print(f"  ⚠️  Objetivo no alcanzado completamente")
        
        # Pequeña pausa entre pruebas
        import time
        time.sleep(1)

    print(f"\n{'='*60}")
    print("DEMOSTRACIÓN COMPLETA")
    print("="*60)

def run_interactive_demo(robot_chain, robot_name):
    """Ejecuta la demostración interactiva con visualización 3D."""
    print("=== MODO INTERACTIVO 3D ===")
    
    # Objetivo inicial razonable para el robot
    total_length = robot_chain.get_total_length()
    initial_target = [total_length * 0.6, total_length * 0.2, total_length * 0.4]
    
    # Crear visualizador
    visualizer = FABRIK3DVisualizer(robot_chain, robot_name, initial_target)
    
    # Iniciar visualización interactiva
    animation = visualizer.start_visualization()
    
    return animation

if __name__ == '__main__':
    print("=== INICIO: FABRIK-R ALGORITHM IMPLEMENTATION ===")
    print("Ejecutando FABRIK-R con robot cargado desde YAML...")

    # 1. Cargar un robot desde un archivo de configuración
    robot_config_path = 'config/robot-niryo.yaml'
    print(f"\nCargando robot desde: '{robot_config_path}'")
    
    try:
        robot_structure = cargar_robot_desde_yaml(robot_config_path)
        print("Robot cargado exitosamente desde YAML")
    except Exception as e:
        print(f"Error al cargar robot: {e}")
        print("Creando robot simple de ejemplo...")
        
        # Crear un robot simple de 3 articulaciones para demostración
        joints = [
            Joint(JointType.REVOLUTE, axis=np.array([0, 0, 1]), limits=(-np.pi, np.pi), length=0.1),
            Joint(JointType.REVOLUTE, axis=np.array([0, 1, 0]), limits=(-np.pi/2, np.pi/2), length=0.08),
            Joint(JointType.REVOLUTE, axis=np.array([0, 1, 0]), limits=(-np.pi/2, np.pi/2), length=0.06)
        ]
        robot_chain = SerialChain(joints)
        robot_name = "simple_robot"
    else:
        # 2. Crear la cadena cinemática para FABRIK-R
        robot_chain = SerialChain(robot_structure)
        robot_name = robot_structure.name
    
    # 3. Preguntar al usuario qué modo prefiere
    print(f"\nRobot '{robot_name}' cargado exitosamente.")
    print(f"Número de articulaciones: {robot_chain.n_joints}")
    print(f"Longitud total: {robot_chain.get_total_length():.3f} m")
    
    mode = input("\n¿Qué modo prefieres? \n  [1] Demostración estática (por defecto)\n  [2] Visualización interactiva 3D\nSelección (1-2): ").strip()
    
    if mode == '2':
        try:
            # Modo interactivo
            print("Iniciando modo interactivo...")
            animation = run_interactive_demo(robot_chain, robot_name)
        except KeyboardInterrupt:
            print("\n\nVisualización interrumpida por el usuario.")
        except Exception as e:
            print(f"\nError en la visualización: {e}")
            print("Ejecutando modo estático como alternativa...")
            run_static_demo(robot_chain, robot_name)

    else:
        # Modo estático (por defecto)
        print("Iniciando modo estático...")
        run_static_demo(robot_chain, robot_name)
    
    print("\n=== FIN: FABRIK-R ALGORITHM IMPLEMENTATION ===")
