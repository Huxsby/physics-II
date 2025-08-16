"""
Módulo de grabación para el sistema FABRIK 3D.

Este módulo maneja la captura, grabación y exportación de animaciones
del sistema FABRIK de forma independiente a la visualización principal.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from animation import guardar_animacion


class RecordingSystem:
    """
    Sistema de grabación avanzado para animaciones FABRIK 3D.
    
    Maneja la captura de frames, grabación en tiempo real y exportación
    de animaciones con buffer circular para recap de últimos segundos.
    También incluye el manejo de controles de teclado para la interfaz.
    """
    
    def __init__(self, fps=20, dpi=300, max_buffer_seconds=10):
        """
        Inicializa el sistema de grabación.
        
        Args:
            fps (int): Frames por segundo para la grabación
            dpi (int): DPI para la exportación de animaciones
            max_buffer_seconds (int): Segundos máximos en el buffer circular
        """
        # Estado de grabación
        self.recording_state = 'stopped'  # 'stopped', 'recording', 'paused'
        self.recording_frames = []  # Buffer para frames de grabación
        self.frame_buffer = []  # Buffer circular para últimos N segundos
        self.current_frame_count = 0
        
        # Configuración de grabación
        self.recording_fps = fps
        self.recording_dpi = dpi
        self.max_buffer_size = fps * max_buffer_seconds  # N segundos a X fps
        
    def start_recording(self):
        """Inicia la grabación de la animación."""
        if self.recording_state == 'recording':
            print("ADVERTENCIA: Ya se está grabando")
            return
        
        self.recording_state = 'recording'
        self.recording_frames = []
        self.current_frame_count = 0
        print("GRABACION INICIADA")
        print("   Presiona P para pausar, X para parar")
    
    def pause_recording(self):
        """Pausa o reanuda la grabación."""
        if self.recording_state == 'recording':
            self.recording_state = 'paused'
            print("GRABACION PAUSADA")
            print("   Presiona P para reanudar")
        elif self.recording_state == 'paused':
            self.recording_state = 'recording'
            print("GRABACION REANUDADA")
        else:
            print("ADVERTENCIA: No hay grabación activa para pausar")
    
    def stop_recording(self, robot_name="default", base_point=None, ax=None):
        """
        Detiene la grabación y guarda el archivo.
        
        Args:
            robot_name (str): Nombre del robot para el archivo
            base_point (np.ndarray, optional): Punto base del robot para visualización
            ax (matplotlib.axes.Axes, optional): Eje 3D para obtener límites automáticamente
        """
        if self.recording_state == 'stopped':
            print("ADVERTENCIA: No hay grabación activa")
            return
        
        if len(self.recording_frames) == 0:
            print("ADVERTENCIA: No hay frames grabados")
            self.recording_state = 'stopped'
            return
        
        self.recording_state = 'stopped'
        frame_count = len(self.recording_frames)
        duration = frame_count / self.recording_fps
        
        print(f"GRABACION DETENIDA")
        print(f"   Frames capturados: {frame_count}")
        print(f"   Duración: {duration:.1f} segundos")
        
        # Extraer límites del eje si se proporciona
        ax_limits = None
        if ax is not None and hasattr(ax, 'get_xlim'):
            ax_limits = {
                'x': ax.get_xlim(),
                'y': ax.get_ylim(),
                'z': ax.get_zlim()
            }
        
        # Crear animación temporal con los frames grabados
        self._save_recorded_frames("recording", robot_name, base_point, ax_limits)
    
    def capture_recap(self, robot_name="default", base_point=None, ax=None):
        """
        Captura los últimos N segundos del buffer.
        
        Args:
            robot_name (str): Nombre del robot para el archivo
            base_point (np.ndarray, optional): Punto base del robot para visualización
            ax (matplotlib.axes.Axes, optional): Eje 3D para obtener límites automáticamente
        """
        if len(self.frame_buffer) == 0:
            print("ADVERTENCIA: No hay frames en el buffer para recap")
            return
        
        frame_count = len(self.frame_buffer)
        duration = frame_count / self.recording_fps
        
        print(f"CAPTURANDO RECAP")
        print(f"   Frames disponibles: {frame_count}")
        print(f"   Duración: {duration:.1f} segundos")
        
        # Extraer límites del eje si se proporciona
        ax_limits = None
        if ax is not None and hasattr(ax, 'get_xlim'):
            ax_limits = {
                'x': ax.get_xlim(),
                'y': ax.get_ylim(),
                'z': ax.get_zlim()
            }
        
        # Usar todos los frames del buffer
        self._save_buffer_frames("recap", robot_name, base_point, ax_limits)
    
    def capture_frame(self, joints, target, timestamp):
        """
        Captura un frame del estado actual del robot.
        
        Args:
            joints (list): Lista de posiciones de articulaciones
            target (np.ndarray): Posición del target
            timestamp (int): Timestamp del frame
        """
        current_frame = {
            'joints': [joint.copy() for joint in joints],
            'target': target.copy(),
            'timestamp': timestamp
        }
        
        # Agregar al buffer circular (últimos N segundos)
        self.frame_buffer.append(current_frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)  # Eliminar frame más antiguo
        
        # Si estamos grabando, agregar al buffer de grabación
        if self.recording_state == 'recording':
            self.recording_frames.append(current_frame.copy())
            self.current_frame_count += 1
    
    def get_recording_status(self):
        """
        Obtiene el estado actual de la grabación.
        
        Returns:
            dict: Información del estado de grabación
        """
        return {
            'state': self.recording_state,
            'frames_recorded': len(self.recording_frames),
            'duration': len(self.recording_frames) / self.recording_fps if len(self.recording_frames) > 0 else 0,
            'buffer_frames': len(self.frame_buffer),
            'buffer_duration': len(self.frame_buffer) / self.recording_fps if len(self.frame_buffer) > 0 else 0
        }
    
    def _save_recorded_frames(self, prefix, robot_name, base_point=None, ax_limits=None):
        """
        Guarda los frames grabados como animación.
        
        Args:
            prefix (str): Prefijo para el nombre del archivo
            robot_name (str): Nombre del robot
            base_point (np.ndarray, optional): Punto base del robot
            ax_limits (dict, optional): Límites de los ejes para el plot
        """
        if len(self.recording_frames) == 0:
            return
        
        if base_point is None:
            base_point = np.array([0.0, 0.0, 0.0])
        
        # Crear una animación temporal con frames específicos
        temp_fig, temp_ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        
        # Configurar límites si se proporcionan
        if ax_limits:
            temp_ax.set_xlim(ax_limits['x'])
            temp_ax.set_ylim(ax_limits['y'])
            temp_ax.set_zlim(ax_limits['z'])
        else:
            # Límites por defecto basados en los datos
            all_points = []
            for frame in self.recording_frames:
                all_points.extend(frame['joints'])
                all_points.append(frame['target'])
            
            if all_points:
                points_array = np.array(all_points)
                margin = 50
                temp_ax.set_xlim(points_array[:, 0].min() - margin, points_array[:, 0].max() + margin)
                temp_ax.set_ylim(points_array[:, 1].min() - margin, points_array[:, 1].max() + margin)
                temp_ax.set_zlim(points_array[:, 2].min() - margin, points_array[:, 2].max() + margin)
        
        temp_ax.set_xlabel('X')
        temp_ax.set_ylabel('Y')
        temp_ax.set_zlabel('Z')
        temp_ax.set_title('FABRIK 3D - Animación Grabada')
        
        def animate_recorded(frame_idx):
            temp_ax.clear()
            if ax_limits:
                temp_ax.set_xlim(ax_limits['x'])
                temp_ax.set_ylim(ax_limits['y'])
                temp_ax.set_zlim(ax_limits['z'])
            temp_ax.set_xlabel('X')
            temp_ax.set_ylabel('Y')
            temp_ax.set_zlabel('Z')
            temp_ax.set_title('FABRIK 3D - Animación Grabada')
            
            # Obtener el frame grabado
            frame_data = self.recording_frames[frame_idx]
            joints_data = frame_data['joints']
            target_data = frame_data['target']
            
            # Dibujar robot
            points = np.array(joints_data)
            temp_ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o-', 
                        color='blue', lw=2, markersize=6, markerfacecolor='red')
            
            # Dibujar target
            temp_ax.plot([target_data[0]], [target_data[1]], [target_data[2]], 
                        'o', color='green', markersize=10, alpha=0.7)
            
            # Dibujar base
            temp_ax.plot([base_point[0]], [base_point[1]], [base_point[2]], 
                        'o', color='black', markersize=8, markerfacecolor='yellow')
        
        # Crear animación
        temp_anim = FuncAnimation(temp_fig, animate_recorded, frames=len(self.recording_frames),
                                 interval=50, blit=False, cache_frame_data=False, repeat=False)
        
        # Guardar
        filename = f'FABRIK/fabrik_3d_{prefix}_{robot_name}'
        try:
            print(f"Guardando {filename.split('/')[-1]}...")
            guardar_animacion(temp_anim, filename, fps=self.recording_fps, dpi=self.recording_dpi)
        except Exception as e:
            print(f"Error guardando animación: {e}")
        finally:
            plt.close(temp_fig)
    
    def _save_buffer_frames(self, prefix, robot_name, base_point=None, ax_limits=None):
        """
        Guarda los frames del buffer como animación.
        
        Args:
            prefix (str): Prefijo para el nombre del archivo
            robot_name (str): Nombre del robot
            base_point (np.ndarray, optional): Punto base del robot
            ax_limits (dict, optional): Límites de los ejes para el plot
        """
        if len(self.frame_buffer) == 0:
            return
        
        if base_point is None:
            base_point = np.array([0.0, 0.0, 0.0])
        
        # Crear una animación temporal con frames del buffer
        temp_fig, temp_ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        
        # Configurar límites si se proporcionan
        if ax_limits:
            temp_ax.set_xlim(ax_limits['x'])
            temp_ax.set_ylim(ax_limits['y'])
            temp_ax.set_zlim(ax_limits['z'])
        else:
            # Límites por defecto basados en los datos
            all_points = []
            for frame in self.frame_buffer:
                all_points.extend(frame['joints'])
                all_points.append(frame['target'])
            
            if all_points:
                points_array = np.array(all_points)
                margin = 50
                temp_ax.set_xlim(points_array[:, 0].min() - margin, points_array[:, 0].max() + margin)
                temp_ax.set_ylim(points_array[:, 1].min() - margin, points_array[:, 1].max() + margin)
                temp_ax.set_zlim(points_array[:, 2].min() - margin, points_array[:, 2].max() + margin)
        
        temp_ax.set_xlabel('X')
        temp_ax.set_ylabel('Y')
        temp_ax.set_zlabel('Z')
        temp_ax.set_title('FABRIK 3D - Recap Últimos 10s')
        
        def animate_buffer(frame_idx):
            temp_ax.clear()
            if ax_limits:
                temp_ax.set_xlim(ax_limits['x'])
                temp_ax.set_ylim(ax_limits['y'])
                temp_ax.set_zlim(ax_limits['z'])
            temp_ax.set_xlabel('X')
            temp_ax.set_ylabel('Y')
            temp_ax.set_zlabel('Z')
            temp_ax.set_title('FABRIK 3D - Recap Últimos 10s')
            
            # Obtener el frame del buffer
            frame_data = self.frame_buffer[frame_idx]
            joints_data = frame_data['joints']
            target_data = frame_data['target']
            
            # Dibujar robot
            points = np.array(joints_data)
            temp_ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o-', 
                        color='blue', lw=2, markersize=6, markerfacecolor='red')
            
            # Dibujar target
            temp_ax.plot([target_data[0]], [target_data[1]], [target_data[2]], 
                        'o', color='green', markersize=10, alpha=0.7)
            
            # Dibujar base
            temp_ax.plot([base_point[0]], [base_point[1]], [base_point[2]], 
                        'o', color='black', markersize=8, markerfacecolor='yellow')
        
        # Crear animación
        temp_anim = FuncAnimation(temp_fig, animate_buffer, frames=len(self.frame_buffer),
                                 interval=50, blit=False, cache_frame_data=False, repeat=False)
        
        # Guardar
        filename = f'FABRIK/fabrik_3d_{prefix}_{robot_name}'
        try:
            print(f"Guardando {filename.split('/')[-1]}...")
            guardar_animacion(temp_anim, filename, fps=self.recording_fps, dpi=self.recording_dpi)
        except Exception as e:
            print(f"Error guardando recap: {e}")
        finally:
            plt.close(temp_fig)

    # MANEJO DE CONTROLES DE INTERFAZ

    def on_key_press(self, event, fabrik_instance):
        """
        Manejador de eventos para las teclas presionadas.
        
        Args:
            event: Evento de teclado de matplotlib
            fabrik_instance: Instancia del sistema FABRIK para controlar
        """
        if not event.key:
            return
            
        key = event.key.lower()
        
        match key:
            # === MOVIMIENTO DEL TARGET ===
            case 'up' | 'w':     fabrik_instance.target[1] += fabrik_instance.target_step # Y+
            case 'down' | 's':   fabrik_instance.target[1] -= fabrik_instance.target_step # Y-
            case 'left' | 'a':   fabrik_instance.target[0] -= fabrik_instance.target_step # X-
            case 'right' | 'd':  fabrik_instance.target[0] += fabrik_instance.target_step # X+
            case 'u' | 'q':      fabrik_instance.target[2] += fabrik_instance.target_step # Z+
            case 'j' | 'e':      fabrik_instance.target[2] -= fabrik_instance.target_step # Z-
            # === CONTROLES ESPECIALES ===
            case 'r':       fabrik_instance.target = fabrik_instance.initial_target.copy() # Reset a posición inicial del efector final
            case '+' | '=': self.adjust_speed(fabrik_instance, 1.5)     # Aumentar velocidad
            case '-':       self.adjust_speed(fabrik_instance, 1/1.5)   # Disminuir velocidad  
            case 'h':       self.print_help()           # Ayuda
            # === SISTEMA DE GRABACIÓN ===
            case 'g': self.start_recording()    # Iniciar
            case 'p': self.pause_recording()    # Pausar/Reanudar
            case 'x': self.stop_recording(fabrik_instance.name, fabrik_instance.base_point, fabrik_instance.ax)     # Parar y guardar
            case 'c': self.capture_recap(fabrik_instance.name, fabrik_instance.base_point, fabrik_instance.ax)      # Recap
            
        # Limitar el target dentro del área visible
        plot_range = fabrik_instance.limbs_len * 1.2
        fabrik_instance.target = np.clip(fabrik_instance.target, -plot_range, plot_range)
    
    def adjust_speed(self, fabrik_instance, factor: float):
        """
        Ajusta la velocidad de movimiento.
        
        Args:
            fabrik_instance: Instancia del sistema FABRIK
            factor (float): Factor de multiplicación para la velocidad
        """
        fabrik_instance.target_step = np.clip(
            fabrik_instance.target_step * factor, 
            fabrik_instance.limbs_len * 0.01, 
            fabrik_instance.limbs_len * 0.2
        )
        print(f"\rVelocidad {'aumentada:' if factor > 1 else 'reducida: '} {fabrik_instance.target_step:6.1f}", 
              end='', flush=True)

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
        status = self.get_recording_status()
        if status['state'] != 'stopped':
            estado_indicador = "[REC]" if status['state'] == 'recording' else "[PAUSADO]"
            print(f"\n{estado_indicador} Estado: {status['state'].upper()}")
            print(f"Frames grabados: {status['frames_recorded']} ({status['duration']:.1f}s)")
