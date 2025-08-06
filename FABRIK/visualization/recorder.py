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
    
    def stop_recording(self, robot_name="default"):
        """
        Detiene la grabación y guarda el archivo.
        
        Args:
            robot_name (str): Nombre del robot para el archivo
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
        
        # Crear animación temporal con los frames grabados
        self._save_recorded_frames("recording", robot_name)
    
    def capture_recap(self, robot_name="default"):
        """
        Captura los últimos N segundos del buffer.
        
        Args:
            robot_name (str): Nombre del robot para el archivo
        """
        if len(self.frame_buffer) == 0:
            print("ADVERTENCIA: No hay frames en el buffer para recap")
            return
        
        frame_count = len(self.frame_buffer)
        duration = frame_count / self.recording_fps
        
        print(f"CAPTURANDO RECAP")
        print(f"   Frames disponibles: {frame_count}")
        print(f"   Duración: {duration:.1f} segundos")
        
        # Usar todos los frames del buffer
        self._save_buffer_frames("recap", robot_name)
    
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
