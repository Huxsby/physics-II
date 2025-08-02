#!/usr/bin/env python3
"""
Demo automático de los controles de teclado para FABRIK 3D.
Muestra el robot siguiendo una trayectoria predefinida.
"""

import numpy as np
import sys
import os

# Agregar el path para importar el módulo FABRIK
sys.path.append(os.path.dirname(__file__))

try:
    from fabrik_paper_constrained_3d import FabrikIK3D
    print("Módulo FABRIK importado correctamente")
except ImportError as e:
    print(f"Error al importar el módulo FABRIK: {e}")
    sys.exit(1)

def demo_keyboard_controls():
    """
    Demostración de los controles de teclado.
    """
    print("DEMO: Controles de Teclado para FABRIK 3D")
    print("=" * 60)
    
    # Crear instancia del sistema
    ik_system = FabrikIK3D()
    
    print("Mapa de Controles:")
    print("┌─────────────────────────────────────────┐")
    print("│  WASD   │ Movimiento primario XY       │")
    print("│  Q/E    │ Eje Z (arriba/abajo)         │")
    print("│ Flechas │ Movimiento alternativo XY    │")
    print("│  U/J    │ Eje Z alternativo            │")
    print("│   R     │ Reset a posición inicial     │")
    print("│  +/-    │ Velocidad de movimiento      │")
    print("│   H     │ Ayuda completa               │")
    print("└─────────────────────────────────────────┘")
    
    print("\nNota: Los controles Q/S ahora funcionan sin conflictos")
    print("      gracias a mpl.rc_context() que deshabilita keymaps reservados")
    print("\nEl mouse controla la vista 3D (rotar/zoom)")
    print("El target se mueve con el teclado únicamente")
    
    print("\nIniciando visualización 3D...")
    print("   Nota: Asegúrate de que la ventana tenga foco para usar el teclado")
    
    # Ejecutar automáticamente algunos comandos para demostración
    print("\nDemo automático en 3 segundos...")
    
    # Iniciar la visualización
    ik_system.setup_plot()

def create_trajectory_demo():
    """
    Crea una demostración con trayectoria automática.
    """
    print("DEMO: Trayectoria Automática 3D")
    print("=" * 50)
    
    # Crear instancia del sistema
    ik_system = FabrikIK3D()
    
    # Definir una trayectoria en 3D (helicoidal)
    t_values = np.linspace(0, 4*np.pi, 100)
    trajectory = []
    
    for t in t_values:
        x = 80 * np.cos(t)
        y = 80 * np.sin(t)
        z = 30 + 20 * np.sin(t/2)  # Componente vertical
        trajectory.append(np.array([x, y, z]))
    
    print("Trayectoria generada:")
    print(f"   • {len(trajectory)} puntos")
    print(f"   • Rango X: [{min(p[0] for p in trajectory):.1f}, {max(p[0] for p in trajectory):.1f}]")
    print(f"   • Rango Y: [{min(p[1] for p in trajectory):.1f}, {max(p[1] for p in trajectory):.1f}]")
    print(f"   • Rango Z: [{min(p[2] for p in trajectory):.1f}, {max(p[2] for p in trajectory):.1f}]")
    
    print("Durante la demo puedes usar los controles de teclado para tomar el control:")
    print("   WASD (primario), QE para Z, o flechas direccionales, UJ para Z")
    print("Iniciando demo automático...")
    
    # Agregar la trayectoria al sistema
    ik_system.demo_trajectory = trajectory
    ik_system.demo_index = 0
    ik_system.demo_active = True
    
    # Modificar el método animate para incluir la demo
    original_animate = ik_system.animate
    
    def animate_with_demo(frame):
        # Si la demo está activa, usar puntos de la trayectoria
        if hasattr(ik_system, 'demo_active') and ik_system.demo_active:
            if hasattr(ik_system, 'demo_trajectory') and hasattr(ik_system, 'demo_index'):
                if ik_system.demo_index < len(ik_system.demo_trajectory):
                    ik_system.target = ik_system.demo_trajectory[ik_system.demo_index].copy()
                    ik_system.demo_index += 1
                else:
                    # Reiniciar la trayectoria
                    ik_system.demo_index = 0
        
        return original_animate(frame)
    
    ik_system.animate = animate_with_demo
    ik_system.setup_plot()

if __name__ == "__main__":
    print("FABRIK 3D - Demos de Control")
    print("=" * 40)
    print("1. Demo de controles de teclado")
    print("2. Demo con trayectoria automática")
    print("=" * 40)
    
    choice = input("Selecciona demo (1 o 2): ").strip()
    
    if choice == "1":
        demo_keyboard_controls()
    elif choice == "2":
        create_trajectory_demo()
    else:
        print("Opción inválida. Ejecutando demo por defecto...")
        demo_keyboard_controls()
