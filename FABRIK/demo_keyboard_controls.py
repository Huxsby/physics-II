#!/usr/bin/env python3
"""
demo_keyboard_controls.py
=========================
Demo de seguimiento de trayectoria para FabrikSerialSolver con el robot Niryo One.

Dos modos:
  1. Trayectoria automatica: interpolacion suave entre configuraciones articulares
     aleatorias validas. Cada waypoint se obtiene con thetas_aleatorias() y se
     convierte a Cartesiano con CinematicaDirecta(). El solver sigue la trayectoria
     en modo encadenado (sin reset entre frames).

  2. Control interactivo: el usuario mueve el target con teclado y el solver
     sigue en tiempo real desde la ultima configuracion resuelta.

Ejecucion desde la raiz del proyecto:
    python FABRIK/demo_keyboard_controls.py

Controles interactivos:
    W / S  ->  target +X / -X
    A / D  ->  target -Y / +Y
    Q / E  ->  target +Z / -Z
    R      ->  reiniciar solver y target
"""

import sys
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FABRIK_DIR   = os.path.abspath(os.path.dirname(__file__))

for _p in [_PROJECT_ROOT, _FABRIK_DIR, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, thetas_limite
from calculations.class_helicoidales import CinematicaDirecta
from animation import exportar_trayectoria_cartesian, importar_trayectoria_cartesian
from fabrik_core.fabrik_serial_solver import FabrikSerialSolver

# ---------------------------------------------------------------------------
# Generacion de trayectoria suave via FK
# ---------------------------------------------------------------------------

def generate_smooth_trajectory(robot, num_waypoints=6, frames_per_segment=50):
    """
    Genera una trayectoria suave del efector final interpolando configuraciones
    articulares aleatorias validas con suavizado cosenoidal (ease in-out).

    Flujo:
      1. Generar num_waypoints configuraciones aleatorias validas con thetas_aleatorias().
      2. Para cada par de waypoints consecutivos, interpolar frames_per_segment
         configuraciones intermedias usando parametro coseno.
      3. Convertir cada configuracion a posicion cartesiana con CinematicaDirecta().

    El loop se cierra volviendo al primer waypoint.

    Returns:
        list[np.ndarray]: Posiciones [x, y, z] en metros del efector final.
    """
    print(f"  Generando {num_waypoints} waypoints aleatorios validos...")
    waypoints = [thetas_aleatorias(robot)[0] for _ in range(num_waypoints)]
    waypoints.append(waypoints[0])  # cerrar el loop

    trajectory = []
    for i in range(num_waypoints):
        t_start = waypoints[i]
        t_end   = waypoints[i + 1]
        for j in range(frames_per_segment):
            alpha        = j / frames_per_segment
            alpha_smooth = 0.5 - 0.5 * np.cos(alpha * np.pi)
            thetas_interp  = t_start * (1.0 - alpha_smooth) + t_end * alpha_smooth
            thetas_clipped = np.array(thetas_limite(robot, thetas_interp.tolist()))
            T = CinematicaDirecta(robot.ejes_helicoidales, thetas_clipped, robot.M)
            trajectory.append(T[:3, 3].copy())

    print(f"  Trayectoria generada: {len(trajectory)} puntos")
    return trajectory


# ---------------------------------------------------------------------------
# Demo 1: Trayectoria automatica
# ---------------------------------------------------------------------------

def run_trajectory_demo(robot, solver):
    """
    Anima el seguimiento de una trayectoria FK suave usando FabrikSerialSolver
    en modo encadenado: sin reset entre frames para suavidad natural.
    """
    print("\nDemo 1: Trayectoria automatica suave")
    print("-" * 40)

    trajectory = generate_smooth_trajectory(robot, num_waypoints=6, frames_per_segment=50)

    # exportar_trayectoria_cartesian prepende "data/motion/cartesian/" internamente
    exportar_trayectoria_cartesian(path="trayectoria_new.xyz", trayectoria=trajectory)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    r = solver.total_length * 1.1
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(0, r * 1.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("FabrikSerialSolver - Seguimiento de trayectoria FK")

    traj_arr = np.array(trajectory)
    ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2],
            "--", color="gray", lw=0.8, alpha=0.5, label="trayectoria FK")

    chain_line, = ax.plot([], [], [], "o-", color="steelblue", lw=2,
                          markersize=5, markerfacecolor="white")
    target_dot, = ax.plot([], [], [], "o", color="green", markersize=10, alpha=0.8)
    ax.plot([0], [0], [0], "o", color="black", markersize=8)
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                          bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    frame_idx = [0]
    solver.reset_to_initial()

    def animate(_frame):
        idx    = frame_idx[0] % len(trajectory)
        target = trajectory[idx]
        frame_idx[0] += 1
        result = solver.solve(target)
        pts    = np.array(result.joint_positions)
        chain_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        target_dot.set_data_3d([target[0]], [target[1]], [target[2]])
        conv_str = "OK" if result.converged else "NO"
        info_text.set_text(
            f"Frame {idx+1}/{len(trajectory)}  "
            f"iter={result.iterations}  "
            f"err={result.final_error * 1000:.2f} mm  conv={conv_str}"
        )
        return chain_line, target_dot, info_text

    _anim = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo 2: Control interactivo por teclado
# ---------------------------------------------------------------------------

def run_interactive_demo(robot, solver):
    """
    Mueve el target con teclado WASD/QE en tiempo real.
    El solver sigue desde la ultima configuracion resuelta (sin reset).
    """
    print("\nDemo 2: Control interactivo")
    print("  W/S=+/-X  A/D=-/+Y  Q/E=+/-Z  R=reset")
    print("-" * 40)

    step   = solver.total_length * 0.05
    target = [np.array([solver.total_length * 0.3, 0.0, solver.total_length * 0.5])]

    with mpl.rc_context({"keymap.quit": [], "keymap.save": []}):
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")

        r = solver.total_length * 1.1
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(0, r * 1.5)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("FabrikSerialSolver - Control de teclado\n"
                     "[W/S: +/-X   A/D: -/+Y   Q/E: +/-Z   R: reset]")

        chain_line, = ax.plot([], [], [], "o-", color="steelblue", lw=2,
                              markersize=5, markerfacecolor="white")
        target_dot, = ax.plot([], [], [], "o", color="green", markersize=10, alpha=0.8)
        ax.plot([0], [0], [0], "o", color="black", markersize=8)
        info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        def on_key(event):
            t = target[0]
            if   event.key == "w": t[0] += step
            elif event.key == "s": t[0] -= step
            elif event.key == "a": t[1] -= step
            elif event.key == "d": t[1] += step
            elif event.key == "q": t[2] += step
            elif event.key == "e": t[2] -= step
            elif event.key == "r":
                target[0] = np.array([solver.total_length * 0.3, 0.0,
                                       solver.total_length * 0.5])
                solver.reset_to_initial()

        fig.canvas.mpl_connect("key_press_event", on_key)

        def animate(_frame):
            t      = target[0]
            result = solver.solve(t)
            pts    = np.array(result.joint_positions)
            chain_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            target_dot.set_data_3d([t[0]], [t[1]], [t[2]])
            conv_str = "OK" if result.converged else "NO"
            info_text.set_text(
                f"Target: [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]\n"
                f"iter={result.iterations}  "
                f"err={result.final_error * 1000:.2f} mm  conv={conv_str}"
            )
            return chain_line, target_dot, info_text

        _anim = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    yaml_path = os.path.join(_PROJECT_ROOT, "config", "robot-niryo.yaml")
    print(f"Cargando robot desde: {yaml_path}")

    robot = cargar_robot_desde_yaml(yaml_path)
    if robot is None:
        print("ERROR: No se pudo cargar el robot.")
        sys.exit(1)

    solver = FabrikSerialSolver.from_robot(robot)
    print(solver)

    while True:
        print("=" * 50)
        print("1. Trayectoria automatica suave")
        print("2. Control interactivo (teclado)")
        print("0. Salir")
        print("=" * 50)
        opcion = input("Selecciona demo: ").strip()

        if opcion == "1":
            run_trajectory_demo(robot, solver)
        elif opcion == "2":
            run_interactive_demo(robot, solver)
        elif opcion == "0":
            print("Saliendo...")
            break
        else:
            print("Opcion no valida.")
