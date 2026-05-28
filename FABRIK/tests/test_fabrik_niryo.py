"""
test_fabrik_niryo.py
====================
Script de prueba para el FabrikSerialSolver con el robot Niryo One.

Carga el robot desde config/robot-niryo.yaml usando la infraestructura
del proyecto (cargar_robot_desde_yaml), construye el solver y ejecuta
dos baterias de pruebas:

  Bateria 1 - FK targets:
    Genera configuraciones aleatorias validas con thetas_aleatorias(), calcula
    la posicion cartesiana del efector con CinematicaDirecta() y usa esa
    posicion como target para FABRIK. Garantiza que el target es alcanzable
    y que la solucion esperada se conoce de antemano.

  Bateria 2 - Robustez encadenada:
    Genera N targets aleatorios consecutivos y ejecuta solve() en secuencia
    sin reiniciar entre llamadas. Evalua cuantos convergen y si el solver
    se atasca al pasar de una pose a otra.

Ejecucion desde la raiz del proyecto:
    python FABRIK/tests/test_fabrik_niryo.py

Dependencias: numpy, matplotlib, scipy, pyyaml (ver pyproject.toml)
"""

import sys
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Ajuste de path para importar modulos del proyecto
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FABRIK_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in [_PROJECT_ROOT, _FABRIK_DIR, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias
from calculations.class_helicoidales import CinematicaDirecta
from fabrik_core.fabrik_serial_solver import FabrikSerialSolver

# ---------------------------------------------------------------------------
# Configuracion de la prueba
# ---------------------------------------------------------------------------

N_FK_TARGETS    = 8    # Casos en Bateria 1 (FK targets aleatorios)
N_CHAIN_TARGETS = 12   # Targets en Bateria 2 (robustez encadenada)
SEED            = 42   # Semilla para reproducibilidad


# ---------------------------------------------------------------------------
# Generacion de targets mediante cinematica directa
# ---------------------------------------------------------------------------

def generar_fk_target(robot):
    """
    Genera una configuracion aleatoria valida y calcula la posicion cartesiana
    del efector final usando cinematica directa (producto de exponenciales).

    Returns:
        (thetas, posicion_xyz): configuracion articular y posicion del efector.
    """
    thetas, _ = thetas_aleatorias(robot)
    T = CinematicaDirecta(robot.ejes_helicoidales, thetas, robot.M)
    posicion = T[:3, 3]
    return thetas, posicion


# ---------------------------------------------------------------------------
# Bateria 1: targets via FK
# ---------------------------------------------------------------------------

def run_fk_battery(robot, solver: FabrikSerialSolver, n: int) -> list:
    """
    Ejecuta n pruebas con targets generados por cinematica directa.
    Cada target tiene solucion conocida (la configuracion que lo genero).

    Returns:
        Lista de dicts con: thetas_ref, target, result.
    """
    print()
    print("=" * 80)
    print("  Bateria 1 - Targets generados por cinematica directa (FK)")
    print("  Target garantizado alcanzable; solucion de referencia conocida")
    print("=" * 80)
    print(f"  {'Caso':<6} {'iter':>5} {'error (m)':>12} {'conv':>6}  target [x, y, z]")
    print("-" * 80)

    casos = []
    for i in range(n):
        thetas_ref, target = generar_fk_target(robot)
        solver.reset_to_initial()
        result = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"
        print(
            f"  FK-{i+1:<3}  {result.iterations:>5} {result.final_error:>12.6f} {conv_str:>6}"
            f"  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )
        casos.append({"thetas_ref": thetas_ref, "target": target, "result": result})

    n_conv = sum(1 for c in casos if c["result"].converged)
    print("-" * 80)
    print(f"  Convergencia: {n_conv}/{n}  ({100*n_conv/n:.0f}%)")
    print("=" * 80)
    return casos


# ---------------------------------------------------------------------------
# Bateria 2: robustez encadenada (sin reset entre targets)
# ---------------------------------------------------------------------------

def run_chained_battery(robot, solver: FabrikSerialSolver, n: int) -> list:
    """
    Genera n targets consecutivos y llama a solve() en secuencia SIN reiniciar
    la configuracion entre llamadas. Simula movimiento real del robot donde
    la posicion anterior es el punto de partida para la siguiente.

    Returns:
        Lista de dicts con: target, result, desde (posicion inicial del efector).
    """
    print()
    print("=" * 80)
    print("  Bateria 2 - Robustez encadenada (sin reset entre targets)")
    print("  El solver parte de la configuracion final de la llamada anterior")
    print("=" * 80)
    print(f"  {'Paso':<6} {'iter':>5} {'error (m)':>12} {'conv':>6}  "
          f"{'desde [efector]':<30}  target [x, y, z]")
    print("-" * 80)

    # Partir de la configuracion inicial
    solver.reset_to_initial()
    casos = []

    for i in range(n):
        _, target = generar_fk_target(robot)
        desde = solver.joints[-1].copy()  # efector antes de resolver
        result = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"
        desde_str = f"[{desde[0]:+.3f}, {desde[1]:+.3f}, {desde[2]:+.3f}]"
        print(
            f"  T-{i+1:<3}   {result.iterations:>5} {result.final_error:>12.6f} {conv_str:>6}"
            f"  {desde_str:<30}  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )
        casos.append({"target": target, "result": result, "desde": desde})

    n_conv = sum(1 for c in casos if c["result"].converged)
    print("-" * 80)
    print(f"  Convergencia: {n_conv}/{n}  ({100*n_conv/n:.0f}%)")
    print("=" * 80)
    return casos


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

def _draw_chain(ax, joint_positions, color="steelblue"):
    pts = np.array(joint_positions)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color=color,
            linewidth=1.5, markersize=3, markerfacecolor="white")


def _set_equal_axes(ax, points):
    pts = np.array(points)
    if len(pts) == 0:
        return
    center = pts.mean(axis=0)
    r = max(np.max(np.abs(pts - center)) * 1.2, 0.05)
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def plot_fk_battery(solver, casos):
    """Muestra la posicion final de cada caso FK en subplots 3D."""
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("Bateria 1 - FK targets (Niryo One)", fontsize=13)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]

        _draw_chain(ax, res.joint_positions)

        ax.scatter(*solver.base_position, color="black", s=15, zorder=5,
                   label="base")

        if res.converged:
            # Efector alcanza el target: marcador estrella con relleno verde
            # y borde rojo para indicar que ambos puntos coinciden.
            ax.scatter(*caso["target"],
                       color="limegreen", edgecolors="crimson", linewidths=1.5,
                       s=55, zorder=6, marker="*",
                       label="target = efector (conv)")
        else:
            ax.scatter(*caso["target"],
                       color="green", s=18, zorder=5, marker="o",
                       label="target")
            ax.scatter(*res.end_effector,
                       color="crimson", s=18, zorder=5, marker="x",
                       linewidths=1.5, label="efector final")

        all_pts = list(res.joint_positions) + [caso["target"]]
        _set_equal_axes(ax, all_pts)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        conv_str = "OK" if res.converged else "NO"
        ax.set_title(
            f"FK-{idx+1}  conv={conv_str}\n"
            f"iter={res.iterations}  err={res.final_error:.4f} m",
            fontsize=8,
        )
        ax.legend(fontsize=6, loc="upper left", markerscale=1.2)

    plt.tight_layout()
    plt.show()


def plot_chained_battery(solver, casos):
    """
    Muestra la trayectoria encadenada: para cada paso dibuja la posicion
    anterior del efector y la posicion final alcanzada.
    """
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle("Bateria 2 - Robustez encadenada (Niryo One)", fontsize=13)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]

        _draw_chain(ax, res.joint_positions)

        ax.scatter(*solver.base_position, color="black",  s=15, zorder=5,
                   label="base")
        ax.scatter(*caso["desde"],        color="orange", s=18, zorder=5,
                   marker="^", label="efector anterior")

        if res.converged:
            # Efector alcanza el target: estrella verde/rojo combinada
            ax.scatter(*caso["target"],
                       color="limegreen", edgecolors="crimson", linewidths=1.5,
                       s=55, zorder=6, marker="*",
                       label="target = efector (conv)")
        else:
            ax.scatter(*caso["target"],
                       color="green", s=18, zorder=5, marker="o",
                       label="target")
            ax.scatter(*res.end_effector,
                       color="crimson", s=18, zorder=5, marker="x",
                       linewidths=1.5, label="efector final")

        all_pts = list(res.joint_positions) + [caso["target"], caso["desde"]]
        _set_equal_axes(ax, all_pts)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        conv_str = "OK" if res.converged else "NO"
        ax.set_title(
            f"T-{idx+1}  conv={conv_str}\n"
            f"iter={res.iterations}  err={res.final_error:.4f} m",
            fontsize=8,
        )
        ax.legend(fontsize=6, loc="upper left", markerscale=1.2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Ejecucion principal
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    yaml_path = os.path.join(_PROJECT_ROOT, "config", "robot-niryo.yaml")
    print(f"Cargando robot desde: {yaml_path}")
    robot = cargar_robot_desde_yaml(yaml_path)
    if robot is None:
        print("ERROR: No se pudo cargar el robot.")
        sys.exit(1)

    print(robot)

    print("\nConstruyendo FabrikSerialSolver...")
    solver = FabrikSerialSolver.from_robot(robot)
    print(solver)

    # Bateria 1: targets con FK garantizada alcanzables
    casos_fk = run_fk_battery(robot, solver, N_FK_TARGETS)

    # Bateria 2: robustez encadenada
    casos_chain = run_chained_battery(robot, solver, N_CHAIN_TARGETS)

    plot_fk_battery(solver, casos_fk)
    plot_chained_battery(solver, casos_chain)


if __name__ == "__main__":
    main()

