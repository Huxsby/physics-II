"""
comparacion_ik_vs_fabrik.py
===========================
Animación comparativa lado a lado de dos métodos de cinemática inversa
siguiendo una trayectoria circular en el espacio 3D:

  - Izquierda : Newton-Jacobiano  (IK_Jacobian)
  - Derecha   : FABRIK             (FabrikSerialSolver)

Robot usado: Niryo One  (config/robot-niryo.yaml)

Ejecución desde la raíz del proyecto:
    python tests/comparacion_ik_vs_fabrik.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC     = os.path.join(_ROOT, "src")
_FABRIK  = os.path.join(_ROOT, "FABRIK")

for _p in [_ROOT, _SRC, _FABRIK]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from core import cargar_robot_desde_yaml
from calculations import IK_Jacobian
from calculations.class_jacobian import calcular_jacobiana
from animation import guardar_animacion
from animation.class_robot_plotter import calcular_transformaciones

from fabrik_core.fabrik_serial_solver import FabrikSerialSolver

# ---------------------------------------------------------------------------
# Parámetros de la trayectoria circular
# ---------------------------------------------------------------------------
NUM_PUNTOS     = 72          # puntos del círculo (= frames de la animación)
RADIO          = 0.18        # metros
Z_ALTURA       = 0.30        # metros
RPY_JACOBIAN   = [0, np.pi, 0]   # orientación fija para IK Jacobiana

# Parámetros de visualización
ANIMATION_INTERVAL_MS = 80   # ms entre frames (≈12 fps)
SAVE_ANIMATION        = False  # Cambia a True para guardar el video
OUTPUT_NAME           = "comparacion_ik_vs_fabrik"

# ---------------------------------------------------------------------------
# Generar trayectoria circular
# ---------------------------------------------------------------------------
angulos = np.linspace(0, 2 * np.pi, NUM_PUNTOS, endpoint=False)
puntos  = np.array([[RADIO * np.cos(a), RADIO * np.sin(a), Z_ALTURA]
                    for a in angulos])

# ---------------------------------------------------------------------------
# Cargar robot
# ---------------------------------------------------------------------------
print("Cargando robot...")
robot = cargar_robot_desde_yaml("config/robot-niryo.yaml")
print()

# ---------------------------------------------------------------------------
# Pre-computar Newton-Jacobiano
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Pre-computando Newton-Jacobiano...")
print("=" * 60)

Jacobiana_tuple = calcular_jacobiana(robot)
jacobian_positions = []   # [(n+1) x 3]  posiciones cartesianas de cadena
jacobian_metrics   = []   # {'iters', 'converged'}

# Warm-start: IK al primer punto para obtener una configuración inicial cercana
thetas_guess = np.zeros(robot.num_links)
primer_sol = IK_Jacobian(
    robot, Jacobiana_tuple,
    thetas_actuales=thetas_guess,
    p_xyz=puntos[0], RPY=RPY_JACOBIAN,
    show=False
)
if primer_sol:
    thetas_guess = np.array(primer_sol[-1])

for idx, punto in enumerate(puntos):
    sol = IK_Jacobian(
        robot, Jacobiana_tuple,
        thetas_actuales=thetas_guess,
        p_xyz=punto, RPY=RPY_JACOBIAN,
        show=False
    )
    if sol:
        thetas_final  = np.array(sol[-1])
        thetas_guess  = thetas_final
        iters         = len(sol)
        converged     = True
    else:
        thetas_final  = thetas_guess.copy()
        iters         = -1
        converged     = False

    transforms = calcular_transformaciones(robot, thetas_final)
    pos = np.array([T[:3, 3] for T in transforms])
    jacobian_positions.append(pos)
    jacobian_metrics.append({'iters': iters, 'converged': converged})

    if (idx + 1) % 10 == 0 or idx == NUM_PUNTOS - 1:
        n_conv = sum(1 for m in jacobian_metrics if m['converged'])
        print(f"  [{idx+1:3d}/{NUM_PUNTOS}]  convergidos: {n_conv}/{idx+1}")

# ---------------------------------------------------------------------------
# Pre-computar FABRIK
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  Pre-computando FABRIK...")
print("=" * 60)

fabrik_solver    = FabrikSerialSolver.from_robot(robot)
fabrik_positions = []   # [(n+1) x 3]
fabrik_metrics   = []   # {'iters', 'converged', 'error'}

for idx, punto in enumerate(puntos):
    result = fabrik_solver.solve(punto)
    fabrik_positions.append(np.array(result.joint_positions))
    fabrik_metrics.append({
        'iters'    : result.iterations,
        'converged': result.converged,
        'error'    : result.final_error,
    })

    if (idx + 1) % 10 == 0 or idx == NUM_PUNTOS - 1:
        n_conv = sum(1 for m in fabrik_metrics if m['converged'])
        print(f"  [{idx+1:3d}/{NUM_PUNTOS}]  convergidos: {n_conv}/{idx+1}")

# ---------------------------------------------------------------------------
# Resumen previo a la animación
# ---------------------------------------------------------------------------
n_jac_conv  = sum(1 for m in jacobian_metrics if m['converged'])
n_fab_conv  = sum(1 for m in fabrik_metrics   if m['converged'])
avg_jac_it  = np.mean([m['iters'] for m in jacobian_metrics if m['converged']])
avg_fab_it  = np.mean([m['iters'] for m in fabrik_metrics   if m['converged']])
avg_fab_err = np.mean([m['error'] for m in fabrik_metrics   if m['converged']])

print()
print("=" * 60)
print("  RESUMEN")
print("=" * 60)
print(f"  {'Método':<22} {'Conv':>6}  {'Iter media':>10}")
print(f"  {'-'*22} {'-'*6}  {'-'*10}")
print(f"  {'Newton-Jacobiano':<22} {n_jac_conv:>3}/{NUM_PUNTOS}  {avg_jac_it:>10.1f}")
print(f"  {'FABRIK':<22} {n_fab_conv:>3}/{NUM_PUNTOS}  {avg_fab_it:>10.1f}  (err medio conv: {avg_fab_err:.2e} m)")
print()

# ---------------------------------------------------------------------------
# Cálculo de límites globales para ambos subplots
# ---------------------------------------------------------------------------
all_pts = np.vstack(jacobian_positions + fabrik_positions)
margin  = 0.05
xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
zlim = (max(0, all_pts[:, 2].min() - margin), all_pts[:, 2].max() + margin)

# ---------------------------------------------------------------------------
# Animación
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7))
fig.suptitle("Comparación IK: Newton-Jacobiano  vs  FABRIK\n"
             f"Robot Niryo One — Trayectoria circular (r={RADIO} m, z={Z_ALTURA} m)",
             fontsize=12, fontweight='bold')

ax_jac  = fig.add_subplot(121, projection='3d')
ax_fab  = fig.add_subplot(122, projection='3d')

def _setup_ax(ax):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    # Trazar trayectoria de referencia (en gris punteado)
    ax.plot(puntos[:, 0], puntos[:, 1], puntos[:, 2],
            '--', color='lightgray', linewidth=1, zorder=0)

def _draw_chain(ax, positions, color):
    pos = np.asarray(positions)
    # Eslabones
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            '-o', color=color, linewidth=2.5, markersize=4, zorder=3)
    # Base
    ax.scatter(*pos[0], color='black', s=70, marker='s', zorder=5)
    # Efector final
    ax.scatter(*pos[-1], color='limegreen', s=90, marker='*', zorder=5)

# Trazas acumuladas (se limpian al reiniciar desde frame 0)
trace_jac_pts = []
trace_fab_pts = []

def update(frame):
    if frame == 0:
        trace_jac_pts.clear()
        trace_fab_pts.clear()

    ax_jac.cla()
    ax_fab.cla()
    _setup_ax(ax_jac)
    _setup_ax(ax_fab)

    target = puntos[frame]

    # ---- Newton-Jacobiano ----
    trace_jac_pts.append(jacobian_positions[frame][-1].copy())
    if len(trace_jac_pts) > 1:
        tr = np.array(trace_jac_pts)
        ax_jac.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                    '-', color='steelblue', linewidth=1, alpha=0.55, zorder=1)
    _draw_chain(ax_jac, jacobian_positions[frame], color='steelblue')
    ax_jac.scatter(*target, color='red', s=100, marker='x', linewidths=2, zorder=6)

    mj = jacobian_metrics[frame]
    conv_str = "SI" if mj['converged'] else "NO"
    ax_jac.set_title(
        f"Newton-Jacobiano\n"
        f"Frame {frame+1}/{NUM_PUNTOS}  |  iter: {mj['iters']}  |  conv: {conv_str}",
        fontsize=9
    )

    # ---- FABRIK ----
    trace_fab_pts.append(fabrik_positions[frame][-1].copy())
    if len(trace_fab_pts) > 1:
        tr = np.array(trace_fab_pts)
        ax_fab.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                    '-', color='darkorchid', linewidth=1, alpha=0.55, zorder=1)
    _draw_chain(ax_fab, fabrik_positions[frame], color='darkorchid')
    ax_fab.scatter(*target, color='red', s=100, marker='x', linewidths=2, zorder=6)

    mf = fabrik_metrics[frame]
    conv_str = "SI" if mf['converged'] else "NO"
    ax_fab.set_title(
        f"FABRIK\n"
        f"Frame {frame+1}/{NUM_PUNTOS}  |  iter: {mf['iters']}  |  conv: {conv_str}"
        f"  |  err: {mf['error']:.2e} m",
        fontsize=9
    )

    # Leyenda compacta solo en el primer frame para no reconstruirla cada vez
    for ax, col in [(ax_jac, 'steelblue'), (ax_fab, 'darkorchid')]:
        ax.scatter([], [], [], color=col,       marker='o', s=30,  label='Articulación')
        ax.scatter([], [], [], color='black',   marker='s', s=50,  label='Base')
        ax.scatter([], [], [], color='limegreen', marker='*', s=60, label='Efector')
        ax.scatter([], [], [], color='red',     marker='x', s=60,  label='Target')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.6)

anim = FuncAnimation(
    fig, update,
    frames=NUM_PUNTOS,
    interval=ANIMATION_INTERVAL_MS,
    repeat=True
)

plt.tight_layout()

if SAVE_ANIMATION:
    print(f"Guardando animación '{OUTPUT_NAME}'...")
    guardar_animacion(anim, OUTPUT_NAME)
    print("Guardada.")

plt.show()
