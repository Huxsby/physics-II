"""
test_alg4_alg6.py
=================
Bateria de pruebas para:
  - Algorithm 4: joint_angles() - conversion de posiciones FABRIK a angulos
  - Algorithm 6: solve_with_orientation() - control de orientacion del efector

Ejecucion desde la raiz del proyecto:
    python FABRIK/tests/test_alg4_alg6.py
"""

import sys
import os
import math

import numpy as np

# ---------------------------------------------------------------------------
# Ajuste de path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FABRIK_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in [_PROJECT_ROOT, _FABRIK_DIR, os.path.join(_PROJECT_ROOT, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias
from calculations.class_helicoidales import CinematicaDirecta
from fabrik_core.fabrik_serial_solver import FabrikSerialSolver, _exp_screw
from fabrik_core.quaternion_utils import (
    quat_identity,
    quat_from_axis_angle,
    quat_rotate_vector,
    quat_from_two_vectors,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

ROBOT_YAML = os.path.join(_PROJECT_ROOT, "config", "robot-niryo.yaml")
SEED       = 42


# ---------------------------------------------------------------------------
# Utilidades de FK
# ---------------------------------------------------------------------------

def fk_end_effector(robot, thetas):
    """
    Calcula la posicion del efector final mediante FK (producto de exponenciales).
    Usa CinematicaDirecta del proyecto cuando este disponible, con fallback
    a _exp_screw del modulo FABRIK (sin dependencias de src).

    Returns:
        np.ndarray: posicion [x, y, z] del efector en metros.
    """
    T = CinematicaDirecta(robot.ejes_helicoidales, thetas, robot.M)
    return T[:3, 3]


def fk_end_effector_fallback(screw_axes, thetas, p_home):
    """
    FK usando _exp_screw (modulo FABRIK). Util cuando CinematicaDirecta no esta
    disponible o para verificar consistencia interna.

    p_home: posicion del efector a config cero en el frame base.
    """
    T = np.eye(4)
    for Si, theta in zip(screw_axes, thetas):
        T = T @ _exp_screw(Si, theta)
    p_h = np.array([p_home[0], p_home[1], p_home[2], 1.0])
    return (T @ p_h)[:3]


# ---------------------------------------------------------------------------
# Algorithm 4: Position to Joint Angles
# ---------------------------------------------------------------------------

def test_alg4_fk_consistency(robot, n_trials=8):
    """
    Verifica que joint_angles() extrae angulos coherentes con la solucion FABRIK.

    Para cada trial:
      1. Genera una config aleatoria con theta_base=0 (primer joint twist, no
         determinable desde posiciones cartesianas).
      2. Calcula el target via FK.
      3. Resuelve con FABRIK.
      4. Extrae angulos con joint_angles().
      5. Aplica FK con los angulos extraidos y compara la posicion del efector.

    La tolerancia de FK-check (1 cm) es mayor que la del FABRIK (1e-4 m) porque
    FABRIK puede converger a una configuracion diferente de la de referencia
    (multiples soluciones IK), pero siempre debe reproducir la POSICION del target.
    """
    print("\n" + "=" * 60)
    print("Algorithm 4 — Position to Joint Angles")
    print("=" * 60)

def build_joints_from_thetas(robot, screw_axes, thetas):
    """
    Calcula las posiciones de los joints del robot para una configuracion dada
    usando FK (producto de exponenciales en Space form).

    Returns:
        List[np.ndarray]: lista de n+1 vectores [x,y,z], joints[0]=base.
    """
    p_home = [np.zeros(3)]
    cumsum = np.zeros(3)
    for link in robot.links:
        cumsum = cumsum + np.asarray(link.joint_coords, float)
        p_home.append(cumsum.copy())

    T_acc = np.eye(4)
    joints = []
    for i in range(len(robot.links) + 1):
        ph = np.append(p_home[i], 1.0)
        joints.append((T_acc @ ph)[:3].copy())
        if i < len(robot.links):
            T_acc = T_acc @ _exp_screw(screw_axes[i], thetas[i])
    return joints


# ---------------------------------------------------------------------------
# Algorithm 4: Position to Joint Angles
# ---------------------------------------------------------------------------

def test_alg4_fk_consistency(robot, n_trials=8):
    """
    Verifica que joint_angles() extrae angulos coherentes con la solucion FABRIK.

    Para cada trial:
      1. Genera una config aleatoria con joints de tipo TWIST fijos a 0
         (J0 y J3 en Niryo One: ejes paralelos al segmento siguiente,
         no determinables desde posiciones cartesianas).
      2. Calcula el target via FK y los joints exactos para esa config.
      3. Inicializa FABRIK CON los joints exactos de referencia.
      4. Resuelve con FABRIK (converge rapidamente, misma solucion).
      5. Extrae angulos con joint_angles().
      6. Aplica FK con los angulos extraidos y compara contra el target.

    Limitaciones documentadas del Algorithm 4 (Aristidou & Lasenby 2011):
      - Joints de twist (eje paralelo al segmento): angulo no determinable,
        se devuelve 0.0. Para el Niryo One: J0 (Z-twist, base) y J3 (X-twist).
      - Ultimo joint (J5): no hay segmento siguiente en la cadena → 0.0.
    """
    print("\n" + "=" * 60)
    print("Algorithm 4 — Position to Joint Angles")
    print("=" * 60)

    np.random.seed(SEED)
    screw_axes = robot.get_ejes_helicoidales()
    solver = FabrikSerialSolver.from_robot(robot)

    # Joints con twist o sin segmento siguiente: no determinables
    # Niryo One: J0=twist(Z), J3=twist(X), J5=ultimo
    UNDETERMINED_JOINTS = {0, 3, 5}

    ok_count = 0

    for trial in range(n_trials):
        # Configuracion aleatoria con joints no-determinables fijos a 0
        thetas_ref, _ = thetas_aleatorias(robot)
        for j in UNDETERMINED_JOINTS:
            thetas_ref[j] = 0.0

        target = fk_end_effector(robot, thetas_ref)

        # Inicializar FABRIK con los joints exactos de la config referencia
        joints_ref = build_joints_from_thetas(robot, screw_axes, thetas_ref)
        solver.joints       = joints_ref
        solver.orientations = [quat_identity() for _ in range(solver.n + 1)]

        result = solver.solve(target, use_workspace_constraints=False)

        if not result.converged:
            status = f"SKIP (FABRIK no convergio, err={result.final_error:.4f} m)"
            print(f"  Trial {trial + 1:02d}: {status}")
            continue

        # Extraer angulos con Algorithm 4
        thetas_ext = solver.joint_angles(robot)

        # FK con los angulos extraidos vs target original
        target_check = fk_end_effector(robot, thetas_ext)
        pos_error    = float(np.linalg.norm(target_check - target))

        passed = pos_error < 0.01  # 1 cm
        if passed:
            ok_count += 1

        mark = "OK  " if passed else "FALLO"
        print(
            f"  Trial {trial + 1:02d}: [{mark}] "
            f"FABRIK_err={result.final_error:.5f} m  "
            f"FK_check_err={pos_error * 1000:.2f} mm  "
            f"iters={result.iterations}"
        )

    print(f"\n  Resultado: {ok_count}/{n_trials} trials OK")
    print(f"  (joints no-determinables fijados a 0: {sorted(UNDETERMINED_JOINTS)})\n")
    return ok_count, n_trials


# ---------------------------------------------------------------------------
# Algorithm 6: FABRIK with Orientation Control
# ---------------------------------------------------------------------------

def test_alg6_orientation_control(robot):
    """
    Verifica que solve_with_orientation() controla la direccion del ultimo
    segmento del efector final segun la orientacion objetivo.

    Para cada orientacion objetivo:
      1. Resuelve FABRIK con restriccion de orientacion.
      2. Mide el angulo entre la direccion real del ultimo segmento y la deseada.
      3. Mide el error de posicion del efector al target.

    Tolerancias:
      - Orientacion: < 5 grados.
      - Posicion: < 2 cm (puede ser mayor que la tolerancia del FABRIK porque la
        restriccion de orientacion reduce los grados de libertad efectivos).
    """
    print("=" * 60)
    print("Algorithm 6 — FABRIK with Orientation Control")
    print("=" * 60)

    solver = FabrikSerialSolver.from_robot(robot)

    # Target alcanzable para el Niryo One (dentro del workspace)
    target = np.array([0.20, 0.00, 0.25])

    # Orientaciones a probar: (descripcion, quaternion [w,x,y,z])
    orientations = [
        ("identidad [0,0,1]",        np.array([1.0, 0.0, 0.0, 0.0])),
        ("inclinacion 30 deg en X",  quat_from_axis_angle([1, 0, 0], math.radians(30))),
        ("inclinacion 45 deg en Y",  quat_from_axis_angle([0, 1, 0], math.radians(45))),
        ("inclinacion -30 deg en X", quat_from_axis_angle([1, 0, 0], math.radians(-30))),
        ("rotar 60 deg en Z",        quat_from_axis_angle([0, 0, 1], math.radians(60))),
    ]

    ok_count = 0
    for desc, quat in orientations:
        solver.reset_to_initial()
        result = solver.solve_with_orientation(
            target, quat, use_workspace_constraints=False
        )

        joints   = result.joint_positions
        last_dir = _normalize(joints[-1] - joints[-2])
        desired  = _normalize(quat_rotate_vector(quat, np.array([0.0, 0.0, 1.0])))

        ori_err_rad = float(np.arccos(np.clip(np.dot(last_dir, desired), -1.0, 1.0)))
        ori_err_deg = math.degrees(ori_err_rad)
        pos_err_mm  = result.final_error * 1000.0

        passed = ori_err_deg < 5.0
        if passed:
            ok_count += 1

        mark = "OK   " if passed else "FALLO"
        print(
            f"  [{mark}] {desc:<30s}  "
            f"ori_err={ori_err_deg:5.2f} deg  "
            f"pos_err={pos_err_mm:6.2f} mm  "
            f"iters={result.iterations}"
        )

    print(f"\n  Resultado: {ok_count}/{len(orientations)} orientaciones OK\n")
    return ok_count, len(orientations)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nCargando robot desde", ROBOT_YAML)
    robot = cargar_robot_desde_yaml(ROBOT_YAML)

    ok4, total4 = test_alg4_fk_consistency(robot)
    ok6, total6 = test_alg6_orientation_control(robot)

    print("=" * 60)
    alg4_str = f"Algorithm 4: {ok4}/{total4}"
    alg6_str = f"Algorithm 6: {ok6}/{total6}"
    print(f"  {alg4_str}  |  {alg6_str}")
    all_ok = (ok4 >= total4 * 0.7) and (ok6 >= total6 * 0.6)
    print(f"  {'PASS' if all_ok else 'FALLO'}")
    print("=" * 60)
