"""
test_fabrik_niryo.py
====================
Bateria de pruebas de la Fase 5 (Extraccion de Angulos) para FABRIKRSolver
con el robot Niryo One.

Estructura:
  verify_known_configurations()
    Prueba offline con una cadena 3-joint planar (bisagras en Y).
    Verifica que extract_joint_angles devuelve los angulos geometricamente
    correctos para tres configuraciones conocidas:
      Caso 1: brazo estirado (todos theta = 0).
      Caso 2: brazo en L  (joint 0 a +90 grados, joints 1 y 2 a 0).
      Caso 3: angulos arbitrarios [+30, +45, -15] grados.
      Caso 4: get_hardware_command con offset no nulo.

  run_fk_battery(robot, solver, n)
    Genera n configuraciones aleatorias con thetas_aleatorias(), calcula la
    posicion del efector via CinematicaDirecta, y usa esa posicion como target
    para FABRIKRSolver. Verifica convergencia y limites articulares.

  run_chained_battery(robot, solver, n)
    Genera n targets consecutivos sin reiniciar el solver entre llamadas.
    Simula movimiento continuo del brazo.

Ejecucion desde la raiz del proyecto:
    python FABRIK/tests/test_fabrik_niryo.py

Dependencias: numpy, matplotlib, pyyaml (ver pyproject.toml)
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

from core.class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, limits
from calculations.class_helicoidales import CinematicaDirecta
from fabrik_core.fabrik_r_solver import (
    FABRIKRSolver,
    RevoluteJoint,
    extract_joint_angle,
    _safe_normalize,
    EPS,
)

# _exp_screw se usa solo en _ik_newton_position (verificacion de accesibilidad)
# Se importa desde legacy para no crear una dependencia nueva en el solver principal.
from legacy.fabrik_serial_solver import _exp_screw

# ---------------------------------------------------------------------------
# Configuracion de la prueba
# ---------------------------------------------------------------------------

N_FK_TARGETS    = 8    # Casos en Bateria 1 (FK targets aleatorios)
N_CHAIN_TARGETS = 12   # Targets en Bateria 2 (robustez encadenada)
SEED            = 42   # Semilla para reproducibilidad


# ---------------------------------------------------------------------------
# Verificacion con configuraciones geometricamente conocidas
# ---------------------------------------------------------------------------

def verify_known_configurations() -> bool:
    """
    Verifica que extract_joint_angles devuelve los angulos correctos para cuatro
    configuraciones con solucion geometrica exactamente conocida.

    Cadena usada: 3 joints con bisagra en Y (eje [0,1,0]), ref [0,0,1].
    Configuracion inicial: brazo estirado a lo largo del eje Z.

    Referencia: AGENTS.md Sec. 5; SANTOS21 Sec. V (extraccion de variables).
    """
    L = 0.2     # longitud de cada eslabon (m)
    TOL = 1e-5  # tolerancia angular (rad)

    joints = [
        RevoluteJoint(
            length=L,
            axis_local=[0., 1., 0.],
            ref_axis_local=[0., 0., 1.],
            theta_min=-math.pi,
            theta_max=math.pi,
        )
        for _ in range(3)
    ]
    init_pos = [
        np.array([0., 0., 0.]),
        np.array([0., 0., L]),
        np.array([0., 0., 2 * L]),
        np.array([0., 0., 3 * L]),
    ]
    solver = FABRIKRSolver(joints, init_pos)

    all_ok = True

    print()
    print("=" * 70)
    print("  Verificacion: extract_joint_angles  (cadena 3-joint, bisagra Y)")
    print("=" * 70)
    print(f"  {'Caso':<40} {'Resultado':>8}  angulos obtenidos (grados)")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Caso 1: brazo estirado (todos theta = 0)
    # ------------------------------------------------------------------
    p0 = [np.array([0., 0., 0.]),
          np.array([0., 0., L]),
          np.array([0., 0., 2 * L]),
          np.array([0., 0., 3 * L])]
    got  = solver.extract_joint_angles(p0)
    exp  = [0.0, 0.0, 0.0]
    ok1  = all(abs(g - e) < TOL for g, e in zip(got, exp))
    all_ok = all_ok and ok1
    got_str = [f"{math.degrees(a):+.3f}" for a in got]
    print(f"  {'1: estirado  (theta=[0,0,0] deg)':<40} {'OK' if ok1 else 'FAIL':>8}  {got_str}")

    # ------------------------------------------------------------------
    # Caso 2: brazo en L (joint 0 a +90 deg, joints 1 y 2 a 0 deg)
    # Los tres eslabones apuntan en +X; el efector esta en [3L, 0, 0].
    # theta_0 = 90 deg (Z -> X), theta_1 = 0 deg, theta_2 = 0 deg.
    # ------------------------------------------------------------------
    p1 = [np.array([0., 0., 0.]),
          np.array([L,  0., 0.]),
          np.array([2 * L, 0., 0.]),
          np.array([3 * L, 0., 0.])]
    got  = solver.extract_joint_angles(p1)
    exp  = [math.pi / 2, 0.0, 0.0]
    ok2  = all(abs(g - e) < TOL for g, e in zip(got, exp))
    all_ok = all_ok and ok2
    got_str = [f"{math.degrees(a):+.3f}" for a in got]
    print(f"  {'2: L-shape   (theta=[90,0,0] deg)':<40} {'OK' if ok2 else 'FAIL':>8}  {got_str}")

    # ------------------------------------------------------------------
    # Caso 3: angulos arbitrarios [+30, +45, -15] grados
    # Angulos acumulados (absolutos): [30, 75, 60] grados.
    # Cada eslabon apunta en [sin(alpha), 0, cos(alpha)].
    # ------------------------------------------------------------------
    th     = [math.radians(30.), math.radians(45.), math.radians(-15.)]
    alphas = [sum(th[:k + 1]) for k in range(3)]
    p2 = [np.zeros(3)]
    for alpha in alphas:
        p2.append(p2[-1] + L * np.array([math.sin(alpha), 0., math.cos(alpha)]))
    got  = solver.extract_joint_angles(p2)
    exp  = th
    ok3  = all(abs(g - e) < TOL for g, e in zip(got, exp))
    all_ok = all_ok and ok3
    got_str = [f"{math.degrees(a):+.3f}" for a in got]
    exp_str = [f"{math.degrees(e):+.3f}" for e in exp]
    print(f"  {'3: arbitrario (theta=[30,45,-15] deg)':<40} {'OK' if ok3 else 'FAIL':>8}  {got_str}")
    print(f"  {'   esperado':<40} {'':>8}  {exp_str}")

    # ------------------------------------------------------------------
    # Caso 4: get_hardware_command con offset no nulo
    # offset = [10, -5, 3] grados -> theta_hw = theta_calc - offset
    # ------------------------------------------------------------------
    offsets_deg = [10., -5., 3.]
    joints_off = [
        RevoluteJoint(
            length=L,
            axis_local=[0., 1., 0.],
            ref_axis_local=[0., 0., 1.],
            theta_min=-math.pi,
            theta_max=math.pi,
            offset=math.radians(od),
        )
        for od in offsets_deg
    ]
    solver_off = FABRIKRSolver(joints_off, init_pos)
    # Asignar posiciones del caso 3 directamente
    solver_off.positions = [np.array(pv, dtype=float) for pv in p2]
    hw_cmd   = solver_off.get_hardware_command()
    exp_hw   = [th[i] - math.radians(offsets_deg[i]) for i in range(3)]
    ok4      = all(abs(g - e) < TOL for g, e in zip(hw_cmd, exp_hw))
    all_ok   = all_ok and ok4
    got_str  = [f"{math.degrees(a):+.3f}" for a in hw_cmd]
    exp_str  = [f"{math.degrees(e):+.3f}" for e in exp_hw]
    print(f"  {'4: hardware_cmd con offset [10,-5,3] deg':<40} {'OK' if ok4 else 'FAIL':>8}  {got_str}")
    print(f"  {'   esperado':<40} {'':>8}  {exp_str}")

    print("-" * 70)
    print(f"  Resultado global: {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 70)
    return all_ok


# ---------------------------------------------------------------------------
# Generacion de targets mediante cinematica directa
# ---------------------------------------------------------------------------

def generar_fk_target(robot):
    """
    Genera una configuracion aleatoria valida y calcula la posicion del efector
    usando cinematica directa (producto de exponenciales).
    """
    thetas, _ = thetas_aleatorias(robot)
    T = CinematicaDirecta(robot.ejes_helicoidales, thetas, robot.M)
    return thetas, T[:3, 3]


# ---------------------------------------------------------------------------
# IK Newton-Jacobian numerica (verificacion de accesibilidad del target)
# ---------------------------------------------------------------------------

def _ik_newton_position(robot, target_xyz, thetas_init=None, tol=1e-3, max_iter=40):
    """
    Verifica si un punto es alcanzable mediante IK de posicion Newton-Jacobian.
    Usa FK producto de exponenciales con _exp_screw del modulo legacy.

    Returns:
        (converged: bool, final_error_m: float, iterations: int)
        converged solo es True si la solucion cumple posicion y limites.
    """
    S  = robot.get_ejes_helicoidales()
    M  = robot.M
    n  = len(S)
    th = np.zeros(n) if thetas_init is None else np.array(thetas_init, dtype=float).copy()

    lims = []
    for i in range(n):
        key = f'joint_{i + 1}'
        if robot.limits_dict and key in robot.limits_dict:
            lo, hi = robot.limits_dict[key]
            lims.append((float(lo), float(hi)))
        else:
            lims.append((-math.pi, math.pi))

    def _clip_limits(th_):
        th_clip = th_.copy()
        for i, (lo, hi) in enumerate(lims):
            th_clip[i] = float(np.clip(th_clip[i], lo, hi))
        return th_clip

    def _within_limits(th_):
        for i, (lo, hi) in enumerate(lims):
            if not (lo - 1e-9 <= th_[i] <= hi + 1e-9):
                return False
        return True

    th = _clip_limits(th)

    def _fk_pos(th_):
        T = np.eye(4)
        for Si, ti in zip(S, th_):
            T = T @ _exp_screw(np.asarray(Si, float), ti)
        return (T @ M)[:3, 3]

    def _jac_pos(th_, eps=1e-6):
        p0 = _fk_pos(th_)
        J  = np.zeros((3, n))
        for k in range(n):
            th_p = th_.copy(); th_p[k] += eps
            J[:, k] = (_fk_pos(th_p) - p0) / eps
        return J

    for it in range(max_iter):
        delta = target_xyz - _fk_pos(th)
        err   = float(np.linalg.norm(delta))
        if err < tol and _within_limits(th):
            return True, err, it
        J  = _jac_pos(th)
        # Newton de posicion con proyeccion a caja de limites por iteracion.
        # Esto evita reportar targets como accesibles con configuraciones no fisicas.
        th = _clip_limits(th + np.linalg.pinv(J) @ delta)

    final_err = float(np.linalg.norm(target_xyz - _fk_pos(th)))
    return (final_err < tol) and _within_limits(th), final_err, max_iter


# ---------------------------------------------------------------------------
# Detalle de limites articulares
# ---------------------------------------------------------------------------

def _limits_verbose(robot, thetas):
    result = []
    if robot.limits_dict is None:
        return result
    for i, theta in enumerate(thetas):
        key = f'joint_{i + 1}'
        if key not in robot.limits_dict:
            continue
        lo, hi = robot.limits_dict[key]
        ok     = (lo <= theta <= hi)
        excess = max(0.0, lo - theta, theta - hi)
        result.append({
            'joint':  i,
            'theta':  float(theta),
            'lower':  float(lo),
            'upper':  float(hi),
            'ok':     ok,
            'excess': excess,
        })
    return result


def _print_limits_violations(label, casos):
    violating = [(i, c) for i, c in enumerate(casos)
                 if c.get('lim_ok') is False and c.get('lim_detail')]
    if not violating:
        return
    print()
    print(f"  Detalle de violaciones de limites ({label}):")
    print(f"  {'Caso':<8} {'Joint':>5} {'theta(deg)':>12} {'[lo,hi](deg)':>22} {'exceso(deg)':>12}")
    print("  " + "-" * 64)
    for case_idx, caso in violating:
        for jd in caso['lim_detail']:
            if not jd['ok']:
                t_deg  = math.degrees(jd['theta'])
                lo_deg = math.degrees(jd['lower'])
                hi_deg = math.degrees(jd['upper'])
                ex_deg = math.degrees(jd['excess'])
                prefix = f"{label}-{case_idx + 1}"
                print(f"  {prefix:<8}   J{jd['joint']} {t_deg:>+12.2f}"
                      f"  [{lo_deg:>+8.2f}, {hi_deg:>+8.2f}]  {ex_deg:>+11.2f}")


# ---------------------------------------------------------------------------
# Bateria 1: targets via FK
# ---------------------------------------------------------------------------

def run_fk_battery(robot, solver: FABRIKRSolver, n: int) -> list:
    """
    Ejecuta n pruebas con targets generados por cinematica directa.

    Returns:
        Lista de dicts con: thetas_ref, target, result, thetas_hw, lim_ok, etc.
    """
    print()
    print("=" * 90)
    print("  Bateria 1 - Targets generados por cinematica directa (FK)")
    print("=" * 90)
    print(f"  {'Caso':<6} {'iter':>5} {'error(m)':>12} {'conv':>6}  {'lim':>3}  {'nwt':>3}  {'exit':<14}  target [x, y, z]")
    print("-" * 104)

    casos = []
    for i in range(n):
        thetas_ref, target = generar_fk_target(robot)
        solver.reset_to_initial()
        result   = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"
        exit_str = result.exit_reason
        thetas_hw  = None
        lim_ok     = None
        lim_detail = []
        if result.converged:
            try:
                thetas_hw  = solver.get_hardware_command()
                lim_ok, _  = limits(robot, thetas_hw)
                lim_detail = _limits_verbose(robot, thetas_hw)
            except Exception as exc:
                print(f"  [WARN] get_hardware_command fallo en caso {i}: {exc}")
        lim_str = "--" if lim_ok is None else ("OK" if lim_ok else "NO")

        nwt_conv, nwt_err, _ = _ik_newton_position(robot, target)
        nwt_str = "OK" if nwt_conv else "NO"

        print(
            f"  FK-{i + 1:<3}  {result.iterations:>5} {result.final_error:>12.6f}"
            f" {conv_str:>6}  {lim_str:>3}  {nwt_str:>3}  {exit_str:<14}"
            f"  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )
        casos.append({
            "thetas_ref":  thetas_ref,
            "target":      target,
            "result":      result,
            "thetas_hw":   thetas_hw,
            "lim_ok":      lim_ok,
            "lim_detail":  lim_detail,
            "newton_conv": nwt_conv,
            "newton_err":  nwt_err,
        })

    n_conv = sum(1 for c in casos if c["result"].converged)
    n_lim  = sum(1 for c in casos if c.get("lim_ok") is True)
    n_nwt  = sum(1 for c in casos if c.get("newton_conv"))
    n_stab = sum(1 for c in casos if c["result"].exit_reason == "stable")
    print("-" * 104)
    print(f"  Convergencia FABRIK-R: {n_conv}/{n}  |  Limites OK: {n_lim}/{n}"
          f"  |  Accesible (Newton+lim): {n_nwt}/{n}  |  Estables (early-exit): {n_stab}/{n}")
    _print_limits_violations("FK", casos)

    # Tabla comparativa: thetas_ref vs get_hardware_command()
    print()
    print("  Angulos de referencia (FK) vs hardware_command()  [grados]:")
    print(f"  {'Caso':<6} {'Joint':>5} {'ref(deg)':>10} {'hw(deg)':>10} {'diff':>9}"
          f"  {'ref_lim':>7}  {'hw_lim':>7}")
    print("  " + "-" * 60)
    for i, caso in enumerate(casos):
        if not caso["result"].converged:
            continue
        thetas_hw = caso.get("thetas_hw")
        if thetas_hw is None:
            continue
        thetas_ref = caso["thetas_ref"]
        for ji, (tr, tf) in enumerate(zip(thetas_ref, thetas_hw)):
            key = f'joint_{ji + 1}'
            lo, hi = (robot.limits_dict.get(key, (-math.pi, math.pi))
                      if robot.limits_dict else (-math.pi, math.pi))
            ref_ok = "OK" if lo <= tr <= hi else "NO"
            fab_ok = "OK" if lo <= tf <= hi else "NO"
            diff   = math.degrees(tf - tr)
            marker = " <--" if abs(diff) > 15 else ""
            print(f"  FK-{i + 1:<3}    J{ji}  {math.degrees(tr):>+10.2f}"
                  f" {math.degrees(tf):>+10.2f} {diff:>+9.2f}"
                  f"  {ref_ok:>7}  {fab_ok:>7}{marker}")
    print("=" * 90)
    return casos


# ---------------------------------------------------------------------------
# Bateria 2: robustez encadenada (sin reset entre targets)
# ---------------------------------------------------------------------------

def run_chained_battery(robot, solver: FABRIKRSolver, n: int) -> list:
    """
    Genera n targets consecutivos y llama a solve() en secuencia SIN reiniciar.

    Returns:
        Lista de dicts con: target, result, desde, lim_ok, etc.
    """
    print()
    print("=" * 100)
    print("  Bateria 2 - Robustez encadenada (sin reset entre targets)")
    print("=" * 100)
    print(f"  {'Paso':<6} {'iter':>5} {'error(m)':>12} {'conv':>6}  "
          f"{'lim':>3}  {'nwt':>3}  {'exit':<14}  {'desde [efector]':<30}  target [x, y, z]")
    print("-" * 116)

    solver.reset_to_initial()
    casos = []

    for i in range(n):
        _, target = generar_fk_target(robot)
        desde  = solver.positions[-1].copy()
        result   = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"
        exit_str = result.exit_reason
        lim_ok     = None
        lim_detail = []
        if result.converged:
            try:
                thetas_hw  = solver.get_hardware_command()
                lim_ok, _  = limits(robot, thetas_hw)
                lim_detail = _limits_verbose(robot, thetas_hw)
            except Exception:
                lim_ok = None
        lim_str = "--" if lim_ok is None else ("OK" if lim_ok else "NO")

        nwt_conv, nwt_err, _ = _ik_newton_position(robot, target)
        nwt_str = "OK" if nwt_conv else "NO"

        desde_str = f"[{desde[0]:+.3f}, {desde[1]:+.3f}, {desde[2]:+.3f}]"
        print(
            f"  T-{i + 1:<3}   {result.iterations:>5} {result.final_error:>12.6f}"
            f" {conv_str:>6}  {lim_str:>3}  {nwt_str:>3}  {exit_str:<14}"
            f"  {desde_str:<30}"
            f"  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )
        casos.append({
            "target":      target,
            "result":      result,
            "desde":       desde,
            "lim_ok":      lim_ok,
            "lim_detail":  lim_detail,
            "newton_conv": nwt_conv,
            "newton_err":  nwt_err,
        })

    n_conv = sum(1 for c in casos if c["result"].converged)
    n_lim  = sum(1 for c in casos if c.get("lim_ok") is True)
    n_nwt  = sum(1 for c in casos if c.get("newton_conv"))
    n_stab = sum(1 for c in casos if c["result"].exit_reason == "stable")
    print("-" * 116)
    print(f"  Convergencia FABRIK-R: {n_conv}/{n}  |  Limites OK: {n_lim}/{n}"
          f"  |  Accesible (Newton+lim): {n_nwt}/{n}  |  Estables (early-exit): {n_stab}/{n}")
    _print_limits_violations("T", casos)
    print("=" * 116)
    return casos


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------

def _draw_chain(ax, joint_positions, color="steelblue"):
    pts = np.array(joint_positions)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color=color,
            linewidth=1.5, markersize=3, markerfacecolor="white")


def _compute_global_bounds(casos_list: list):
    all_pts = []
    for caso in casos_list:
        all_pts.extend(list(caso["result"].joint_positions))
        all_pts.append(caso["target"])
        if "desde" in caso:
            all_pts.append(caso["desde"])
    if not all_pts:
        return np.zeros(3), 0.4
    pts    = np.array(all_pts)
    center = pts.mean(axis=0)
    r      = max(np.max(np.abs(pts - center)) * 1.2, 0.1)
    return center, r


def _apply_global_bounds(ax, center, r):
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def plot_fk_battery(solver, casos):
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    fig    = plt.figure(figsize=(5 * n_cols, 5.8 * n_rows))
    fig.suptitle("Bateria 1 - FK targets (Niryo One / FABRIKRSolver)", fontsize=11, y=0.995)
    g_center, g_r = _compute_global_bounds(casos)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]
        _draw_chain(ax, res.joint_positions)
        ax.scatter(*solver.base, color="black", s=12, zorder=5, label="base")

        if res.converged:
            ax.scatter(*caso["target"], color="limegreen", edgecolors="crimson",
                       linewidths=1.2, s=45, zorder=6, marker="*", label="target=EE")
        else:
            ax.scatter(*caso["target"], color="green",   s=14, zorder=5,
                       marker="o", label="target")
            ax.scatter(*res.end_effector, color="crimson", s=14, zorder=5,
                       marker="x", linewidths=1.2, label="EE")

        _apply_global_bounds(ax, g_center, g_r)
        ax.set_xlabel("X", fontsize=6, labelpad=0)
        ax.set_ylabel("Y", fontsize=6, labelpad=0)
        ax.set_zlabel("Z", fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5)

        conv_str = "OK" if res.converged else "NO"
        lo       = caso.get("lim_ok")
        lim_str  = "--" if lo is None else ("OK" if lo else "NO")
        nwt_str  = "OK" if caso.get("newton_conv") else "NO"
        ax.set_title(
            f"FK-{idx + 1} | conv={conv_str} lim={lim_str} nwt={nwt_str}\n"
            f"i={res.iterations}  e={res.final_error * 1000:.1f}mm",
            fontsize=7, pad=2,
        )
        ax.legend(fontsize=5, loc="upper left", markerscale=1.0,
                  borderpad=0.2, labelspacing=0.15, handlelength=1.0)

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=1.5)
    plt.show()


def plot_chained_battery(solver, casos):
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    fig    = plt.figure(figsize=(5 * n_cols, 5.8 * n_rows))
    fig.suptitle("Bateria 2 - Robustez encadenada (Niryo One / FABRIKRSolver)",
                 fontsize=11, y=0.995)
    g_center, g_r = _compute_global_bounds(casos)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]
        _draw_chain(ax, res.joint_positions)
        ax.scatter(*solver.base, color="black",  s=12, zorder=5, label="base")
        ax.scatter(*caso["desde"], color="orange", s=14, zorder=5,
                   marker="^", label="ant.")

        if res.converged:
            ax.scatter(*caso["target"], color="limegreen", edgecolors="crimson",
                       linewidths=1.2, s=45, zorder=6, marker="*", label="target=EE")
        else:
            ax.scatter(*caso["target"], color="green",   s=14, zorder=5,
                       marker="o", label="target")
            ax.scatter(*res.end_effector, color="crimson", s=14, zorder=5,
                       marker="x", linewidths=1.2, label="EE")

        _apply_global_bounds(ax, g_center, g_r)
        ax.set_xlabel("X", fontsize=6, labelpad=0)
        ax.set_ylabel("Y", fontsize=6, labelpad=0)
        ax.set_zlabel("Z", fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5)

        conv_str = "OK" if res.converged else "NO"
        lo       = caso.get("lim_ok")
        lim_str  = "--" if lo is None else ("OK" if lo else "NO")
        nwt_str  = "OK" if caso.get("newton_conv") else "NO"
        ax.set_title(
            f"T-{idx + 1} | conv={conv_str} lim={lim_str} nwt={nwt_str}\n"
            f"i={res.iterations}  e={res.final_error * 1000:.1f}mm",
            fontsize=7, pad=2,
        )
        ax.legend(fontsize=5, loc="upper left", markerscale=1.0,
                  borderpad=0.2, labelspacing=0.15, handlelength=1.0)

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=1.5)
    plt.show()


# ---------------------------------------------------------------------------
# Ejecucion principal
# ---------------------------------------------------------------------------

def main():
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 0. Verificacion geometrica offline (sin robot fisico)
    # ------------------------------------------------------------------
    ok = verify_known_configurations()
    if not ok:
        print("\nERROR: Verificacion de configuraciones conocidas FALLO.")
        print("       Revisar extract_joint_angles antes de continuar.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Cargar robot Niryo One desde YAML
    # ------------------------------------------------------------------
    yaml_path = os.path.join(_PROJECT_ROOT, "config", "robot-niryo.yaml")
    print(f"\nCargando robot desde: {yaml_path}")
    robot = cargar_robot_desde_yaml(yaml_path)
    if robot is None:
        print("ERROR: No se pudo cargar el robot.")
        sys.exit(1)
    print(robot)

    # ------------------------------------------------------------------
    # 2. Construir FABRIKRSolver desde el robot
    # ------------------------------------------------------------------
    print("\nConstruyendo FABRIKRSolver.from_robot()...")
    solver = FABRIKRSolver.from_robot(robot, max_iterations=200, tolerance=1e-3)
    print(f"  Joints: {solver.n}")
    print(f"  Longitudes: {[f'{l:.4f}' for l in solver.lengths]}")
    print(f"  Base: {solver.base}")
    solver.get_debug_state()

    # ------------------------------------------------------------------
    # 3. Baterias de prueba
    # ------------------------------------------------------------------
    casos_fk    = run_fk_battery(robot, solver, N_FK_TARGETS)
    casos_chain = run_chained_battery(robot, solver, N_CHAIN_TARGETS)

    # ------------------------------------------------------------------
    # 4. Visualizacion
    # ------------------------------------------------------------------
    plot_fk_battery(solver, casos_fk)
    plot_chained_battery(solver, casos_chain)


if __name__ == "__main__":
    main()
