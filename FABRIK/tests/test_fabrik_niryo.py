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

from core.class_robot_structure import cargar_robot_desde_yaml, thetas_aleatorias, limits
from calculations.class_helicoidales import CinematicaDirecta
from fabrik_core.fabrik_serial_solver import FabrikSerialSolver, _exp_screw

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
# IK Newton-Jacobian numerica (verificacion de accesibilidad)
# ---------------------------------------------------------------------------

def _ik_newton_position(robot, target_xyz, thetas_init=None, tol=1e-3, max_iter=40):
    """
    Verifica si un punto es alcanzable mediante IK de posicion Newton-Jacobian.
    Usa la Jacobiana analitica de posicion (diferencias finitas) sobre la FK
    producto de exponenciales. No necesita Jacobiana simbolica.

    Args:
        robot       : instancia de Robot.
        target_xyz  : array [x, y, z] del punto objetivo.
        thetas_init : configuracion inicial (None = todo ceros).
        tol         : tolerancia de posicion en metros (default 1 mm).
        max_iter    : maximo de iteraciones Newton.

    Returns:
        (converged: bool, final_error_m: float, iterations: int)
    """
    S  = robot.get_ejes_helicoidales()
    M  = robot.M
    n  = len(S)
    th = np.zeros(n) if thetas_init is None else np.array(thetas_init, dtype=float).copy()

    def _fk_pos(th):
        T = np.eye(4)
        for Si, ti in zip(S, th):
            T = T @ _exp_screw(np.asarray(Si, float), ti)
        return (T @ M)[:3, 3]

    def _jac_pos_numerical(th, eps=1e-6):
        """Jacobiana de posicion 3xn por diferencias finitas."""
        p0 = _fk_pos(th)
        J  = np.zeros((3, n))
        for i in range(n):
            th_p = th.copy(); th_p[i] += eps
            J[:, i] = (_fk_pos(th_p) - p0) / eps
        return J

    for it in range(max_iter):
        delta = target_xyz - _fk_pos(th)
        err   = float(np.linalg.norm(delta))
        if err < tol:
            return True, err, it
        J  = _jac_pos_numerical(th)
        th = th + np.linalg.pinv(J) @ delta

    final_err = float(np.linalg.norm(target_xyz - _fk_pos(th)))
    return final_err < tol, final_err, max_iter


# ---------------------------------------------------------------------------
# Verificacion detallada de limites articulares
# (activa el bloque DEBUG comentado en class_robot_structure.py)
# ---------------------------------------------------------------------------

def _limits_verbose(robot, thetas):
    """
    Version detallada de limits(): devuelve una lista de dicts por articulacion
    con el valor extraido, los limites y el exceso (positivo = violacion).

    Reemplica el bloque DEBUG comentado en:
        src/core/class_robot_structure.py  lineas ~534-540

    Returns:
        list[dict]: un dict por articulacion con:
            joint   : indice (0-based)
            theta   : angulo extraido (rad)
            lower   : limite inferior (rad)
            upper   : limite superior (rad)
            ok      : True si esta dentro del rango
            excess  : rad fuera de rango (0 si ok; >0 si viola)
    """
    result = []
    if robot.limits_dict is None:
        return result
    for i, theta in enumerate(thetas):
        key = f'joint_{i+1}'
        if key not in robot.limits_dict:
            continue
        lo, hi = robot.limits_dict[key]
        ok     = (lo <= theta <= hi)
        excess = max(0.0, lo - theta, theta - hi)  # rad fuera del rango
        result.append({
            'joint' : i,
            'theta' : float(theta),
            'lower' : float(lo),
            'upper' : float(hi),
            'ok'    : ok,
            'excess': excess,
        })
    return result


def _print_limits_violations(label, casos):
    """
    Imprime tabla de violaciones de limites articulares para todos los casos
    donde lim_ok is False.
    """
    violating = [(i, c) for i, c in enumerate(casos)
                 if c.get('lim_ok') is False and c.get('lim_detail')]
    if not violating:
        return
    print()
    print(f"  Detalle de violaciones de limites ({label}):")
    print(f"  {'Caso':<8} {'Joint':>5} {'theta (deg)':>12} {'[lo, hi] (deg)':>22} {'exceso (deg)':>13}")
    print("  " + "-" * 66)
    for case_idx, caso in violating:
        for jd in caso['lim_detail']:
            if not jd['ok']:
                t_deg  = math.degrees(jd['theta'])
                lo_deg = math.degrees(jd['lower'])
                hi_deg = math.degrees(jd['upper'])
                ex_deg = math.degrees(jd['excess'])
                prefix = f"{label}-{case_idx+1}"
                print(f"  {prefix:<8}   J{jd['joint']} {t_deg:>+12.2f}  [{lo_deg:>+8.2f}, {hi_deg:>+8.2f}]  {ex_deg:>+12.2f}")


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
    print(f"  {'Caso':<6} {'iter':>5} {'error (m)':>12} {'conv':>6}  {'lim':>3}  {'nwt':>3}  target [x, y, z]")
    print("-" * 90)

    casos = []
    for i in range(n):
        thetas_ref, target = generar_fk_target(robot)
        solver.reset_to_initial()
        result = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"

        # Verificar limites articulares (solo si FABRIK convergio)
        lim_ok     = None
        lim_detail = []
        thetas_ext = None
        if result.converged:
            try:
                thetas_ext = solver.joint_angles(robot)
                lim_ok, _ = limits(robot, thetas_ext)
                lim_detail = _limits_verbose(robot, thetas_ext)
            except Exception:
                lim_ok = None
        lim_str = "--" if lim_ok is None else ("OK" if lim_ok else "NO")

        # Verificar accesibilidad del target via Newton-Jacobian IK
        nwt_conv, nwt_err, _ = _ik_newton_position(robot, target)
        nwt_str = "OK" if nwt_conv else "NO"

        print(
            f"  FK-{i+1:<3}  {result.iterations:>5} {result.final_error:>12.6f} {conv_str:>6}"
            f"  {lim_str:>3}  {nwt_str:>3}"
            f"  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )
        casos.append({
            "thetas_ref": thetas_ref, "target": target, "result": result,
            "thetas_ext": thetas_ext,
            "lim_ok": lim_ok, "lim_detail": lim_detail,
            "newton_conv": nwt_conv, "newton_err": nwt_err,
        })

    n_conv = sum(1 for c in casos if c["result"].converged)
    n_lim  = sum(1 for c in casos if c.get("lim_ok") is True)
    n_nwt  = sum(1 for c in casos if c.get("newton_conv"))
    print("-" * 90)
    print(f"  Convergencia FABRIK: {n_conv}/{n}  |  Limites OK: {n_lim}/{n}  |  Accesible (Newton): {n_nwt}/{n}")
    _print_limits_violations("FK", casos)

    # Comparacion thetas_ref (generacion FK) vs thetas_ext (Algorithm 4 sobre FABRIK)
    # Permite distinguir si las violaciones provienen de Algorithm 4 o de la
    # estrategia de constraints activa en el solver.
    print()
    print("  Comparacion thetas_ref vs thetas_FABRIK (Algorithm 4)  [grados]:")
    print(f"  {'Caso':<6} {'Joint':>5} {'ref':>9} {'fabrik':>9} {'diff':>8}  {'ref_lim':>7}  {'fab_lim':>7}")
    print("  " + "-" * 60)
    for i, caso in enumerate(casos):
        if not caso["result"].converged:
            continue
        thetas_ref = caso["thetas_ref"]
        thetas_ext = caso.get("thetas_ext")
        if thetas_ext is None:
            continue
        for ji, (tr, tf) in enumerate(zip(thetas_ref, thetas_ext)):
            key = f'joint_{ji+1}'
            lo_hi = robot.limits_dict.get(key, (-math.pi, math.pi)) if robot.limits_dict else (-math.pi, math.pi)
            lo, hi = lo_hi
            ref_ok = "OK" if lo <= tr <= hi else "NO"
            fab_ok = "OK" if lo <= tf <= hi else "NO"
            diff   = math.degrees(tf - tr)
            marker = " <--" if abs(diff) > 10 else ""
            print(f"  FK-{i+1:<3}    J{ji}  {math.degrees(tr):>+9.2f} {math.degrees(tf):>+9.2f} {diff:>+8.2f}  {ref_ok:>7}  {fab_ok:>7}{marker}")
    print("=" * 90)
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
          f"{'lim':>3}  {'nwt':>3}  {'desde [efector]':<30}  target [x, y, z]")
    print("-" * 100)

    # Partir de la configuracion inicial
    solver.reset_to_initial()
    debug_hinge = os.getenv("FABRIK_HINGE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
    casos = []

    for i in range(n):
        _, target = generar_fk_target(robot)
        desde = solver.joints[-1].copy()  # efector antes de resolver
        result = solver.solve(target)
        conv_str = "SI" if result.converged else "NO"

        # Verificar limites articulares (solo si FABRIK convergio)
        lim_ok     = None
        lim_detail = []
        if result.converged:
            try:
                thetas_ext = solver.joint_angles(robot)
                lim_ok, _ = limits(robot, thetas_ext)
                lim_detail = _limits_verbose(robot, thetas_ext)
            except Exception:
                lim_ok = None
        lim_str = "--" if lim_ok is None else ("OK" if lim_ok else "NO")

        # Verificar accesibilidad del target via Newton-Jacobian IK
        nwt_conv, nwt_err, _ = _ik_newton_position(robot, target)
        nwt_str = "OK" if nwt_conv else "NO"

        desde_str = f"[{desde[0]:+.3f}, {desde[1]:+.3f}, {desde[2]:+.3f}]"
        print(
            f"  T-{i+1:<3}   {result.iterations:>5} {result.final_error:>12.6f} {conv_str:>6}"
            f"  {lim_str:>3}  {nwt_str:>3}"
            f"  {desde_str:<30}  [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
        )

        if debug_hinge and not result.converged and hasattr(solver, "get_last_hinge_debug"):
            dbg = solver.get_last_hinge_debug()
            counts = dbg.get("counts", {})
            if counts:
                pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                top = ", ".join([f"J{ji}:{cnt}" for ji, cnt in pairs[:4]])
                print(f"    Debug HINGE backward saturations -> {top}")
            else:
                print("    Debug HINGE backward saturations -> none")

        casos.append({
            "target": target, "result": result, "desde": desde,
            "lim_ok": lim_ok, "lim_detail": lim_detail,
            "newton_conv": nwt_conv, "newton_err": nwt_err,
        })

    n_conv = sum(1 for c in casos if c["result"].converged)
    n_lim  = sum(1 for c in casos if c.get("lim_ok") is True)
    n_nwt  = sum(1 for c in casos if c.get("newton_conv"))
    print("-" * 100)
    print(f"  Convergencia FABRIK: {n_conv}/{n}  |  Limites OK: {n_lim}/{n}  |  Accesible (Newton): {n_nwt}/{n}")
    _print_limits_violations("T", casos)
    print("=" * 100)
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


def _compute_global_bounds(casos_list: list) -> tuple:
    """
    Calcula el bounding box global de todos los puntos de todos los casos.
    Devuelve (center, radius) para usarlos como escala uniforme en todos
    los subplots de una bateria.

    Mantiene la interactividad: los limites son fijos pero el usuario puede
    rotar la camara libremente (azimut, elevacion) con el raton.
    """
    all_pts = []
    for caso in casos_list:
        res = caso["result"]
        all_pts.extend(list(res.joint_positions))
        all_pts.append(caso["target"])
        if "desde" in caso:
            all_pts.append(caso["desde"])
    if not all_pts:
        return np.zeros(3), 0.4
    pts = np.array(all_pts)
    center = pts.mean(axis=0)
    r = max(np.max(np.abs(pts - center)) * 1.2, 0.1)
    return center, r


def _apply_global_bounds(ax, center: np.ndarray, r: float) -> None:
    """Aplica limites uniformes a un subplot 3D manteniendo el aspecto."""
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def plot_fk_battery(solver, casos):
    """Muestra la posicion final de cada caso FK en subplots 3D."""
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 5.8 * n_rows))
    fig.suptitle("Bateria 1 - FK targets (Niryo One)", fontsize=11, y=0.995)

    # Escala uniforme: mismo bounding box para todos los subplots
    g_center, g_r = _compute_global_bounds(casos)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]

        _draw_chain(ax, res.joint_positions)

        ax.scatter(*solver.base_position, color="black", s=12, zorder=5,
                   label="base")

        if res.converged:
            ax.scatter(*caso["target"],
                       color="limegreen", edgecolors="crimson", linewidths=1.2,
                       s=45, zorder=6, marker="*",
                       label="target=efector")
        else:
            ax.scatter(*caso["target"],
                       color="green", s=14, zorder=5, marker="o",
                       label="target")
            ax.scatter(*res.end_effector,
                       color="crimson", s=14, zorder=5, marker="x",
                       linewidths=1.2, label="efector")

        _apply_global_bounds(ax, g_center, g_r)

        ax.set_xlabel("X", fontsize=6, labelpad=0)
        ax.set_ylabel("Y", fontsize=6, labelpad=0)
        ax.set_zlabel("Z", fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5)

        conv_str = "OK" if res.converged else "NO"
        lo = caso.get("lim_ok")
        lim_str = "--" if lo is None else ("OK" if lo else "NO")
        nwt_str = "OK" if caso.get("newton_conv") else "NO"
        ax.set_title(
            f"FK-{idx+1} | conv={conv_str} lim={lim_str} nwt={nwt_str}\n"
            f"i={res.iterations}  e={res.final_error*1000:.1f}mm",
            fontsize=7, pad=2,
        )
        ax.legend(fontsize=5, loc="upper left", markerscale=1.0,
                  borderpad=0.2, labelspacing=0.15, handlelength=1.0)

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=1.5)
    plt.show()


def plot_chained_battery(solver, casos):
    """
    Muestra la trayectoria encadenada: para cada paso dibuja la posicion
    anterior del efector y la posicion final alcanzada.
    """
    n      = len(casos)
    n_cols = 4
    n_rows = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 5.8 * n_rows))
    fig.suptitle("Bateria 2 - Robustez encadenada (Niryo One)", fontsize=11, y=0.995)

    # Escala uniforme: mismo bounding box para todos los subplots
    g_center, g_r = _compute_global_bounds(casos)

    for idx, caso in enumerate(casos):
        ax  = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        res = caso["result"]

        _draw_chain(ax, res.joint_positions)

        ax.scatter(*solver.base_position, color="black",  s=12, zorder=5,
                   label="base")
        ax.scatter(*caso["desde"],        color="orange", s=14, zorder=5,
                   marker="^", label="ant.")

        if res.converged:
            ax.scatter(*caso["target"],
                       color="limegreen", edgecolors="crimson", linewidths=1.2,
                       s=45, zorder=6, marker="*",
                       label="target=efector")
        else:
            ax.scatter(*caso["target"],
                       color="green", s=14, zorder=5, marker="o",
                       label="target")
            ax.scatter(*res.end_effector,
                       color="crimson", s=14, zorder=5, marker="x",
                       linewidths=1.2, label="efector")

        _apply_global_bounds(ax, g_center, g_r)

        ax.set_xlabel("X", fontsize=6, labelpad=0)
        ax.set_ylabel("Y", fontsize=6, labelpad=0)
        ax.set_zlabel("Z", fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5)

        conv_str = "OK" if res.converged else "NO"
        lo = caso.get("lim_ok")
        lim_str = "--" if lo is None else ("OK" if lo else "NO")
        nwt_str = "OK" if caso.get("newton_conv") else "NO"
        ax.set_title(
            f"T-{idx+1} | conv={conv_str} lim={lim_str} nwt={nwt_str}\n"
            f"i={res.iterations}  e={res.final_error*1000:.1f}mm",
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

    yaml_path = os.path.join(_PROJECT_ROOT, "config", "robot-niryo.yaml")
    print(f"Cargando robot desde: {yaml_path}")
    robot = cargar_robot_desde_yaml(yaml_path)
    if robot is None:
        print("ERROR: No se pudo cargar el robot.")
        sys.exit(1)

    print(robot)

    constraint_policy = os.getenv("FABRIK_CONSTRAINT_POLICY", "transitional").strip().lower()
    debug_hinge = os.getenv("FABRIK_HINGE_DEBUG", "0").strip().lower() in ("1", "true", "yes")
    print("\nConstruyendo FabrikSerialSolver...")
    print(f"Politica de constraints: {constraint_policy}")
    solver = FabrikSerialSolver.from_robot(robot, constraint_policy=constraint_policy)
    if hasattr(solver, "enable_hinge_debug"):
        solver.enable_hinge_debug(debug_hinge)
    if debug_hinge:
        print("Debug HINGE backward saturations: enabled")
    print(solver)

    # Bateria 1: targets con FK garantizada alcanzables
    casos_fk = run_fk_battery(robot, solver, N_FK_TARGETS)

    # Bateria 2: robustez encadenada
    casos_chain = run_chained_battery(robot, solver, N_CHAIN_TARGETS)

    plot_fk_battery(solver, casos_fk)
    plot_chained_battery(solver, casos_chain)


if __name__ == "__main__":
    main()

