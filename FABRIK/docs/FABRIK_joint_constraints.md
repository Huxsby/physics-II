# Restricciones de Articulación en FABRIK: Ball vs Hinge (Revolute)

**Fuentes primarias:**
- [AL11] Aristidou & Lasenby, "FABRIK: A fast, iterative solver for the IK problem", *Graphical Models* 73(5), 2011 (DOI: 10.1016/j.gmod.2011.05.003, local: docs/02.1-FABRIK.pdf)
- [ACL16] Aristidou, Chrysanthou & Lasenby, "Extending FABRIK with model constraints", *CAVW* 27(1), 2016 (DOI: 10.1002/cav.1630, local: docs/01.1.2-Extending FABRIK with model constraints.pdf)
- Implementación de referencia: `FABRIK_chain_3D-master` (CALIKO port en Python)

---

## 1. Tipos de Joint en FABRIK

FABRIK distingue tres tipos de articulación, con semánticas muy distintas:

### 1.1 BALL (ball-and-socket)
- **DoF**: 3 (rotación libre en cualquier dirección dentro de un cono)
- **Parámetros**: `ball_max` — ángulo máximo de deflexión desde la dirección del segmento anterior
- **Restricción geométrica**: el ángulo entre el segmento entrante y el segmento saliente debe ser ≤ `ball_max`
- **Implementación**: en cada paso (forward/backward), si el ángulo entre los vectores dirección excede `ball_max`, el vector se rota hacia el vector de referencia hasta el límite
- **Ejemplo anatómico**: hombro, cadera

### 1.2 GLOBAL_HINGE
- **DoF**: 1 (rotación en un plano fijo en el frame del mundo)
- **Parámetros**: `rotation_axis` (eje en frame global), `reference_axis` (dirección cero dentro del plano), `cw_degs`, `acw_degs`
- **Restricción geométrica**:
  1. El vector dirección del hueso se **proyecta** sobre el plano perpendicular al `rotation_axis` en frame global
  2. El ángulo signado (en ese plano) relativo a `reference_axis` se acota a `[−cw_degs, acw_degs]`
- **Cuándo usar**: cuando el eje de bisagra NO cambia con el movimiento de las articulaciones padre (e.g. base fija al suelo)

### 1.3 LOCAL_HINGE
- **DoF**: 1 (rotación en un plano relativo al hueso padre)
- **Parámetros**: igual que GLOBAL_HINGE pero los ejes están en el frame **local del hueso anterior**
- **Restricción geométrica**:
  1. El `rotation_axis` local se transforma al frame global usando la matriz de rotación del hueso anterior: `R = create_rotation_matrix(prev_bone_inner_to_outer_uv)`; luego `relative_axis = R @ local_rotation_axis`
  2. El vector dirección del hueso se proyecta sobre el plano perpendicular a `relative_axis`
  3. El ángulo signado se acota con los límites CW/ACW, usando `relative_reference_axis = R @ local_reference_axis`
- **Cuándo usar**: articulaciones revoluta que NO están en la base — su eje efectivo depende de la postura de las articulaciones padre

---

## 2. Por qué los joints revolutos NO son BALL

Las articulaciones revoluto (como todas las del Niryo One excepto la base) permiten **rotación en un único plano**. Tratarlas como BALL genera errores físicos:

| Propiedad | BALL (incorrecto) | HINGE (correcto) |
|-----------|-------------------|------------------|
| Plano de movimiento | Cualquier dirección dentro del cono | Solo en el plano perpendicular al eje |
| Ángulo que se limita | Deflexión local entre segmentos (ángulo de codo) | Ángulo signado dentro del plano de rotación |
| Relación con ángulo absoluto | **Ninguna** — se puede acumular rotación ilimitada | Directamente el ángulo de la articulación |
| Genera poses no físicas | Sí (el brazo se puede torcer lateralmente) | No |

**Consecuencia observada en el proyecto**: con BALL, el ángulo de deflexión local de J4 puede ser pequeño (e.g. 45°) mientras el ángulo absoluto de rotación medido por `joint_angles()` es grande (e.g. −143°). Esto ocurre porque BALL no restringe en qué plano se produce la deflexión.

---

## 3. Implementación correcta en FABRIK (según referencia CALIKO)

### Paso genérico (backward pass, articulación `i`)

```
prev_dir = inner_to_outer_uv del hueso (i-1)   # dirección del segmento anterior
this_dir = inner_to_outer_uv del hueso (i)      # dirección actual del segmento

if joint_type == BALL:
    angle = angle_between(prev_dir, this_dir)
    if angle > ball_max:
        this_dir = rotate(this_dir, toward=prev_dir, by=angle - ball_max)

elif joint_type == GLOBAL_HINGE:
    this_dir = project_onto_plane(this_dir, normal=rotation_axis)  # proyectar
    signed_angle = signed_angle(reference_axis, this_dir, rotation_axis)
    if signed_angle > acw_degs:
        this_dir = rotate(reference_axis, acw_degs, around=rotation_axis)
    elif signed_angle < -cw_degs:
        this_dir = rotate(reference_axis, -cw_degs, around=rotation_axis)

elif joint_type == LOCAL_HINGE:
    R = create_rotation_matrix(prev_dir)        # frame del hueso anterior
    rel_axis = normalize(R @ local_rotation_axis)
    rel_ref  = normalize(R @ local_reference_axis)
    this_dir = project_onto_plane(this_dir, normal=rel_axis)
    signed_angle = signed_angle(rel_ref, this_dir, rel_axis)
    if signed_angle > acw_degs:
        this_dir = rotate(rel_ref, acw_degs, around=rel_axis)
    elif signed_angle < -cw_degs:
        this_dir = rotate(rel_ref, -cw_degs, around=rel_axis)
```

El paso **forward** es idéntico pero con la dirección invertida (`outer_to_inner_uv`).

---

## 4. Aplicación al Niryo One

| Joint | Eje local | Tipo correcto en FABRIK | Tipo actual | Problema |
|-------|-----------|------------------------|-------------|----------|
| J0 Base | [0,0,1] (yaw) | GLOBAL_HINGE o caso especial | BALL (+ yaw postprocess) | Parcialmente correcto; se resuelve con `_last_theta0` |
| J1 Hombro | [0,−1,0] | **LOCAL_HINGE** | BALL | Deflexión lateral no fisica |
| J2 Brazo | [0,−1,0] | **LOCAL_HINGE** | BALL | Idem |
| J3 Codo | [1,0,0] | **LOCAL_HINGE** | BALL | Idem |
| J4 Antebrazo | [0,−1,0] | **LOCAL_HINGE** | BALL | Causa principal de violaciones de límites |
| J5 Muñeca | [1,0,0] | **LOCAL_HINGE** | BALL | Idem |

**¿Por qué LOCAL_HINGE y no GLOBAL_HINGE para J1–J5?**
Los ejes [0,−1,0] y [1,0,0] son ejes en el frame local de cada articulación. Cuando J0 rota, el eje efectivo de J1 rota con él. LOCAL_HINGE transforma el eje al frame global en cada iteración usando la dirección del segmento anterior, lo cual es físicamente correcto.

---

## 5. Estado actual del solver (`fabrik_serial_solver.py`)

El solver usa `JointType.BALL` para todos los joints (incluyendo los revolutos). Los intentos previos de usar `HINGE_LOCAL` y `HINGE_GLOBAL` causaron regresión de convergencia (5/8 y 7/8 vs 8/8 baseline BALL) porque la implementación de HINGE en el solver **no sigue el algoritmo de la referencia**:

- No usa `project_onto_plane` (proyección sobre el plano de la bisagra)
- No implementa `signed_angle` con un eje de referencia para acotar CW/ACW
- No transforma el eje local usando `create_rotation_matrix(prev_dir)` en LOCAL_HINGE

Una implementación correcta de LOCAL_HINGE requiere:
1. `project_on_to_plane(this_dir, normal=rel_axis)` — la proyección es la operación central
2. `get_signed_angle_between_degs(rel_ref, projected_dir, rel_axis)` — ángulo signado para CW/ACW
3. `create_rotation_matrix(prev_dir)` — construir la matriz de rotación del frame del hueso anterior

---

## 6. Funciones utilitarias necesarias (de `Utils.py` de referencia)

```python
def project_on_to_plane(v, normal):
    """Proyecta v sobre el plano perpendicular a normal."""
    n = normalize(normal)
    return normalize(v - dot(v, n) * n)

def get_signed_angle_between_degs(ref, v, axis):
    """Ángulo signado de ref a v en el plano perpendicular a axis."""
    unsigned = angle_between_degs(ref, v)
    cross = np.cross(ref, v)
    sign = np.sign(dot(cross, axis))
    return unsigned * (sign if sign != 0 else 1)

def create_rotation_matrix(inner_to_outer_uv):
    """Matriz de rotación que lleva [0,0,1] al vector dado (frame del hueso)."""
    # Implementación via Gram-Schmidt o quaternion
    ...
```

---

## 7. Pendiente

- [ ] Reimplementar `HINGE_LOCAL` en `fabrik_serial_solver.py` siguiendo el algoritmo de referencia (proyección + ángulo signado + transformación de eje por frame anterior)
- [ ] Añadir `project_onto_plane` y `get_signed_angle_between_degs` a `math_utils.py`
- [ ] Implementar `create_rotation_matrix(prev_dir)` para transformar ejes locales al frame global
- [ ] Reemplazar todos los joints J1–J5 del Niryo One por `LOCAL_HINGE` con sus ejes y límites correctos
- [ ] Re-evaluar convergencia con la implementación correcta
