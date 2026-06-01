# AGENTS.md - Gobernanza del Subárbol FABRIK-R

## 1) Misión y Enfoque Directivo
Este documento rige de forma estricta las acciones de cualquier agente autónomo o desarrollador que trabaje en el subárbol `FABRIK/`. El objetivo principal es migrar el sistema desde el paradigma clásico de FABRIK (orientado a Computer Graphics y articulaciones esféricas de 3-DOF) hacia el paradigma de **FABRIK-R** (Santos et al., 2021, 2022), diseñado específicamente para manipuladores robóticos reales compuestos por articulaciones de 1-DOF (revolutas/bisagras) bajo restricciones mecánicas estrictas.

Queda estrictamente prohibido utilizar atajos de conveniencia heurística o aproximaciones que rompan las restricciones físicas intrínsecas de los enlaces de 1-DOF. Toda modificación de código debe ser matemáticamente consistente con las formulaciones de los papers normativos.

## 2) Fuentes Normativas (Orden de Prioridad Estricto)
1. **SANTOS21 (FABRIK-R Core):** Santos, M. C., et al. "FABRIK-R: An Extension Developed Based on FABRIK for Robotics Manipulators." *IEEE Access*, vol. 9, 2021. DOI: 10.1109/ACCESS.2021.3070693.
2. **SANTOS22 (Subsea Application):** Santos, P. C., et al. "Inverse kinematics of a subsea constrained manipulator based on FABRIK-R." *OCEANS 2022, Hampton Roads*. DOI: 10.1109/OCEANS47191.2022.9977290.
3. **AL11 (Base Histórica):** Aristidou, A., Lasenby, J. "FABRIK: A fast, iterative solver for the Inverse Kinematics problem." *Graphical Models*, 2011. (Solo como referencia del bucle fundamental Forward/Backward).

## 3) Reglas de Oro para el Agente (Líneas Rojas)
* **No Suposiciones Globales:** Las articulaciones tipo *Hinge* de 1-DOF operan en sistemas de coordenadas locales que rotan solidariamente con los eslabones anteriores. Tratar un eje de bisagra como un vector estático global `[0, 0, 1]` será motivo de rechazo automático del código.
* **Preservación de Longitudes:** Tras cualquier proyección geométrica sobre el plano de rotación permitido de la articulación, la distancia euclídea entre $p_i$ y $p_{i-1}$ DEBE ser re-normalizada exactamente a la longitud nominal del eslabón ($l_i$). Las distorsiones de longitud en la cadena cinemática destruyen la convergencia.
* **Sincronización de Documentación:** No se aceptará ninguna línea de código matemático en `fabrik_r_solver.py` que no contenga un comentario explícito citando la sección o ecuación correspondiente de **SANTOS21** o **SANTOS22**.

## 4) Protocolo de Limpieza y Aislamiento del Entorno
Antes de escribir la nueva implementación de FABRIK-R, el agente debe ejecutar las siguientes acciones de ordenamiento en el espacio de trabajo:
1. Crear un directorio llamado `FABRIK/legacy/`.
2. Mover los archivos de la sesión anterior (`fabrik_serial_solver.py`, `fabrik_paper_constrained_3d.py`) a dicha carpeta.
3. Actualizar `FABRIK_README.md` para reflejar el estado "Archivado" de la versión clásica y enlazar al nuevo `FABRIK_R_README.md`.

## 5) Orden de Ejecución de Tareas (Roadmap de Actualización)
El agente procesará la refactorización siguiendo este orden secuencial inalterable:
* **Fase 1: Aislamiento del Workspace:** Aplicar el protocolo de limpieza del punto 4.
* **Fase 2: Implementación Matemática Base:** Programar las utilidades de proyección planar de FABRIK-R empleando matrices de rotación locales y cuaterniones.
* **Fase 3: Bucle de Solución FABRIK-R:** Desarrollar el script de producción `fabrik_r_solver.py` con las pasadas modificadas (Forward/Backward de 1-DOF).
* **Fase 4: Restricciones de Límites de Ángulo:** Integrar el clamping angular directo sobre el plano cinemático según las secciones de límites físicos de SANTOS22.
* **Fase 5: Extracción de Ángulos de Articulación:** Implementar el algoritmo de lectura final de variables de junta ($	heta_i$) para su envío al hardware físico.
* **Fase 6: Batería de Tests:** Validar contra perfiles de robots con articulaciones secuenciales ortogonales (estilo Niryo One o Schilling Titan 2).

## 6) Criterios de Rechazo de Código
El código será rechazado de inmediato si:
1. Sufre de "deadlocks" o bucles infinitos en configuraciones alineadas (singularidades de plano).
2. Utiliza funciones de optimización genéricas de caja negra (como `scipy.optimize`) dentro del bucle interno de FABRIK-R para resolver las proyecciones, rompiendo la naturaleza analítica y rápida del algoritmo.
3. No maneja la consistencia de signos ($sgn$) al proyectar vectores en orientaciones opuestas (antiparalelas).