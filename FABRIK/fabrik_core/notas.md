1. El Objetivo Original (La Idea)
El propósito: Implementar un solver de Cinemática Inversa (IK) que fuera extremadamente rápido, computacionalmente ligero (sin inversión de matrices Jacobianas) y fácil de implementar para un brazo robótico real (Niryo One de 6-DOF).
El punto de partida: El algoritmo FABRIK clásico. Es el estándar de oro en videojuegos y animación por su velocidad y simplicidad geométrica.

2. El Choque con la Realidad (El Problema)
La limitación teórica: FABRIK fue diseñado para gráficos 3D, asumiendo que las articulaciones son esféricas (Ball-and-Socket, 3-DOF). El algoritmo simplemente tira de puntos en el espacio a lo largo de líneas rectas.
La restricción física: Los robots industriales como el Niryo One están compuestos por bisagras y juntas de revolución de 1-DOF. Al intentar forzar a FABRIK a respetar un único plano de rotación estricto, el algoritmo colapsa:

Se rompen las distancias de los eslabones al proyectar los puntos de forma ingenua.

Se producen violaciones "fantasma" de los límites porque los ejes de las bisagras cambian con la postura del robot.

El algoritmo entra en bucles infinitos (deadlocks o chattering) al rebotar contra los topes mecánicos.

3. La Solución Implementada (Lo que tienes hecho)
Para resolver esto, implementaste FABRIK-R (basado en los papers recientes de Santos et al., 2021/2022). Tu código actual cuenta con un motor geométrico riguroso:

Marcos de Referencia Dinámicos: Has reemplazado los ejes estáticos por propagación de Cinemática Directa (FK) acumulativa. El solver ahora entiende que si el hombro gira, el plano de rotación del codo gira con él en el espacio global.

Proyección Estricta: En cada pasada, los vectores se proyectan ortogonalmente al plano de la bisagra y se re-normalizan, garantizando que el error de longitud de los eslabones se mantenga en cero absoluto (0.00e+00).

Gestión de Límites Geométricos: Se implementó la fórmula de rotación de Rodrigues para realizar el clamping angular directamente en el plano válido, extrayendo ángulos finales idénticos a los que requiere el hardware del robot.

Mecanismo Anti-Estancamiento (Fase 7): Un sistema de paciencia adaptativa que detecta cuando el brazo se atasca contra sus límites físicos. Introduce micro-perturbaciones (reseed) para intentar escapar del mínimo local y, si falla, devuelve la mejor postura segura alcanzada (best_state_fallback).

4. Resultados (La Realidad de los Datos)
Lo positivo (Robustez Matemática):

Geometría perfecta: La ortogonalidad y la longitud de los eslabones no sufren degradación numérica.

Seguridad del hardware: El algoritmo respeta los límites articulares rigurosamente y no genera comandos imposibles.

Estabilidad: Ya no hay bucles infinitos; el algoritmo sabe cuándo rendirse ordenadamente.

El área de mejora (La Convergencia Estricta):

La tasa de convergencia estricta en rutas encadenadas complejas es baja (4 de 12 casos alcanzan la tolerancia milimétrica exacta).

El solver recurre frecuentemente al fallback.

5. El Análisis Crítico (El "Por qué")
Aquí es donde demuestras tu comprensión del problema:
FABRIK-R es un algoritmo codicioso (greedy). Toma decisiones óptimas locales en cada articulación durante las pasadas Forward/Backward. Cuando un target obliga a una articulación a saturar su límite físico, la cadena pierde grados de libertad. Como el algoritmo no tiene una visión global del gradiente de error (algo que sí tiene un método Jacobiano), le cuesta encontrar las configuraciones "exóticas" donde retroceder una junta permitiría avanzar a otra para alcanzar el objetivo final.

6. Próximos Pasos (El Futuro del Proyecto)
Para concluir, puedes plantearle a tu tutor las vías que tienes pensadas para mejorar esa tasa de convergencia en las próximas semanas:

Afinar el Reseed Heurístico: Mejorar la lógica de perturbación para que el brazo "salte" inteligentemente los mínimos locales.

Tolerancias Blandas (Soft-Tolerance): Implementar criterios donde quedarse a 3 milímetros del objetivo por un bloqueo mecánico se considere un éxito funcional de agarre, en lugar de un fallo matemático.

Enfoque Híbrido: Utilizar FABRIK-R como un generador de postura inicial ultrarrápido (warm-start) y dejar que un solver Jacobiano clásico refine los últimos milímetros, combinando lo mejor de ambos mundos.