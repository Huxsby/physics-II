import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Valores dados
L1 = 1
L2 = 0.2
m = 0.5
b = 0.2

# Definir la ecuación g(θ1, θ2) = 0
def funcion_g(rad_tita_2, rad_tita_1):
    return m * (np.cos(rad_tita_1) + L2 * np.cos(rad_tita_1 + rad_tita_2)) - \
           (np.sin(rad_tita_1) + L2 * np.sin(rad_tita_1 + rad_tita_2)) + b

# Lista de valores de θ1 en grados y radianes
tita_1 = np.arange(0, 361, 30)  # de 0° a 360° cada 30°
rad_tita_1 = np.radians(tita_1)  # Convertir a radianes

# Resolver para θ2
soluciones_tita2 = []
for t1 in rad_tita_1:
    # Resolver g(t1, t2) = 0 con una estimación inicial
    sol = fsolve(funcion_g, x0=[0], args=(t1))  # x0=[0] como punto de inicio
    soluciones_tita2.append(np.degrees(sol[0]))  # Convertir a grados

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(tita_1, soluciones_tita2, marker='o', linestyle='-', color='b', label=r'Solución $\theta_2$')
plt.xlabel(r'$\theta_1$ (grados)')
plt.ylabel(r'$\theta_2$ (grados)')
plt.title(r'Solución de $\theta_2$ en función de $\theta_1$')
plt.legend()
plt.grid()
plt.show()