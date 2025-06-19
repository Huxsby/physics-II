
library(ggplot2)
library(broom) # Para tidy()

# Leer los datos
datos <- read.csv('tiempos_workspace.csv')

# Ajustar un modelo lineal (puedes probar otros, ej. cuadrático: y ~ x + I(x^2))
# Para predecir tiempo (y) en función de N (x)
modelo <- lm(tiempo ~ N, data=datos)

# Obtener resumen y coeficientes
resumen_modelo <- summary(modelo)
coeficientes <- tidy(modelo)

print('Resumen del Modelo Lineal:')
print(resumen_modelo)
print('Coeficientes del Modelo:')
print(coeficientes)

# Extraer la ecuación para el título del gráfico
intercepto <- coef(modelo)[1]
pendiente_N <- coef(modelo)[2]
ecuacion <- sprintf("Tiempo ≈ %.4f + %.4f * N (R² = %.3f)", 
                    intercepto, pendiente_N, resumen_modelo$r.squared)

# Crear el gráfico con ggplot2
grafico <- ggplot(datos, aes(x=N, y=tiempo)) +
  geom_point(color='blue', size=3) +
  geom_smooth(method='lm', se=TRUE, color='red', formula=y ~ x) + # Línea de regresión
  labs(title=paste("Predicción del Tiempo de Cálculo de graficar_workspace", ecuacion, sep="
"),
       x="Número de Puntos (N)",
       y="Tiempo de Cálculo Promedio (s)") +
  theme_minimal()

# Guardar el gráfico
ggsave('c:\\Users\\Huxsby\\Documents\\repgit\\physics-II\\prediccion_tiempo_workspace.png', plot=grafico, width=10, height=6)

print(paste("Gráfico guardado en: c:\\Users\\Huxsby\\Documents\\repgit\\physics-II\\prediccion_tiempo_workspace.png"))
