# ==========================================================
# PRACTICA: RED NEURONAL PARA CLASIFICAR ESTUDIANTES
# Unidad 3. Redes Neuronales
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------
# 1. DATOS DE ENTRENAMIENTO
# Cada fila representa un estudiante.
# Columna 1 = actividades.
# Columna 2 = examen.
# ----------------------------------------------------------

# Permite reproducir los mismos resultados
np.random.seed(42)

# Crear matriz de datos con 24 estudiantes de 4 posiciones
datos = np.array([
    [90, 85], 
], dtype=float)


# ----------------------------------------------------------
# 2. GENERAR LAS ETIQUETAS DE ENTRENAMIENTO
# ----------------------------------------------------------
# Calcular el promedio ponderado:
# Actividades = 40%
# Examen = 60%

promedios = (
    datos[:, 0] * 0.40 +
    datos[:, 1] * 0.60
)

# Clasificacion:
# 1 = satisfactorio
# 0 = requiere atencion

y = (promedios >= 70).astype(float)

# Convertir las etiquetas en una matriz columna
y = y.reshape(-1, 1)


# ----------------------------------------------------------
# 3. NORMALIZAR LOS DATOS
# Convertir calificaciones de 0-100 a 0-1
# ----------------------------------------------------------
X = datos / 100.0

print("       DATOS DE ENTRENAMIENTO")
print("Total de estudiantes:", len(datos))
print("Estudiantes satisfactorios:",
      int(np.sum(y == 1)))
print("Estudiantes que requieren atencion:",
      int(np.sum(y == 0)))

# ----------------------------------------------------------
# 4. ARQUITECTURA DE LA RED NEURONAL
# 2 entradas, 5 neuronas ocultas, 1 salida
# ----------------------------------------------------------

# Crear pesos de entrada a capa oculta
W1 = np.random.randn(2, 5) * np.sqrt(2 / 7)

# Crear sesgos de la capa oculta
b1 = np.zeros((1, 5))

# Crear pesos de capa oculta a salida
W2 = np.random.randn(5, 1) * np.sqrt(2 / 6)

# Crear sesgo de la capa de salida
b2 = np.zeros((1, 1))

# Mostrar arquitectura
print("----ARQUITECTURA DE LA RED"------)
print("Neuronas de entrada: 2")
print("Neuronas ocultas: 5")
print("Neuronas de salida: 1")

# ----------------------------------------------------------
# 5. FUNCION DE ACTIVACION SIGMOIDE
# ----------------------------------------------------------
def sigmoid(x):

    # Evitar desbordamientos numericos
    x = np.clip(x, -500, 500)

    # Aplicar funcion sigmoide
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------
# 6. FUNCION PARA CALCULAR EL ERROR
# Entropia cruzada binaria
# ----------------------------------------------------------
def calcular_error(y_real, y_predicha):

    # Evitar logaritmos de cero
    epsilon = 1e-9

    # Calcular el error
    error = -np.mean(
        y_real * np.log(y_predicha + epsilon) +
        (1 - y_real) * np.log(1 - y_predicha + epsilon)
    )
    return error

# ----------------------------------------------------------
# 7. ENTRENAMIENTO DE LA RED NEURONAL
# Algoritmo Backpropagation
# ----------------------------------------------------------
# Numero de epocas
epocas = 10000

# Tasa de aprendizaje
tasa_aprendizaje = 1.0

# Numero de estudiantes de entrenamiento
m = len(X)

# Lista para almacenar los errores
historial_error = []

print("       INICIANDO ENTRENAMIENTO")

# Iniciar ciclo de entrenamiento
for epoca in range(epocas):

    # --------------------------------------
    # 7.1 PROPAGACION HACIA ADELANTE
    # --------------------------------------
    # Calcular entradas de la capa oculta
    z1 = np.dot(X, W1) + b1

    # Aplicar funcion de activacion
    a1 = sigmoid(z1)

    # Calcular entradas de la capa de salida
    z2 = np.dot(a1, W2) + b2

    # Obtener prediccion final
    a2 = sigmoid(z2)


    # --------------------------------------
    # 7.2 CALCULAR EL ERROR
    # --------------------------------------
    error = calcular_error(y, a2)

    # Guardar error
    historial_error.append(error)

    # --------------------------------------
    # 7.3 BACKPROPAGATION
    # Gradiente de la capa de salida
    # --------------------------------------

    dz2 = (a2 - y) / m

    # Gradiente de los pesos de salida
    dW2 = np.dot(a1.T, dz2)

    # Gradiente de los sesgos de salida
    db2 = np.sum(
        dz2,
        axis=0,
        keepdims=True
    )


    # --------------------------------------
    # 7.4 PROPAGAR ERROR A CAPA OCULTA
    # --------------------------------------
    dz1 = (
        np.dot(dz2, W2.T) *
        a1 *
        (1 - a1)
    )

    # Gradiente de los pesos ocultos
    dW1 = np.dot(X.T, dz1)

    # Gradiente de los sesgos ocultos
    db1 = np.sum(
        dz1,
        axis=0,
        keepdims=True
    )

    # --------------------------------------
    # 7.5 ACTUALIZAR PESOS Y SESGOS
    # Descenso del gradiente
    # --------------------------------------
    W2 -= tasa_aprendizaje * dW2
    b2 -= tasa_aprendizaje * db2
    W1 -= tasa_aprendizaje * dW1
    b1 -= tasa_aprendizaje * db1

    # --------------------------------------
    # 7.6 MOSTRAR AVANCES
    # --------------------------------------
    if (epoca + 1) in [
        1, 100, 500, 1000,
        2000, 5000, 10000
    ]:

        # Obtener predicciones de entrenamiento
        predicciones_entrenamiento = (
            a2 >= 0.5
        ).astype(int)

        # Calcular exactitud
        exactitud = np.mean(
            predicciones_entrenamiento == y
        )

        print(
            f"Epoca: {epoca + 1:5d} | "
            f"Error: {error:.6f} | "
            f"Exactitud: {exactitud * 100:.2f}%"
        )


print("\nEntrenamiento finalizado.")

# ----------------------------------------------------------
# 8. MOSTRAR GRAFICA DEL ERROR
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(historial_error)
plt.title(
    "Error durante el entrenamiento de la red neuronal"
)
plt.xlabel("Epocas")
plt.ylabel("Entropia cruzada binaria")
plt.grid(True)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# 9. FUNCION PARA SOLICITAR CALIFICACIONES
# ----------------------------------------------------------
def leer_calificacion(mensaje):
    while True:
        try:

            # Solicitar dato al usuario
            calificacion = float(
                input(mensaje).strip().replace(",", ".")
            )

            # Validar que sea un numero entre 0 y 100
            if (
                np.isfinite(calificacion) and
                0 <= calificacion <= 100
            ):
                return calificacion
            else:
                print(
                    "Error: la calificacion debe "
                    "estar entre 0 y 100."
                )
        except ValueError:
            print(
                "Error: debes ingresar un valor numerico."
            )

# ----------------------------------------------------------
# 10. CAPTURAR CINCO ESTUDIANTES NUEVOS
# ----------------------------------------------------------
print("\n==========================================")
print("    CAPTURA DE CINCO ESTUDIANTES NUEVOS")
print("==========================================")

print("Ingresa las calificaciones de cada estudiante.")
print("Los valores deben estar entre 0 y 100.")

# Lista para almacenar los estudiantes
estudiantes_nuevos = []

# Solicitar datos de cinco estudiantes
for i in range(5):
    print(f"ESTUDIANTE {i + 1}")  

    # Solicitar calificacion de actividades
    actividades = leer_calificacion( "Calificacion de actividades: ")

    # Solicitar calificacion del examen
    examen = leer_calificacion( "Calificacion del examen: ")

    # Guardar los datos
    estudiantes_nuevos.append([
        actividades,
        examen
    ])

# Convertir lista en matriz NumPy
estudiantes_nuevos = np.array(
    estudiantes_nuevos,
    dtype=float
)

# ----------------------------------------------------------
# 11. NORMALIZAR LOS NUEVOS DATOS
# ----------------------------------------------------------
# Convertir calificaciones a valores de 0-1
X_nuevos = estudiantes_nuevos / 100.0

# ----------------------------------------------------------
# 12. PROPAGACION HACIA ADELANTE
# UTILIZAR PESOS APRENDIDOS
# ----------------------------------------------------------

# Calcular capa oculta
z1_nuevos = np.dot(X_nuevos, W1) + b1
a1_nuevos = sigmoid(z1_nuevos)

# Calcular capa de salida
z2_nuevos = np.dot(a1_nuevos, W2) + b2

# Obtener probabilidades
probabilidades = sigmoid(z2_nuevos).flatten()

# ----------------------------------------------------------
# 13. GENERAR PREDICCIONES
# ----------------------------------------------------------
# Aplicar umbral de 0.5

# 1 = satisfactorio
# 0 = requiere atencion
predicciones = (
    probabilidades >= 0.5
).astype(int)

# ----------------------------------------------------------
# 14. CALCULAR CLASIFICACIONES ESPERADAS
# ----------------------------------------------------------
# Calcular promedio ponderado de cada estudiante
promedios_nuevos = (
    estudiantes_nuevos[:, 0] * 0.40 +
    estudiantes_nuevos[:, 1] * 0.60
)

# Generar clasificaciones esperadas
esperados = (
    promedios_nuevos >= 70
).astype(int)

# ----------------------------------------------------------
# 15. MOSTRAR RESULTADOS
# ----------------------------------------------------------
print("---------------RESULTADOS DE LOS ESTUDIANTES------------------")

# Encabezados de la tabla
print(
    f"{'No.':<5}"
    f"{'Activ.':<10}"
    f"{'Examen':<10}"
    f"{'Promedio':<11}"
    f"{'Prob.':<10}"
    f"{'Esperado':<22}"
    f"{'Prediccion':<22}"
    f"{'Comparacion'}"
)

print("-" * 110)
# Recorrer los cinco estudiantes

for i in range(5):
    # Obtener datos del estudiante
    actividades = estudiantes_nuevos[i, 0]
    examen = estudiantes_nuevos[i, 1]
    promedio = promedios_nuevos[i]
    probabilidad = probabilidades[i]
    esperado = esperados[i]
    predicho = predicciones[i]


    # Clasificacion esperada
    if esperado == 1:
        resultado_esperado = "SATISFACTORIO"
    else:
        resultado_esperado = "REQUIERE ATENCION"


    # Prediccion de la red neuronal
    if predicho == 1:
        resultado_predicho = "SATISFACTORIO"
    else:
        resultado_predicho = "REQUIERE ATENCION"

    # Comparar resultados
    if esperado == predicho:
        comparacion = "CORRECTO"
    else:
        comparacion = "INCORRECTO"


    # Imprimir resultados
    print(
        f"{i + 1:<5}"
        f"{actividades:<10.1f}"
        f"{examen:<10.1f}"
        f"{promedio:<11.2f}"
        f"{probabilidad:<10.4f}"
        f"{resultado_esperado:<22}"
        f"{resultado_predicho:<22}"
        f"{comparacion}"
    )

# ----------------------------------------------------------
# 16. CALCULAR EXACTITUD DE LAS PRUEBAS
# ----------------------------------------------------------
exactitud_pruebas = np.mean(
    predicciones == esperados
) * 100

print("\n==============================================================")
print(
    f"Exactitud en los cinco casos de prueba: "
    f"{exactitud_pruebas:.2f}%"
)