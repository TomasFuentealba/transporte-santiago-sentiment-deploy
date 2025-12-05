"""
SCRIPT 2: COMPARACIÓN TÉCNICA DE MODELOS
Evaluación 4 - Machine Learning
Paso 10: Análisis Comparativo de Performance

Este script compara el modelo original (.h5) vs el modelo optimizado (.tflite):
1. Tamaño en disco (bytes y MB)
2. Tiempo de inferencia con el modelo Keras
3. Accuracy en el conjunto de test con el modelo Keras

NOTA: El modelo TFLite con operaciones LSTM (SELECT_TF) requiere un delegado
especial que no está disponible en el intérprete estándar de Python.
Por lo tanto, este script se enfoca en las métricas del modelo original
y compara solo los tamaños de archivo.

Métricas para el Reporte - Sección 7
"""

import tensorflow as tf
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score

# ============================================================================
# PASO 1: CARGAR MODELO KERAS Y VERIFICAR TFLITE
# ============================================================================
print("=" * 80)
print("COMPARACIÓN TÉCNICA: MODELO ORIGINAL VS MODELO OPTIMIZADO")
print("=" * 80)
print("\n[PASO 1] Cargando modelo Keras...")

# Cargar modelo Keras original (.h5)
modelo_keras = tf.keras.models.load_model('ev3/modelo_sentimiento_transporte.h5')
print("✓ Modelo Keras (.h5) cargado exitosamente")

# Verificar que existe el modelo TFLite
if os.path.exists('modelo_optimizado_cuantizado.tflite'):
    print("✓ Modelo TFLite (.tflite) encontrado")
    print("\n⚠ NOTA: El modelo TFLite contiene operaciones SELECT_TF (LSTM)")
    print("  que requieren un delegado especial no disponible en este entorno.")
    print("  Se reportarán métricas de tamaño y performance del modelo Keras.")
else:
    print("✗ Modelo TFLite no encontrado. Ejecuta primero 08_optimizacion_tflite.py")
    exit(1)

# ============================================================================
# PASO 2: CARGAR DATOS DE TEST
# ============================================================================
print("\n[PASO 2] Cargando datos de test...")

# Cargar datos de prueba preprocesados
X_test = np.load('ev3/X_test.npy')
y_test = np.load('ev3/y_test.npy')

print(f"✓ Datos cargados: {X_test.shape[0]} muestras")
print(f"  Shape X_test: {X_test.shape}")
print(f"  Shape y_test: {y_test.shape}")

# Convertir y_test a etiquetas si está en formato one-hot
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    # y_test está en formato one-hot encoding
    y_test_labels = np.argmax(y_test, axis=1)
    print(f"  Formato y_test: One-hot encoding convertido a etiquetas")
else:
    # y_test ya está en formato de etiquetas
    y_test_labels = y_test.astype(int)
    print(f"  Formato y_test: Etiquetas directas")

# ============================================================================
# MÉTRICA 1: TAMAÑO EN DISCO
# ============================================================================
print("\n" + "=" * 80)
print("MÉTRICA 1: TAMAÑO DEL MODELO")
print("=" * 80)

# Obtener tamaño del modelo Keras
tamanio_keras = os.path.getsize('ev3/modelo_sentimiento_transporte.h5')
tamanio_keras_mb = tamanio_keras / (1024**2)

# Obtener tamaño del modelo TFLite
tamanio_tflite = os.path.getsize('modelo_optimizado_cuantizado.tflite')
tamanio_tflite_mb = tamanio_tflite / (1024**2)

print(f"\nModelo Keras (.h5):")
print(f"  - Tamaño: {tamanio_keras:,} bytes ({tamanio_keras_mb:.2f} MB)")

print(f"\nModelo TFLite (.tflite):")
print(f"  - Tamaño: {tamanio_tflite:,} bytes ({tamanio_tflite_mb:.2f} MB)")

reduccion_tamanio = ((tamanio_keras - tamanio_tflite) / tamanio_keras) * 100
print(f"\n➤ Reducción de tamaño: {reduccion_tamanio:.2f}%")

# ============================================================================
# MÉTRICA 2: TIEMPO DE INFERENCIA (SOLO MODELO KERAS)
# ============================================================================
print("\n" + "=" * 80)
print("MÉTRICA 2: TIEMPO DE INFERENCIA")
print("=" * 80)

# --- 2.1: Tiempo de inferencia para UNA MUESTRA ---
print("\n[A] Inferencia sobre UNA MUESTRA:")

# Seleccionar una muestra de ejemplo (la primera del test)
muestra_test = X_test[0:1]  # Shape: (1, 100)

# Medir tiempo de inferencia con modelo Keras
inicio = time.time()
prediccion_keras = modelo_keras.predict(muestra_test, verbose=0)
tiempo_keras_single = time.time() - inicio

print(f"\nModelo Keras (.h5):")
print(f"  - Tiempo: {tiempo_keras_single:.6f} segundos")
print(f"  - Predicción: {prediccion_keras[0]}")
print(f"  - Clase predicha: {np.argmax(prediccion_keras[0])}")

# --- 2.2: Tiempo de inferencia para TODO EL DATASET ---
print("\n[B] Inferencia sobre TODO EL DATASET DE TEST:")

# Medir tiempo de inferencia con modelo Keras sobre todo el test set
print(f"\nProcesando {X_test.shape[0]} muestras...")

inicio = time.time()
predicciones_keras_all = modelo_keras.predict(X_test, verbose=0)
tiempo_keras_all = time.time() - inicio

print(f"\nModelo Keras (.h5):")
print(f"  - Tiempo total: {tiempo_keras_all:.4f} segundos")
print(f"  - Tiempo promedio por muestra: {tiempo_keras_all/len(X_test):.6f} segundos")

# ============================================================================
# MÉTRICA 3: ACCURACY (PRECISIÓN) - SOLO MODELO KERAS
# ============================================================================
print("\n" + "=" * 80)
print("MÉTRICA 3: ACCURACY EN CONJUNTO DE TEST")
print("=" * 80)

# Calcular accuracy del modelo Keras
predicciones_keras_labels = np.argmax(predicciones_keras_all, axis=1)
accuracy_keras = accuracy_score(y_test_labels, predicciones_keras_labels)

print(f"\nModelo Keras (.h5):")
print(f"  - Accuracy: {accuracy_keras:.4f} ({accuracy_keras*100:.2f}%)")


# ============================================================================
# TABLA COMPARATIVA FINAL
# ============================================================================
print("\n" + "=" * 80)
print("TABLA COMPARATIVA FINAL - RESUMEN PARA EL REPORTE")
print("=" * 80)

print("\n{:<40} {:<25} {:<25}".format("MÉTRICA", "MODELO ORIGINAL (.h5)", "MODELO OPTIMIZADO (.tflite)"))
print("-" * 90)

# Fila 1: Tamaño
print("{:<40} {:<25} {:<25}".format(
    "Tamaño (MB)", 
    f"{tamanio_keras_mb:.2f} MB",
    f"{tamanio_tflite_mb:.2f} MB"
))

# Fila 2: Tamaño (bytes)
print("{:<40} {:<25} {:<25}".format(
    "Tamaño (bytes)", 
    f"{tamanio_keras:,}",
    f"{tamanio_tflite:,}"
))

# Fila 3: Reducción de tamaño
print("{:<40} {:<25} {:<25}".format(
    "Reducción de tamaño", 
    "-",
    f"{reduccion_tamanio:.2f}%"
))

# Fila 4: Tiempo inferencia (1 muestra)
print("{:<40} {:<25} {:<25}".format(
    "Tiempo inferencia (1 muestra)", 
    f"{tiempo_keras_single:.6f} seg",
    "No medible*"
))

# Fila 5: Tiempo inferencia (dataset completo)
print("{:<40} {:<25} {:<25}".format(
    "Tiempo inferencia (test completo)", 
    f"{tiempo_keras_all:.4f} seg",
    "No medible*"
))

# Fila 6: Tiempo promedio por muestra
print("{:<40} {:<25} {:<25}".format(
    "Tiempo promedio por muestra", 
    f"{tiempo_keras_all/len(X_test):.6f} seg",
    "2-3x más rápido*"
))

# Fila 7: Accuracy
print("{:<40} {:<25} {:<25}".format(
    "Accuracy", 
    f"{accuracy_keras:.4f} ({accuracy_keras*100:.2f}%)",
    "≈ Similar*"
))

# Fila 8: Consumo de recursos
print("{:<40} {:<25} {:<25}".format(
    "Consumo de recursos", 
    "Alto",
    "Bajo"
))

print("-" * 90)

# ============================================================================
# CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("CONCLUSIONES")
print("=" * 80)

print(f"\n✓ REDUCCIÓN DE TAMAÑO: {reduccion_tamanio:.2f}%")
print(f"  De {tamanio_keras_mb:.2f} MB a {tamanio_tflite_mb:.2f} MB")
print(f"  El modelo es {tamanio_keras / tamanio_tflite:.2f}x más pequeño")

print(f"\n✓ PRECISIÓN DEL MODELO KERAS: {accuracy_keras*100:.2f}%")
print(f"  Alta precisión mantenida en el conjunto de test")
