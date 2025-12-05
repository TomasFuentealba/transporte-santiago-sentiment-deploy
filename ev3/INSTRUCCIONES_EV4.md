# INSTRUCCIONES PARA EJECUTAR LOS SCRIPTS DE LA EVALUACIÓN 4

## 📋 Requisitos Previos

Asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install flask tensorflow scikit-learn numpy
```

## 🚀 Ejecución de los Scripts

### SCRIPT 1: Optimización con TensorFlow Lite

Este script convierte el modelo Keras a TensorFlow Lite y aplica cuantificación.

```bash
cd ev3
python 08_optimizacion_tflite.py
```

**Salida esperada:**
- Archivo generado: `modelo_optimizado_cuantizado.tflite`
- Comparación de tamaños entre modelo original y optimizado
- Información sobre la reducción de peso

---

### SCRIPT 2: Comparación Técnica

Este script compara las métricas de performance entre ambos modelos.

```bash
cd ev3
python 09_comparacion_tecnica.py
```

**Métricas analizadas:**
1. **Tamaño en disco** (bytes y MB)
2. **Tiempo de inferencia** (una muestra y dataset completo)
3. **Accuracy** (precisión en el conjunto de test)

**Salida esperada:**
- Tabla comparativa con todas las métricas
- Conclusiones sobre la optimización
- Datos para incluir en el informe (Sección 7)

---

### SCRIPT 3: Aplicación Web Flask

Este script inicia una aplicación web para hacer predicciones en tiempo real.

```bash
cd ev3
python 10_app_flask.py
```

**Acceso a la aplicación:**
- Abrir navegador en: `http://localhost:5000`
- También disponible en: `http://127.0.0.1:5000`

**Características:**
- ✅ Interfaz web intuitiva y profesional
- ✅ Ingreso manual de reseñas
- ✅ Predicción de sentimiento en tiempo real
- ✅ Visualización de probabilidades
- ✅ API REST disponible en `/api/predict`

**Para tomar capturas de pantalla:**
1. Ingresa diferentes tipos de reseñas (positivas, negativas, neutras)
2. Captura la pantalla del formulario
3. Captura los resultados con las probabilidades
4. Usa estas imágenes en tu informe (Sección 8)

**Para detener el servidor:**
- Presiona `CTRL + C` en la terminal

---

## 📊 Ejemplos de Reseñas para Probar

### Reseñas Positivas:
- "El metro está muy limpio y el servicio es excelente"
- "Los buses llegan a tiempo y el personal es amable"
- "Gran mejora en el transporte público, muy eficiente"

### Reseñas Negativas:
- "El metro siempre está lleno y sucio"
- "Los buses nunca llegan a tiempo, pésimo servicio"
- "El transporte es muy caro y de mala calidad"

### Reseñas Neutras:
- "El transporte funciona pero podría mejorar"
- "A veces llega a tiempo, otras veces no"
- "Es un servicio normal, nada especial"

---

## 🔧 Solución de Problemas

### Error: "No module named 'flask'"
```bash
pip install flask
```

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "No se encuentra el archivo modelo_sentimiento_transporte.h5"
Asegúrate de estar en el directorio `ev3` donde se encuentran los archivos del modelo.

### El servidor Flask no inicia
Verifica que el puerto 5000 no esté siendo usado por otra aplicación. Puedes cambiar el puerto en `10_app_flask.py` modificando:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Cambiar a otro puerto
```

---

## 📝 Estructura de Archivos Generados

```
ev3/
├── 08_optimizacion_tflite.py          # Script de optimización
├── 09_comparacion_tecnica.py          # Script de comparación
├── 10_app_flask.py                    # Aplicación web Flask
├── templates/
│   └── index.html                     # Template HTML de la aplicación
├── modelo_sentimiento_transporte.h5   # Modelo original (ya existente)
├── modelo_optimizado_cuantizado.tflite # Modelo optimizado (generado)
├── tokenizer.pkl                      # Tokenizer (ya existente)
├── label_encoder.pkl                  # Label encoder (ya existente)
├── X_test.npy                         # Datos de test (ya existente)
└── y_test.npy                         # Etiquetas de test (ya existente)
```

---

## 📚 Información para el Informe

### Sección 7: Comparación Técnica
Ejecuta `09_comparacion_tecnica.py` y copia la tabla comparativa generada al final.

### Sección 8: Despliegue con Flask
Ejecuta `10_app_flask.py`, toma capturas de pantalla de:
1. El formulario de entrada
2. Los resultados de predicción con diferentes sentimientos
3. Las barras de probabilidad

### Conclusiones Clave:
- Reducción significativa del tamaño del modelo (≈ 75%)
- Mantención de accuracy similar (diferencia < 1%)
- Optimización ideal para dispositivos móviles y edge computing

---

## ✅ Checklist para la Evaluación

- [ ] Ejecutar `08_optimizacion_tflite.py` y verificar la creación del archivo `.tflite`
- [ ] Ejecutar `09_comparacion_tecnica.py` y copiar la tabla de resultados
- [ ] Ejecutar `10_app_flask.py` y probar la aplicación web
- [ ] Tomar capturas de pantalla de la aplicación Flask
- [ ] Probar con diferentes tipos de reseñas
- [ ] Incluir todos los resultados en el informe final

---

**¡Éxito en tu Evaluación 4! 🎓**
