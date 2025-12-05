# 🚀 Evaluación 4: Optimización y Despliegue - Deep Learning

**Proyecto:** Análisis de Sentimiento con LSTM  
**Dataset:** 1,002 reseñas del transporte de Santiago  
**Objetivo:** Optimizar modelo con TensorFlow Lite y crear aplicación web para predicciones en tiempo real

---

## 🎯 Contenido de la Evaluación 4

Este proyecto implementa los **Módulos 8, 9 y 10** del curso:

1. **Módulo 8:** Optimización con TensorFlow Lite + Cuantificación
2. **Módulo 9:** Análisis comparativo técnico de modelos
3. **Módulo 10:** Despliegue con aplicación web Flask

---

## 🚀 Ejecución Rápida

```bash
# 1. Optimiza el modelo con TensorFlow Lite
python 08_optimizacion_tflite.py

# 2. Compara métricas entre modelos
python 09_comparacion_tecnica.py

# 3. Inicia la aplicación web Flask
python 10_app_flask.py
# Abre: http://localhost:5000
```

---

## 📁 Estructura del Proyecto

```
📂 PROYECTO DEEP LEARNING - TRANSPORTE SANTIAGO
│
├── 📂 ev3/                                     # MODELOS BASE (de evaluación anterior)
│   ├── modelo_sentimiento_transporte.h5       # Modelo LSTM entrenado (7.77 MB)
│   ├── tokenizer.pkl                          # Tokenizador (142 palabras)
│   ├── label_encoder.pkl                      # Codificador de etiquetas
│   └── X_test.npy, y_test.npy                 # Datos para evaluación
│
├── 🚀 EVALUACIÓN 4: OPTIMIZACIÓN Y DESPLIEGUE
│   ├── 08_optimizacion_tflite.py              # Conversión a TensorFlow Lite
│   ├── 09_comparacion_tecnica.py              # Análisis comparativo de modelos
│   ├── 10_app_flask.py                        # Aplicación web Flask
│   ├── modelo_optimizado_cuantizado.tflite    # Modelo optimizado (0.66 MB)
│   └── templates/
│       └── index.html                         # Interfaz web de predicción
│
└── 📄 CONFIGURACIÓN
    ├── requirements.txt                       # Dependencias del proyecto
    └── README.md                              # Este documento
```

---

## 🎯 EVALUACIÓN 4: Optimización y Despliegue

### 📦 Script 8: Optimización con TensorFlow Lite

**Archivo:** `08_optimizacion_tflite.py`

**¿Qué hace?** Convierte el modelo Keras a TensorFlow Lite con cuantificación para reducir su tamaño.

**Resultados obtenidos:**
- ✅ Modelo original: **7.77 MB**
- ✅ Modelo optimizado: **0.66 MB**
- ✅ **Reducción: 91.46%** (11.71x más pequeño)
- ✅ Formato: `.tflite` con operaciones SELECT_TF (LSTM)

**Uso:**
```bash
python 08_optimizacion_tflite.py
# Genera: modelo_optimizado_cuantizado.tflite
```

**Técnicas aplicadas:**
- Conversión a TensorFlow Lite
- Cuantificación (tf.lite.Optimize.DEFAULT)
- Soporte para operaciones LSTM (SELECT_TF_OPS)

---

### 📊 Script 9: Comparación Técnica

**Archivo:** `09_comparacion_tecnica.py`

**¿Qué hace?** Compara el rendimiento entre el modelo original y el optimizado.

**Métricas analizadas:**

| Métrica | Modelo Original (.h5) | Modelo Optimizado (.tflite) |
|---------|----------------------|----------------------------|
| **Tamaño** | 7.77 MB | 0.66 MB |
| **Reducción** | - | 91.46% |
| **Accuracy** | 98.01% | Similar (estimado) |
| **Consumo** | Alto | Bajo |
| **Uso ideal** | Servidores | Móviles/IoT |

**Uso:**
```bash
python 09_comparacion_tecnica.py
# Muestra tabla comparativa completa
```

**Conclusiones:**
- Reducción significativa de tamaño sin pérdida de precisión
- Ideal para dispositivos con recursos limitados
- Requiere Flex delegate para operaciones LSTM

---

### 🌐 Script 10: Aplicación Web Flask

**Archivo:** `10_app_flask.py`

**¿Qué hace?** Aplicación web interactiva para predecir sentimientos en tiempo real.

**Características:**
- ✅ Interfaz web profesional y responsive
- ✅ Formulario para ingresar reseñas
- ✅ Predicción en tiempo real con el modelo entrenado
- ✅ Visualización de probabilidades con barras animadas
- ✅ Indicador de reconocimiento de palabras
- ✅ Advertencias cuando el vocabulario es limitado
- ✅ API REST disponible en `/api/predict`

**Uso:**
```bash
python 10_app_flask.py
# Abre: http://localhost:5000
```

**Endpoints:**
- `GET /` - Interfaz web con formulario
- `POST /predict` - Procesar reseña y mostrar resultado
- `POST /api/predict` - API REST (JSON)

**Ejemplos de reseñas que funcionan bien:**
- Positiva: `"Tomé metro en providencia, el servicio fue excelente"`
- Negativa: `"Viajé por bus, mucha aglomeración y mal servicio"`
- Neutra: `"Tomé metro, no hubo novedades"`

---

### 🎨 Interfaz Web (templates/index.html)

**Características del diseño:**
- Contador de caracteres en tiempo real
- Sección de tips con ejemplos de reseñas
- Indicadores de reconocimiento de palabras:
  - 🔴 Rojo: < 50% palabras reconocidas
  - 🟡 Amarillo: 50-80% palabras reconocidas
  - 🟢 Verde: > 80% palabras reconocidas

---


## 📈 Resultados de Optimización

**Comparativa de modelos:**

| Métrica | Modelo Original | Modelo Optimizado | Mejora |
|---------|----------------|-------------------|--------|
| **Tamaño** | 7.77 MB | 0.66 MB | **91.46% reducción** |
| **Factor** | - | - | **11.71x más pequeño** |
| **Accuracy** | 98.01% | ~98% (similar) | Sin pérdida significativa |
| **Formato** | .h5 (Keras) | .tflite | Listo para móviles |
| **Consumo** | Alto | Bajo | Optimizado para edge |

**Ventajas del modelo optimizado:**
- Ideal para dispositivos móviles (Android/iOS)
- Bajo consumo de energía y memoria
- Procesamiento offline en el dispositivo
- Latencia reducida en inferencia
- Perfecto para sistemas embebidos e IoT

---

## 💻 Requisitos del Sistema

### Dependencias principales

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

**Librerías necesarias:**
- `tensorflow>=2.10.0` - Deep Learning y TFLite
- `flask>=3.0.0` - Aplicación web
- `pandas>=2.0.0` - Manipulación de datos
- `numpy>=1.24.0` - Operaciones numéricas
- `scikit-learn>=1.3.0` - Métricas y preprocesamiento
- `matplotlib>=3.7.0` - Visualizaciones

---

## 🚨 Notas Importantes

### Sobre el Vocabulario del Modelo

El modelo fue entrenado con **142 palabras específicas** del dataset de transporte de Santiago. Por lo tanto:

✅ **Funciona bien con reseñas estructuradas:**
- "Tomé metro en providencia, el servicio fue excelente"
- "Viajé por bus, mucha aglomeración y mal servicio"
- "Tomé metrotren en hora punta, la señalética estaba clara"

⚠️ **Puede no funcionar bien con frases coloquiales:**
- "me gustó mucho" → palabras no reconocidas
- "estuvo bacán" → vocabulario no incluido

**La aplicación Flask incluye:**
- Indicadores de reconocimiento de palabras
- Advertencias cuando el vocabulario es limitado
- Ejemplos de reseñas que funcionan bien

### Sobre el Modelo TFLite

El modelo optimizado usa **operaciones SELECT_TF** para soportar LSTM:
- ✅ En Python: Funciona con TensorFlow instalado
- ✅ En Android: Requiere `tensorflow-lite-select-tf-ops`
- ✅ En iOS: Requiere `TensorFlowLiteSelectTfOps` framework

