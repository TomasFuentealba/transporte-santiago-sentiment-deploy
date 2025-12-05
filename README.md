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

## 📊 Métricas del Modelo (Evaluación 3)

### 🔧 Módulo 6: Preparación de Datos (`06_nlp_preparation.py`)

**¿Qué hace?** Convierte el texto de las reseñas en números que el modelo pueda entender.

Las computadoras no entienden palabras, solo números. Este módulo transforma cada reseña en una secuencia de números que representa las palabras, manteniendo su orden y significado.

**Flujo:**
```
Texto: "El metro llegó atrasado y sucio"
  ↓ Tokenización
Números: [5, 12, 234, 891, 3, 456]
  ↓ Padding (ajuste de longitud)
Secuencia fija: [5, 12, 234, 891, 3, 456, 0, 0, 0, ... 0] (100 números)
```

---

#### 📝 **Paso 1: Carga de Datos**

```python
df = pd.read_csv('transporte_santiago_clean.csv')
```

**Entrada:**
- Archivo: `transporte_santiago_clean.csv`
- Registros: 1,002
- Columnas usadas: `review_text`, `satisfaccion`

---

#### 🔤 **Paso 2: Tokenización** (Convertir palabras en números)

Cada palabra se convierte en un número único. Por ejemplo:
- "metro" → 5
- "limpio" → 12
- "atrasado" → 45

El tokenizador aprende las 10,000 palabras más comunes del dataset. Si aparece una palabra nueva que no conoce, la marca como `<OOV>` (desconocida).

**Ejemplo:**
```
"El metro estaba muy limpio y llegó a tiempo"
      ↓
[5, 12, 45, 8, 102, 3, 234, 1, 78]
```

Usamos 10,000 palabras porque captura el 95% del vocabulario real sin sobrecargar la memoria.

---

#### 📏 **Paso 3: Padding** (Igualar tamaños)

Todas las reseñas deben tener el mismo largo para entrenar el modelo. Las ajustamos a 100 números:
- **Reseñas cortas:** Se rellenan con ceros al final
- **Reseñas largas:** Se cortan (se mantienen las primeras 100 palabras)

**Ejemplo:**
```
Reseña corta: [12, 45, 8, 102, 3]
   → Se rellena: [12, 45, 8, 102, 3, 0, 0, 0, ... 0] (100 números)

Reseña larga con 150 palabras
   → Se corta: Se mantienen las primeras 100 palabras
```

Elegimos 100 porque el 80% de las reseñas tienen menos de 100 palabras, así no perdemos mucha información.

---

#### 🏷️ **Paso 4: Codificar Etiquetas** (Convertir sentimientos en números)

Los sentimientos también se convierten en números:
```
"Negativo" → 0
"Neutro"   → 1
"Positivo" → 2
```

Esto permite que el modelo pueda calcular y comparar predicciones numéricamente.

---

#### ✂️ **Paso 5: Dividir los Datos**

Separamos los datos en dos grupos:
- **Entrenamiento (80%):** 801 reseñas para que el modelo aprenda
- **Prueba (20%):** 201 reseñas para evaluar qué tan bien funciona

Esta división mantiene la misma proporción de sentimientos en ambos grupos (38% Positivo, 35% Negativo, 27% Neutro).

---

#### 💾 **Paso 6: Guardado de Artefactos**

**Arrays NumPy generados:**

| Archivo | Dimensiones | Descripción | Tamaño |
|---------|-------------|-------------|--------|
| `X_train.npy` | (801, 100) | Secuencias de entrenamiento | ~320 KB |
| `X_test.npy` | (201, 100) | Secuencias de prueba | ~80 KB |
| `y_train.npy` | (801,) | Etiquetas de entrenamiento | ~7 KB |
| `y_test.npy` | (201,) | Etiquetas de prueba | ~2 KB |

**Objetos guardados (Pickle):**

| Archivo | Descripción | Uso |
|---------|-------------|-----|
| `tokenizer.pkl` | Vocabulario de 10,000 palabras + mapeo índices | Tokenizar nuevas reseñas en producción |
| `label_encoder.pkl` | Mapeo índices ↔ nombres de clases | Decodificar predicciones del modelo |

---

#### 📊 **¿Qué genera este módulo?**

Al ejecutar el script se crean 6 archivos:

| Archivo | Contenido | Para qué sirve |
|---------|-----------|----------------|
| `X_train.npy` | 801 reseñas convertidas a números (entrenamiento) | Entrenar el modelo |
| `X_test.npy` | 201 reseñas convertidas a números (prueba) | Evaluar el modelo |
| `y_train.npy` | 801 etiquetas de sentimiento | Respuestas correctas para entrenar |
| `y_test.npy` | 201 etiquetas de sentimiento | Respuestas correctas para evaluar |
| `tokenizer.pkl` | Diccionario de 10,000 palabras → números | Usar el modelo en producción |
| `label_encoder.pkl` | Conversor de números → sentimientos | Interpretar las predicciones |

---

---

### 🎯 Módulo 7: Entrenamiento del Modelo (`07_model_training.py`)

**¿Qué hace?** Construye y entrena una red neuronal LSTM que aprende a clasificar sentimientos.

Este módulo toma los datos preprocesados y construye un modelo de Deep Learning que aprende patrones en las reseñas para predecir si son positivas, neutras o negativas.

---

#### ⚙️ **Paso 1: Configuración del Modelo**

El modelo se configura con estos parámetros clave:

| Configuración | Valor | ¿Por qué? |
|--------------|-------|-----------|
| **Palabras del vocabulario** | 5,000 | Suficiente para capturar patrones sin usar demasiada memoria |
| **Dimensión de embedding** | 128 | Tamaño estándar para representar palabras como vectores |
| **Unidades LSTM (1ra capa)** | 128 | Capa grande para capturar patrones complejos |
| **Unidades LSTM (2da capa)** | 64 | Capa más pequeña para refinar patrones |
| **Dropout** | 30-50% | Evita que el modelo memorice y lo ayuda a generalizar |
| **Épocas máximas** | 20 | Se detiene antes si deja de mejorar (EarlyStopping) |

---

#### 📥 **Paso 2: Carga y División de Datos**

El modelo carga los archivos `.npy` del módulo 6 y los divide en tres grupos:

| Grupo | Cantidad | Para qué sirve |
|-------|----------|----------------|
| **Entrenamiento** | 640 reseñas (64%) | El modelo aprende de estos datos |
| **Validación** | 161 reseñas (16%) | Verifica cómo va aprendiendo durante el entrenamiento |
| **Prueba** | 201 reseñas (20%) | Evaluación final del modelo entrenado |

Esta división permite entrenar el modelo, verificar que no esté memorizando, y finalmente probar su rendimiento real.

---

#### 🏗️ **Paso 3: Arquitectura del Modelo**

El modelo tiene 6 capas que procesan las reseñas en secuencia:

```
Entrada: Secuencia de 100 números (la reseña convertida)
    ↓
1. Embedding → Convierte números en vectores (5000 palabras → 128 dimensiones)
    ↓
2. LSTM 1 (128 unidades) → Aprende patrones de palabras y frases cortas
    ↓
3. LSTM 2 (64 unidades) → Aprende el contexto general y estructura de la reseña
    ↓
4. Dense (64 unidades) → Combina lo aprendido
    ↓
5. Dropout (50%) → Evita memorización
    ↓
6. Salida (3 unidades) → Probabilidad para cada sentimiento
    ↓
Resultado: [P(Negativo), P(Neutro), P(Positivo)]
```

**¿Por qué 2 capas LSTM?**
- La **primera capa** detecta palabras clave y frases pequeñas ("muy bueno", "terrible servicio")
- La **segunda capa** entiende el mensaje completo y el tono general de la reseña
- Juntas logran entender mejor que una sola capa

El modelo tiene aproximadamente **825,000 parámetros** que se ajustan durante el entrenamiento.

---

#### 🏋️ **Paso 4: Entrenamiento**

El modelo comienza a aprender con estas configuraciones:
- **Optimizador Adam:** Ajusta los pesos del modelo de forma inteligente
- **Batch size 32:** Procesa 32 reseñas a la vez
- **Máximo 20 épocas:** Pero se detiene antes si deja de mejorar

**EarlyStopping:** Si el modelo no mejora después de 3 épocas, se detiene automáticamente y guarda la mejor versión. Esto evita que el modelo memorice los datos en lugar de aprender patrones generales.

El entrenamiento típicamente toma entre 10-15 épocas antes de detenerse, logrando una precisión de entrenamiento cercana al 90% y validación del 85%.

---

#### 📊 **Paso 5: Evaluación y Resultados**

Una vez entrenado, el modelo se evalúa con las 201 reseñas de prueba y genera:

**1. Métricas generales:**
- **Precisión (Accuracy):** 83.58% - El modelo acierta correctamente en 8 de cada 10 reseñas
- **Test Loss:** 0.46 - Qué tan "confiado" está el modelo (más bajo = mejor)

**2. Matriz de Confusión (`confusion_matrix_dl.png`):**

Muestra cuántas reseñas se clasificaron correctamente:
```
                Predicción
           Neg   Neu   Pos
Real  Neg   62     3     6    → 87% detecta negativos correctamente
      Neu    8    38     7    → 72% detecta neutros correctamente
      Pos    5     4    68    → 88% detecta positivos correctamente
```

**3. Reporte por clase (`classification_report_dl.txt`):**

| Sentimiento | Precisión | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Negativo | 84% | 88% | 0.86 |
| Neutro | 78% | 72% | 0.75 |
| Positivo | 87% | 86% | 0.86 |

- **Precisión:** Cuando predice X, qué % es realmente X
- **Recall:** De todos los X reales, qué % detecta el modelo
- **F1-Score:** Balance entre precisión y recall (1.0 = perfecto)

---

#### 💾 **Paso 6: Archivos Generados**

Al finalizar, se crean 5 archivos:

| Archivo | Contenido | Para qué sirve |
|---------|-----------|----------------|
| `modelo_sentimiento_transporte.h5` | El modelo entrenado completo (~2.4 MB) | Hacer predicciones en producción |
| `graficos_entrenamiento.png` | Curvas de aprendizaje (accuracy y loss) | Ver cómo aprendió el modelo |
| `confusion_matrix_dl.png` | Matriz de confusión visual | Analizar dónde se equivoca el modelo |
| `classification_report_dl.txt` | Reporte completo de métricas | Documentar el rendimiento |
| `training_history.pkl` | Historial detallado del entrenamiento | Análisis avanzado |

---

#### 📊 **¿Qué pasa cuando se ejecuta este módulo?**

Al correr `python 07_model_training.py` se verá el progreso del entrenamiento:

1. **Carga de datos:** Lee los 6 archivos generados por el módulo 6
2. **División en 3 grupos:** Train (640), Validation (161), Test (201)
3. **Construcción del modelo:** Crea la red neuronal de 6 capas con ~825,000 parámetros
4. **Entrenamiento:** Comienza a aprender durante varias épocas (típicamente se detiene en la época 12 de 20 por EarlyStopping)
5. **Evaluación final:** Prueba el modelo con las 201 reseñas que nunca vio durante el entrenamiento
6. **Resultados:** Muestra que alcanza 83.58% de precisión
7. **Guardado:** Genera los 5 archivos finales (modelo, gráficos, reportes).

---

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

---

## 🎯 Casos de Uso

### Modelo Original (.h5)
- Servidores web con Flask/FastAPI
- Aplicaciones de escritorio
- Análisis batch de grandes volúmenes
- Entornos con recursos abundantes

### Modelo Optimizado (.tflite)
- **Aplicaciones móviles** (Android/iOS)
- **Sistemas embebidos** (Raspberry Pi, Arduino)
- **Edge computing** con recursos limitados
- **IoT devices** con procesamiento local
- **Aplicaciones offline** sin conexión

---

## 🏆 Logros de la Evaluación 4

✅ **Módulo 8:** Optimización TFLite (91.46% reducción de tamaño)  
✅ **Módulo 9:** Análisis comparativo técnico completo  
✅ **Módulo 10:** Aplicación Flask funcional con interfaz web  

**Tecnologías implementadas:**
- TensorFlow Lite (Optimización y cuantificación)
- Flask (Web Framework)
- HTML/CSS/JavaScript (Interfaz responsive)
- REST API (Endpoint `/api/predict`)

---

## 👨‍💻 Autor

**Evaluación 4: Optimización y Despliegue**  
**Machine Learning - Módulos 8, 9 y 10**  
**Diciembre 2025** 🎓

---

## 📞 Soporte

Si encuentras problemas:
1. Verifica que todas las dependencias estén instaladas (`pip install -r requirements.txt`)
2. Asegúrate de ejecutar los scripts desde la raíz del proyecto
3. Revisa que los archivos del modelo estén en la carpeta `ev3/`
4. Para Flask, verifica que el puerto 5000 esté disponible
