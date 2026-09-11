# transporte-santiago-sentiment-deploy

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-optimized-brightgreen?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> ⚠️ **Disclaimer educativo**: Este dataset fue provisto por el curso [Nombre del Curso/Profesor] con fines académicos.
> Los datos son anonimizados y se usan solo para demostración de técnicas de ML/NLP.
> No representan datos de producción ni deben usarse comercialmente.

---

Optimización TFLite (91% reducción) + Despliegue Flask para predicción en tiempo real.

**Modelo original: 7.77 MB** → **Optimizado: 0.66 MB** (91.46% reducción)

## 🚀 Ejecución

```bash
pip install -r requirements.txt

# 1. Optimizar modelo (TFLite + cuantificación)
python 08_optimizacion_tflite.py

# 2. Comparar métricas
python 09_comparacion_tecnica.py

# 3. Iniciar app web Flask
python 10_app_flask.py
# Abre: http://localhost:5000
```

## 📁 Estructure

```
08_optimizacion_tflite.py    # Conversión a TFLite + cuantificación
09_comparacion_tecnica.py    # Análisis comparativo
10_app_flask.py              # App web Flask
modelo_optimizado_cuantizado.tflite    # Modelo optimizado (0.66 MB)
ptemplates/index.html         # Interfaz web
ev3/                         # Modelos base (h5, pkl, npy)
```

Ver código completo en: https://github.com/TomasFuentealba/transporte-santiago-sentiment-deploy
