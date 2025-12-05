"""
SCRIPT 3: APLICACIÓN WEB CON FLASK
Evaluación 4 - Machine Learning
Paso 11: Despliegue del Modelo - Interfaz Web

Esta aplicación Flask permite:
1. Ingresar reseñas de transporte manualmente
2. Predecir el sentimiento (Negativo, Neutro, Positivo)
3. Mostrar resultados en tiempo real

Para ejecutar:
    python 10_app_flask.py

Luego abrir en navegador: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN FLASK
# ============================================================================
app = Flask(__name__)

# Configuración de hiperparámetros (deben coincidir con el entrenamiento)
MAX_LENGTH = 100  # Longitud máxima de las secuencias

# ============================================================================
# CARGA DE MODELOS Y OBJETOS AL INICIO
# ============================================================================
print("=" * 70)
print("INICIANDO APLICACIÓN FLASK - ANÁLISIS DE SENTIMIENTOS")
print("=" * 70)
print("\n[INFO] Cargando modelo y objetos preprocesados...")

# Cargar el modelo Keras entrenado
modelo = tf.keras.models.load_model('ev3/modelo_sentimiento_transporte.h5')
print("✓ Modelo cargado: modelo_sentimiento_transporte.h5")

# Cargar el tokenizer (para convertir texto a secuencias numéricas)
with open('ev3/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
print("✓ Tokenizer cargado: tokenizer.pkl")

# Cargar el label encoder (para decodificar las predicciones)
with open('ev3/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("✓ Label Encoder cargado: label_encoder.pkl")

# Obtener las clases disponibles
clases = label_encoder.classes_
print(f"✓ Clases disponibles: {clases}")

# Diccionario manual para mapear índices a etiquetas de sentimiento
# Esto es útil si el label_encoder tiene problemas de decodificación
SENTIMIENTOS = {
    0: "Negativo",
    1: "Neutro", 
    2: "Positivo"
}

print("\n[INFO] Sistema listo para recibir solicitudes")
print("=" * 70)

# ============================================================================
# FUNCIÓN DE PREPROCESAMIENTO
# ============================================================================
def preprocesar_texto(texto):
    """
    Preprocesa el texto ingresado por el usuario para que sea compatible
    con el modelo entrenado.
    
    Args:
        texto (str): Texto de la reseña ingresada por el usuario
        
    Returns:
        numpy.array: Secuencia numérica preprocesada lista para predicción
    """
    # Convertir el texto a minúsculas para consistencia
    texto = texto.lower().strip()
    
    # Tokenizar: Convertir texto a secuencia de números
    secuencia = tokenizer.texts_to_sequences([texto])
    
    # Padding: Ajustar la longitud de la secuencia a MAX_LENGTH
    # IMPORTANTE: Usar 'pre' para que coincida con el entrenamiento
    # Si es más corta, se rellena con ceros AL INICIO
    # Si es más larga, se trunca
    secuencia_padded = pad_sequences(secuencia, maxlen=MAX_LENGTH, padding='pre')
    
    return secuencia_padded

# ============================================================================
# RUTA PRINCIPAL (GET): PÁGINA DE INICIO CON FORMULARIO
# ============================================================================
@app.route('/')
def index():
    """
    Ruta principal que renderiza la página de inicio con el formulario
    para ingresar reseñas.
    """
    return render_template('index.html')

# ============================================================================
# RUTA DE PREDICCIÓN (POST): PROCESAR RESEÑA Y PREDECIR SENTIMIENTO
# ============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta que recibe la reseña del formulario, la procesa y devuelve
    la predicción del sentimiento.
    
    Returns:
        HTML: Página con el resultado de la predicción
    """
    try:
        # Obtener el texto del formulario
        texto_resena = request.form.get('resena', '').strip()
        
        # Validar que se haya ingresado texto
        if not texto_resena:
            return render_template('index.html', 
                                 error="Por favor, ingresa una reseña para analizar.")
        
        # Log para debugging
        print(f"\n[PREDICCIÓN] Texto recibido: {texto_resena[:50]}...")
        
        # Preprocesar el texto (tokenizar y aplicar padding)
        secuencia_procesada = preprocesar_texto(texto_resena)
        
        # Analizar cuántas palabras fueron reconocidas
        secuencia_original = tokenizer.texts_to_sequences([texto_resena.lower()])[0]
        palabras_totales = len(texto_resena.split())
        palabras_reconocidas = len([x for x in secuencia_original if x > 1])  # Excluir <OOV>
        porcentaje_reconocimiento = (palabras_reconocidas / palabras_totales * 100) if palabras_totales > 0 else 0
        
        print(f"[PREDICCIÓN] Shape de la secuencia: {secuencia_procesada.shape}")
        print(f"[PREDICCIÓN] Palabras reconocidas: {palabras_reconocidas}/{palabras_totales} ({porcentaje_reconocimiento:.1f}%)")
        print(f"[PREDICCIÓN] Secuencia: {secuencia_original[:20]}...")  # Mostrar primeros 20 tokens
        
        # Realizar la predicción con el modelo
        prediccion = modelo.predict(secuencia_procesada, verbose=0)
        print(f"[PREDICCIÓN] Probabilidades: {prediccion[0]}")
        
        # Obtener el índice de la clase con mayor probabilidad
        clase_predicha_idx = np.argmax(prediccion[0])
        
        # Obtener las probabilidades para cada clase
        prob_negativo = float(prediccion[0][0]) * 100
        prob_neutro = float(prediccion[0][1]) * 100
        prob_positivo = float(prediccion[0][2]) * 100
        
        # Decodificar la predicción usando el diccionario manual
        sentimiento = SENTIMIENTOS.get(clase_predicha_idx, "Desconocido")
        
        # Alternativamente, se puede usar el label_encoder:
        # sentimiento = label_encoder.inverse_transform([clase_predicha_idx])[0]
        
        print(f"[PREDICCIÓN] Sentimiento detectado: {sentimiento} ({prob_positivo:.1f}%)")
        
        # Determinar el color para mostrar el resultado
        colores = {
            "Negativo": "#dc3545",   # Rojo
            "Neutro": "#ffc107",     # Amarillo
            "Positivo": "#28a745"    # Verde
        }
        color = colores.get(sentimiento, "#6c757d")
        
        # Renderizar la página con los resultados
        return render_template('index.html',
                             resena_original=texto_resena,
                             sentimiento=sentimiento,
                             prob_negativo=prob_negativo,
                             prob_neutro=prob_neutro,
                             prob_positivo=prob_positivo,
                             color=color,
                             palabras_reconocidas=palabras_reconocidas,
                             palabras_totales=palabras_totales,
                             porcentaje_reconocimiento=porcentaje_reconocimiento,
                             mostrar_resultado=True)
    
    except Exception as e:
        # Manejo de errores
        print(f"[ERROR] {str(e)}")
        return render_template('index.html', 
                             error=f"Error al procesar la reseña: {str(e)}")

# ============================================================================
# API REST (OPCIONAL): ENDPOINT JSON PARA PREDICCIONES
# ============================================================================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint API REST que recibe JSON y devuelve la predicción en JSON.
    Útil para integraciones con otras aplicaciones.
    
    Ejemplo de uso:
        POST http://localhost:5000/api/predict
        Body: {"texto": "El metro está muy limpio y llega a tiempo"}
        
    Returns:
        JSON: Predicción y probabilidades
    """
    try:
        # Obtener datos del JSON
        data = request.get_json()
        texto_resena = data.get('texto', '').strip()
        
        if not texto_resena:
            return jsonify({'error': 'Texto vacío'}), 400
        
        # Preprocesar y predecir
        secuencia_procesada = preprocesar_texto(texto_resena)
        prediccion = modelo.predict(secuencia_procesada, verbose=0)
        
        # Obtener resultados
        clase_predicha_idx = np.argmax(prediccion[0])
        sentimiento = SENTIMIENTOS.get(clase_predicha_idx, "Desconocido")
        
        # Preparar respuesta JSON
        respuesta = {
            'sentimiento': sentimiento,
            'confianza': float(np.max(prediccion[0])) * 100,
            'probabilidades': {
                'negativo': float(prediccion[0][0]) * 100,
                'neutro': float(prediccion[0][1]) * 100,
                'positivo': float(prediccion[0][2]) * 100
            }
        }
        
        return jsonify(respuesta), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SERVIDOR FLASK INICIADO")
    print("=" * 70)
    print("\n Accede a la aplicación en: http://localhost:5000")
    print("\n⚠  Presiona CTRL+C para detener el servidor")
    print("=" * 70 + "\n")
    
    # Iniciar el servidor Flask
    # debug=True permite ver errores detallados y auto-reload
    app.run(debug=True, host='0.0.0.0', port=5000)
