from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime
from extraer_texto import extraer_texto_pdf
from procesar_texto import procesar_texto
from generar_embeddings import cargar_modelo_embeddings, generar_embeddings
from buscar_contexto import buscar_oraciones_similares
from generar_respuesta import configurar_openai, generar_respuesta
import numpy as np
import os

app = Flask(__name__)

# Configurar la API de OpenAI
configurar_openai()

# Cargar modelo y embeddings una vez al iniciar el servidor
modelo = cargar_modelo_embeddings()

# Verificar si los embeddings ya existen
if os.path.exists('embeddings.npy') and os.path.exists('oraciones.npy'):
    print("Cargando embeddings y oraciones desde archivos...")
    embeddings_oraciones = np.load('embeddings.npy')
    oraciones = np.load('oraciones.npy', allow_pickle=True)
else:
    # Ruta del PDF
    ruta_pdf = 'asignatura.pdf'
    # Extraer texto del PDF
    print("Extrayendo texto del PDF...")
    texto_pdf = extraer_texto_pdf(ruta_pdf)
    # Procesar y segmentar el texto
    print("Procesando texto...")
    oraciones = procesar_texto(texto_pdf)
    # Generar embeddings
    print("Generando embeddings...")
    embeddings_oraciones = generar_embeddings(oraciones, modelo)
    # Guardar embeddings y oraciones
    np.save('embeddings.npy', embeddings_oraciones)
    np.save('oraciones.npy', oraciones)

@app.route('/set_language/<language>')
def set_language(language):
    session['idioma'] = language
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.now().year
    idioma = session.get('idioma', 'es')  # Idioma por defecto es español
    if 'historial' not in session:
        session['historial'] = []
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        rol = request.form['rol']
        try:
            contexto = buscar_oraciones_similares(pregunta, embeddings_oraciones, oraciones, modelo)
            respuesta = generar_respuesta(contexto, pregunta, rol, idioma)
        except Exception as e:
            respuesta = "Lo siento, ocurrió un error al procesar tu solicitud." if idioma == 'es' else "Sorry, an error occurred while processing your request."
            print(f"Error: {e}")
        # Capitalizar el rol y traducir si es necesario
        if idioma == 'en':
            roles_traducidos = {
                'mentor': 'Mentor',
                'tutor': 'Tutor',
                'entrenador': 'Coach',
                'companero': 'Peer',
                'estudiante': 'Student',
                'simulador': 'Simulator',
                'herramienta': 'Tool'
            }
            rol_capitalizado = roles_traducidos.get(rol, rol).capitalize()
        else:
            rol_capitalizado = rol.capitalize()
        # Agregar al historial
        session['historial'].append({
            'pregunta': pregunta,
            'respuesta': respuesta,
            'rol': rol_capitalizado
        })
        session.modified = True
    else:
        rol = ''
        rol_capitalizado = ''
    return render_template('index.html', historial=session['historial'], rol=rol, rol_capitalizado=rol_capitalizado, idioma=idioma, current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)

