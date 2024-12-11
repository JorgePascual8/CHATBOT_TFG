import numpy as np
import os
from extraer_texto import extraer_texto_pdf
from procesar_texto import procesar_texto
from generar_embeddings import cargar_modelo_embeddings, generar_embeddings
from buscar_contexto import buscar_oraciones_similares
from generar_respuesta import configurar_openai, generar_respuesta

def main():
    # Ruta del PDF
    ruta_pdf = 'asignatura.pdf'

    # Cargar el modelo de embeddings
    print("Cargando modelo de embeddings...")
    modelo = cargar_modelo_embeddings()

    # Verificar si los embeddings ya existen
    if os.path.exists('embeddings.npy') and os.path.exists('oraciones.npy'):
        print("Cargando embeddings y oraciones desde archivos...")
        embeddings_oraciones = np.load('embeddings.npy')
        oraciones = np.load('oraciones.npy', allow_pickle=True)
    else:
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

    # Configurar la API de OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: No se encontró la clave de API de OpenAI en las variables de entorno.")
        return
    configurar_openai()

    # Iniciar el chatbot
    print("Hola, soy tu tutor virtual. ¿En qué puedo ayudarte?")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() in ['salir', 'adiós', 'gracias']:
            print("Tutor: ¡Hasta luego!")
            break
        # Buscar contexto relevante
        contexto = buscar_oraciones_similares(pregunta, embeddings_oraciones, oraciones, modelo)
        # Generar respuesta
        respuesta = generar_respuesta(contexto, pregunta)
        print(f"Tutor: {respuesta}")

if __name__ == '__main__':
    main()
