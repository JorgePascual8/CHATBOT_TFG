from sentence_transformers import SentenceTransformer

def cargar_modelo_embeddings():
    modelo = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    return modelo

def generar_embeddings(oraciones, modelo):
    embeddings = modelo.encode(oraciones)
    return embeddings
