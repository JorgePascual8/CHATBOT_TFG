import numpy as np

def buscar_oraciones_similares(pregunta, embeddings_oraciones, oraciones, modelo, top_k=5):
    embedding_pregunta = modelo.encode([pregunta])[0]
    distancias = np.linalg.norm(embeddings_oraciones - embedding_pregunta, axis=1)
    indices_ordenados = np.argsort(distancias)
    oraciones_similares = [oraciones[i] for i in indices_ordenados[:top_k]]
    return ' '.join(oraciones_similares)
