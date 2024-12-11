import nltk
from nltk.tokenize import sent_tokenize

def procesar_texto(texto):
    oraciones = sent_tokenize(texto, language='spanish')
    return oraciones

