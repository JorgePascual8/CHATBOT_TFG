import PyPDF2

def extraer_texto_pdf(ruta_pdf):
    texto = ""
    with open(ruta_pdf, 'rb') as archivo:
        lector_pdf = PyPDF2.PdfReader(archivo)
        for pagina in lector_pdf.pages:
            texto += pagina.extract_text()
    return texto
