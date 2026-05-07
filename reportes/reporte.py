import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

def generar_reporte(resultado, nombre_archivo="reporte_diagnostico.pdf"):
    # 1. Asegurar que la carpeta 'resultados' exista
    carpeta_salida = "resultados"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    # 2. Configurar el documento
    doc = SimpleDocTemplate(ruta_completa, pagesize=letter)
    estilos = getSampleStyleSheet()
    elementos = []

    # 3. Contenido del reporte
    titulo = Paragraph("Informe de Análisis de Tumor Cerebral", estilos['Title'])
    elementos.append(titulo)
    elementos.append(Spacer(1, 20))

    # Formateamos el texto con etiquetas HTML simples de ReportLab
    texto = f"""
    <b>DETALLES DEL ANÁLISIS:</b><br/><br/>
    <b>• Pixeles afectados:</b> {resultado['pixeles_tumor']}<br/>
    <b>• Porcentaje de ocupación:</b> {resultado['porcentaje']}%<br/>
    <b>• Nivel de riesgo:</b> {resultado['riesgo']}<br/>
    <b>• Ubicación anatómica:</b> {resultado['ubicacion']}<br/><br/>
    <i>Nota: Este informe es generado por una Inteligencia Artificial (U-Net) y debe ser validado por un especialista médico.</i>
    """

    elementos.append(Paragraph(texto, estilos['BodyText']))

    # 4. Construir el PDF
    try:
        doc.build(elementos)
        print(f" Reporte generado exitosamente en: {ruta_completa}")
    except Exception as e:
        print(f" Error al generar el PDF: {e}")

if __name__ == "__main__":
    # Prueba rápida con datos ficticios
    datos_prueba = {
        "pixeles_tumor": 1200,
        "porcentaje": 7.3,
        "riesgo": "Moderado",
        "ubicacion": "Hemisferio Izquierdo - Zona Frontal"
    }
    generar_reporte(datos_prueba)