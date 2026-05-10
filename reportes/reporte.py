import os
from fpdf import FPDF
import datetime

class ReporteRadiologico(FPDF):
    def header(self):
        # Fondo del encabezado
        self.set_fill_color(11, 14, 20)
        self.rect(0, 0, 210, 30, 'F')
        self.set_y(10)
        self.set_font('Arial', 'B', 15)
        # CORRECCIÓN: set_text_color (con guiones) y valores individuales
        self.set_text_color(76, 201, 240) 
        self.cell(0, 10, 'SISTEMA DE ANALISIS DE NEUROIMAGEN', 0, 1, 'C')
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def generar_reporte_final(datos, nombre_paciente, img_panel_path):
    pdf = ReporteRadiologico()
    pdf.add_page()
    pdf.ln(25)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, '1. RESULTADOS DEL ANALISIS DIGITAL', 0, 1, 'L')
    
    pdf.set_font('Courier', '', 11)
    # Sin emojis para mantener profesionalismo
    resumen = (
        f"ARCHIVO:       {nombre_paciente}\n"
        f"LOCALIZACION:  {datos['ubicacion']}\n"
        f"DIAMETRO:      {datos['diametro']}\n"
        f"VOLUMEN:       {datos['volumen']}\n"
        f"RIESGO:        {datos['riesgo']}"
    )
    pdf.multi_cell(0, 8, resumen)
    pdf.ln(5)
    
    pdf.cell(0, 10, '2. EVIDENCIA GRAFICA (MRI + SEGMENTACION)', 0, 1, 'L')
    # Ajustamos la imagen al ancho del PDF
    pdf.image(img_panel_path, x=10, w=190)
    
    out_path = f"reportes/Reporte_{nombre_paciente.replace('.nii.gz', '')}.pdf"
    pdf.output(out_path)
    return out_path