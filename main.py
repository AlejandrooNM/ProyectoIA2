import os
import matplotlib.pyplot as plt
import numpy as np
from analisis.predictor import predecir_tumor
from analisis.analizador import obtener_diagnostico_clinico
from reportes.reporte import generar_reporte_final

def iniciar_sistema():
    print("\n" + "="*50)
    print("SISTEMA DE ANALISIS DE NEUROIMAGEN PROFESIONAL")
    print("="*50)

    entrada = input("\nIntroduce la ruta del archivo .nii: ").strip().replace('"', '').replace("'", "")

    if not os.path.exists(entrada):
        return print(f"Error: No se encontro el archivo en la ruta especificada.")

    try:
        # 1. Inferencia de la IA
        print("[1/3] Procesando imagen con red neuronal U-Net...")
        # FIX: predecir_tumor ahora devuelve 4 valores
        original, mascara, mascara_prob, slices_mascara = predecir_tumor(entrada)

        mascara_limpia = mascara.squeeze()
        original_rot   = original
        mascara_rot    = mascara_limpia

        # 2. Analisis Clinico
        print("[2/3] Analizando resultados clinicos...")
        # FIX: se pasan mascara_prob y slices_mascara para volumen 3D y confianza
        datos = obtener_diagnostico_clinico(mascara_rot, mascara_prob, slices_mascara)

        # 3. Visualizacion
        print("[3/3] Generando visualizacion...")
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='#0b0e14')

        mask_masked = np.ma.masked_where(mascara_rot < 0.5, mascara_rot)

        # PANEL 1: MRI Original
        axes[0].imshow(original_rot, cmap='gray')
        axes[0].set_title("VISTA AXIAL T1WCE", color='#4cc9f0', pad=15)
        axes[0].axis('off')

        # PANEL 2: Segmentacion IA
        axes[1].imshow(original_rot, cmap='gray')
        axes[1].imshow(mask_masked, cmap='cool', alpha=0.7, interpolation='none')
        axes[1].set_title("SEGMENTACION IA (LOCALIZACION)", color='#f72585', pad=15)
        axes[1].axis('off')
        axes[1].set_xlim(axes[0].get_xlim())
        axes[1].set_ylim(axes[0].get_ylim())

        # PANEL 3: Reporte — incluye confianza y slices si hay tumor
        if datos.get("tumor_presente", True):
            info_panel = (
                "INFORME ANALITICO\n"
                "---------------------------\n\n"
                f"LOCALIZACION: {datos['ubicacion']}\n\n"
                f"DIAMETRO:     {datos['diametro']}\n\n"
                f"VOLUMEN:      {datos['volumen']}\n\n"
                f"RIESGO:       {datos['riesgo']}\n\n"
                f"CONFIANZA IA: {datos['confianza']}\n\n"
                f"SLICES:       {datos['slices_con_tumor']}"
            )
        else:
            info_panel = (
                "INFORME ANALITICO\n"
                "---------------------------\n\n"
                "SIN TUMOR DETECTADO\n\n"
                "No se encontro region\n"
                "tumoral en el volumen\n"
                "analizado."
            )

        axes[2].text(0.1, 0.5, info_panel, color='white', family='monospace', fontsize=12,
                     linespacing=1.8, verticalalignment='center',
                     bbox=dict(boxstyle='round,pad=1.5', facecolor='#1a1e26',
                     edgecolor='#4361ee', linewidth=2))
        axes[2].set_title("METRICAS CLINICAS", color='#4361ee', pad=15)
        axes[2].axis('off')

        plt.tight_layout()

        # 4. Reporte PDF
        os.makedirs("reportes", exist_ok=True)
        img_temp = "reportes/temp_panel.png"
        plt.savefig(img_temp, dpi=300, facecolor='#0b0e14')

        pdf_path = generar_reporte_final(datos, os.path.basename(entrada), img_temp)

        print(f"\n[OK] Analisis completado.")
        print(f"[OK] Reporte PDF generado en: {pdf_path}")

        plt.show()

        if os.path.exists(img_temp):
            os.remove(img_temp)

    except Exception as e:
        print(f"\n[ERROR] Fallo en la ejecucion: {e}")
        raise  # muestra el traceback completo para facilitar debugging

if __name__ == "__main__":
    iniciar_sistema()