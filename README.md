NEUROAI — Sistema de Análisis de Neuroimagen con IA

Descripción:
NeuroAI es un proyecto desarrollado en Python que utiliza inteligencia artificial para detectar y segmentar tumores cerebrales en imágenes MRI.

El sistema analiza archivos médicos en formato .nii o .nii.gz utilizando una red neuronal U-Net entrenada con la base de datos BraTS 2021.

Funciones principales:
- Procesamiento de imágenes MRI
- Detección automática de tumores
- Segmentación mediante IA
- Cálculo de volumen y diámetro
- Generación de reportes PDF
- Interfaz gráfica profesional

Tecnologías utilizadas:
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- NiBabel

Base de datos:
BraTS 2021 (Brain Tumor Segmentation Challenge)

Modelo utilizado:
U-Net

¿Por qué U-Net?
Porque es uno de los modelos más utilizados en segmentación médica debido a su alta precisión y capacidad para detectar regiones tumorales pequeñas.

Cómo ejecutar el proyecto:

1. Instalar dependencias:
pip install torch numpy matplotlib opencv-python nibabel scikit-learn pillow fpdf

2. Ejecutar interfaz gráfica:
python app.py

3. Ejecutar entrenamiento:
python entrenamiento/entrenar.py

Estructura del proyecto:

/analisis
/entrenamiento
/modelos
/reportes
/dataset
app.py
main.py

Autor:
Equipo 3 de la materia de IA
Ingeniería en Sistemas Computacionales
Instituto Tecnológico de Tijuana

Nota:
Este proyecto tiene fines educativos y no reemplaza un diagnóstico médico profesional.
