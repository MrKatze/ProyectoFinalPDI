# 🎓 Detección de Rasgos Faciales - Avance V (Sin Landmarks)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Academic-orange.svg)]()

Sistema completo de **detección y segmentación de rasgos faciales** (ojos, nariz, boca) **sin usar landmarks de dlib**, implementado con técnicas de Procesamiento Digital de Imágenes.

![Pipeline](docs/banner.png)

---

## 🎯 Características Principales

- ✅ **Sin Landmarks**: No usa dlib ni modelos preentrenados
- ✅ **8 Métodos Independientes**: 3 para ojos + 2 para nariz + 3 para boca
- ✅ **Pipeline Completo**: Integra Avances II → III → IV → V
- ✅ **Visualización Automática**: Genera 20 paneles por imagen
- ✅ **Métricas Automáticas**: Calcula tasas de detección y tiempos
- ✅ **Basado en MATLAB**: Implementa Canny, Marr-Hildreth, morfología del curso

---

## 📊 Resultados

**Tasas de Detección (Dataset de 15 imágenes):**
- 🟢 Rostros: **86.7%** (13/15)
- 🟢 Ojos: **86.7%** (13/15)
- 🟢 Nariz: **86.7%** (13/15)
- 🟡 Boca: **66.7%** (10/15)

**Rendimiento:**
- ⚡ 0.738 segundos/imagen
- 🚀 1.35 FPS

---

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/leKeevin/DetectorFacial.git
cd DetectorFacial/ProyectoFinalPDI

# Instalar (automático)
./instalar.sh

# O manualmente:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📸 Uso

### 1. Organizar Imágenes

```
images/
├── persona1/
│   ├── foto1.jpg
│   └── foto2.jpg
├── persona2/
│   └── ...
└── persona3/
    └── ...
```

### 2. Ejecutar

```bash
# Prueba rápida (1 imagen)
./prueba_rapida.py

# Dataset completo
./avance_v_pipeline.py

# Imagen específica
./avance_v_pipeline.py --imagen ruta/imagen.jpg
```

---

## 🔬 Metodología

### **Avance II - Preprocesamiento**
- Detección de rostros (Haar Cascade)
- Alineación basada en ojos
- Normalización 256×256
- Filtros: Mediana, Gaussiano, Laplaciano, Highboost

### **Avance III - Segmentación Fondo-Rostro**
- Umbralización Global y Otsu
- Segmentación de piel (YCrCb)
- Limpieza morfológica

### **Avance IV - Regiones de Rasgos**
- División por proporciones faciales estándar
- ROIs para ojos, nariz y boca

### **Avance V - Segmentación Avanzada (Sin Landmarks)**

#### 👁️ **Ojos (3 métodos)**
1. **Haar Cascade** + validación geométrica
2. **Proyección horizontal** + análisis de varianza
3. **Canny + Hough circular** (detección de pupilas)

#### 👃 **Nariz (2 métodos)**
1. **Gradientes Sobel** en región central
2. **Filtros Gabor** para análisis de textura

#### 👄 **Boca (3 métodos)**
1. **Color YCrCb** (labios rojizos)
2. **Canny + morfología horizontal**
3. **Proyección vertical** + mínimos locales

#### 🔍 **Detección de Bordes**
- **Canny**: Supresión de no-máximos + histéresis
- **Marr-Hildreth**: LoG + cruces por cero

#### 🔄 **Morfología**
- Erosión, dilatación, apertura, cierre
- Gradiente, top-hat, black-hat

---

## 📁 Estructura del Proyecto

```
ProyectoFinalPDI/
├── src/
│   ├── avance_ii/        # Preprocesamiento
│   ├── avance_iii/       # Umbralización
│   ├── avance_iv/        # Regiones de rasgos
│   ├── avance_v/         # Segmentación avanzada ⭐
│   └── utils/            # Utilidades
│
├── images/               # Dataset (no incluido)
├── resultados/           # Resultados (no incluido)
├── docs/                 # Documentación
│
├── avance_v_pipeline.py  # Script principal
├── prueba_rapida.py      # Prueba rápida
└── instalar.sh           # Instalador
```

---

## 📊 Visualizaciones

El sistema genera **20 paneles por imagen**:

| Paneles | Contenido |
|---------|-----------|
| 1-4 | Preprocesamiento (original, detectado, alineado, normalizado) |
| 5-8 | Filtros (mediana, gaussiano, laplaciano, highboost) |
| 9-12 | Umbralización (global, Otsu, piel, segmentación) |
| 13-16 | Bordes y morfología (Canny, Marr-Hildreth, apertura, cierre) |
| 17-20 | Rasgos (ojos, nariz, boca, resultado final) |

---

## 🛠️ Tecnologías

- **Python 3.8+**
- **OpenCV** - Procesamiento de imágenes
- **NumPy** - Operaciones numéricas
- **Matplotlib** - Visualización
- **SciPy** - Algoritmos científicos
- **scikit-image** - Procesamiento adicional

---

## 📚 Documentación

- [`README_COMPLETO.md`](README_COMPLETO.md) - Metodología detallada
- [`EMPEZAR_AQUI.md`](EMPEZAR_AQUI.md) - Guía de inicio rápido
- [`docs/GUIA_USO.md`](docs/GUIA_USO.md) - Manual de usuario
- [`PROYECTO_COMPLETO.md`](PROYECTO_COMPLETO.md) - Resumen ejecutivo

---

## 🎓 Proyecto Académico

Este proyecto fue desarrollado para la materia de **Procesamiento Digital de Imágenes** (Noveno Semestre), implementando técnicas PDI sin el uso de landmarks o modelos preentrenados.

**Autores:** Equipo PDI  
**Fecha:** Diciembre 2025

---

## 📝 Licencia

Proyecto académico - Universidad

---

## 🙏 Agradecimientos

- Profesora y equipo de PDI
- Códigos MATLAB del curso como referencia
- OpenCV y comunidad de visión por computadora

---

## ⭐ Si te fue útil

Si este proyecto te ayudó en tu aprendizaje, ¡dale una estrella! ⭐

---

**¿Preguntas?** Abre un issue o consulta la documentación completa.
