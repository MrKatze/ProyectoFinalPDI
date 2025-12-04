# Proyecto Final - Detección y Reconocimiento de Rostros
## Procesamiento Digital de Imágenes

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

---

## 📋 Descripción del Proyecto

Sistema completo de **detección y reconocimiento de rostros** implementado desde cero utilizando técnicas de Procesamiento Digital de Imágenes, sin dependencias de modelos pre-entrenados de landmarks.

### Avances del Proyecto

#### ✅ Avance I: Obtención del Conocimiento
- Investigación de características faciales representativas
- Técnicas de alineación y normalización de rostros
- Referencias bibliográficas

#### ✅ Avance II: Obtención de Imágenes y Preprocesamiento
- 5 fotos por persona desde diferentes ángulos
- Aplicación de alineación y normalización
- Filtros estadísticos, suavizantes y realzantes

#### ✅ Avance III: Segmentación en dos Regiones
- Umbralización global
- Umbralización de Otsu
- Segmentación fondo-rostro

#### ✅ Avance IV: Rasgos Importantes
- Identificación de rasgos clave (ojos, nariz, boca)
- Algoritmos de detección sin landmarks

#### 🔄 Avance V: Segmentación de Rasgos (EN DESARROLLO)
- **Sin usar landmarks de dlib**
- Combinación de todas las técnicas anteriores
- Detección de bordes (Canny, Marr-Hildreth)
- Morfología matemática
- Segmentación por color y gradientes

#### 🔜 Avance VI: Extracción de Descriptores
- Descriptores para reconocimiento facial
- Compacidad
- Distancia radial normalizada
- Cruces por cero
- Rugosidad

---

## 🎯 Objetivo del Avance V

Implementar un sistema completo que:

1. **Detecte rostros** en imágenes
2. **Alinee y normalice** los rostros detectados
3. **Aplique preprocesamiento** (filtros)
4. **Segmente el fondo del rostro** (umbralización)
5. **Detecte rasgos faciales** (ojos, nariz, boca) **SIN landmarks**
6. **Use detección de bordes** (Canny, Marr-Hildreth)
7. **Aplique morfología** para mejorar la segmentación

---

## 🏗️ Estructura del Proyecto

```
ProyectoFinalPDI/
│
├── src/                              # Código fuente
│   ├── avance_i/                     # Investigación (referencias)
│   ├── avance_ii/                    # Preprocesamiento
│   │   ├── alineacion.py
│   │   └── filtros.py
│   ├── avance_iii/                   # Segmentación fondo-rostro
│   │   └── umbralizacion.py
│   ├── avance_iv/                    # Detección de rasgos
│   │   └── deteccion_rasgos.py
│   ├── avance_v/                     # ⭐ ACTUAL - Segmentación completa
│   │   ├── segmentador_ojos.py
│   │   ├── segmentador_nariz.py
│   │   ├── segmentador_boca.py
│   │   ├── deteccion_bordes.py
│   │   └── morfologia.py
│   └── utils/                        # Utilidades comunes
│       ├── visualizacion.py
│       └── validacion.py
│
├── images/                           # Imágenes de entrada
│   ├── persona1/
│   ├── persona2/
│   └── persona3/
│
├── resultados/                       # Resultados por avance
│   ├── avance_ii/
│   ├── avance_iii/
│   ├── avance_iv/
│   └── avance_v/
│
├── notebooks/                        # Notebooks para análisis
│   ├── avance_v_desarrollo.ipynb
│   └── avance_v_pruebas.ipynb
│
├── docs/                             # Documentación
│   ├── referencias_avance_i.md
│   ├── metodologia_avance_v.md
│   └── resultados_avance_v.md
│
├── avance_v_pipeline.py              # Script principal Avance V
├── requirements.txt
└── README.md
```

---

## 🚀 Instalación

### 1. Crear entorno virtual

```bash
cd ProyectoFinalPDI
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- OpenCV >= 4.5
- NumPy >= 1.19
- Matplotlib >= 3.3
- SciPy >= 1.5

---

## 💻 Uso del Avance V

### Método 1: Pipeline Completo

```bash
python avance_v_pipeline.py
```

Este script:
1. Lee imágenes de `images/`
2. Aplica todo el pipeline (Avances II, III, IV, V)
3. Guarda resultados en `resultados/avance_v/`
4. Genera visualizaciones comparativas

### Método 2: Uso Programático

```python
from src.avance_v.segmentador_completo import SegmentadorRasgos
import cv2

# Cargar imagen
imagen = cv2.imread('images/persona1/foto1.jpg')

# Crear segmentador
segmentador = SegmentadorRasgos()

# Procesar
resultados = segmentador.procesar_completo(imagen)

# Resultados incluyen:
# - Rostro alineado y normalizado
# - Filtros aplicados
# - Segmentación fondo-rostro
# - Rasgos detectados (ojos, nariz, boca)
# - Bordes detectados
# - Máscaras morfológicas
```

---

## 📊 Técnicas Implementadas

### Avance II: Preprocesamiento
- ✅ Alineación basada en detección de ojos
- ✅ Normalización de tamaño e iluminación
- ✅ Filtro estadístico (mediana)
- ✅ Filtro suavizante (Gaussiano)
- ✅ Filtro realzante (Laplaciano, Highboost)

### Avance III: Segmentación Fondo-Rostro
- ✅ Umbralización global iterativa
- ✅ Umbralización de Otsu
- ✅ Comparación de métodos

### Avance IV: Identificación de Rasgos
- ✅ Región de ojos (tercio superior)
- ✅ Región de nariz (centro)
- ✅ Región de boca (tercio inferior)

### Avance V: Segmentación Avanzada (SIN Landmarks)

#### 🔍 Detección de Ojos
**Método 1: Haar Cascade + Refinamiento**
- Clasificador Haar para detección inicial
- Análisis de simetría bilateral
- Validación por distancia entre ojos

**Método 2: Proyección Horizontal + Morfología**
- Proyección de intensidad por filas
- Detección de regiones oscuras (ojos)
- Operadores morfológicos para limpiar

**Método 3: Detección de Bordes + Hough**
- Canny para detectar contornos
- Transformada de Hough circular (pupilas)
- Agrupación por proximidad

#### 👃 Detección de Nariz
**Método 1: Análisis de Gradientes**
- Gradiente de Sobel en X e Y
- Magnitud máxima en región central
- Morfología para definir contorno

**Método 2: Análisis de Textura**
- Filtros de Gabor multi-orientación
- Análisis de varianza local
- Región con mayor complejidad

#### 👄 Detección de Boca
**Método 1: Análisis de Color (YCrCb)**
- Conversión a espacio YCrCb
- Canal Cr para detectar tonos rojizos
- Umbralización adaptativa

**Método 2: Detección de Bordes Horizontales**
- Canny + morfología horizontal
- Detección de línea entre labios
- Validación por posición relativa

**Método 3: Proyección Vertical**
- Suma de intensidades por columnas
- Mínimo local indica cavidad bucal
- Expansión a región completa

#### 🔲 Detección de Bordes
**Canny:**
- Suavizado Gaussiano
- Gradiente con Sobel
- Supresión de no-máximos
- Histéresis dual-threshold

**Marr-Hildreth (LoG):**
- Laplaciano de Gaussiano
- Detección de cruces por cero
- Independiente de orientación

#### 🎭 Morfología Matemática
- **Erosión:** Eliminar ruido pequeño
- **Dilatación:** Rellenar huecos
- **Apertura:** Suavizar contornos externos
- **Cierre:** Unir regiones fragmentadas
- **Gradiente morfológico:** Resaltar bordes

---

## 📈 Resultados Esperados

Para cada imagen, el sistema genera:

### Visualización de 20 Paneles:

```
┌───────────────────────────────────────────────────┐
│ AVANCE II: PREPROCESAMIENTO                      │
├───────────────────────────────────────────────────┤
│ 1. Original │ 2. Alineado │ 3. Normalizado       │
│ 4. Mediana  │ 5. Gaussiano │ 6. Laplaciano        │
├───────────────────────────────────────────────────┤
│ AVANCE III: SEGMENTACIÓN FONDO-ROSTRO            │
├───────────────────────────────────────────────────┤
│ 7. Umbral Global │ 8. Umbral Otsu │ 9. Máscara   │
├───────────────────────────────────────────────────┤
│ AVANCE IV-V: RASGOS (SIN LANDMARKS)              │
├───────────────────────────────────────────────────┤
│ 10. Ojos Det. │ 11. Nariz Det. │ 12. Boca Det.   │
│ 13. Ojo Izq   │ 14. Ojo Der    │ 15. Nariz       │
│ 16. Boca      │ 17. Máscara Rasgos               │
├───────────────────────────────────────────────────┤
│ AVANCE V: DETECCIÓN DE BORDES                    │
├───────────────────────────────────────────────────┤
│ 18. Canny     │ 19. Marr-Hildreth │ 20. Morf.   │
└───────────────────────────────────────────────────┘
```

### Métricas:
- Tasa de detección de rostros
- Tasa de detección de rasgos (ojos, nariz, boca)
- Tiempo de procesamiento por imagen
- Comparación de métodos

---

## 🔬 Metodología (Basada en Clases de MATLAB)

### De los códigos de MATLAB aprendimos:

#### Filtros (2do Parcial):
- `filtros_suavizantes.m` → Implementado en `filtros.py`
- `gradiente_laplaciano.m` → Usado en detección de bordes
- `practica4_hightboost.m` → Filtro realzante

#### Detección de Bordes (3er Parcial):
- `Canny.m` → `deteccion_bordes.py` (Canny)
- `MarrHildreht.m` → `deteccion_bordes.py` (Marr-Hildreth)
- `DeteccionBordes.m` → Comparación de métodos

#### Morfología (3er Parcial):
- `ProcMorfoUmbra.m` → `morfologia.py`
- `mejora_morfologica.m` → Operadores morfológicos

---

## 📝 Documentación para Reporte

### Sección 1: Introducción
- Objetivo del Avance V
- Relación con avances anteriores
- Justificación de no usar landmarks

### Sección 2: Marco Teórico
- Detección de bordes (Canny, Marr-Hildreth)
- Morfología matemática
- Espacios de color (RGB, YCrCb)
- Transformada de Hough

### Sección 3: Metodología
- Pipeline completo (Avances II → III → IV → V)
- Algoritmos implementados para cada rasgo
- Parámetros utilizados

### Sección 4: Resultados
- Imágenes procesadas
- Comparación de métodos
- Métricas de desempeño
- Casos exitosos y fallidos

### Sección 5: Conclusiones
- Efectividad de cada método
- Ventajas vs landmarks
- Trabajo futuro (Avance VI)

---

## 👥 Equipo

- **Proyecto:** Detección de Rostros para Reconocimiento de Personas
- **Materia:** Procesamiento Digital de Imágenes
- **Semestre:** Noveno
- **Fecha:** Diciembre 2025

---

## 📚 Referencias

Ver `docs/referencias_avance_i.md` para bibliografía completa.

Principales fuentes:
1. Gonzalez & Woods - "Digital Image Processing" (4th Ed)
2. Szeliski - "Computer Vision: Algorithms and Applications"
3. Papers de detección facial sin landmarks

---

## 🎯 Próximos Pasos

### Para Avance VI:
- [ ] Implementar descriptores (compacidad, distancia radial, etc.)
- [ ] Crear dataset de características por persona
- [ ] Implementar clasificador para reconocimiento
- [ ] Evaluar precisión del sistema completo

---

## 📞 Soporte

Para dudas sobre el código:
- Revisar comentarios en cada módulo
- Consultar `docs/metodologia_avance_v.md`
- Ejecutar notebooks de desarrollo

---

**Última actualización:** 4 de Diciembre de 2025
