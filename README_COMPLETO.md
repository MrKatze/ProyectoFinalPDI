# 🎓 Proyecto Final PDI - Avance V
## Sistema de Detección y Segmentación de Rasgos Faciales SIN Landmarks

**Equipo PDI - Noveno Semestre**  
**Fecha:** Diciembre 2025  
**Materia:** Procesamiento Digital de Imágenes

---

## 📋 Descripción

Sistema completo de detección y segmentación de rasgos faciales (ojos, nariz, boca) **sin usar landmarks de dlib**, implementando técnicas de PDI aprendidas en clase:

- ✅ Preprocesamiento (Avance II)
- ✅ Umbralización (Avance III)
- ✅ Identificación de regiones (Avance IV)
- ✅ Segmentación avanzada (Avance V)

### 🎯 Características Principales

1. **Sin Landmarks**: No usa dlib ni modelos preentrenados
2. **8 Métodos de Detección**: 
   - 3 para ojos (Haar Cascade, Proyección, Hough)
   - 2 para nariz (Gradientes, Gabor)
   - 3 para boca (Color, Canny, Proyección)
3. **Basado en MATLAB del Curso**: Implementa Canny, Marr-Hildreth, morfología
4. **Visualizaciones Completas**: 20 paneles mostrando todo el pipeline
5. **Métricas Automáticas**: Tasas de detección y tiempos

---

## 🚀 Instalación y Uso

### 1. Requisitos

```bash
# Python 3.8 o superior
python3 --version

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Organizar Imágenes

```
images/
├── persona1/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
├── persona2/
│   └── ...
└── persona3/
    └── ...
```

### 3. Ejecutar

#### Prueba Rápida (recomendado para empezar)
```bash
./prueba_rapida.py
```

#### Procesar Dataset Completo
```bash
./avance_v_pipeline.py
```

#### Procesar Imagen Específica
```bash
./avance_v_pipeline.py --imagen images/persona1/foto1.jpg
```

---

## 📊 Resultados

El sistema genera:

### Visualizaciones (20 Paneles)
```
resultados/avance_v/
├── persona1/
│   ├── persona1_img1_pipeline.png  ← 20 paneles
│   ├── persona1_img2_pipeline.png
│   └── ...
├── persona2/
│   └── ...
└── metricas.csv                     ← Métricas Excel
```

### Paneles Generados

| Panel | Contenido | Avance |
|-------|-----------|--------|
| 1-4 | Original, Rostro detectado, Alineado, Normalizado | II |
| 5-8 | Filtros: Mediana, Gaussiano, Laplaciano, Highboost | II |
| 9-12 | Umbral Global, Otsu, Máscara Piel, Segmentación | III |
| 13-16 | Canny, Marr-Hildreth, Apertura, Cierre | V |
| 17-20 | Ojos, Nariz, Boca, Resultado Final | V |

---

## 🔬 Metodología

### Avance II: Preprocesamiento

1. **Detección y Alineación**
   ```python
   - Haar Cascade para rostro
   - Detección de ojos
   - Rotación según ángulo de ojos
   - Normalización 256x256
   ```

2. **Filtros Aplicados**
   - **Mediana**: Reduce ruido impulsivo
   - **Gaussiano**: Suavizado
   - **Laplaciano**: Detección de bordes
   - **Highboost**: Realce de detalles

### Avance III: Segmentación Fondo-Rostro

```python
- Umbralización Global (127)
- Método de Otsu (automático)
- Segmentación de piel (YCrCb)
- Combinación AND de máscaras
- Limpieza morfológica
```

### Avance IV: Regiones de Rasgos

Proporciones faciales estándar:
- Ojos: 20%-45% altura, todo el ancho
- Nariz: 35%-65% altura, centro
- Boca: 60%-85% altura, centro

### Avance V: Segmentación Sin Landmarks

#### 👁️ OJOS (3 Métodos)

**Método 1: Haar Cascade + Validación**
```python
- Detecta ojos con clasificadores Haar
- Valida posición (mitad superior)
- Valida tamaño y proporción (0.5 < w/h < 3.0)
```

**Método 2: Proyección Horizontal**
```python
- Analiza varianza horizontal por fila
- Regiones con alta varianza = ojos
- Agrupa regiones cercanas
```

**Método 3: Canny + Hough Circular**
```python
- Detecta bordes con Canny
- Encuentra círculos (pupilas) con Hough
- Radio típico: 5-30 píxeles
```

#### 👃 NARIZ (2 Métodos)

**Método 1: Gradientes (Sobel)**
```python
- Calcula gradientes X e Y
- Nariz tiene gradientes fuertes en ambas direcciones
- Busca en región central (1/3 ancho y alto)
```

**Método 2: Textura (Gabor)**
```python
- Banco de filtros Gabor (4 orientaciones)
- Textura característica de la nariz
- Prefiere formas verticales (h/w > 0.8)
```

#### 👄 BOCA (3 Métodos)

**Método 1: Color (YCrCb)**
```python
- Detecta componente Cr alta (rojo)
- Rango: Cr [140-180], Cb [90-130]
- Labios tienen tonos rojizos
```

**Método 2: Canny + Morfología**
```python
- Detecta bordes con Canny
- Dilatación horizontal para conectar labios
- Cierre morfológico
```

**Método 3: Proyección Vertical**
```python
- Suma columnas (proyección vertical)
- Boca tiene baja intensidad (oscura)
- Busca mínimos locales en centro inferior
```

#### 🔍 Detección de Bordes

**Canny** (como `Canny.m`):
1. Suavizado Gaussiano
2. Cálculo de gradientes (Sobel)
3. Supresión de no-máximos
4. Histéresis (doble umbral)

**Marr-Hildreth** (como `MarrHildreht.m`):
1. Filtro Gaussiano (σ=1.0)
2. Operador Laplaciano
3. Detección de cruces por cero

#### 🔄 Morfología (como `ProcMorfoUmbra.m`)

```python
- Erosión: Reduce regiones blancas
- Dilatación: Expande regiones blancas
- Apertura: Elimina ruido (erosión + dilatación)
- Cierre: Rellena huecos (dilatación + erosión)
- Gradiente: Resalta bordes
- Top-hat: Estructuras brillantes pequeñas
- Black-hat: Estructuras oscuras pequeñas
```

---

## 📈 Métricas

El sistema calcula automáticamente:

### Tasas de Detección
```
Rostros: X / Total (XX%)
Ojos:    X / Total (XX%)
Nariz:   X / Total (XX%)
Boca:    X / Total (XX%)
```

### Tiempos de Procesamiento
```
Tiempo total:    X.XX segundos
Tiempo promedio: X.XXX segundos/imagen
FPS estimado:    X.XX imágenes/segundo
```

---

## 📁 Estructura del Proyecto

```
ProyectoFinalPDI/
│
├── src/                          # Código fuente
│   ├── avance_ii/                # Preprocesamiento
│   │   ├── alineacion.py         # Detección y alineación
│   │   └── filtros.py            # Filtros de imagen
│   │
│   ├── avance_iii/               # Umbralización
│   │   └── umbralizacion.py      # Métodos de umbralización
│   │
│   ├── avance_iv/                # Regiones
│   │   └── deteccion_rasgos.py   # ROIs faciales
│   │
│   ├── avance_v/                 # Segmentación avanzada
│   │   ├── deteccion_bordes.py   # Canny, Marr-Hildreth
│   │   ├── morfologia.py         # Operadores morfológicos
│   │   ├── segmentador_ojos.py   # 3 métodos
│   │   ├── segmentador_nariz.py  # 2 métodos
│   │   └── segmentador_boca.py   # 3 métodos
│   │
│   └── utils/                    # Utilidades
│       ├── cargador_imagenes.py  # Carga de dataset
│       ├── visualizacion.py      # Generación de gráficos
│       └── metricas.py           # Cálculo de métricas
│
├── images/                       # Dataset de imágenes
├── resultados/                   # Resultados generados
├── docs/                         # Documentación
├── notebooks/                    # Notebooks Jupyter
│
├── avance_v_pipeline.py          # 🚀 Script principal
├── prueba_rapida.py              # 🧪 Prueba rápida
├── requirements.txt              # Dependencias
├── README.md                     # Este archivo
└── EMPEZAR_AQUI.md               # Guía de inicio
```

---

## 🔧 Solución de Problemas

### Error: "No se detectó rostro"
- Verificar que el rostro sea frontal
- Mejorar iluminación de la imagen
- Usar imagen con mayor resolución

### Error: "No se detectaron ojos/nariz/boca"
- Normal en imágenes difíciles
- El sistema usa consenso de múltiples métodos
- Revisar visualización intermedia

### Errores de importación
```bash
pip install --upgrade opencv-python numpy matplotlib scipy scikit-image
```

---

## 📚 Referencias

### Códigos MATLAB del Curso
- `filtros_suavizantes.m` → `filtros.py`
- `practica4_hightboost.m` → Método highboost
- `Canny.m` → `deteccion_bordes.py`
- `MarrHildreht.m` → `deteccion_bordes.py`
- `ProcMorfoUmbra.m` → `morfologia.py`
- `DeteccionBordes.m` → Comparación de métodos

### Técnicas Implementadas
1. **Filtrado**: Mediana, Gaussiano, Laplaciano, Highboost
2. **Umbralización**: Global, Otsu, Adaptativa
3. **Detección de Bordes**: Canny, Marr-Hildreth, Sobel, Prewitt
4. **Morfología**: Erosión, Dilatación, Apertura, Cierre
5. **Segmentación**: Color (YCrCb, HSV), Textura (Gabor)
6. **Detección**: Haar Cascade, Hough, Proyección

---

## 🎯 Diferencias con Versión Anterior

| Aspecto | Versión Anterior | Esta Versión |
|---------|------------------|--------------|
| Landmarks | ✗ Usaba dlib | ✅ Sin landmarks |
| Estructura | Mezclada | ✅ Por Avances |
| Métodos | Pre-entrenados | ✅ PDI desde cero |
| Visualización | Básica | ✅ 20 paneles |
| Documentación | Mínima | ✅ Completa |

---

## 📞 Para el Reporte

Usar:
1. **Visualizaciones**: `resultados/avance_v/persona*/` (20 paneles)
2. **Métricas**: `resultados/avance_v/metricas.csv`
3. **Reporte**: `resultados/avance_v/reporte_metricas.txt`
4. **Este README**: Documentación de metodología

---

## ✨ Características Destacadas

✅ **Sin Landmarks**: Cumple con requisito de la profesora  
✅ **8 Métodos**: 3+2+3 para ojos, nariz, boca  
✅ **Consenso**: Vota entre múltiples métodos  
✅ **Completo**: Integra Avances II→III→IV→V  
✅ **Documentado**: README, guías, comentarios  
✅ **Basado en MATLAB**: Adapta códigos del curso  
✅ **Automático**: Procesa dataset completo  
✅ **Visualizable**: 20 paneles por imagen  

---

## 📝 Licencia

Este proyecto es parte del trabajo académico para la materia de Procesamiento Digital de Imágenes, Noveno Semestre.

---

## 👥 Autores

**Equipo PDI** - Noveno Semestre  
Procesamiento Digital de Imágenes  
Diciembre 2025

---

**¡Proyecto completo y listo para usar! 🎉**
