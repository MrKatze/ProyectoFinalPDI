# Guía de Uso - Proyecto Final PDI

## 🚀 Inicio Rápido

### 1. Configurar el entorno

```bash
# Navegar al directorio del proyecto
cd "ProyectoFinalPDI"

# Crear entorno virtual (si no existe)
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
# O en Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Organizar las imágenes

Estructura requerida:
```
images/
├── persona1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── persona2/
│   ├── img1.jpg
│   └── ...
└── persona3/
    └── ...
```

### 3. Ejecutar el sistema

#### Opción A: Prueba rápida (1 imagen)
```bash
python3 prueba_rapida.py
```

#### Opción B: Procesar dataset completo
```bash
python3 avance_v_pipeline.py --imagenes images --salida resultados/avance_v
```

#### Opción C: Procesar una imagen específica
```bash
python3 avance_v_pipeline.py --imagen images/persona1/img1.jpg --salida resultados
```

## 📊 Resultados Generados

El sistema genera:

### 1. Visualizaciones (20 paneles)
Para cada imagen procesada se crea una visualización mostrando:
- Paneles 1-4: Preprocesamiento (Avance II)
- Paneles 5-8: Filtros aplicados
- Paneles 9-12: Umbralización y segmentación (Avance III)
- Paneles 13-16: Detección de bordes y morfología (Avance V)
- Paneles 17-20: Rasgos detectados (ojos, nariz, boca, resultado final)

### 2. Métricas
- `reporte_metricas.txt`: Reporte textual completo
- `metricas.csv`: Métricas en formato CSV para Excel

## 🔧 Estructura del Código

```
src/
├── avance_ii/          # Preprocesamiento
│   ├── alineacion.py   # Detección y alineación de rostros
│   └── filtros.py      # Filtros (mediana, gaussiano, laplaciano, etc.)
│
├── avance_iii/         # Umbralización
│   └── umbralizacion.py # Métodos de umbralización (global, Otsu)
│
├── avance_iv/          # Regiones de rasgos
│   └── deteccion_rasgos.py # Identificación de regiones faciales
│
├── avance_v/           # Segmentación avanzada
│   ├── deteccion_bordes.py    # Canny y Marr-Hildreth
│   ├── morfologia.py          # Operadores morfológicos
│   ├── segmentador_ojos.py    # 3 métodos para ojos
│   ├── segmentador_nariz.py   # 2 métodos para nariz
│   └── segmentador_boca.py    # 3 métodos para boca
│
└── utils/              # Utilidades
    ├── cargador_imagenes.py # Carga del dataset
    ├── visualizacion.py     # Generación de visualizaciones
    └── metricas.py          # Cálculo de métricas
```

## 📈 Metodología

### Avance II: Preprocesamiento
1. Detección de rostros con Haar Cascade
2. Detección de ojos para alineación
3. Rotación basada en ángulo de ojos
4. Normalización de tamaño (256x256)
5. Normalización de iluminación (CLAHE)
6. Filtros:
   - Mediana (reduce ruido impulsivo)
   - Gaussiano (suavizado)
   - Laplaciano (detección de bordes)
   - Highboost (realce de detalles)

### Avance III: Segmentación Fondo-Rostro
1. Umbralización global (valor fijo)
2. Método de Otsu (umbral automático)
3. Segmentación de piel (YCrCb)
4. Combinación de máscaras
5. Limpieza morfológica

### Avance IV: Regiones de Rasgos
- División del rostro en regiones basadas en proporciones faciales estándar
- Ojos: 20%-45% altura
- Nariz: 30%-65% altura, centrado
- Boca: 60%-85% altura

### Avance V: Segmentación Sin Landmarks

#### Ojos (3 métodos):
1. **Haar Cascade** + validación geométrica
2. **Proyección horizontal** + análisis de varianza
3. **Canny + Hough circular** (detección de pupilas)

#### Nariz (2 métodos):
1. **Análisis de gradientes** (Sobel) en región central
2. **Análisis de textura** con filtros de Gabor

#### Boca (3 métodos):
1. **Segmentación por color** (YCrCb - labios rojizos)
2. **Canny + morfología** + análisis horizontal
3. **Proyección vertical** + análisis de intensidad

#### Detección de Bordes:
- **Canny**: Supresión de no-máximos + histéresis
- **Marr-Hildreth**: Laplaciano de Gaussiana + cruces por cero

#### Morfología:
- Erosión, dilatación, apertura, cierre
- Gradiente morfológico, top-hat, black-hat
- Limpieza de componentes, relleno de huecos

## 💡 Tips para Mejores Resultados

1. **Calidad de imágenes**: Usar fotos con buena iluminación y rostros frontales
2. **Tamaño**: Las imágenes se redimensionan automáticamente, pero evitar imágenes muy pequeñas (<200x200)
3. **Fondo**: Fondos simples facilitan la segmentación
4. **Expresión**: Rostros neutros dan mejores resultados que expresiones extremas

## 🐛 Solución de Problemas

### "No se detectó rostro"
- Verificar que la imagen tenga un rostro visible
- Probar con otra imagen con mejor iluminación
- El rostro debe estar relativamente frontal

### "No se detectaron ojos/nariz/boca"
- Normal en algunas imágenes difíciles
- El sistema usa consenso de múltiples métodos
- Revisar visualización intermedia para debugging

### Errores de importación
```bash
# Asegurarse de estar en el directorio correcto
cd ProyectoFinalPDI

# Reinstalar dependencias
pip install -r requirements.txt
```

## 📚 Para el Reporte

El sistema genera automáticamente:

1. **Visualizaciones completas**: Muestran cada paso del proceso
2. **Métricas cuantitativas**: Tasas de detección y tiempos
3. **Comparaciones**: Entre diferentes métodos de cada rasgo

Para el reporte pueden usar:
- Las imágenes de `resultados/avance_v/`
- Las métricas de `metricas.csv`
- El reporte textual `reporte_metricas.txt`

## 📞 Contacto

Para dudas o problemas, revisar:
- `README.md`: Documentación completa del proyecto
- `EMPEZAR_AQUI.md`: Guía de inicio rápido
- Este archivo: Guía de uso detallada
