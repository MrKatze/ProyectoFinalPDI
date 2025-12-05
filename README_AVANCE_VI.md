# 📊 Avance VI - Descriptores de Forma

## 🎯 Objetivo

Extraer **descriptores geométricos** de los rasgos faciales detectados (ojos, nariz, boca) para caracterizar sus formas mediante medidas cuantitativas.

---

## 📐 Descriptores Implementados

### **1. Compacidad**

Mide qué tan "circular" es una forma comparando área y perímetro.

$$
\text{Compacidad} = \frac{P^2}{4\pi A}
$$

**Interpretación:**
- **1.0**: Círculo perfecto
- **>1.0**: Forma alargada o irregular
- **~1.1-1.3**: Formas ovaladas (ojos)
- **>2.0**: Formas muy alargadas (boca)

---

### **2. Distancia Radial Normalizada**

Mide distancias desde el **centroide** hasta el borde del contorno en 360 ángulos.

#### a) **Media Radial**
Promedio de todas las distancias:

$$
\mu_r = \frac{1}{N} \sum_{i=1}^{N} r_i
$$

Indica el "radio promedio" de la forma.

#### b) **Desviación Estándar Radial**
Variabilidad de las distancias:

$$
\sigma_r = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i - \mu_r)^2}
$$

**Interpretación:**
- **Baja (< 0.1)**: Forma muy regular/circular
- **Media (0.1-0.2)**: Forma moderadamente uniforme
- **Alta (> 0.2)**: Forma irregular con salientes

#### c) **Cruces por Cero**
Cuenta cuántas veces la distancia radial cruza su valor medio.

**Interpretación:**
- **< 10**: Forma suave (círculo, elipse)
- **10-30**: Algunos salientes/entrantes
- **> 30**: Múltiples irregularidades (forma estrellada)

---

### **3. Índice de Área**

Compara el área real con el área de un círculo equivalente:

$$
\text{Índice de Área} = \frac{A_{\text{contorno}}}{\pi \mu_r^2}
$$

**Interpretación:**
- **~0.785**: Forma cuadrada
- **~1.0**: Forma circular
- **<0.7**: Forma cóncava o con huecos
- **>1.0**: No debería ocurrir (error numérico)

---

### **4. Índice de Rugosidad**

Mide la "rugosidad" del borde comparando con su envolvente convexa:

$$
\text{Rugosidad} = \frac{P_{\text{contorno}}}{P_{\text{convex hull}}}
$$

**Interpretación:**
- **~1.0**: Borde completamente suave (convexo)
- **1.0-1.1**: Borde con leves ondulaciones
- **>1.1**: Borde rugoso/dentado

---

## 🛠️ Uso del Pipeline

### **Instalación**

Si aún no lo has hecho:

```bash
cd ProyectoFinalPDI
./instalar.sh
source venv/bin/activate
```

### **Procesar Dataset Completo**

```bash
./avance_vi_pipeline.py
```

### **Procesar Una Imagen**

```bash
./avance_vi_pipeline.py --imagen images/persona1/foto1.jpg
```

### **Especificar Directorios**

```bash
./avance_vi_pipeline.py --imagenes mi_dataset --resultados mi_resultados
```

---

## 📂 Estructura de Resultados

```
resultados/avance_vi/
├── persona1_foto1_descriptores.png   # Visualizaciones
├── persona1_foto2_descriptores.png
├── ...
├── reporte_descriptores.txt          # Reporte textual completo
├── descriptores.csv                  # Datos en CSV (Excel)
└── descriptores.json                 # Datos en JSON
```

---

## 📊 Visualizaciones Generadas

Cada imagen procesada genera un **panel de análisis completo** con:

### **Para Cada Rasgo (Ojos, Nariz, Boca):**

1. **Imagen con Contorno Detectado**
   - Región segmentada con contorno en verde

2. **Gráfica Polar de Distancia Radial**
   - Visualización circular de las distancias normalizadas
   - Muestra la "firma" geométrica de la forma

3. **Contorno vs Convex Hull**
   - Verde: Contorno original
   - Rojo: Envolvente convexa
   - Permite ver concavidades y rugosidad

4. **Tabla de Descriptores**
   - Todos los valores numéricos
   - Formato claro y legible

---

## 📈 Formato de Datos CSV

```csv
Imagen,Rasgo,Area,Perimetro,Compacidad,Media_Radial,Desviacion_Radial,Cruces_Cero,Indice_Area,Indice_Rugosidad,Centroide_X,Centroide_Y
persona1_foto1.jpg,ojos,2345.67,189.45,1.2345,0.8765,0.1234,18,0.9876,1.0234,128.45,87.23
persona1_foto1.jpg,nariz,3456.78,234.56,1.5678,0.7654,0.1543,24,0.8765,1.0456,128.00,145.67
persona1_foto1.jpg,boca,4567.89,278.90,2.1234,0.6543,0.1876,32,0.7654,1.0789,128.12,198.45
...
```

**Fácil de importar en:**
- ✅ Excel / LibreOffice Calc
- ✅ Python (pandas)
- ✅ MATLAB
- ✅ R

---

## 📊 Formato JSON

```json
[
  {
    "nombre_imagen": "persona1_foto1.jpg",
    "rostro_detectado": true,
    "rasgos": {
      "ojos": {
        "area": 2345.67,
        "perimetro": 189.45,
        "compacidad": 1.2345,
        "media_radial": 0.8765,
        "desviacion_radial": 0.1234,
        "cruces_por_cero": 18,
        "indice_area": 0.9876,
        "indice_rugosidad": 1.0234,
        "centroide_x": 128.45,
        "centroide_y": 87.23
      },
      "nariz": { ... },
      "boca": { ... }
    }
  },
  ...
]
```

---

## 🔬 Interpretación de Resultados

### **Ejemplo: Análisis de Ojos**

```
Compacidad: 1.25
→ Forma ligeramente ovalada (esperado para ojos)

Desviación Radial: 0.12
→ Forma moderadamente uniforme

Cruces por Cero: 18
→ Contorno relativamente suave

Índice de Área: 0.95
→ Área cercana a un círculo equivalente

Índice de Rugosidad: 1.05
→ Borde con leves ondulaciones
```

**Conclusión:** Ojos bien detectados con forma regular característica.

---

### **Ejemplo: Análisis de Boca**

```
Compacidad: 2.45
→ Forma muy alargada (esperado para boca cerrada)

Desviación Radial: 0.25
→ Forma con irregularidades notables

Cruces por Cero: 35
→ Múltiples salientes/entrantes

Índice de Rugosidad: 1.18
→ Borde rugoso (labios con textura)
```

**Conclusión:** Boca detectada con características típicas (alargada, bordes irregulares).

---

## 📊 Análisis Estadístico

### **Valores Típicos por Rasgo**

| Rasgo | Compacidad | Desv. Radial | Cruces Cero | Rugosidad |
|-------|------------|--------------|-------------|-----------|
| **Ojos** | 1.1 - 1.4 | 0.08 - 0.15 | 12 - 25 | 1.02 - 1.08 |
| **Nariz** | 1.3 - 1.8 | 0.12 - 0.20 | 18 - 30 | 1.04 - 1.12 |
| **Boca** | 1.8 - 3.0 | 0.15 - 0.30 | 25 - 45 | 1.08 - 1.20 |

*Valores basados en dataset de prueba con rostros frontales bien iluminados.*

---

## 🧪 Aplicaciones

### **1. Reconocimiento Facial**
Usar descriptores como características para clasificación de personas.

### **2. Detección de Emociones**
Cambios en descriptores de boca pueden indicar emociones:
- **Sonrisa**: Mayor compacidad, menor rugosidad
- **Enojo**: Menor compacidad (boca apretada)

### **3. Control de Calidad**
Validar que las detecciones sean correctas:
- Compacidad extrema (>4.0) → Posible error
- Área muy pequeña (<500 px²) → Región dudosa

### **4. Análisis Biométrico**
Descriptores pueden ser invariantes a:
- ✅ Rotación (distancia radial normalizada)
- ✅ Escala (normalización)
- ⚠️ Iluminación (puede afectar segmentación)

---

## 📚 Fundamentos Teóricos

### **Momentos de Hu**

Los descriptores implementados son relacionados con los **Momentos de Hu**, invariantes geométricos usados en reconocimiento de patrones.

### **Análisis de Fourier**

La distancia radial normalizada es una **firma 1D** que puede analizarse con **Descriptores de Fourier** para mayor robustez.

### **Morfología Matemática**

El convex hull usado en rugosidad es un operador morfológico fundamental.

---

## 🐛 Troubleshooting

### **"No se detectaron rostros"**
- Verificar que la imagen tenga rostros frontales
- Mejorar iluminación
- Verificar que Haar Cascade esté cargado correctamente

### **"Sin contornos válidos"**
- La segmentación no encontró regiones suficientemente grandes
- Ajustar umbral mínimo de área (actualmente 100 px²)
- Revisar que la máscara binaria tenga regiones

### **Descriptores con valores extremos**
- **Compacidad > 10**: Contorno muy irregular, posible error de segmentación
- **Rugosidad > 2**: Borde extremadamente dentado, revisar detección
- **Área < 100**: Región demasiado pequeña

---

## 🔧 Personalización

### **Cambiar Número de Puntos Radiales**

En `descriptores_forma.py`:

```python
extractor = DescriptoresForma(num_puntos_radiales=720)  # Mayor resolución
```

### **Filtrar Contornos Pequeños**

En `descriptores_forma.py`, línea ~376:

```python
if cv2.contourArea(contorno) < 500:  # Aumentar umbral
    continue
```

### **Exportar Otros Formatos**

Agregar en `avance_vi_pipeline.py`:

```python
# Excel directo con pandas
import pandas as pd
df = pd.DataFrame(datos)
df.to_excel('descriptores.xlsx', index=False)
```

---

## 📖 Referencias

1. **Gonzalez & Woods** - *Digital Image Processing* (Cap. 11: Representación y Descripción)
2. **Sonka et al.** - *Image Processing, Analysis and Machine Vision* (Cap. 8: Shape Representation)
3. **OpenCV Docs** - [Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)

---

## ✅ Checklist de Uso

- [ ] Dataset organizado en `images/persona*/`
- [ ] Entorno virtual activado (`source venv/bin/activate`)
- [ ] Ejecutar pipeline: `./avance_vi_pipeline.py`
- [ ] Verificar resultados en `resultados/avance_vi/`
- [ ] Revisar reporte textual: `reporte_descriptores.txt`
- [ ] Abrir CSV en Excel: `descriptores.csv`
- [ ] Analizar visualizaciones PNG
- [ ] Interpretar descriptores según tabla de valores típicos

---

## 🎓 Créditos

**Proyecto:** Procesamiento Digital de Imágenes - Avance VI  
**Curso:** PDI Noveno Semestre  
**Fecha:** Diciembre 2025  

---

## 📧 Contacto

Para dudas o reportar issues en el código, revisa la documentación completa o el código fuente con comentarios detallados.

---

**¡Listo para analizar formas faciales! 🚀**
