#!/usr/bin/env python3
"""
================================================================================
SEGMENTACIÓN ROSTRO-FONDO Y RECORTE AUTOMÁTICO
================================================================================
Sistema que implementa:
1. Segmentación rostro vs fondo (máscara binaria)
2. Detección de transiciones negro→blanco (bordes de ROI)
3. Recorte automático de región facial
4. Procesamiento completo con la región recortada

Técnicas implementadas desde cero:
- Umbralización adaptativa (Otsu manual)
- Operaciones morfológicas (erosión/dilatación)
- Detección de componentes conexas
- Extracción de bounding box
================================================================================
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import sys


# ============================================================================
# MÓDULO 1: SEGMENTACIÓN ROSTRO-FONDO
# ============================================================================

class SegmentadorRostroFondo:
    """Separa rostro del fondo usando múltiples técnicas"""
    
    @staticmethod
    def convertir_a_gris(imagen: np.ndarray) -> np.ndarray:
        """Conversión manual a escala de grises"""
        if len(imagen.shape) == 2:
            return imagen
        gris = (0.299 * imagen[:,:,2] + 
                0.587 * imagen[:,:,1] + 
                0.114 * imagen[:,:,0])
        return gris.astype(np.uint8)
    
    @staticmethod
    def calcular_umbral_otsu(imagen: np.ndarray) -> int:
        """
        Método de Otsu implementado desde cero
        Encuentra umbral óptimo maximizando varianza entre clases
        """
        # Calcular histograma
        histograma = np.zeros(256)
        h, w = imagen.shape
        total_pixeles = h * w
        
        for i in range(h):
            for j in range(w):
                histograma[imagen[i, j]] += 1
        
        # Normalizar histograma (probabilidades)
        probabilidades = histograma / total_pixeles
        
        # Calcular media global
        media_global = 0
        for i in range(256):
            media_global += i * probabilidades[i]
        
        # Buscar umbral óptimo
        mejor_umbral = 0
        max_varianza = 0
        
        w0 = 0  # Peso clase 0 (fondo)
        suma0 = 0  # Suma ponderada clase 0
        
        for t in range(256):
            w0 += probabilidades[t]
            if w0 == 0:
                continue
            
            w1 = 1 - w0  # Peso clase 1 (objeto)
            if w1 == 0:
                break
            
            suma0 += t * probabilidades[t]
            
            media0 = suma0 / w0
            media1 = (media_global - suma0) / w1
            
            # Varianza entre clases
            varianza_entre = w0 * w1 * (media0 - media1) ** 2
            
            if varianza_entre > max_varianza:
                max_varianza = varianza_entre
                mejor_umbral = t
        
        return mejor_umbral
    
    @staticmethod
    def umbralizar_otsu(imagen: np.ndarray) -> np.ndarray:
        """
        Aplica umbralización de Otsu
        Rostro (tonos medios/claros) → blanco (255)
        Fondo (muy oscuro o muy claro) → negro (0)
        """
        umbral = SegmentadorRostroFondo.calcular_umbral_otsu(imagen)
        print(f"  → Umbral de Otsu calculado: {umbral}")
        
        # Crear máscara inicial
        mascara = np.zeros_like(imagen)
        mascara[imagen > umbral] = 255
        
        # Verificar si debemos invertir (contar píxeles blancos)
        # Si más del 60% es blanco, probablemente el fondo es blanco → invertir
        porcentaje_blanco = np.sum(mascara == 255) / mascara.size
        print(f"  → Porcentaje de píxeles blancos: {porcentaje_blanco*100:.1f}%")
        
        if porcentaje_blanco > 0.6:
            print(f"  → Invirtiendo máscara (fondo detectado como blanco)")
            mascara = 255 - mascara
        
        return mascara
    
    @staticmethod
    def erosion(imagen: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Erosión morfológica manual
        Elimina ruido pequeño (píxeles aislados)
        """
        h, w = imagen.shape
        resultado = np.zeros_like(imagen)
        pad = kernel_size // 2
        img_pad = np.pad(imagen, pad, mode='constant', constant_values=0)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+kernel_size, j:j+kernel_size]
                # Si todos los píxeles en la ventana son 255, mantener
                if np.all(ventana == 255):
                    resultado[i, j] = 255
        
        return resultado
    
    @staticmethod
    def dilatacion(imagen: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Dilatación morfológica manual
        Rellena huecos pequeños
        """
        h, w = imagen.shape
        resultado = np.zeros_like(imagen)
        pad = kernel_size // 2
        img_pad = np.pad(imagen, pad, mode='constant', constant_values=0)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+kernel_size, j:j+kernel_size]
                # Si al menos un píxel en la ventana es 255, activar
                if np.any(ventana == 255):
                    resultado[i, j] = 255
        
        return resultado
    
    @staticmethod
    def apertura(imagen: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apertura morfológica: erosión seguida de dilatación
        Elimina ruido pero preserva forma
        """
        erosionada = SegmentadorRostroFondo.erosion(imagen, kernel_size)
        abierta = SegmentadorRostroFondo.dilatacion(erosionada, kernel_size)
        return abierta
    
    @staticmethod
    def cierre(imagen: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Cierre morfológico: dilatación seguida de erosión
        Rellena huecos pero preserva forma
        """
        dilatada = SegmentadorRostroFondo.dilatacion(imagen, kernel_size)
        cerrada = SegmentadorRostroFondo.erosion(dilatada, kernel_size)
        return cerrada
    
    @staticmethod
    def obtener_componente_principal(mascara: np.ndarray) -> np.ndarray:
        """
        Extrae la componente conexa más grande (el rostro)
        Elimina ruido de fondo
        """
        # Etiquetar componentes conexas (algoritmo básico)
        h, w = mascara.shape
        etiquetas = np.zeros_like(mascara, dtype=np.int32)
        etiqueta_actual = 1
        
        # Primera pasada: asignar etiquetas preliminares
        equivalencias = {}
        
        for i in range(h):
            for j in range(w):
                if mascara[i, j] == 255:
                    vecinos = []
                    
                    # Revisar vecinos ya procesados (arriba e izquierda)
                    if i > 0 and etiquetas[i-1, j] > 0:
                        vecinos.append(etiquetas[i-1, j])
                    if j > 0 and etiquetas[i, j-1] > 0:
                        vecinos.append(etiquetas[i, j-1])
                    
                    if len(vecinos) == 0:
                        # Nuevo componente
                        etiquetas[i, j] = etiqueta_actual
                        equivalencias[etiqueta_actual] = etiqueta_actual
                        etiqueta_actual += 1
                    else:
                        # Usar etiqueta mínima de vecinos
                        min_etiqueta = min(vecinos)
                        etiquetas[i, j] = min_etiqueta
                        
                        # Registrar equivalencias
                        for v in vecinos:
                            if v != min_etiqueta:
                                equivalencias[v] = min_etiqueta
        
        # Resolver equivalencias transitivas
        for key in equivalencias:
            while equivalencias[key] != equivalencias[equivalencias[key]]:
                equivalencias[key] = equivalencias[equivalencias[key]]
        
        # Segunda pasada: aplicar equivalencias
        for i in range(h):
            for j in range(w):
                if etiquetas[i, j] > 0:
                    etiquetas[i, j] = equivalencias[etiquetas[i, j]]
        
        # Contar tamaño de cada componente
        tamanios = {}
        for i in range(h):
            for j in range(w):
                if etiquetas[i, j] > 0:
                    tamanios[etiquetas[i, j]] = tamanios.get(etiquetas[i, j], 0) + 1
        
        if len(tamanios) == 0:
            return mascara
        
        # Encontrar componente más grande
        etiqueta_principal = max(tamanios, key=tamanios.get)
        print(f"  → Componente principal: {tamanios[etiqueta_principal]} píxeles")
        
        # Crear máscara con solo componente principal
        resultado = np.zeros_like(mascara)
        resultado[etiquetas == etiqueta_principal] = 255
        
        return resultado
    
    @staticmethod
    def detectar_piel_rgb(imagen_color: np.ndarray) -> np.ndarray:
        """
        Detecta piel usando rangos RGB empíricos
        Método complementario a Otsu
        """
        # Rangos RGB típicos de piel humana ajustados
        R = imagen_color[:,:,2].astype(np.float32)
        G = imagen_color[:,:,1].astype(np.float32)
        B = imagen_color[:,:,0].astype(np.float32)
        
        mascara = np.zeros((imagen_color.shape[0], imagen_color.shape[1]), dtype=np.uint8)
        
        # Condiciones más estrictas para piel
        cond1 = (R > 95) & (G > 40) & (B > 20)
        cond2 = (R > G) & (R > B) & (G > B)
        cond3 = np.abs(R - G) > 15
        cond4 = (R < 240) & (G < 230) & (B < 220)  # Evitar zonas muy claras
        cond5 = (R > 120) | (G > 80)  # Al menos cierta intensidad
        
        mascara[cond1 & cond2 & cond3 & cond4 & cond5] = 255
        
        return mascara
    
    @staticmethod
    def enfoque_region_central(mascara: np.ndarray, factor: float = 0.7) -> np.ndarray:
        """
        Prioriza la región central de la imagen (donde suele estar el rostro)
        Aplica un peso mayor al centro usando una máscara gaussiana
        """
        h, w = mascara.shape
        centro_y, centro_x = h // 2, w // 2
        
        # Crear máscara de peso gaussiano centrada
        Y, X = np.ogrid[:h, :w]
        dist_centro = np.sqrt((X - centro_x)**2 + (Y - centro_y)**2)
        max_dist = np.sqrt(centro_x**2 + centro_y**2)
        
        # Peso gaussiano (mayor en el centro)
        peso = np.exp(-(dist_centro**2) / (2 * (max_dist * factor)**2))
        
        # Aplicar peso a la máscara
        mascara_ponderada = (mascara.astype(np.float32) / 255.0) * peso
        
        # Umbralizar para obtener máscara binaria
        umbral_central = np.percentile(mascara_ponderada[mascara_ponderada > 0], 50)
        resultado = np.zeros_like(mascara)
        resultado[mascara_ponderada > umbral_central] = 255
        
        return resultado
    
    @staticmethod
    def segmentar_rostro(imagen_gris: np.ndarray, imagen_color: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pipeline completo de segmentación rostro-fondo
        Usa Otsu + detección de piel (si hay imagen color)
        
        Returns:
            Máscara binaria (rostro=255, fondo=0)
        """
        print("\n[SEGMENTACIÓN ROSTRO-FONDO]")
        
        # 1. Umbralización de Otsu
        print("  1. Aplicando umbralización de Otsu...")
        mascara_otsu = SegmentadorRostroFondo.umbralizar_otsu(imagen_gris)
        
        # 2. Si hay imagen color, usar detección de piel
        if imagen_color is not None:
            print("  2. Aplicando detección de piel RGB...")
            mascara_piel = SegmentadorRostroFondo.detectar_piel_rgb(imagen_color)
            
            # Combinar máscaras (intersección para mayor precisión)
            print("  3. Combinando máscaras (Otsu ∩ Piel)...")
            mascara = np.zeros_like(mascara_otsu)
            mascara[(mascara_otsu == 255) & (mascara_piel == 255)] = 255
            
            porcentaje_piel = np.sum(mascara_piel == 255) / mascara_piel.size * 100
            porcentaje_combinado = np.sum(mascara == 255) / mascara.size * 100
            print(f"     → Detección de piel: {porcentaje_piel:.1f}%")
            print(f"     → Intersección: {porcentaje_combinado:.1f}%")
        else:
            mascara = mascara_otsu
        
        # 2.5 Enfoque en región central
        print("  3.5. Priorizando región central...")
        mascara = SegmentadorRostroFondo.enfoque_region_central(mascara, factor=0.6)
        
        # 2.6 INVERTIR ANTES de morfología: lo que era piel (blanco) ahora es el rostro
        # Pero necesitamos que el ROSTRO sea blanco, y está como NEGRO después del enfoque
        # Verificar qué es mayoría
        porcentaje_blanco_actual = np.sum(mascara == 255) / mascara.size
        if porcentaje_blanco_actual > 0.5:
            # Si más del 50% es blanco, significa que el fondo está en blanco → invertir
            print("  3.6. Invirtiendo máscara (rostro como objeto principal)...")
            mascara = 255 - mascara
        
        # 3. Operaciones morfológicas para limpiar
        print("  4. Aplicando apertura morfológica (eliminar ruido)...")
        mascara = SegmentadorRostroFondo.apertura(mascara, kernel_size=5)
        
        print("  5. Aplicando cierre morfológico (rellenar huecos)...")
        mascara = SegmentadorRostroFondo.cierre(mascara, kernel_size=7)
        
        # 4. Extraer componente principal
        print("  6. Extrayendo componente conexa principal...")
        mascara = SegmentadorRostroFondo.obtener_componente_principal(mascara)
        
        # 5. Dilatación final para incluir bordes
        print("  7. Dilatación final para incluir bordes del rostro...")
        mascara = SegmentadorRostroFondo.dilatacion(mascara, kernel_size=9)
        
        print("  ✓ Segmentación completada")
        
        return mascara


# ============================================================================
# MÓDULO 2: DETECCIÓN DE TRANSICIONES Y RECORTE
# ============================================================================

class DetectorTransiciones:
    """Detecta bordes de transición negro→blanco y extrae bounding box"""
    
    @staticmethod
    def detectar_bordes_mascara(mascara: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detecta transiciones de negro (0) a blanco (255)
        Encuentra el bounding box mínimo que contiene el rostro
        
        Returns:
            (x, y, ancho, alto) - Coordenadas del rectángulo de recorte
        """
        print("\n[DETECCIÓN DE TRANSICIONES]")
        
        # Encontrar píxeles blancos (rostro)
        filas_blancas = np.any(mascara == 255, axis=1)
        cols_blancas = np.any(mascara == 255, axis=0)
        
        if not np.any(filas_blancas) or not np.any(cols_blancas):
            print("  ✗ No se detectó región blanca")
            return (0, 0, mascara.shape[1], mascara.shape[0])
        
        # Encontrar límites
        y_min = np.argmax(filas_blancas)
        y_max = len(filas_blancas) - np.argmax(filas_blancas[::-1]) - 1
        
        x_min = np.argmax(cols_blancas)
        x_max = len(cols_blancas) - np.argmax(cols_blancas[::-1]) - 1
        
        ancho = x_max - x_min + 1
        alto = y_max - y_min + 1
        
        print(f"  → Transición detectada en:")
        print(f"    Superior: y = {y_min}")
        print(f"    Inferior: y = {y_max}")
        print(f"    Izquierda: x = {x_min}")
        print(f"    Derecha: x = {x_max}")
        print(f"  ✓ Bounding box: ({x_min}, {y_min}, {ancho}, {alto})")
        
        return (x_min, y_min, ancho, alto)
    
    @staticmethod
    def aplicar_margen(bbox: Tuple[int, int, int, int], 
                      margen_porcentaje: float,
                      limite_h: int, 
                      limite_w: int) -> Tuple[int, int, int, int]:
        """
        Añade margen alrededor del bounding box
        
        Args:
            bbox: (x, y, ancho, alto)
            margen_porcentaje: Margen como % del tamaño (ej: 0.1 = 10%)
            limite_h, limite_w: Dimensiones de la imagen
        """
        x, y, w, h = bbox
        
        # Calcular margen
        margen_w = int(w * margen_porcentaje)
        margen_h = int(h * margen_porcentaje)
        
        # Aplicar margen con límites
        x_nuevo = max(0, x - margen_w)
        y_nuevo = max(0, y - margen_h)
        w_nuevo = min(limite_w - x_nuevo, w + 2*margen_w)
        h_nuevo = min(limite_h - y_nuevo, h + 2*margen_h)
        
        print(f"  → Margen añadido: {margen_porcentaje*100:.0f}% ({margen_w}px H, {margen_h}px V)")
        print(f"  ✓ BBox con margen: ({x_nuevo}, {y_nuevo}, {w_nuevo}, {h_nuevo})")
        
        return (x_nuevo, y_nuevo, w_nuevo, h_nuevo)
    
    @staticmethod
    def recortar_imagen(imagen: np.ndarray, 
                       bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Recorta imagen según bounding box
        """
        x, y, w, h = bbox
        
        # Validar límites
        if x < 0 or y < 0 or x+w > imagen.shape[1] or y+h > imagen.shape[0]:
            print("  ⚠ BBox fuera de límites, ajustando...")
            x = max(0, x)
            y = max(0, y)
            w = min(imagen.shape[1] - x, w)
            h = min(imagen.shape[0] - y, h)
        
        recortada = imagen[y:y+h, x:x+w]
        print(f"  ✓ Imagen recortada: {recortada.shape}")
        
        return recortada


# ============================================================================
# MÓDULO 3: PIPELINE INTEGRADO
# ============================================================================

class PipelineSegmentacionRecorte:
    """Pipeline completo: segmentar → detectar transiciones → recortar → procesar"""
    
    def __init__(self):
        self.segmentador = SegmentadorRostroFondo()
        self.detector = DetectorTransiciones()
        self.resultados = {}
    
    def procesar_imagen(self, ruta_imagen: str, margen: float = 0.15) -> Dict:
        """
        Pipeline completo
        
        Args:
            ruta_imagen: Path a la imagen
            margen: Margen alrededor del rostro (0.15 = 15%)
        
        Returns:
            Diccionario con resultados
        """
        print("\n" + "="*80)
        print("PIPELINE: SEGMENTACIÓN ROSTRO-FONDO Y RECORTE AUTOMÁTICO")
        print("="*80)
        print(f"\nProcesando: {os.path.basename(ruta_imagen)}")
        
        # ===== 1. CARGAR IMAGEN =====
        print("\n[1/5] Cargando imagen...")
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"  ✗ Error al cargar {ruta_imagen}")
            return {}
        
        h_orig, w_orig = imagen.shape[:2]
        print(f"  ✓ Imagen cargada: {w_orig}x{h_orig}")
        
        gris = self.segmentador.convertir_a_gris(imagen)
        
        self.resultados['original'] = imagen
        self.resultados['gris'] = gris
        
        # ===== 2. SEGMENTAR ROSTRO-FONDO =====
        print("\n[2/5] Segmentando rostro del fondo...")
        mascara = self.segmentador.segmentar_rostro(gris, imagen_color=imagen)
        self.resultados['mascara'] = mascara
        
        # ===== 3. DETECTAR TRANSICIONES =====
        print("\n[3/5] Detectando transiciones negro→blanco...")
        bbox = self.detector.detectar_bordes_mascara(mascara)
        
        # Añadir margen
        bbox_con_margen = self.detector.aplicar_margen(
            bbox, margen, h_orig, w_orig
        )
        
        self.resultados['bbox_original'] = bbox
        self.resultados['bbox_margen'] = bbox_con_margen
        
        # ===== 4. RECORTAR IMAGEN =====
        print("\n[4/5] Recortando región facial...")
        imagen_recortada = self.detector.recortar_imagen(imagen, bbox_con_margen)
        gris_recortada = self.detector.recortar_imagen(gris, bbox_con_margen)
        mascara_recortada = self.detector.recortar_imagen(mascara, bbox_con_margen)
        
        self.resultados['imagen_recortada'] = imagen_recortada
        self.resultados['gris_recortada'] = gris_recortada
        self.resultados['mascara_recortada'] = mascara_recortada
        
        # ===== 5. APLICAR MÁSCARA =====
        print("\n[5/5] Aplicando máscara a imagen recortada...")
        # Crear imagen con fondo negro
        imagen_con_fondo_negro = imagen_recortada.copy()
        
        # Convertir mascara recortada a 3 canales
        if len(imagen_recortada.shape) == 3:
            mascara_3ch = np.stack([mascara_recortada]*3, axis=-1)
            imagen_con_fondo_negro[mascara_3ch == 0] = 0
        else:
            imagen_con_fondo_negro[mascara_recortada == 0] = 0
        
        self.resultados['imagen_fondo_negro'] = imagen_con_fondo_negro
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETADO")
        print("="*80)
        print(f"\nResumen:")
        print(f"  • Imagen original: {w_orig}x{h_orig}")
        print(f"  • Región detectada: {bbox[2]}x{bbox[3]}")
        print(f"  • Región recortada: {bbox_con_margen[2]}x{bbox_con_margen[3]}")
        print(f"  • Reducción: {100*(1-bbox_con_margen[2]*bbox_con_margen[3]/(w_orig*h_orig)):.1f}%")
        
        return self.resultados
    
    def visualizar_resultados(self, nombre_archivo: str):
        """
        Genera visualización completa del proceso
        """
        print("\n[GENERANDO VISUALIZACIÓN]")
        
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Pipeline: Segmentación Rostro-Fondo y Recorte Automático', 
                    fontsize=16, fontweight='bold')
        
        # Fila 1: Proceso de segmentación
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(self.resultados['original'], cv2.COLOR_BGR2RGB))
        plt.title('1. Imagen Original', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(self.resultados['gris'], cmap='gray')
        plt.title('2. Escala de Grises', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 3)
        plt.imshow(self.resultados['mascara'], cmap='gray')
        plt.title('3. Máscara (Rostro=Blanco)', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 4)
        # Dibujar bounding boxes
        img_bbox = self.resultados['original'].copy()
        x1, y1, w1, h1 = self.resultados['bbox_original']
        cv2.rectangle(img_bbox, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 3)
        x2, y2, w2, h2 = self.resultados['bbox_margen']
        cv2.rectangle(img_bbox, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
        plt.title('4. BBox Detectado\nRojo=sin margen, Verde=con margen', fontweight='bold')
        plt.axis('off')
        
        # Fila 2: Resultados del recorte
        plt.subplot(2, 4, 5)
        plt.imshow(cv2.cvtColor(self.resultados['imagen_recortada'], cv2.COLOR_BGR2RGB))
        plt.title('5. Imagen Recortada', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        plt.imshow(self.resultados['gris_recortada'], cmap='gray')
        plt.title('6. Gris Recortada', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        plt.imshow(self.resultados['mascara_recortada'], cmap='gray')
        plt.title('7. Máscara Recortada', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        plt.imshow(cv2.cvtColor(self.resultados['imagen_fondo_negro'], cv2.COLOR_BGR2RGB))
        plt.title('8. Resultado Final\n(Rostro + Fondo Negro)', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Guardar
        nombre_base = os.path.splitext(os.path.basename(nombre_archivo))[0]
        ruta_salida = f"segmentacion_recorte_{nombre_base}.png"
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualización guardada: {ruta_salida}")
        
        # Guardar imagen recortada individual
        ruta_recortada = f"rostro_recortado_{nombre_base}.png"
        cv2.imwrite(ruta_recortada, self.resultados['imagen_recortada'])
        print(f"  ✓ Imagen recortada guardada: {ruta_recortada}")
        
        plt.close()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del programa"""
    
    print("="*80)
    print("SEGMENTACIÓN ROSTRO-FONDO Y RECORTE AUTOMÁTICO")
    print("="*80)
    print("\nTécnicas implementadas:")
    print("  ✓ Umbralización de Otsu (desde cero)")
    print("  ✓ Operaciones morfológicas (erosión, dilatación, apertura, cierre)")
    print("  ✓ Detección de componentes conexas")
    print("  ✓ Extracción de bounding box por transiciones")
    print("  ✓ Recorte automático con margen configurable")
    print("="*80)
    
    # Buscar imágenes
    carpeta_images = "images"
    
    if not os.path.exists(carpeta_images):
        print(f"\n❌ Error: No existe la carpeta '{carpeta_images}'")
        return
    
    # Buscar imágenes
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp']
    imagenes = []
    
    for root, dirs, files in os.walk(carpeta_images):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensiones):
                imagenes.append(os.path.join(root, file))
    
    if len(imagenes) == 0:
        print(f"\n❌ No se encontraron imágenes en '{carpeta_images}'")
        return
    
    print(f"\n📁 Encontradas {len(imagenes)} imágenes")
    
    # ===== SELECCIÓN INTERACTIVA DE IMAGEN =====
    imagen_seleccionada = None
    
    if len(imagenes) == 1:
        imagen_seleccionada = imagenes[0]
        print(f"\n→ Solo hay una imagen, procesando: {os.path.basename(imagen_seleccionada)}")
    else:
        print("\n" + "="*60)
        print("IMÁGENES DISPONIBLES:")
        print("="*60)
        
        # Mostrar todas las imágenes con números
        for i, img in enumerate(imagenes, 1):
            print(f"  {i:2d}. {os.path.basename(img)}")
        
        print("="*60)
        
        # Pedir selección al usuario
        try:
            seleccion = input(f"\nSelecciona una imagen (1-{len(imagenes)}, Enter=1): ").strip()
            
            if seleccion == "":
                indice = 0
            else:
                indice = int(seleccion) - 1
                
            if 0 <= indice < len(imagenes):
                imagen_seleccionada = imagenes[indice]
                print(f"✓ Seleccionada: {os.path.basename(imagen_seleccionada)}")
            else:
                print(f"⚠ Índice fuera de rango, usando imagen 1")
                imagen_seleccionada = imagenes[0]
        except (ValueError, KeyboardInterrupt):
            print(f"\n⚠ Entrada inválida, usando imagen 1")
            imagen_seleccionada = imagenes[0]
    
    # Configuración de margen
    print("\n⚙️  Configuración:")
    try:
        margen_input = input("Margen alrededor del rostro (0.0-0.5, Enter=0.15): ").strip()
        margen = float(margen_input) if margen_input else 0.15
        margen = max(0.0, min(0.5, margen))  # Limitar entre 0 y 0.5
    except:
        margen = 0.15
    
    print(f"  → Margen: {margen*100:.0f}%")
    
    # Crear pipeline y procesar
    pipeline = PipelineSegmentacionRecorte()
    resultados = pipeline.procesar_imagen(imagen_seleccionada, margen=margen)
    
    if resultados:
        # Generar visualización
        pipeline.visualizar_resultados(imagen_seleccionada)
        
        print("\n✅ Procesamiento completado exitosamente")
        print("\n📊 Archivos generados:")
        print("  • segmentacion_recorte_*.png (visualización completa)")
        print("  • rostro_recortado_*.png (imagen recortada)")
        print("\n💡 La imagen recortada puede usarse con proyecto_completo_simplificado.py")
        print("   para un procesamiento más eficiente y preciso!")
    else:
        print("\n❌ Error durante el procesamiento")


if __name__ == "__main__":
    main()
