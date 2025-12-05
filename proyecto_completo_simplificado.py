#!/usr/bin/env python3
"""
================================================================================
PROYECTO COMPLETO PDI - DETECCIÓN Y ANÁLISIS DE RASGOS FACIALES
================================================================================
Sistema integral que implementa todas las etapas del procesamiento:
1. Normalización y ecualización
2. Preprocesamiento (filtros estadísticos, suavizantes, realzantes)
3. Segmentación de bordes (Canny y Marr-Hildreth implementados desde cero)
4. Detección de rasgos faciales (sin métodos pre-hechos)
5. Extracción de descriptores de forma
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
# ETAPA 1: NORMALIZACIÓN Y ECUALIZACIÓN
# ============================================================================

class Normalizador:
    """Normalización y ecualización de imágenes"""
    
    @staticmethod
    def cargar_imagen(ruta: str) -> Optional[np.ndarray]:
        """Carga una imagen desde archivo"""
        if not os.path.exists(ruta):
            print(f"❌ Error: No existe {ruta}")
            return None
        img = cv2.imread(ruta)
        if img is None:
            print(f"❌ Error al cargar {ruta}")
            return None
        return img
    
    @staticmethod
    def convertir_a_gris(imagen: np.ndarray) -> np.ndarray:
        """Conversión manual a escala de grises usando fórmula estándar"""
        if len(imagen.shape) == 2:
            return imagen
        # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
        gris = (0.299 * imagen[:,:,2] + 
                0.587 * imagen[:,:,1] + 
                0.114 * imagen[:,:,0])
        return gris.astype(np.uint8)
    
    @staticmethod
    def normalizar_tamano(imagen: np.ndarray, tamano: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Normalización de tamaño usando interpolación bilineal manual
        """
        h_src, w_src = imagen.shape[:2]
        h_dst, w_dst = tamano
        
        # Factores de escala
        scale_x = w_src / w_dst
        scale_y = h_src / h_dst
        
        # Crear imagen de salida
        if len(imagen.shape) == 3:
            normalizada = np.zeros((h_dst, w_dst, imagen.shape[2]), dtype=np.uint8)
        else:
            normalizada = np.zeros((h_dst, w_dst), dtype=np.uint8)
        
        # Interpolación bilineal
        for y in range(h_dst):
            for x in range(w_dst):
                # Coordenadas en imagen original
                src_x = x * scale_x
                src_y = y * scale_y
                
                # Píxeles vecinos
                x1 = int(src_x)
                y1 = int(src_y)
                x2 = min(x1 + 1, w_src - 1)
                y2 = min(y1 + 1, h_src - 1)
                
                # Pesos para interpolación
                wx = src_x - x1
                wy = src_y - y1
                
                # Interpolación bilineal
                if len(imagen.shape) == 3:
                    for c in range(imagen.shape[2]):
                        val = (imagen[y1, x1, c] * (1-wx) * (1-wy) +
                               imagen[y1, x2, c] * wx * (1-wy) +
                               imagen[y2, x1, c] * (1-wx) * wy +
                               imagen[y2, x2, c] * wx * wy)
                        normalizada[y, x, c] = int(val)
                else:
                    val = (imagen[y1, x1] * (1-wx) * (1-wy) +
                           imagen[y1, x2] * wx * (1-wy) +
                           imagen[y2, x1] * (1-wx) * wy +
                           imagen[y2, x2] * wx * wy)
                    normalizada[y, x] = int(val)
        
        return normalizada
    
    @staticmethod
    def ecualizar_histograma(imagen: np.ndarray) -> np.ndarray:
        """
        Ecualización de histograma implementada desde cero
        Mejora el contraste redistribuyendo intensidades
        """
        # Calcular histograma
        hist, _ = np.histogram(imagen.flatten(), 256, [0, 256])
        
        # Función de distribución acumulativa (CDF)
        cdf = hist.cumsum()
        
        # Normalizar CDF
        cdf_normalizado = cdf * 255 / cdf[-1]
        
        # Mapear intensidades usando CDF normalizada
        ecualizada = np.interp(imagen.flatten(), range(256), cdf_normalizado)
        
        return ecualizada.reshape(imagen.shape).astype(np.uint8)


# ============================================================================
# ETAPA 2: PREPROCESAMIENTO - FILTROS
# ============================================================================

class FiltrosPreprocesamiento:
    """Implementación de filtros estadísticos, suavizantes y realzantes"""
    
    @staticmethod
    def aplicar_padding(imagen: np.ndarray, pad: int) -> np.ndarray:
        """Añade padding replicando bordes"""
        return np.pad(imagen, pad, mode='edge')
    
    # ----- FILTROS ESTADÍSTICOS -----
    
    @staticmethod
    def filtro_mediana(imagen: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Filtro de mediana para reducir ruido impulsivo (sal y pimienta)
        Implementación manual sin usar cv2.medianBlur
        """
        h, w = imagen.shape
        pad = kernel_size // 2
        img_pad = FiltrosPreprocesamiento.aplicar_padding(imagen, pad)
        resultado = np.zeros_like(imagen)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+kernel_size, j:j+kernel_size]
                resultado[i, j] = np.median(ventana)
        
        return resultado.astype(np.uint8)
    
    # ----- FILTROS SUAVIZANTES -----
    
    @staticmethod
    def crear_kernel_gaussiano(size: int, sigma: float) -> np.ndarray:
        """
        Crea kernel gaussiano desde cero
        G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
        """
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalizar
        kernel = kernel / kernel.sum()
        return kernel
    
    @staticmethod
    def filtro_gaussiano(imagen: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Filtro gaussiano para suavizado preservando bordes mejor que promedio
        """
        kernel = FiltrosPreprocesamiento.crear_kernel_gaussiano(kernel_size, sigma)
        h, w = imagen.shape
        pad = kernel_size // 2
        img_pad = FiltrosPreprocesamiento.aplicar_padding(imagen, pad)
        resultado = np.zeros_like(imagen, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+kernel_size, j:j+kernel_size]
                resultado[i, j] = np.sum(ventana * kernel)
        
        return resultado.astype(np.uint8)
    
    @staticmethod
    def filtro_bilateral(imagen: np.ndarray, d: int = 9, sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Filtro bilateral: suaviza preservando bordes
        Combina cercanía espacial y similitud de intensidad
        """
        h, w = imagen.shape
        pad = d // 2
        img_pad = FiltrosPreprocesamiento.aplicar_padding(imagen, pad)
        resultado = np.zeros_like(imagen, dtype=np.float32)
        
        # Kernel espacial gaussiano
        kernel_espacial = FiltrosPreprocesamiento.crear_kernel_gaussiano(d, sigma_space)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+d, j:j+d].astype(np.float32)
                intensidad_central = imagen[i, j]
                
                # Kernel de intensidad
                dif_intensidad = ventana - intensidad_central
                kernel_intensidad = np.exp(-(dif_intensidad**2) / (2 * sigma_color**2))
                
                # Combinar kernels
                kernel_total = kernel_espacial * kernel_intensidad
                kernel_total = kernel_total / kernel_total.sum()
                
                resultado[i, j] = np.sum(ventana * kernel_total)
        
        return resultado.astype(np.uint8)
    
    # ----- FILTROS REALZANTES -----
    
    @staticmethod
    def filtro_laplaciano(imagen: np.ndarray) -> np.ndarray:
        """
        Filtro Laplaciano para detección de bordes
        Kernel: [0 -1 0; -1 4 -1; 0 -1 0]
        """
        kernel = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
        
        h, w = imagen.shape
        img_pad = FiltrosPreprocesamiento.aplicar_padding(imagen, 1)
        resultado = np.zeros_like(imagen, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+3, j:j+3]
                resultado[i, j] = np.sum(ventana * kernel)
        
        # Normalizar a rango 0-255
        resultado = np.abs(resultado)
        resultado = (resultado / resultado.max() * 255).astype(np.uint8)
        return resultado
    
    @staticmethod
    def filtro_highboost(imagen: np.ndarray, k: float = 1.5) -> np.ndarray:
        """
        Highboost = Original + k * (Original - Suavizada)
        Realza bordes manteniendo la imagen original
        """
        suavizada = FiltrosPreprocesamiento.filtro_gaussiano(imagen, 5, 1.0)
        
        # Calcular máscara de alta frecuencia
        mascara = imagen.astype(np.float32) - suavizada.astype(np.float32)
        
        # Aplicar highboost
        realzada = imagen.astype(np.float32) + k * mascara
        
        # Clip a rango válido
        realzada = np.clip(realzada, 0, 255)
        return realzada.astype(np.uint8)
    
    @staticmethod
    def unsharp_masking(imagen: np.ndarray, sigma: float = 1.0, 
                       amount: float = 1.5) -> np.ndarray:
        """
        Unsharp masking para realce de detalles
        Sharp = Original + amount * (Original - Blur)
        """
        blur = FiltrosPreprocesamiento.filtro_gaussiano(imagen, 5, sigma)
        mascara = imagen.astype(np.float32) - blur.astype(np.float32)
        realzada = imagen.astype(np.float32) + amount * mascara
        realzada = np.clip(realzada, 0, 255)
        return realzada.astype(np.uint8)


# ============================================================================
# ETAPA 3: SEGMENTACIÓN DE BORDES
# ============================================================================

class SegmentadorBordes:
    """Implementación de Canny y Marr-Hildreth desde cero"""
    
    @staticmethod
    def gradiente_sobel(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula gradientes usando operador Sobel
        Returns: magnitud, gradiente_x, gradiente_y
        """
        # Kernels Sobel
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
        
        h, w = imagen.shape
        img_pad = np.pad(imagen, 1, mode='edge')
        
        grad_x = np.zeros_like(imagen, dtype=np.float32)
        grad_y = np.zeros_like(imagen, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+3, j:j+3]
                grad_x[i, j] = np.sum(ventana * sobel_x)
                grad_y[i, j] = np.sum(ventana * sobel_y)
        
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        
        return magnitud, grad_x, grad_y
    
    @staticmethod
    def supresion_no_maximos(magnitud: np.ndarray, grad_x: np.ndarray, 
                            grad_y: np.ndarray) -> np.ndarray:
        """
        Supresión de no-máximos: adelgaza bordes
        Mantiene solo píxeles que son máximos locales en dirección del gradiente
        """
        h, w = magnitud.shape
        resultado = np.zeros_like(magnitud)
        
        # Calcular ángulo del gradiente
        angulo = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angulo[angulo < 0] += 180
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                q = 255
                r = 255
                
                # Determinar vecinos según dirección del gradiente
                if (0 <= angulo[i,j] < 22.5) or (157.5 <= angulo[i,j] <= 180):
                    q = magnitud[i, j+1]
                    r = magnitud[i, j-1]
                elif 22.5 <= angulo[i,j] < 67.5:
                    q = magnitud[i+1, j-1]
                    r = magnitud[i-1, j+1]
                elif 67.5 <= angulo[i,j] < 112.5:
                    q = magnitud[i+1, j]
                    r = magnitud[i-1, j]
                elif 112.5 <= angulo[i,j] < 157.5:
                    q = magnitud[i-1, j-1]
                    r = magnitud[i+1, j+1]
                
                # Mantener solo si es máximo local
                if magnitud[i,j] >= q and magnitud[i,j] >= r:
                    resultado[i,j] = magnitud[i,j]
        
        return resultado
    
    @staticmethod
    def umbral_histeresis(imagen: np.ndarray, umbral_bajo: float, 
                         umbral_alto: float) -> np.ndarray:
        """
        Umbralización por histéresis: clasifica bordes fuertes/débiles
        - Bordes fuertes: > umbral_alto (255)
        - Bordes débiles: umbral_bajo < x < umbral_alto (conectados a fuertes)
        - No bordes: < umbral_bajo (0)
        """
        h, w = imagen.shape
        resultado = np.zeros_like(imagen, dtype=np.uint8)
        
        # Clasificar píxeles
        fuertes = imagen >= umbral_alto
        debiles = (imagen >= umbral_bajo) & (imagen < umbral_alto)
        
        resultado[fuertes] = 255
        
        # Conectar bordes débiles a fuertes
        for i in range(1, h-1):
            for j in range(1, w-1):
                if debiles[i, j]:
                    # Verificar si hay un borde fuerte en vecindad
                    if np.any(resultado[i-1:i+2, j-1:j+2] == 255):
                        resultado[i, j] = 255
        
        return resultado
    
    @staticmethod
    def canny(imagen: np.ndarray, umbral_bajo: float = 50, 
             umbral_alto: float = 150) -> np.ndarray:
        """
        Detector de bordes Canny completo implementado desde cero
        Etapas:
        1. Suavizado gaussiano
        2. Cálculo de gradientes (Sobel)
        3. Supresión de no-máximos
        4. Umbralización por histéresis
        """
        # 1. Suavizado
        suavizada = FiltrosPreprocesamiento.filtro_gaussiano(imagen, 5, 1.4)
        
        # 2. Gradientes
        magnitud, grad_x, grad_y = SegmentadorBordes.gradiente_sobel(suavizada)
        
        # 3. Supresión de no-máximos
        suprimida = SegmentadorBordes.supresion_no_maximos(magnitud, grad_x, grad_y)
        
        # 4. Histéresis
        bordes = SegmentadorBordes.umbral_histeresis(suprimida, umbral_bajo, umbral_alto)
        
        return bordes
    
    @staticmethod
    def laplaciano_gaussiano(imagen: np.ndarray, sigma: float = 1.4) -> np.ndarray:
        """
        Laplaciano de Gaussiana (LoG) para Marr-Hildreth
        LoG = ∇²(G * I)
        """
        # Suavizar con Gaussiano
        suavizada = FiltrosPreprocesamiento.filtro_gaussiano(imagen, 5, sigma)
        
        # Aplicar Laplaciano
        kernel_log = np.array([[0, 0, -1, 0, 0],
                              [0, -1, -2, -1, 0],
                              [-1, -2, 16, -2, -1],
                              [0, -1, -2, -1, 0],
                              [0, 0, -1, 0, 0]])
        
        h, w = suavizada.shape
        img_pad = np.pad(suavizada, 2, mode='edge')
        resultado = np.zeros_like(suavizada, dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                ventana = img_pad[i:i+5, j:j+5]
                resultado[i, j] = np.sum(ventana * kernel_log)
        
        return resultado
    
    @staticmethod
    def detectar_cruces_por_cero(log_imagen: np.ndarray, 
                                 umbral: float = 0) -> np.ndarray:
        """
        Detecta cruces por cero en imagen LoG
        Indica cambios bruscos de intensidad (bordes)
        """
        h, w = log_imagen.shape
        bordes = np.zeros_like(log_imagen, dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Verificar cruces por cero en vecindad 3x3
                vecindad = log_imagen[i-1:i+2, j-1:j+2]
                
                # Si hay cambio de signo
                if (vecindad.min() < -umbral) and (vecindad.max() > umbral):
                    bordes[i, j] = 255
        
        return bordes
    
    @staticmethod
    def marr_hildreth(imagen: np.ndarray, sigma: float = 1.4, 
                     umbral: float = 0) -> np.ndarray:
        """
        Detector de bordes Marr-Hildreth completo
        Basado en cruces por cero del Laplaciano de Gaussiana
        """
        # 1. Calcular LoG
        log_img = SegmentadorBordes.laplaciano_gaussiano(imagen, sigma)
        
        # 2. Detectar cruces por cero
        bordes = SegmentadorBordes.detectar_cruces_por_cero(log_img, umbral)
        
        return bordes


# ============================================================================
# ETAPA 4: DETECCIÓN DE RASGOS FACIALES (SIN MÉTODOS PRE-HECHOS)
# ============================================================================

class DetectorRasgos:
    """Detección de ojos, nariz y boca usando métodos propios"""
    
    @staticmethod
    def proyeccion_horizontal(imagen: np.ndarray) -> np.ndarray:
        """Suma de intensidades por fila"""
        return np.sum(imagen, axis=1)
    
    @staticmethod
    def proyeccion_vertical(imagen: np.ndarray) -> np.ndarray:
        """Suma de intensidades por columna"""
        return np.sum(imagen, axis=0)
    
    @staticmethod
    def detectar_ojos_proyeccion(imagen: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta región de ojos usando proyecciones y varianza
        Los ojos están en región superior (20-45% de altura)
        """
        h, w = imagen.shape
        
        # Región de interés para ojos (tercio superior)
        roi_inicio = int(h * 0.2)
        roi_fin = int(h * 0.45)
        roi = imagen[roi_inicio:roi_fin, :]
        
        # Proyección horizontal en ROI
        proyeccion = DetectorRasgos.proyeccion_horizontal(roi)
        
        # Calcular varianza por fila (ojos tienen alta varianza)
        varianzas = []
        for i in range(roi.shape[0]):
            varianzas.append(np.var(roi[i, :]))
        varianzas = np.array(varianzas)
        
        # Encontrar máximo de varianza (línea de ojos)
        if len(varianzas) == 0:
            return None
            
        linea_ojos = np.argmax(varianzas) + roi_inicio
        
        # Altura de región de ojos (aproximadamente 15% de altura facial)
        altura_ojos = int(h * 0.15)
        y_inicio = max(0, linea_ojos - altura_ojos//2)
        y_fin = min(h, linea_ojos + altura_ojos//2)
        
        # Ancho completo
        x_inicio = 0
        x_fin = w
        
        return (x_inicio, y_inicio, x_fin - x_inicio, y_fin - y_inicio)
    
    @staticmethod
    def detectar_nariz_gradientes(imagen: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta nariz usando análisis de gradientes en región central
        La nariz está en región central (35-65% altura, 30-70% ancho)
        """
        h, w = imagen.shape
        
        # ROI para nariz (región central)
        roi_y1 = int(h * 0.35)
        roi_y2 = int(h * 0.65)
        roi_x1 = int(w * 0.30)
        roi_x2 = int(w * 0.70)
        
        roi = imagen[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Calcular gradientes
        magnitud, _, _ = SegmentadorBordes.gradiente_sobel(roi)
        
        # La nariz tiene gradientes fuertes en región central
        # Buscar región con mayor densidad de gradientes
        umbral = np.percentile(magnitud, 70)
        mascara = magnitud > umbral
        
        # Encontrar región conectada más grande (componente principal)
        if not np.any(mascara):
            return None
        
        # Calcular bounding box de región con gradientes fuertes
        filas_con_gradiente = np.any(mascara, axis=1)
        cols_con_gradiente = np.any(mascara, axis=0)
        
        if not np.any(filas_con_gradiente) or not np.any(cols_con_gradiente):
            return None
        
        y_min = np.argmax(filas_con_gradiente)
        y_max = len(filas_con_gradiente) - np.argmax(filas_con_gradiente[::-1])
        x_min = np.argmax(cols_con_gradiente)
        x_max = len(cols_con_gradiente) - np.argmax(cols_con_gradiente[::-1])
        
        # Convertir a coordenadas de imagen completa
        x = roi_x1 + x_min
        y = roi_y1 + y_min
        ancho = x_max - x_min
        alto = y_max - y_min
        
        return (x, y, ancho, alto)
    
    @staticmethod
    def detectar_boca_proyeccion(imagen: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta boca usando proyección vertical y análisis de intensidad
        La boca está en región inferior (60-85% altura)
        """
        h, w = imagen.shape
        
        # ROI para boca (región inferior)
        roi_y1 = int(h * 0.60)
        roi_y2 = int(h * 0.85)
        roi = imagen[roi_y1:roi_y2, :]
        
        # Proyección vertical
        proyeccion = DetectorRasgos.proyeccion_vertical(roi)
        
        # Suavizar proyección
        proyeccion_suave = np.convolve(proyeccion, np.ones(5)/5, mode='same')
        
        # Encontrar mínimos locales (la boca suele ser más oscura)
        minimos = []
        for i in range(1, len(proyeccion_suave)-1):
            if proyeccion_suave[i] < proyeccion_suave[i-1] and \
               proyeccion_suave[i] < proyeccion_suave[i+1]:
                minimos.append(i)
        
        if len(minimos) == 0:
            # Si no hay mínimos claros, usar región central
            x_centro = w // 2
            ancho_boca = int(w * 0.4)
            x = x_centro - ancho_boca // 2
        else:
            # Usar el mínimo más profundo en región central
            minimos_centrales = [m for m in minimos if w*0.3 < m < w*0.7]
            if len(minimos_centrales) == 0:
                minimos_centrales = minimos
            
            x_centro = minimos_centrales[np.argmin([proyeccion_suave[m] 
                                                    for m in minimos_centrales])]
            ancho_boca = int(w * 0.4)
            x = max(0, x_centro - ancho_boca // 2)
        
        # Altura de boca
        alto_boca = int(h * 0.20)
        y = roi_y1
        
        return (x, y, ancho_boca, alto_boca)


# ============================================================================
# ETAPA 5: DESCRIPTORES DE FORMA (AVANCE VI)
# ============================================================================

class CalculadorDescriptores:
    """Extrae descriptores geométricos de regiones"""
    
    @staticmethod
    def calcular_area(mascara: np.ndarray) -> int:
        """Cuenta píxeles activos"""
        return np.sum(mascara > 0)
    
    @staticmethod
    def calcular_perimetro(mascara: np.ndarray) -> float:
        """
        Calcula perímetro contando transiciones borde-fondo
        """
        # Detectar bordes de la máscara
        h, w = mascara.shape
        perimetro = 0
        
        for i in range(h):
            for j in range(w):
                if mascara[i, j] > 0:
                    # Contar vecinos que son fondo
                    vecinos_fondo = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                if mascara[ni, nj] == 0:
                                    vecinos_fondo += 1
                            else:
                                vecinos_fondo += 1
                    
                    if vecinos_fondo > 0:
                        perimetro += 1
        
        return float(perimetro)
    
    @staticmethod
    def calcular_compacidad(area: float, perimetro: float) -> float:
        """
        Compacidad = P² / (4π * A)
        Círculo perfecto = 1.0
        """
        if area <= 0:
            return 0.0
        return (perimetro ** 2) / (4 * np.pi * area)
    
    @staticmethod
    def calcular_centroide(mascara: np.ndarray) -> Tuple[float, float]:
        """Calcula centro de masa de la región"""
        y_coords, x_coords = np.where(mascara > 0)
        
        if len(x_coords) == 0:
            return (0.0, 0.0)
        
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        
        return (cx, cy)
    
    @staticmethod
    def calcular_distancias_radiales(mascara: np.ndarray, 
                                     num_angulos: int = 360) -> np.ndarray:
        """
        Calcula distancias desde centroide hasta borde en diferentes ángulos
        """
        cx, cy = CalculadorDescriptores.calcular_centroide(mascara)
        
        # Encontrar puntos del contorno
        y_coords, x_coords = np.where(mascara > 0)
        
        if len(x_coords) == 0:
            return np.zeros(num_angulos)
        
        # Calcular ángulos de cada punto del contorno
        angulos = np.arctan2(y_coords - cy, x_coords - cx)
        distancias_originales = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        
        # Interpolar para obtener distancias uniformes
        angulos_uniformes = np.linspace(-np.pi, np.pi, num_angulos)
        distancias = np.zeros(num_angulos)
        
        for i, ang in enumerate(angulos_uniformes):
            # Encontrar puntos cercanos a este ángulo
            dif_angulos = np.abs(angulos - ang)
            indices_cercanos = dif_angulos < (2*np.pi/num_angulos)
            
            if np.any(indices_cercanos):
                distancias[i] = np.max(distancias_originales[indices_cercanos])
            else:
                # Interpolar del vecino más cercano
                idx_cercano = np.argmin(dif_angulos)
                distancias[i] = distancias_originales[idx_cercano]
        
        return distancias
    
    @staticmethod
    def calcular_descriptores_completos(mascara: np.ndarray) -> Dict[str, float]:
        """
        Calcula todos los descriptores del Avance VI
        """
        # Básicos
        area = CalculadorDescriptores.calcular_area(mascara)
        perimetro = CalculadorDescriptores.calcular_perimetro(mascara)
        
        if area == 0:
            return {
                'area': 0, 'perimetro': 0, 'compacidad': 0,
                'media_radial': 0, 'desv_radial': 0, 'cruces_cero': 0,
                'indice_area': 0, 'indice_rugosidad': 0
            }
        
        # Compacidad
        compacidad = CalculadorDescriptores.calcular_compacidad(area, perimetro)
        
        # Distancias radiales
        distancias = CalculadorDescriptores.calcular_distancias_radiales(mascara)
        
        # Normalizar distancias
        max_dist = np.max(distancias) if np.max(distancias) > 0 else 1
        distancias_norm = distancias / max_dist
        
        # Media y desviación
        media_radial = np.mean(distancias_norm)
        desv_radial = np.std(distancias_norm)
        
        # Cruces por cero
        dif = distancias_norm - media_radial
        cruces = np.sum(np.diff(np.sign(dif)) != 0)
        
        # Índice de área
        radio_medio = np.mean(distancias)
        area_circulo = np.pi * (radio_medio ** 2) if radio_medio > 0 else 1
        indice_area = area / area_circulo
        
        # Índice de rugosidad (simplificado)
        # Ratio entre perímetro real y perímetro de círculo equivalente
        radio_equiv = np.sqrt(area / np.pi)
        perimetro_equiv = 2 * np.pi * radio_equiv
        indice_rugosidad = perimetro / perimetro_equiv if perimetro_equiv > 0 else 1
        
        return {
            'area': float(area),
            'perimetro': float(perimetro),
            'compacidad': float(compacidad),
            'media_radial': float(media_radial),
            'desv_radial': float(desv_radial),
            'cruces_cero': int(cruces),
            'indice_area': float(indice_area),
            'indice_rugosidad': float(indice_rugosidad)
        }


# ============================================================================
# CLASE PRINCIPAL: PIPELINE COMPLETO
# ============================================================================

class ProyectoCompletoPDI:
    """Pipeline completo que integra todas las etapas"""
    
    def __init__(self):
        self.normalizador = Normalizador()
        self.filtros = FiltrosPreprocesamiento()
        self.segmentador_bordes = SegmentadorBordes()
        self.detector_rasgos = DetectorRasgos()
        self.calculador_descriptores = CalculadorDescriptores()
        
        # Almacenar resultados intermedios
        self.resultados = {}
    
    def procesar_imagen_completa(self, ruta_imagen: str, 
                                 usar_canny: bool = True) -> Dict:
        """
        Pipeline completo de procesamiento
        
        Args:
            ruta_imagen: Ruta a la imagen
            usar_canny: True para Canny, False para Marr-Hildreth
        
        Returns:
            Diccionario con todos los resultados
        """
        print("\n" + "="*80)
        print("PROCESANDO IMAGEN:", os.path.basename(ruta_imagen))
        print("="*80)
        
        # ===== ETAPA 1: CARGA Y NORMALIZACIÓN =====
        print("\n[1/6] Carga y Normalización...")
        
        # Cargar
        imagen = self.normalizador.cargar_imagen(ruta_imagen)
        if imagen is None:
            return {}
        
        # Convertir a gris
        gris = self.normalizador.convertir_a_gris(imagen)
        print(f"  ✓ Convertida a escala de grises")
        
        # Normalizar tamaño
        normalizada = self.normalizador.normalizar_tamano(gris, (256, 256))
        print(f"  ✓ Normalizada a 256x256")
        
        # Ecualizar histograma
        ecualizada = self.normalizador.ecualizar_histograma(normalizada)
        print(f"  ✓ Histograma ecualizado")
        
        self.resultados['original'] = gris
        self.resultados['normalizada'] = normalizada
        self.resultados['ecualizada'] = ecualizada
        
        # ===== ETAPA 2: PREPROCESAMIENTO CON FILTROS =====
        print("\n[2/6] Preprocesamiento con Filtros...")
        
        # Mejor combinación teórica:
        # 1. Mediana para ruido impulsivo
        # 2. Bilateral para suavizar preservando bordes
        # 3. Unsharp masking para realzar detalles
        
        print("  → Aplicando filtro mediana (ruido sal y pimienta)...")
        filtro1 = self.filtros.filtro_mediana(ecualizada, kernel_size=3)
        
        print("  → Aplicando filtro bilateral (suavizado preservando bordes)...")
        filtro2 = self.filtros.filtro_bilateral(filtro1, d=5, 
                                                sigma_color=50, sigma_space=50)
        
        print("  → Aplicando unsharp masking (realce de detalles)...")
        filtro3 = self.filtros.unsharp_masking(filtro2, sigma=1.0, amount=1.3)
        
        print(f"  ✓ Filtros aplicados correctamente")
        
        self.resultados['filtro_mediana'] = filtro1
        self.resultados['filtro_bilateral'] = filtro2
        self.resultados['preprocesada'] = filtro3
        
        # ===== ETAPA 3: SEGMENTACIÓN DE BORDES =====
        print(f"\n[3/6] Segmentación de Bordes ({['Marr-Hildreth', 'Canny'][usar_canny]})...")
        
        if usar_canny:
            bordes = self.segmentador_bordes.canny(filtro3, 
                                                   umbral_bajo=50, 
                                                   umbral_alto=150)
            print("  ✓ Canny aplicado (umbral bajo=50, alto=150)")
        else:
            bordes = self.segmentador_bordes.marr_hildreth(filtro3, 
                                                           sigma=1.4, 
                                                           umbral=5)
            print("  ✓ Marr-Hildreth aplicado (sigma=1.4)")
        
        self.resultados['bordes'] = bordes
        
        # ===== ETAPA 4: DETECCIÓN DE RASGOS =====
        print("\n[4/6] Detección de Rasgos Faciales...")
        
        # Detectar ojos
        print("  → Detectando ojos (proyección + varianza)...")
        bbox_ojos = self.detector_rasgos.detectar_ojos_proyeccion(filtro3)
        if bbox_ojos:
            print(f"    ✓ Ojos detectados: x={bbox_ojos[0]}, y={bbox_ojos[1]}, "
                  f"w={bbox_ojos[2]}, h={bbox_ojos[3]}")
            # Crear máscara de ojos
            mascara_ojos = np.zeros_like(filtro3)
            x, y, w, h = bbox_ojos
            mascara_ojos[y:y+h, x:x+w] = 255
        else:
            print("    ✗ No se detectaron ojos")
            mascara_ojos = np.zeros_like(filtro3)
        
        # Detectar nariz
        print("  → Detectando nariz (gradientes en región central)...")
        bbox_nariz = self.detector_rasgos.detectar_nariz_gradientes(filtro3)
        if bbox_nariz:
            print(f"    ✓ Nariz detectada: x={bbox_nariz[0]}, y={bbox_nariz[1]}, "
                  f"w={bbox_nariz[2]}, h={bbox_nariz[3]}")
            mascara_nariz = np.zeros_like(filtro3)
            x, y, w, h = bbox_nariz
            mascara_nariz[y:y+h, x:x+w] = 255
        else:
            print("    ✗ No se detectó nariz")
            mascara_nariz = np.zeros_like(filtro3)
        
        # Detectar boca
        print("  → Detectando boca (proyección vertical + intensidad)...")
        bbox_boca = self.detector_rasgos.detectar_boca_proyeccion(filtro3)
        if bbox_boca:
            print(f"    ✓ Boca detectada: x={bbox_boca[0]}, y={bbox_boca[1]}, "
                  f"w={bbox_boca[2]}, h={bbox_boca[3]}")
            mascara_boca = np.zeros_like(filtro3)
            x, y, w, h = bbox_boca
            mascara_boca[y:y+h, x:x+w] = 255
        else:
            print("    ✗ No se detectó boca")
            mascara_boca = np.zeros_like(filtro3)
        
        self.resultados['bbox_ojos'] = bbox_ojos
        self.resultados['bbox_nariz'] = bbox_nariz
        self.resultados['bbox_boca'] = bbox_boca
        self.resultados['mascara_ojos'] = mascara_ojos
        self.resultados['mascara_nariz'] = mascara_nariz
        self.resultados['mascara_boca'] = mascara_boca
        
        # ===== ETAPA 5: DESCRIPTORES DE FORMA =====
        print("\n[5/6] Extracción de Descriptores de Forma...")
        
        desc_ojos = self.calculador_descriptores.calcular_descriptores_completos(mascara_ojos)
        print("  ✓ Descriptores de OJOS:")
        print(f"    - Compacidad: {desc_ojos['compacidad']:.4f}")
        print(f"    - Media radial: {desc_ojos['media_radial']:.4f}")
        print(f"    - Desv. radial: {desc_ojos['desv_radial']:.4f}")
        print(f"    - Cruces por cero: {desc_ojos['cruces_cero']}")
        print(f"    - Índice de área: {desc_ojos['indice_area']:.4f}")
        print(f"    - Índice rugosidad: {desc_ojos['indice_rugosidad']:.4f}")
        
        desc_nariz = self.calculador_descriptores.calcular_descriptores_completos(mascara_nariz)
        print("  ✓ Descriptores de NARIZ:")
        print(f"    - Compacidad: {desc_nariz['compacidad']:.4f}")
        print(f"    - Índice rugosidad: {desc_nariz['indice_rugosidad']:.4f}")
        
        desc_boca = self.calculador_descriptores.calcular_descriptores_completos(mascara_boca)
        print("  ✓ Descriptores de BOCA:")
        print(f"    - Compacidad: {desc_boca['compacidad']:.4f}")
        print(f"    - Índice rugosidad: {desc_boca['indice_rugosidad']:.4f}")
        
        self.resultados['descriptores_ojos'] = desc_ojos
        self.resultados['descriptores_nariz'] = desc_nariz
        self.resultados['descriptores_boca'] = desc_boca
        
        # ===== ETAPA 6: VISUALIZACIÓN =====
        print("\n[6/6] Generando Visualización...")
        self.generar_visualizacion(os.path.basename(ruta_imagen))
        
        print("\n" + "="*80)
        print("✅ PROCESAMIENTO COMPLETADO")
        print("="*80)
        
        return self.resultados
    
    def generar_visualizacion(self, nombre_imagen: str):
        """Genera visualización completa de resultados"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Análisis Completo PDI - {nombre_imagen}', 
                    fontsize=16, fontweight='bold')
        
        # Crear subplots 4x5
        imagenes = [
            ('Original', self.resultados.get('original')),
            ('Normalizada 256x256', self.resultados.get('normalizada')),
            ('Ecualizada', self.resultados.get('ecualizada')),
            ('Filtro Mediana', self.resultados.get('filtro_mediana')),
            ('Filtro Bilateral', self.resultados.get('filtro_bilateral')),
            ('Preprocesada Final', self.resultados.get('preprocesada')),
            ('Bordes Detectados', self.resultados.get('bordes')),
            ('Región Ojos', self.resultados.get('mascara_ojos')),
            ('Región Nariz', self.resultados.get('mascara_nariz')),
            ('Región Boca', self.resultados.get('mascara_boca')),
        ]
        
        for idx, (titulo, img) in enumerate(imagenes, 1):
            if img is not None:
                plt.subplot(3, 4, idx)
                plt.imshow(img, cmap='gray')
                plt.title(titulo, fontsize=10, fontweight='bold')
                plt.axis('off')
        
        # Panel de resultados (descriptores)
        plt.subplot(3, 4, 11)
        plt.axis('off')
        plt.title('Descriptores Ojos', fontsize=10, fontweight='bold')
        desc = self.resultados.get('descriptores_ojos', {})
        texto = f"""
Área: {desc.get('area', 0):.0f} px²
Perímetro: {desc.get('perimetro', 0):.1f} px
Compacidad: {desc.get('compacidad', 0):.4f}
Media radial: {desc.get('media_radial', 0):.4f}
Desv. radial: {desc.get('desv_radial', 0):.4f}
Cruces/cero: {desc.get('cruces_cero', 0)}
Índice área: {desc.get('indice_area', 0):.4f}
Índice rugos.: {desc.get('indice_rugosidad', 0):.4f}
        """
        plt.text(0.1, 0.5, texto, fontsize=8, family='monospace',
                verticalalignment='center')
        
        plt.subplot(3, 4, 12)
        plt.axis('off')
        plt.title('Descriptores Nariz/Boca', fontsize=10, fontweight='bold')
        desc_n = self.resultados.get('descriptores_nariz', {})
        desc_b = self.resultados.get('descriptores_boca', {})
        texto = f"""
NARIZ:
Compacidad: {desc_n.get('compacidad', 0):.4f}
Área: {desc_n.get('area', 0):.0f} px²

BOCA:
Compacidad: {desc_b.get('compacidad', 0):.4f}
Área: {desc_b.get('area', 0):.0f} px²
        """
        plt.text(0.1, 0.5, texto, fontsize=8, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # Guardar
        ruta_salida = f"resultado_completo_{nombre_imagen.replace('.', '_')}.png"
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualización guardada: {ruta_salida}")
        plt.close()


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del programa"""
    
    print("="*80)
    print("PROYECTO COMPLETO PDI - SISTEMA DE ANÁLISIS FACIAL")
    print("="*80)
    print("\nImplementación completa desde cero:")
    print("  ✓ Normalización y ecualización")
    print("  ✓ Filtros estadísticos, suavizantes y realzantes")
    print("  ✓ Segmentación Canny y Marr-Hildreth")
    print("  ✓ Detección de rasgos faciales (métodos propios)")
    print("  ✓ Descriptores de forma (6 descriptores)")
    print("="*80)
    
    # Buscar imágenes en carpeta images
    carpeta_images = "images"
    
    if not os.path.exists(carpeta_images):
        print(f"\n❌ Error: No existe la carpeta '{carpeta_images}'")
        print("Crea la carpeta y añade imágenes para procesar.")
        return
    
    # Buscar todas las imágenes
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
    
    # Seleccionar primera imagen o pedir al usuario
    if len(imagenes) == 1:
        imagen_seleccionada = imagenes[0]
    else:
        print("\nImágenes disponibles:")
        for i, img in enumerate(imagenes[:10], 1):  # Mostrar primeras 10
            print(f"  {i}. {img}")
        
        if len(imagenes) > 10:
            print(f"  ... y {len(imagenes)-10} más")
        
        # Por defecto usar la primera
        imagen_seleccionada = imagenes[0]
        print(f"\n→ Procesando primera imagen: {imagen_seleccionada}")
    
    # Crear pipeline y procesar
    pipeline = ProyectoCompletoPDI()
    
    # Preguntar método de segmentación
    print("\n¿Qué método de segmentación de bordes deseas usar?")
    print("  1. Canny (recomendado)")
    print("  2. Marr-Hildreth")
    
    try:
        opcion = input("Selecciona (1 o 2, Enter=1): ").strip()
        usar_canny = opcion != '2'
    except:
        usar_canny = True
    
    # Procesar imagen
    resultados = pipeline.procesar_imagen_completa(
        imagen_seleccionada, 
        usar_canny=usar_canny
    )
    
    if resultados:
        print("\n✅ Imagen procesada exitosamente")
        print(f"✅ Visualización generada")
        print("\n💡 Revisa el archivo PNG generado con todos los resultados")
    else:
        print("\n❌ Error durante el procesamiento")


if __name__ == "__main__":
    main()
