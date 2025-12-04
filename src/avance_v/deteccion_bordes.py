"""
Módulo de Detección de Bordes
Avance V - Segmentación Avanzada

Implementa métodos de detección de bordes basados en los códigos MATLAB del curso:
- Canny (Canny.m)
- Marr-Hildreth (MarrHildreht.m)
- Sobel
- Prewitt
- Roberts

Referencias MATLAB: Canny.m, MarrHildreht.m, DeteccionBordes.m
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, convolve
from typing import Tuple, Optional


class DetectorBordes:
    """
    Clase con métodos de detección de bordes para segmentación de rasgos faciales
    """
    
    @staticmethod
    def canny(imagen: np.ndarray,
             umbral_bajo: int = 50,
             umbral_alto: int = 150,
             apertura_sobel: int = 3) -> np.ndarray:
        """
        Implementa el detector de bordes de Canny
        Basado en Canny.m del curso
        
        El algoritmo de Canny:
        1. Suavizado Gaussiano para reducir ruido
        2. Cálculo de gradientes (magnitud y dirección)
        3. Supresión de no-máximos
        4. Umbralización con histéresis (doble umbral)
        
        Args:
            imagen: Imagen en escala de grises
            umbral_bajo: Umbral inferior para histéresis
            umbral_alto: Umbral superior para histéresis
            apertura_sobel: Tamaño de apertura para Sobel
            
        Returns:
            Imagen de bordes binaria
        """
        bordes = cv2.Canny(imagen, umbral_bajo, umbral_alto, 
                          apertureSize=apertura_sobel)
        return bordes
    
    @staticmethod
    def canny_automatico(imagen: np.ndarray, 
                        sigma: float = 0.33) -> np.ndarray:
        """
        Canny con umbrales calculados automáticamente
        
        Args:
            imagen: Imagen en escala de grises
            sigma: Factor para calcular umbrales (0.33 es típico)
            
        Returns:
            Imagen de bordes
        """
        # Calcular mediana
        mediana = np.median(imagen)
        
        # Calcular umbrales automáticamente
        umbral_bajo = int(max(0, (1.0 - sigma) * mediana))
        umbral_alto = int(min(255, (1.0 + sigma) * mediana))
        
        return DetectorBordes.canny(imagen, umbral_bajo, umbral_alto)
    
    @staticmethod
    def marr_hildreth(imagen: np.ndarray,
                     sigma: float = 1.0,
                     tamano_kernel: int = 0) -> np.ndarray:
        """
        Implementa el detector de bordes de Marr-Hildreth (LoG)
        Basado en MarrHildreht.m del curso
        
        Marr-Hildreth usa el Laplaciano de Gaussiana (LoG):
        1. Convolución con filtro Gaussiano
        2. Aplicación del operador Laplaciano
        3. Detección de cruces por cero
        
        Args:
            imagen: Imagen en escala de grises
            sigma: Desviación estándar del filtro Gaussiano
            tamano_kernel: Tamaño del kernel (0 para automático)
            
        Returns:
            Imagen de bordes binaria
        """
        # Calcular tamaño de kernel si es automático
        if tamano_kernel == 0:
            tamano_kernel = int(2 * np.ceil(3 * sigma) + 1)
            if tamano_kernel % 2 == 0:
                tamano_kernel += 1
        
        # 1. Suavizado Gaussiano
        imagen_suave = gaussian_filter(imagen.astype(np.float64), sigma=sigma)
        
        # 2. Aplicar Laplaciano
        # Kernel Laplaciano
        kernel_laplaciano = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=np.float64)
        
        laplaciano = convolve(imagen_suave, kernel_laplaciano)
        
        # 3. Detectar cruces por cero
        cruces_cero = DetectorBordes._detectar_cruces_cero(laplaciano)
        
        return cruces_cero
    
    @staticmethod
    def _detectar_cruces_cero(laplaciano: np.ndarray, 
                             umbral: float = 0.01) -> np.ndarray:
        """
        Detecta cruces por cero en la imagen del Laplaciano
        
        Args:
            laplaciano: Imagen después de aplicar Laplaciano
            umbral: Umbral mínimo para considerar un cruce
            
        Returns:
            Imagen binaria con cruces por cero
        """
        h, w = laplaciano.shape
        cruces = np.zeros((h, w), dtype=np.uint8)
        
        # Buscar cambios de signo en vecindad
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Vecinos
                vecindad = laplaciano[i-1:i+2, j-1:j+2]
                
                # Si hay cambio de signo significativo
                if (vecindad.max() * vecindad.min() < 0 and 
                    abs(vecindad.max() - vecindad.min()) > umbral):
                    cruces[i, j] = 255
        
        return cruces
    
    @staticmethod
    def log_opencv(imagen: np.ndarray, 
                   sigma: float = 1.0) -> np.ndarray:
        """
        Laplaciano de Gaussiana usando OpenCV
        Versión optimizada de Marr-Hildreth
        
        Args:
            imagen: Imagen en escala de grises
            sigma: Desviación estándar
            
        Returns:
            Imagen de bordes
        """
        # Suavizado Gaussiano
        suavizada = cv2.GaussianBlur(imagen, (0, 0), sigma)
        
        # Laplaciano
        laplaciano = cv2.Laplacian(suavizada, cv2.CV_64F)
        
        # Normalizar
        laplaciano = np.absolute(laplaciano)
        laplaciano = np.uint8(np.clip(laplaciano, 0, 255))
        
        # Umbralizar para obtener bordes
        _, bordes = cv2.threshold(laplaciano, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return bordes
    
    @staticmethod
    def sobel(imagen: np.ndarray, 
             tamano_kernel: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detector de bordes de Sobel
        
        Args:
            imagen: Imagen en escala de grises
            tamano_kernel: Tamaño del kernel
            
        Returns:
            Tupla (magnitud, grad_x, grad_y)
        """
        # Gradientes
        grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=tamano_kernel)
        grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=tamano_kernel)
        
        # Magnitud
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        magnitud = np.uint8(np.clip(magnitud, 0, 255))
        
        grad_x_abs = np.uint8(np.absolute(grad_x))
        grad_y_abs = np.uint8(np.absolute(grad_y))
        
        return magnitud, grad_x_abs, grad_y_abs
    
    @staticmethod
    def prewitt(imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detector de bordes de Prewitt
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Tupla (magnitud, grad_x, grad_y)
        """
        # Kernels de Prewitt
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float64)
        
        kernel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], dtype=np.float64)
        
        # Aplicar filtros
        grad_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
        
        # Magnitud
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        magnitud = np.uint8(np.clip(magnitud, 0, 255))
        
        grad_x_abs = np.uint8(np.absolute(grad_x))
        grad_y_abs = np.uint8(np.absolute(grad_y))
        
        return magnitud, grad_x_abs, grad_y_abs
    
    @staticmethod
    def roberts(imagen: np.ndarray) -> np.ndarray:
        """
        Detector de bordes de Roberts
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen de bordes
        """
        # Kernels de Roberts
        kernel_x = np.array([[1, 0],
                            [0, -1]], dtype=np.float64)
        
        kernel_y = np.array([[0, 1],
                            [-1, 0]], dtype=np.float64)
        
        # Aplicar filtros
        grad_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_y)
        
        # Magnitud
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        magnitud = np.uint8(np.clip(magnitud, 0, 255))
        
        return magnitud
    
    @staticmethod
    def comparar_detectores(imagen: np.ndarray) -> dict:
        """
        Compara diferentes detectores de bordes
        Similar a DeteccionBordes.m del curso
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Diccionario con resultados de cada detector
        """
        resultados = {
            'original': imagen.copy(),
            'canny': DetectorBordes.canny(imagen),
            'canny_auto': DetectorBordes.canny_automatico(imagen),
            'marr_hildreth': DetectorBordes.marr_hildreth(imagen, sigma=1.0),
            'marr_hildreth_sigma2': DetectorBordes.marr_hildreth(imagen, sigma=2.0),
            'log_opencv': DetectorBordes.log_opencv(imagen),
            'sobel': DetectorBordes.sobel(imagen)[0],
            'prewitt': DetectorBordes.prewitt(imagen)[0],
            'roberts': DetectorBordes.roberts(imagen)
        }
        
        return resultados
    
    @staticmethod
    def bordes_en_region(imagen: np.ndarray,
                        roi: Tuple[int, int, int, int],
                        metodo: str = 'canny') -> np.ndarray:
        """
        Detecta bordes solo en una región de interés
        
        Args:
            imagen: Imagen completa
            roi: Región (x, y, w, h)
            metodo: 'canny', 'marr_hildreth', 'sobel', etc.
            
        Returns:
            Imagen de bordes en la ROI
        """
        x, y, w, h = roi
        region = imagen[y:y+h, x:x+w]
        
        if metodo == 'canny':
            bordes = DetectorBordes.canny(region)
        elif metodo == 'marr_hildreth':
            bordes = DetectorBordes.marr_hildreth(region)
        elif metodo == 'sobel':
            bordes = DetectorBordes.sobel(region)[0]
        else:
            bordes = DetectorBordes.canny(region)
        
        return bordes
