"""
Módulo de Filtros de Imagen
Avance II - Preprocesamiento

Implementa filtros basados en los códigos MATLAB del curso:
- Filtros estadísticos (mediana)
- Filtros suavizantes (Gaussiano)
- Filtros realzantes (Laplaciano, Highboost)
- Filtros de gradiente (Sobel)

Referencias MATLAB: filtros_suavizantes.m, practica4_hightboost.m, gradiente_laplaciano.m
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class FiltrosImagen:
    """
    Clase con métodos de filtrado para preprocesamiento de imágenes
    """
    
    @staticmethod
    def filtro_mediana(imagen: np.ndarray, tamano_kernel: int = 5) -> np.ndarray:
        """
        Aplica filtro de mediana para reducir ruido impulsivo
        
        Args:
            imagen: Imagen de entrada
            tamano_kernel: Tamaño del kernel (debe ser impar)
            
        Returns:
            Imagen filtrada
        """
        return cv2.medianBlur(imagen, tamano_kernel)
    
    @staticmethod
    def filtro_gaussiano(imagen: np.ndarray, 
                        tamano_kernel: Tuple[int, int] = (5, 5),
                        sigma: float = 1.0) -> np.ndarray:
        """
        Aplica filtro Gaussiano para suavizado
        Similar a filtros_suavizantes.m del curso
        
        Args:
            imagen: Imagen de entrada
            tamano_kernel: Tamaño del kernel (ancho, alto)
            sigma: Desviación estándar
            
        Returns:
            Imagen suavizada
        """
        return cv2.GaussianBlur(imagen, tamano_kernel, sigma)
    
    @staticmethod
    def filtro_bilateral(imagen: np.ndarray,
                        d: int = 9,
                        sigma_color: float = 75,
                        sigma_space: float = 75) -> np.ndarray:
        """
        Aplica filtro bilateral (preserva bordes mientras suaviza)
        
        Args:
            imagen: Imagen de entrada
            d: Diámetro del vecindario
            sigma_color: Filtro en espacio de color
            sigma_space: Filtro en espacio de coordenadas
            
        Returns:
            Imagen filtrada
        """
        return cv2.bilateralFilter(imagen, d, sigma_color, sigma_space)
    
    @staticmethod
    def filtro_promedio(imagen: np.ndarray, 
                       tamano_kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Aplica filtro de promedio simple
        
        Args:
            imagen: Imagen de entrada
            tamano_kernel: Tamaño del kernel
            
        Returns:
            Imagen suavizada
        """
        return cv2.blur(imagen, tamano_kernel)
    
    @staticmethod
    def filtro_laplaciano(imagen: np.ndarray, 
                         tamano_kernel: int = 3) -> np.ndarray:
        """
        Aplica filtro Laplaciano para detección de bordes
        Similar a gradiente_laplaciano.m del curso
        
        Args:
            imagen: Imagen de entrada
            tamano_kernel: Tamaño del kernel (1, 3, 5, o 7)
            
        Returns:
            Imagen con bordes resaltados
        """
        # Aplicar Laplaciano
        laplaciano = cv2.Laplacian(imagen, cv2.CV_64F, ksize=tamano_kernel)
        
        # Convertir a uint8
        laplaciano = np.absolute(laplaciano)
        laplaciano = np.uint8(laplaciano)
        
        return laplaciano
    
    @staticmethod
    def filtro_highboost(imagen: np.ndarray, 
                        k: float = 1.5,
                        tamano_kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Aplica filtro Highboost para realce de detalles
        Similar a practica4_hightboost.m del curso
        
        Fórmula: f_highboost = (k * f_original) - f_suavizada
        
        Args:
            imagen: Imagen de entrada
            k: Factor de amplificación (k > 1)
            tamano_kernel: Tamaño del kernel para suavizado
            
        Returns:
            Imagen realzada
        """
        # Convertir a float para operaciones
        img_float = imagen.astype(np.float64)
        
        # Suavizar imagen
        img_suavizada = cv2.GaussianBlur(imagen, tamano_kernel, 0).astype(np.float64)
        
        # Aplicar fórmula de highboost
        img_highboost = (k * img_float) - img_suavizada
        
        # Normalizar y convertir a uint8
        img_highboost = np.clip(img_highboost, 0, 255)
        img_highboost = img_highboost.astype(np.uint8)
        
        return img_highboost
    
    @staticmethod
    def filtro_unsharp_masking(imagen: np.ndarray, 
                               sigma: float = 1.0,
                               amount: float = 1.5) -> np.ndarray:
        """
        Aplica Unsharp Masking para realce de nitidez
        
        Args:
            imagen: Imagen de entrada
            sigma: Desviación estándar del Gaussiano
            amount: Cantidad de realce
            
        Returns:
            Imagen con nitidez realzada
        """
        # Suavizar
        suavizada = cv2.GaussianBlur(imagen, (0, 0), sigma)
        
        # Calcular máscara
        mascara = cv2.subtract(imagen, suavizada)
        
        # Aplicar realce
        realzada = cv2.addWeighted(imagen, 1.0, mascara, amount, 0)
        
        return realzada
    
    @staticmethod
    def gradiente_sobel(imagen: np.ndarray, 
                       tamano_kernel: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula gradiente usando operador Sobel
        Similar a gradiente.m del curso
        
        Args:
            imagen: Imagen de entrada
            tamano_kernel: Tamaño del kernel Sobel
            
        Returns:
            Tupla (magnitud, gradiente_x, gradiente_y)
        """
        # Gradientes en X y Y
        grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=tamano_kernel)
        grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=tamano_kernel)
        
        # Magnitud del gradiente
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        magnitud = np.uint8(np.clip(magnitud, 0, 255))
        
        # Convertir gradientes a uint8
        grad_x_abs = np.uint8(np.absolute(grad_x))
        grad_y_abs = np.uint8(np.absolute(grad_y))
        
        return magnitud, grad_x_abs, grad_y_abs
    
    @staticmethod
    def filtro_gabor(imagen: np.ndarray,
                    frecuencia: float = 0.1,
                    theta: float = 0,
                    sigma_x: float = 3.0,
                    sigma_y: float = 3.0) -> np.ndarray:
        """
        Aplica filtro de Gabor para análisis de textura
        Útil para detectar patrones direccionales
        
        Args:
            imagen: Imagen de entrada
            frecuencia: Frecuencia de la onda sinusoidal
            theta: Orientación del filtro en radianes
            sigma_x: Desviación estándar en X
            sigma_y: Desviación estándar en Y
            
        Returns:
            Imagen filtrada
        """
        # Crear kernel Gabor
        kernel_size = int(2 * np.ceil(3 * max(sigma_x, sigma_y)) + 1)
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma_x,
            theta,
            frecuencia,
            0.5,
            0,
            ktype=cv2.CV_32F
        )
        
        # Aplicar filtro
        filtrada = cv2.filter2D(imagen, cv2.CV_32F, kernel)
        filtrada = np.absolute(filtrada)
        filtrada = np.uint8(np.clip(filtrada, 0, 255))
        
        return filtrada
    
    @staticmethod
    def banco_filtros_gabor(imagen: np.ndarray,
                           num_orientaciones: int = 8,
                           frecuencias: list = [0.05, 0.1, 0.2]) -> list:
        """
        Aplica un banco de filtros Gabor con múltiples orientaciones y frecuencias
        
        Args:
            imagen: Imagen de entrada
            num_orientaciones: Número de orientaciones
            frecuencias: Lista de frecuencias
            
        Returns:
            Lista de imágenes filtradas
        """
        resultados = []
        
        for freq in frecuencias:
            for i in range(num_orientaciones):
                theta = i * np.pi / num_orientaciones
                filtrada = FiltrosImagen.filtro_gabor(
                    imagen, 
                    frecuencia=freq,
                    theta=theta
                )
                resultados.append(filtrada)
        
        return resultados
    
    @staticmethod
    def aplicar_todos_filtros(imagen: np.ndarray) -> dict:
        """
        Aplica todos los filtros principales y retorna un diccionario
        Útil para comparación y visualización
        
        Args:
            imagen: Imagen de entrada
            
        Returns:
            Diccionario con resultados de cada filtro
        """
        resultados = {
            'original': imagen.copy(),
            'mediana': FiltrosImagen.filtro_mediana(imagen),
            'gaussiano': FiltrosImagen.filtro_gaussiano(imagen),
            'bilateral': FiltrosImagen.filtro_bilateral(imagen),
            'laplaciano': FiltrosImagen.filtro_laplaciano(imagen),
            'highboost': FiltrosImagen.filtro_highboost(imagen),
            'unsharp': FiltrosImagen.filtro_unsharp_masking(imagen),
            'sobel_magnitud': FiltrosImagen.gradiente_sobel(imagen)[0]
        }
        
        return resultados
