"""
Módulo de Umbralización
Avance III - Segmentación Fondo-Rostro

Implementa técnicas de umbralización para separar el rostro del fondo:
- Umbralización global
- Método de Otsu
- Umbralización adaptativa
- Combinación de métodos
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class Umbralizador:
    """
    Clase para aplicar técnicas de umbralización a imágenes de rostros
    """
    
    @staticmethod
    def umbral_global(imagen: np.ndarray, 
                     umbral: int = 127,
                     tipo: int = cv2.THRESH_BINARY) -> Tuple[np.ndarray, int]:
        """
        Aplica umbralización global con valor fijo
        
        Args:
            imagen: Imagen en escala de grises
            umbral: Valor del umbral (0-255)
            tipo: Tipo de umbralización (cv2.THRESH_BINARY, etc.)
            
        Returns:
            Tupla (imagen_umbralizada, valor_umbral_usado)
        """
        _, img_umbral = cv2.threshold(imagen, umbral, 255, tipo)
        return img_umbral, umbral
    
    @staticmethod
    def otsu(imagen: np.ndarray, 
             tipo: int = cv2.THRESH_BINARY) -> Tuple[np.ndarray, int]:
        """
        Aplica método de Otsu para calcular umbral óptimo automáticamente
        
        El método de Otsu minimiza la varianza intra-clase de los píxeles
        en blanco y negro, encontrando el umbral óptimo.
        
        Args:
            imagen: Imagen en escala de grises
            tipo: Tipo de umbralización
            
        Returns:
            Tupla (imagen_umbralizada, umbral_calculado)
        """
        umbral_otsu, img_umbral = cv2.threshold(
            imagen, 0, 255, tipo + cv2.THRESH_OTSU
        )
        return img_umbral, int(umbral_otsu)
    
    @staticmethod
    def umbral_adaptativo_media(imagen: np.ndarray,
                                tamano_bloque: int = 11,
                                c: int = 2) -> np.ndarray:
        """
        Umbralización adaptativa usando media local
        
        Args:
            imagen: Imagen en escala de grises
            tamano_bloque: Tamaño del vecindario para calcular umbral
            c: Constante a restar de la media
            
        Returns:
            Imagen umbralizada
        """
        return cv2.adaptiveThreshold(
            imagen,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            tamano_bloque,
            c
        )
    
    @staticmethod
    def umbral_adaptativo_gaussiano(imagen: np.ndarray,
                                    tamano_bloque: int = 11,
                                    c: int = 2) -> np.ndarray:
        """
        Umbralización adaptativa usando media ponderada gaussiana
        
        Args:
            imagen: Imagen en escala de grises
            tamano_bloque: Tamaño del vecindario
            c: Constante a restar
            
        Returns:
            Imagen umbralizada
        """
        return cv2.adaptiveThreshold(
            imagen,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            tamano_bloque,
            c
        )
    
    @staticmethod
    def segmentar_piel(imagen_color: np.ndarray) -> np.ndarray:
        """
        Segmenta regiones de piel usando espacio de color YCrCb
        
        Args:
            imagen_color: Imagen BGR
            
        Returns:
            Máscara binaria de piel
        """
        # Convertir a YCrCb
        ycrcb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2YCrCb)
        
        # Rangos típicos de piel en YCrCb
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Crear máscara
        mascara_piel = cv2.inRange(ycrcb, lower, upper)
        
        return mascara_piel
    
    @staticmethod
    def segmentar_piel_hsv(imagen_color: np.ndarray) -> np.ndarray:
        """
        Segmenta regiones de piel usando espacio de color HSV
        
        Args:
            imagen_color: Imagen BGR
            
        Returns:
            Máscara binaria de piel
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2HSV)
        
        # Rangos de piel en HSV
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        
        # Crear máscara
        mascara_piel = cv2.inRange(hsv, lower, upper)
        
        return mascara_piel
    
    @staticmethod
    def combinar_mascaras(mascara1: np.ndarray, 
                         mascara2: np.ndarray,
                         operacion: str = 'AND') -> np.ndarray:
        """
        Combina dos máscaras binarias
        
        Args:
            mascara1: Primera máscara
            mascara2: Segunda máscara
            operacion: 'AND', 'OR', 'XOR'
            
        Returns:
            Máscara combinada
        """
        if operacion == 'AND':
            return cv2.bitwise_and(mascara1, mascara2)
        elif operacion == 'OR':
            return cv2.bitwise_or(mascara1, mascara2)
        elif operacion == 'XOR':
            return cv2.bitwise_xor(mascara1, mascara2)
        else:
            raise ValueError(f"Operación no válida: {operacion}")
    
    @staticmethod
    def limpiar_mascara(mascara: np.ndarray,
                       kernel_size: int = 5) -> np.ndarray:
        """
        Limpia una máscara binaria usando operaciones morfológicas
        
        Args:
            mascara: Máscara binaria
            kernel_size: Tamaño del kernel morfológico
            
        Returns:
            Máscara limpia
        """
        # Kernel morfológico
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        # Apertura (elimina ruido pequeño)
        limpia = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        
        # Cierre (rellena huecos)
        limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel)
        
        return limpia
    
    @staticmethod
    def segmentar_rostro_completo(imagen: np.ndarray,
                                  imagen_color: Optional[np.ndarray] = None,
                                  limpiar: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Pipeline completo de segmentación de rostro
        Combina múltiples técnicas para obtener mejor resultado
        
        Args:
            imagen: Imagen en escala de grises
            imagen_color: Imagen BGR original (opcional, para segmentación por color)
            limpiar: Si True, limpia la máscara con morfología
            
        Returns:
            Tupla (mascara_final, info_dict)
        """
        info = {}
        
        # 1. Otsu en escala de grises
        mascara_otsu, umbral_otsu = Umbralizador.otsu(imagen)
        info['umbral_otsu'] = umbral_otsu
        info['mascara_otsu'] = mascara_otsu
        
        # 2. Umbralización adaptativa
        mascara_adaptativa = Umbralizador.umbral_adaptativo_gaussiano(imagen)
        info['mascara_adaptativa'] = mascara_adaptativa
        
        # 3. Combinar máscaras
        mascara_combinada = Umbralizador.combinar_mascaras(
            mascara_otsu, 
            mascara_adaptativa,
            'OR'
        )
        
        # 4. Si tenemos imagen en color, usar segmentación de piel
        if imagen_color is not None:
            mascara_piel = Umbralizador.segmentar_piel(imagen_color)
            info['mascara_piel'] = mascara_piel
            
            # Combinar con máscara anterior
            mascara_combinada = Umbralizador.combinar_mascaras(
                mascara_combinada,
                mascara_piel,
                'AND'
            )
        
        # 5. Limpiar máscara
        if limpiar:
            mascara_final = Umbralizador.limpiar_mascara(mascara_combinada)
        else:
            mascara_final = mascara_combinada
        
        info['mascara_final'] = mascara_final
        
        return mascara_final, info
    
    @staticmethod
    def aplicar_mascara(imagen: np.ndarray, 
                       mascara: np.ndarray) -> np.ndarray:
        """
        Aplica una máscara a una imagen
        
        Args:
            imagen: Imagen original
            mascara: Máscara binaria
            
        Returns:
            Imagen con máscara aplicada
        """
        return cv2.bitwise_and(imagen, imagen, mask=mascara)
    
    @staticmethod
    def comparar_metodos(imagen: np.ndarray) -> dict:
        """
        Compara diferentes métodos de umbralización
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Diccionario con resultados de cada método
        """
        resultados = {}
        
        # Global con diferentes umbrales
        for umbral in [100, 127, 150]:
            resultados[f'global_{umbral}'], _ = Umbralizador.umbral_global(
                imagen, umbral
            )
        
        # Otsu
        resultados['otsu'], umbral_otsu = Umbralizador.otsu(imagen)
        resultados['otsu_umbral'] = umbral_otsu
        
        # Adaptativa media
        resultados['adaptativa_media'] = Umbralizador.umbral_adaptativo_media(imagen)
        
        # Adaptativa gaussiana
        resultados['adaptativa_gaussiana'] = Umbralizador.umbral_adaptativo_gaussiano(imagen)
        
        return resultados
