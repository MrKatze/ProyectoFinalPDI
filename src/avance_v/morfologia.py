"""
Módulo de Operadores Morfológicos
Avance V - Segmentación Avanzada

Implementa operadores morfológicos basados en los códigos MATLAB del curso:
- Erosión y dilatación
- Apertura y cierre
- Gradiente morfológico
- Top-hat y Black-hat
- Operadores de mejora

Referencia MATLAB: ProcMorfoUmbra.m, mejora_morfologica.m
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class OperadoresMorfologicos:
    """
    Clase con operadores morfológicos para procesamiento de máscaras y segmentación
    """
    
    @staticmethod
    def crear_elemento_estructurante(forma: str = 'elipse',
                                     tamano: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Crea un elemento estructurante
        
        Args:
            forma: 'rect', 'elipse', 'cruz'
            tamano: Tamaño (ancho, alto)
            
        Returns:
            Elemento estructurante
        """
        if forma == 'rect':
            return cv2.getStructuringElement(cv2.MORPH_RECT, tamano)
        elif forma == 'elipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano)
        elif forma == 'cruz':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, tamano)
        else:
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano)
    
    @staticmethod
    def erosion(imagen: np.ndarray,
               kernel: Optional[np.ndarray] = None,
               iteraciones: int = 1) -> np.ndarray:
        """
        Aplica erosión morfológica
        Reduce regiones blancas, elimina píxeles aislados
        
        Args:
            imagen: Imagen binaria o escala de grises
            kernel: Elemento estructurante (None para 3x3 por defecto)
            iteraciones: Número de veces a aplicar
            
        Returns:
            Imagen erosionada
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (3, 3))
        
        return cv2.erode(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def dilatacion(imagen: np.ndarray,
                  kernel: Optional[np.ndarray] = None,
                  iteraciones: int = 1) -> np.ndarray:
        """
        Aplica dilatación morfológica
        Expande regiones blancas, rellena huecos pequeños
        
        Args:
            imagen: Imagen binaria o escala de grises
            kernel: Elemento estructurante
            iteraciones: Número de veces a aplicar
            
        Returns:
            Imagen dilatada
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (3, 3))
        
        return cv2.dilate(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def apertura(imagen: np.ndarray,
                kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apertura morfológica (erosión seguida de dilatación)
        Elimina ruido pequeño manteniendo formas grandes
        
        Args:
            imagen: Imagen binaria
            kernel: Elemento estructurante
            
        Returns:
            Imagen con apertura aplicada
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        
        return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def cierre(imagen: np.ndarray,
              kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cierre morfológico (dilatación seguida de erosión)
        Rellena huecos pequeños en regiones
        
        Args:
            imagen: Imagen binaria
            kernel: Elemento estructurante
            
        Returns:
            Imagen con cierre aplicado
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        
        return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def gradiente_morfologico(imagen: np.ndarray,
                             kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gradiente morfológico (dilatación - erosión)
        Resalta bordes de objetos
        
        Args:
            imagen: Imagen binaria o escala de grises
            kernel: Elemento estructurante
            
        Returns:
            Gradiente morfológico
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (3, 3))
        
        return cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)
    
    @staticmethod
    def top_hat(imagen: np.ndarray,
               kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Top-hat (imagen - apertura)
        Resalta estructuras brillantes más pequeñas que el elemento estructurante
        
        Args:
            imagen: Imagen escala de grises
            kernel: Elemento estructurante
            
        Returns:
            Imagen con top-hat aplicado
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (9, 9))
        
        return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    
    @staticmethod
    def black_hat(imagen: np.ndarray,
                 kernel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Black-hat (cierre - imagen)
        Resalta estructuras oscuras más pequeñas que el elemento estructurante
        
        Args:
            imagen: Imagen escala de grises
            kernel: Elemento estructurante
            
        Returns:
            Imagen con black-hat aplicado
        """
        if kernel is None:
            kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (9, 9))
        
        return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
    
    @staticmethod
    def limpiar_mascara(mascara: np.ndarray,
                       tamano_min: int = 100) -> np.ndarray:
        """
        Limpia una máscara binaria eliminando componentes pequeños
        
        Args:
            mascara: Máscara binaria
            tamano_min: Tamaño mínimo de componente a mantener (en píxeles)
            
        Returns:
            Máscara limpia
        """
        # Encontrar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mascara, connectivity=8
        )
        
        # Crear máscara limpia
        mascara_limpia = np.zeros_like(mascara)
        
        # Mantener solo componentes grandes (ignorar fondo en label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= tamano_min:
                mascara_limpia[labels == i] = 255
        
        return mascara_limpia
    
    @staticmethod
    def rellenar_huecos(mascara: np.ndarray) -> np.ndarray:
        """
        Rellena huecos en regiones de una máscara binaria
        
        Args:
            mascara: Máscara binaria
            
        Returns:
            Máscara con huecos rellenados
        """
        # Invertir máscara
        mascara_inv = cv2.bitwise_not(mascara)
        
        # Encontrar contornos externos
        contornos, _ = cv2.findContours(
            mascara_inv,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Rellenar contornos
        mascara_rellena = mascara.copy()
        cv2.drawContours(mascara_rellena, contornos, -1, 0, -1)
        
        # Invertir de nuevo
        mascara_rellena = cv2.bitwise_not(mascara_rellena)
        
        return mascara_rellena
    
    @staticmethod
    def mejorar_bordes(bordes: np.ndarray,
                      operacion: str = 'dilatacion',
                      kernel_size: int = 3) -> np.ndarray:
        """
        Mejora bordes detectados usando morfología
        
        Args:
            bordes: Imagen de bordes binaria
            operacion: 'dilatacion', 'cierre', 'ambos'
            kernel_size: Tamaño del kernel
            
        Returns:
            Bordes mejorados
        """
        kernel = OperadoresMorfologicos.crear_elemento_estructurante(
            'elipse', (kernel_size, kernel_size)
        )
        
        if operacion == 'dilatacion':
            return OperadoresMorfologicos.dilatacion(bordes, kernel)
        elif operacion == 'cierre':
            return OperadoresMorfologicos.cierre(bordes, kernel)
        elif operacion == 'ambos':
            dilatado = OperadoresMorfologicos.dilatacion(bordes, kernel)
            return OperadoresMorfologicos.cierre(dilatado, kernel)
        else:
            return bordes
    
    @staticmethod
    def esqueleto(mascara: np.ndarray) -> np.ndarray:
        """
        Calcula el esqueleto morfológico de una máscara
        
        Args:
            mascara: Máscara binaria
            
        Returns:
            Esqueleto
        """
        # Elemento estructurante
        kernel = OperadoresMorfologicos.crear_elemento_estructurante('cruz', (3, 3))
        
        esqueleto = np.zeros_like(mascara)
        temp = mascara.copy()
        
        while True:
            # Apertura
            abierta = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
            
            # Diferencia
            temp2 = cv2.subtract(temp, abierta)
            
            # Acumular
            esqueleto = cv2.bitwise_or(esqueleto, temp2)
            
            # Erosión
            temp = cv2.erode(temp, kernel)
            
            # Terminar si no quedan píxeles blancos
            if cv2.countNonZero(temp) == 0:
                break
        
        return esqueleto
    
    @staticmethod
    def pipeline_mejora_mascara(mascara: np.ndarray,
                               limpiar: bool = True,
                               rellenar: bool = True,
                               suavizar: bool = True) -> np.ndarray:
        """
        Pipeline completo de mejora de máscara
        Basado en mejora_morfologica.m del curso
        
        Args:
            mascara: Máscara binaria inicial
            limpiar: Si True, elimina componentes pequeños
            rellenar: Si True, rellena huecos
            suavizar: Si True, aplica apertura y cierre
            
        Returns:
            Máscara mejorada
        """
        resultado = mascara.copy()
        
        # 1. Apertura para eliminar ruido
        if suavizar:
            kernel_small = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (3, 3))
            resultado = OperadoresMorfologicos.apertura(resultado, kernel_small)
        
        # 2. Cierre para rellenar huecos pequeños
        if suavizar:
            kernel_medium = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
            resultado = OperadoresMorfologicos.cierre(resultado, kernel_medium)
        
        # 3. Limpiar componentes pequeños
        if limpiar:
            resultado = OperadoresMorfologicos.limpiar_mascara(resultado, tamano_min=50)
        
        # 4. Rellenar huecos grandes
        if rellenar:
            resultado = OperadoresMorfologicos.rellenar_huecos(resultado)
        
        # 5. Suavizado final
        if suavizar:
            kernel_final = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (3, 3))
            resultado = OperadoresMorfologicos.apertura(resultado, kernel_final)
        
        return resultado
