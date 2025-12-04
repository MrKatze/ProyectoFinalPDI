"""
Módulo de Detección de Regiones de Rasgos
Avance IV - Identifica regiones donde buscar ojos, nariz y boca

Usa geometría facial estándar para definir regiones de interés
"""

import cv2
import numpy as np
from typing import Tuple, Dict


class DetectorRegiones:
    """
    Clase para identificar regiones faciales usando proporciones geométricas
    """
    
    @staticmethod
    def calcular_regiones_estandar(ancho: int, 
                                   alto: int) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Calcula regiones estándar basadas en proporciones faciales
        
        Proporciones aproximadas de un rostro:
        - Ojos: Tercio superior (entre 20%-45% de la altura)
        - Nariz: Tercio medio (entre 30%-65% de la altura, centrado)
        - Boca: Tercio inferior (entre 60%-85% de la altura)
        
        Args:
            ancho: Ancho de la imagen del rostro
            alto: Alto de la imagen del rostro
            
        Returns:
            Diccionario con regiones {'ojos': (x,y,w,h), 'nariz': (x,y,w,h), ...}
        """
        regiones = {}
        
        # Región de ojos (mitad superior)
        regiones['ojos'] = (
            0,                    # x
            int(alto * 0.20),     # y (20% desde arriba)
            ancho,                # w (todo el ancho)
            int(alto * 0.25)      # h (25% del alto)
        )
        
        # Región de ojo izquierdo
        regiones['ojo_izquierdo'] = (
            0,                    # x
            int(alto * 0.20),     # y
            ancho // 2,           # w (mitad izquierda)
            int(alto * 0.25)      # h
        )
        
        # Región de ojo derecho
        regiones['ojo_derecho'] = (
            ancho // 2,           # x (mitad derecha)
            int(alto * 0.20),     # y
            ancho // 2,           # w
            int(alto * 0.25)      # h
        )
        
        # Región de nariz (tercio medio, centro)
        regiones['nariz'] = (
            ancho // 4,           # x (centrado)
            int(alto * 0.35),     # y (35% desde arriba)
            ancho // 2,           # w (mitad del ancho)
            int(alto * 0.30)      # h (30% del alto)
        )
        
        # Región de boca (tercio inferior)
        regiones['boca'] = (
            ancho // 4,           # x
            int(alto * 0.60),     # y (60% desde arriba)
            ancho // 2,           # w
            int(alto * 0.25)      # h (25% del alto)
        )
        
        return regiones
    
    @staticmethod
    def visualizar_regiones(imagen: np.ndarray,
                           regiones: Dict[str, Tuple[int, int, int, int]],
                           grosor: int = 2) -> np.ndarray:
        """
        Dibuja las regiones sobre la imagen
        
        Args:
            imagen: Imagen del rostro
            regiones: Diccionario de regiones
            grosor: Grosor de las líneas
            
        Returns:
            Imagen con regiones dibujadas
        """
        # Convertir a color si es escala de grises
        if len(imagen.shape) == 2:
            imagen_vis = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            imagen_vis = imagen.copy()
        
        # Colores para cada región
        colores = {
            'ojos': (0, 255, 0),           # Verde
            'ojo_izquierdo': (0, 255, 255), # Amarillo
            'ojo_derecho': (0, 255, 255),  # Amarillo
            'nariz': (255, 0, 0),          # Azul
            'boca': (0, 0, 255)            # Rojo
        }
        
        # Dibujar cada región
        for nombre, (x, y, w, h) in regiones.items():
            color = colores.get(nombre, (255, 255, 255))
            cv2.rectangle(imagen_vis, (x, y), (x + w, y + h), color, grosor)
            
            # Añadir etiqueta
            cv2.putText(imagen_vis, nombre, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return imagen_vis
    
    @staticmethod
    def extraer_roi(imagen: np.ndarray,
                   region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extrae una región de interés de la imagen
        
        Args:
            imagen: Imagen completa
            region: (x, y, w, h)
            
        Returns:
            Región extraída
        """
        x, y, w, h = region
        
        # Asegurar que las coordenadas están dentro de los límites
        h_img, w_img = imagen.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        return imagen[y:y+h, x:x+w]
    
    @staticmethod
    def ajustar_region_a_deteccion(region: Tuple[int, int, int, int],
                                   deteccion_local: Tuple[int, int, int, int]
                                   ) -> Tuple[int, int, int, int]:
        """
        Ajusta coordenadas de una detección local a coordenadas globales
        
        Args:
            region: Región original (x, y, w, h) en imagen completa
            deteccion_local: Detección (x, y, w, h) relativa a la región
            
        Returns:
            Detección en coordenadas globales
        """
        x_region, y_region, _, _ = region
        x_local, y_local, w_local, h_local = deteccion_local
        
        return (
            x_region + x_local,
            y_region + y_local,
            w_local,
            h_local
        )
    
    @staticmethod
    def crear_mascara_region(forma_imagen: Tuple[int, int],
                            region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crea una máscara binaria para una región
        
        Args:
            forma_imagen: (altura, ancho) de la imagen
            region: (x, y, w, h)
            
        Returns:
            Máscara binaria (255 en región, 0 fuera)
        """
        mascara = np.zeros(forma_imagen, dtype=np.uint8)
        x, y, w, h = region
        mascara[y:y+h, x:x+w] = 255
        return mascara
