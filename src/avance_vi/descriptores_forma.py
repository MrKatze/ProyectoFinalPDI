"""
Descriptores de Forma - Avance VI
==================================

Clase para extraer descriptores geométricos de contornos:
- Compacidad: relación perímetro/área
- Distancia radial normalizada: análisis radial desde centroide
- Índice de área: comparación con círculo equivalente
- Índice de rugosidad: suavidad del borde
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List


class DescriptoresForma:
    """
    Extrae descriptores geométricos de contornos de regiones faciales.
    """
    
    def __init__(self, num_puntos_radiales: int = 360):
        """
        Inicializa el extractor de descriptores.
        
        Args:
            num_puntos_radiales: Número de puntos para análisis radial (default: 360, uno por grado)
        """
        self.num_puntos_radiales = num_puntos_radiales
        
    def calcular_compacidad(self, contorno: np.ndarray) -> float:
        """
        Calcula la compacidad de un contorno.
        
        Compacidad = P² / (4π * A)
        - Círculo perfecto: 1.0
        - Formas irregulares: > 1.0
        
        Args:
            contorno: Contorno en formato OpenCV
            
        Returns:
            Valor de compacidad
        """
        try:
            perimetro = cv2.arcLength(contorno, True)
            area = cv2.contourArea(contorno)
            
            if area <= 0:
                return 0.0
                
            compacidad = (perimetro ** 2) / (4 * np.pi * area)
            return float(compacidad)
            
        except Exception as e:
            print(f"⚠️ Error calculando compacidad: {e}")
            return 0.0
    
    def calcular_centroide(self, contorno: np.ndarray) -> Tuple[float, float]:
        """
        Calcula el centroide (centro de masa) de un contorno.
        
        Args:
            contorno: Contorno en formato OpenCV
            
        Returns:
            Tupla (cx, cy) con coordenadas del centroide
        """
        try:
            M = cv2.moments(contorno)
            if M['m00'] == 0:
                # Si el área es cero, usar bbox center
                x, y, w, h = cv2.boundingRect(contorno)
                return (float(x + w/2), float(y + h/2))
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            return (float(cx), float(cy))
            
        except Exception as e:
            print(f"⚠️ Error calculando centroide: {e}")
            x, y, w, h = cv2.boundingRect(contorno)
            return (float(x + w/2), float(y + h/2))
    
    def calcular_distancias_radiales(
        self, 
        contorno: np.ndarray, 
        centroide: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Calcula distancias desde el centroide hasta el borde en N ángulos.
        
        Args:
            contorno: Contorno en formato OpenCV
            centroide: Centroide (si None, se calcula automáticamente)
            
        Returns:
            Array con distancias radiales
        """
        try:
            if centroide is None:
                centroide = self.calcular_centroide(contorno)
            
            cx, cy = centroide
            
            # Obtener todos los puntos del contorno
            puntos = contorno.reshape(-1, 2)
            
            # Calcular ángulos de todos los puntos respecto al centroide
            angulos = np.arctan2(puntos[:, 1] - cy, puntos[:, 0] - cx)
            distancias_originales = np.sqrt(
                (puntos[:, 0] - cx)**2 + (puntos[:, 1] - cy)**2
            )
            
            # Crear array de ángulos uniformemente espaciados
            angulos_uniformes = np.linspace(-np.pi, np.pi, self.num_puntos_radiales)
            
            # Interpolar distancias en ángulos uniformes
            # Ordenar por ángulo para interpolación correcta
            indices_ordenados = np.argsort(angulos)
            angulos_ordenados = angulos[indices_ordenados]
            distancias_ordenadas = distancias_originales[indices_ordenados]
            
            # Interpolar circularmente
            distancias_interpoladas = np.interp(
                angulos_uniformes,
                angulos_ordenados,
                distancias_ordenadas,
                period=2*np.pi
            )
            
            return distancias_interpoladas
            
        except Exception as e:
            print(f"⚠️ Error calculando distancias radiales: {e}")
            return np.zeros(self.num_puntos_radiales)
    
    def normalizar_distancias(self, distancias: np.ndarray) -> np.ndarray:
        """
        Normaliza distancias radiales dividiéndolas por la distancia máxima.
        
        Args:
            distancias: Array de distancias radiales
            
        Returns:
            Array normalizado (valores entre 0 y 1)
        """
        max_dist = np.max(distancias)
        if max_dist == 0:
            return distancias
        return distancias / max_dist
    
    def calcular_media_radial(self, distancias_norm: np.ndarray) -> float:
        """
        Calcula la media de las distancias radiales normalizadas.
        
        Args:
            distancias_norm: Distancias radiales normalizadas
            
        Returns:
            Media de distancias
        """
        return float(np.mean(distancias_norm))
    
    def calcular_desviacion_radial(self, distancias_norm: np.ndarray) -> float:
        """
        Calcula la desviación estándar de distancias radiales.
        
        Baja desviación: forma regular/circular
        Alta desviación: forma irregular/estrellada
        
        Args:
            distancias_norm: Distancias radiales normalizadas
            
        Returns:
            Desviación estándar
        """
        return float(np.std(distancias_norm))
    
    def calcular_cruces_por_cero(self, distancias_norm: np.ndarray) -> int:
        """
        Calcula el número de veces que la distancia radial cruza su media.
        
        Muchos cruces: forma con salientes/entrantes
        Pocos cruces: forma suave
        
        Args:
            distancias_norm: Distancias radiales normalizadas
            
        Returns:
            Número de cruces por cero
        """
        media = np.mean(distancias_norm)
        diferencias = distancias_norm - media
        
        # Contar cambios de signo
        cruces = np.sum(np.diff(np.sign(diferencias)) != 0)
        return int(cruces)
    
    def calcular_indice_area(
        self, 
        contorno: np.ndarray, 
        distancias: np.ndarray
    ) -> float:
        """
        Calcula el índice de área: relación entre área real y área del círculo
        con radio igual a la media de las distancias radiales.
        
        Índice de Área = A / (π * r_medio²)
        
        Args:
            contorno: Contorno en formato OpenCV
            distancias: Distancias radiales (sin normalizar)
            
        Returns:
            Índice de área
        """
        try:
            area_real = cv2.contourArea(contorno)
            radio_medio = np.mean(distancias)
            area_circulo = np.pi * (radio_medio ** 2)
            
            if area_circulo == 0:
                return 0.0
                
            indice = area_real / area_circulo
            return float(indice)
            
        except Exception as e:
            print(f"⚠️ Error calculando índice de área: {e}")
            return 0.0
    
    def calcular_indice_rugosidad(self, contorno: np.ndarray) -> float:
        """
        Calcula el índice de rugosidad: relación entre perímetro del contorno
        y perímetro de su envolvente convexa.
        
        Rugosidad = P_contorno / P_convex_hull
        - ~1.0: borde suave
        - >1.0: borde rugoso/irregular
        
        Args:
            contorno: Contorno en formato OpenCV
            
        Returns:
            Índice de rugosidad
        """
        try:
            perimetro_original = cv2.arcLength(contorno, True)
            hull = cv2.convexHull(contorno)
            perimetro_convexo = cv2.arcLength(hull, True)
            
            if perimetro_convexo == 0:
                return 1.0
                
            rugosidad = perimetro_original / perimetro_convexo
            return float(rugosidad)
            
        except Exception as e:
            print(f"⚠️ Error calculando índice de rugosidad: {e}")
            return 1.0
    
    def extraer_descriptores_completos(
        self, 
        contorno: np.ndarray
    ) -> Dict[str, float]:
        """
        Extrae TODOS los descriptores de forma de un contorno.
        
        Args:
            contorno: Contorno en formato OpenCV
            
        Returns:
            Diccionario con todos los descriptores:
            {
                'compacidad': float,
                'media_radial': float,
                'desviacion_radial': float,
                'cruces_por_cero': int,
                'indice_area': float,
                'indice_rugosidad': float,
                'area': float,
                'perimetro': float
            }
        """
        try:
            # Calcular área y perímetro básicos
            area = cv2.contourArea(contorno)
            perimetro = cv2.arcLength(contorno, True)
            
            # Compacidad
            compacidad = self.calcular_compacidad(contorno)
            
            # Distancias radiales
            centroide = self.calcular_centroide(contorno)
            distancias = self.calcular_distancias_radiales(contorno, centroide)
            distancias_norm = self.normalizar_distancias(distancias)
            
            # Descriptores de distancia radial
            media_radial = self.calcular_media_radial(distancias_norm)
            desviacion_radial = self.calcular_desviacion_radial(distancias_norm)
            cruces = self.calcular_cruces_por_cero(distancias_norm)
            
            # Índices
            indice_area = self.calcular_indice_area(contorno, distancias)
            indice_rugosidad = self.calcular_indice_rugosidad(contorno)
            
            return {
                'area': float(area),
                'perimetro': float(perimetro),
                'compacidad': compacidad,
                'media_radial': media_radial,
                'desviacion_radial': desviacion_radial,
                'cruces_por_cero': cruces,
                'indice_area': indice_area,
                'indice_rugosidad': indice_rugosidad,
                'centroide_x': centroide[0],
                'centroide_y': centroide[1]
            }
            
        except Exception as e:
            print(f"⚠️ Error extrayendo descriptores completos: {e}")
            return {
                'area': 0.0,
                'perimetro': 0.0,
                'compacidad': 0.0,
                'media_radial': 0.0,
                'desviacion_radial': 0.0,
                'cruces_por_cero': 0,
                'indice_area': 0.0,
                'indice_rugosidad': 0.0,
                'centroide_x': 0.0,
                'centroide_y': 0.0
            }
    
    def extraer_descriptores_de_mascara(
        self, 
        mascara: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Extrae descriptores de todos los contornos en una máscara binaria.
        
        Args:
            mascara: Imagen binaria con regiones detectadas
            
        Returns:
            Lista de diccionarios con descriptores de cada contorno
        """
        try:
            # Encontrar contornos
            contornos, _ = cv2.findContours(
                mascara.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contornos) == 0:
                return []
            
            # Extraer descriptores de cada contorno
            descriptores_lista = []
            for i, contorno in enumerate(contornos):
                # Filtrar contornos muy pequeños
                if cv2.contourArea(contorno) < 100:
                    continue
                    
                descriptores = self.extraer_descriptores_completos(contorno)
                descriptores['contorno_id'] = i
                descriptores_lista.append(descriptores)
            
            return descriptores_lista
            
        except Exception as e:
            print(f"⚠️ Error extrayendo descriptores de máscara: {e}")
            return []
    
    def obtener_distancias_para_visualizacion(
        self, 
        contorno: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Obtiene distancias radiales y ángulos para visualización polar.
        
        Args:
            contorno: Contorno en formato OpenCV
            
        Returns:
            Tupla (ángulos, distancias, distancias_normalizadas)
        """
        try:
            distancias = self.calcular_distancias_radiales(contorno)
            distancias_norm = self.normalizar_distancias(distancias)
            angulos = np.linspace(-np.pi, np.pi, self.num_puntos_radiales)
            
            return angulos, distancias, distancias_norm
            
        except Exception as e:
            print(f"⚠️ Error obteniendo datos para visualización: {e}")
            angulos = np.linspace(-np.pi, np.pi, self.num_puntos_radiales)
            distancias = np.zeros(self.num_puntos_radiales)
            return angulos, distancias, distancias
