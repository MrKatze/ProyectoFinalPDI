"""
Módulo de Segmentación de Boca
Avance V - Sin usar Landmarks de dlib

Implementa 3 métodos de detección de boca sin landmarks:
1. Segmentación por color (YCrCb) - labios rojizos
2. Canny + morfología + análisis horizontal
3. Proyección vertical + análisis de intensidad
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .deteccion_bordes import DetectorBordes
from .morfologia import OperadoresMorfologicos


class SegmentadorBoca:
    """
    Clase para detectar y segmentar boca sin usar landmarks
    """
    
    def metodo_1_color_ycrcb(self,
                            imagen_color: np.ndarray
                            ) -> Optional[Tuple[int, int, int, int]]:
        """
        Método 1: Segmentación por color en espacio YCrCb
        Los labios tienen componentes Cr (rojo) más altos
        
        Args:
            imagen_color: Imagen BGR del rostro
            
        Returns:
            Bounding box de la boca (x, y, w, h) o None
        """
        h, w = imagen_color.shape[:2]
        
        # La boca está en la mitad inferior
        y_inicio = h // 2
        x_inicio = w // 4
        x_fin = 3 * w // 4
        
        roi = imagen_color[y_inicio:h, x_inicio:x_fin]
        
        # Convertir a YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        
        # Rango para labios (valores más altos de Cr - componente roja)
        # Estos valores detectan tonos rojizos de labios
        lower = np.array([0, 140, 90], dtype=np.uint8)
        upper = np.array([255, 180, 130], dtype=np.uint8)
        
        # Crear máscara
        mascara_labios = cv2.inRange(ycrcb, lower, upper)
        
        # Morfología para limpiar
        kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        mascara_labios = cv2.morphologyEx(mascara_labios, cv2.MORPH_CLOSE, kernel)
        mascara_labios = cv2.morphologyEx(mascara_labios, cv2.MORPH_OPEN, kernel)
        
        # Limpiar componentes pequeños
        mascara_labios = OperadoresMorfologicos.limpiar_mascara(mascara_labios, 
                                                                tamano_min=100)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(mascara_labios, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos) == 0:
            return None
        
        # Buscar el contorno más grande y horizontal
        mejor_contorno = None
        mejor_score = -1
        
        for cnt in contornos:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # Validar tamaño
            if area < 100 or area > w * h // 5:
                continue
            
            # La boca es más ancha que alta
            ratio = cw / ch if ch > 0 else 0
            if ratio < 1.2:  # Debe ser horizontal
                continue
            
            # Score basado en área y horizontalidad
            score = area * ratio
            
            if score > mejor_score:
                mejor_score = score
                mejor_contorno = (x, y, cw, ch)
        
        if mejor_contorno is None:
            return None
        
        # Ajustar coordenadas a imagen completa
        x, y, bw, bh = mejor_contorno
        return (x_inicio + x, y_inicio + y, bw, bh)
    
    def metodo_2_canny_morfologia(self,
                                  imagen: np.ndarray
                                  ) -> Optional[Tuple[int, int, int, int]]:
        """
        Método 2: Detección usando Canny + morfología + análisis horizontal
        Los bordes de la boca forman líneas horizontales características
        
        Args:
            imagen: Imagen en escala de grises del rostro
            
        Returns:
            Bounding box de la boca (x, y, w, h) o None
        """
        h, w = imagen.shape
        
        # La boca está en la mitad inferior
        y_inicio = h // 2
        x_inicio = w // 4
        x_fin = 3 * w // 4
        
        roi = imagen[y_inicio:h, x_inicio:x_fin]
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_mejorada = clahe.apply(roi)
        
        # Detectar bordes con Canny
        bordes = DetectorBordes.canny(roi_mejorada, 50, 150)
        
        # Morfología: dilatación horizontal para conectar bordes de labios
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        bordes_dilatados = cv2.dilate(bordes, kernel_horizontal, iterations=2)
        
        # Cerrar huecos
        kernel_cierre = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        bordes_cerrados = cv2.morphologyEx(bordes_dilatados, cv2.MORPH_CLOSE, kernel_cierre)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos) == 0:
            return None
        
        # Buscar contorno horizontal en región inferior
        h_roi, w_roi = roi.shape
        
        mejor_contorno = None
        mejor_score = -1
        
        for cnt in contornos:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # Validar tamaño
            if area < 150 or area > w_roi * h_roi // 4:
                continue
            
            # Debe ser horizontal (más ancho que alto)
            ratio = cw / ch if ch > 0 else 0
            if ratio < 1.5:
                continue
            
            # Preferir regiones en la parte inferior de la ROI
            posicion_vertical = y / h_roi
            if posicion_vertical < 0.3:  # Muy arriba
                continue
            
            # Score combinando área, horizontalidad y posición
            score = area * ratio * posicion_vertical
            
            if score > mejor_score:
                mejor_score = score
                mejor_contorno = (x, y, cw, ch)
        
        if mejor_contorno is None:
            return None
        
        # Ajustar coordenadas a imagen completa
        x, y, bw, bh = mejor_contorno
        return (x_inicio + x, y_inicio + y, bw, bh)
    
    def metodo_3_proyeccion_vertical(self,
                                    imagen: np.ndarray
                                    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Método 3: Proyección vertical + análisis de intensidad
        La boca tiene menor intensidad promedio (oscura) en proyección vertical
        
        Args:
            imagen: Imagen en escala de grises del rostro
            
        Returns:
            Bounding box de la boca (x, y, w, h) o None
        """
        h, w = imagen.shape
        
        # Trabajar en mitad inferior
        y_inicio = h // 2
        x_inicio = w // 4
        x_fin = 3 * w // 4
        
        mitad_inferior = imagen[y_inicio:h, x_inicio:x_fin]
        
        # Calcular proyección vertical (suma por columna)
        proyeccion = np.sum(mitad_inferior, axis=0)
        
        # Normalizar
        proyeccion = proyeccion / proyeccion.max()
        
        # La boca tiene baja intensidad (valores bajos en proyección)
        # Buscar región con mínimo local
        
        # Suavizar proyección para eliminar ruido
        from scipy.signal import savgol_filter
        proyeccion_suave = savgol_filter(proyeccion, 11, 3)
        
        # Encontrar mínimos locales
        from scipy.signal import find_peaks
        picos, _ = find_peaks(-proyeccion_suave, distance=10, prominence=0.1)
        
        if len(picos) == 0:
            return None
        
        # El mínimo más prominente en el centro es probable que sea la boca
        w_roi = mitad_inferior.shape[1]
        centro_roi = w_roi // 2
        
        mejor_pico = min(picos, key=lambda p: abs(p - centro_roi))
        
        # Determinar ancho de la boca alrededor del mínimo
        # Buscar hacia los lados hasta encontrar aumento significativo
        umbral = proyeccion_suave[mejor_pico] * 1.3
        
        # Buscar límite izquierdo
        x_izq = mejor_pico
        while x_izq > 0 and proyeccion_suave[x_izq] < umbral:
            x_izq -= 1
        
        # Buscar límite derecho
        x_der = mejor_pico
        while x_der < len(proyeccion_suave) - 1 and proyeccion_suave[x_der] < umbral:
            x_der += 1
        
        ancho_boca = x_der - x_izq
        
        # Validar ancho
        if ancho_boca < 20 or ancho_boca > w_roi * 0.8:
            return None
        
        # Determinar altura aproximada (generalmente 1/4 a 1/3 del ancho)
        alto_boca = ancho_boca // 3
        
        # Buscar posición vertical óptima en esa columna
        columna = mitad_inferior[:, mejor_pico]
        
        # Buscar región más oscura verticalmente
        h_roi = mitad_inferior.shape[0]
        mejor_y = 0
        min_intensidad = float('inf')
        
        for y in range(h_roi - alto_boca):
            intensidad_prom = np.mean(columna[y:y+alto_boca])
            if intensidad_prom < min_intensidad:
                min_intensidad = intensidad_prom
                mejor_y = y
        
        # Ajustar coordenadas a imagen completa
        x_final = x_inicio + x_izq
        y_final = y_inicio + mejor_y
        
        return (x_final, y_final, ancho_boca, alto_boca)
    
    def segmentar_multimetodo(self,
                             imagen: np.ndarray,
                             imagen_color: Optional[np.ndarray] = None
                             ) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
        """
        Aplica los 3 métodos y combina resultados
        
        Args:
            imagen: Imagen en escala de grises
            imagen_color: Imagen BGR original (requerida para método 1)
            
        Returns:
            Tupla (boca_final, info_dict)
        """
        info = {
            'metodo_1': None,
            'metodo_2': None,
            'metodo_3': None,
            'consenso': None
        }
        
        # Aplicar cada método
        boca_m1 = None
        if imagen_color is not None:
            boca_m1 = self.metodo_1_color_ycrcb(imagen_color)
        
        boca_m2 = self.metodo_2_canny_morfologia(imagen)
        boca_m3 = self.metodo_3_proyeccion_vertical(imagen)
        
        info['metodo_1'] = boca_m1
        info['metodo_2'] = boca_m2
        info['metodo_3'] = boca_m3
        
        # Combinar detecciones válidas
        detecciones_validas = [b for b in [boca_m1, boca_m2, boca_m3] if b is not None]
        
        if len(detecciones_validas) == 0:
            return None, info
        
        elif len(detecciones_validas) == 1:
            boca_final = detecciones_validas[0]
        
        else:
            # Promediar coordenadas de detecciones
            x_prom = int(np.mean([b[0] for b in detecciones_validas]))
            y_prom = int(np.mean([b[1] for b in detecciones_validas]))
            w_prom = int(np.mean([b[2] for b in detecciones_validas]))
            h_prom = int(np.mean([b[3] for b in detecciones_validas]))
            
            boca_final = (x_prom, y_prom, w_prom, h_prom)
        
        info['consenso'] = boca_final
        
        return boca_final, info
