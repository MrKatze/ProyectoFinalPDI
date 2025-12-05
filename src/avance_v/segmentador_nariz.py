"""
Módulo de Segmentación de Nariz
Avance V - Sin usar Landmarks de dlib

Implementa 2 métodos de detección de nariz sin landmarks:
1. Análisis de gradientes en región central (Sobel)
2. Análisis de textura con filtros de Gabor
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..avance_ii.filtros import FiltrosImagen
from .deteccion_bordes import DetectorBordes


class SegmentadorNariz:
    """
    Clase para detectar y segmentar nariz sin usar landmarks
    """
    
    def metodo_1_gradientes(self,
                           imagen: np.ndarray
                           ) -> Optional[Tuple[int, int, int, int]]:
        """
        Método 1: Detección usando análisis de gradientes (Sobel)
        La nariz tiene gradientes fuertes vertical y horizontalmente
        
        Args:
            imagen: Imagen en escala de grises del rostro
            
        Returns:
            Bounding box de la nariz (x, y, w, h) o None
        """
        h, w = imagen.shape
        
        # La nariz está en el centro vertical y tercios medios horizontales
        # Región de interés: centro de la imagen
        y_inicio = h // 3
        y_fin = 2 * h // 3
        x_inicio = w // 3
        x_fin = 2 * w // 3
        
        roi = imagen[y_inicio:y_fin, x_inicio:x_fin]
        
        # Calcular gradientes con Sobel
        magnitud, grad_x, grad_y = FiltrosImagen.gradiente_sobel(roi)
        
        # La nariz tiene gradientes fuertes en ambas direcciones
        gradiente_combinado = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        
        # Umbralizar para obtener regiones de alto gradiente
        _, binaria = cv2.threshold(gradiente_combinado, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplicar morfología para limpiar
        from ..avance_v.morfologia import OperadoresMorfologicos
        
        kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos) == 0:
            return None
        
        # Buscar el contorno más central (la nariz está en el centro)
        h_roi, w_roi = roi.shape
        centro_roi = (w_roi // 2, h_roi // 2)
        
        mejor_contorno = None
        mejor_distancia = float('inf')
        
        for cnt in contornos:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # Validar tamaño
            if area < 200 or area > w_roi * h_roi // 3:
                continue
            
            # Calcular distancia al centro
            centro_cnt = (x + cw // 2, y + ch // 2)
            distancia = np.sqrt((centro_cnt[0] - centro_roi[0])**2 + 
                              (centro_cnt[1] - centro_roi[1])**2)
            
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_contorno = (x, y, cw, ch)
        
        if mejor_contorno is None:
            return None
        
        # Ajustar coordenadas a imagen completa
        x, y, nw, nh = mejor_contorno
        return (x_inicio + x, y_inicio + y, nw, nh)
    
    def metodo_2_textura_gabor(self,
                              imagen: np.ndarray
                              ) -> Optional[Tuple[int, int, int, int]]:
        """
        Método 2: Detección usando análisis de textura con filtros de Gabor
        La nariz tiene una textura característica diferente a mejillas y boca
        
        Args:
            imagen: Imagen en escala de grises del rostro
            
        Returns:
            Bounding box de la nariz (x, y, w, h) o None
        """
        h, w = imagen.shape
        
        # Región de interés: centro de la imagen
        y_inicio = h // 3
        y_fin = 2 * h // 3
        x_inicio = w // 4
        x_fin = 3 * w // 4
        
        roi = imagen[y_inicio:y_fin, x_inicio:x_fin]
        
        # Aplicar banco de filtros de Gabor
        respuestas_gabor = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            filtrada = FiltrosImagen.filtro_gabor(
                roi,
                frecuencia=0.1,
                theta=theta,
                sigma_x=3.0,
                sigma_y=3.0
            )
            respuestas_gabor.append(filtrada)
        
        # Combinar respuestas
        respuesta_combinada = np.maximum.reduce(respuestas_gabor)
        
        # Umbralizar
        _, binaria = cv2.threshold(respuesta_combinada, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfología
        from ..avance_v.morfologia import OperadoresMorfologicos
        
        kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (7, 7))
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
        
        # Limpiar componentes pequeños
        binaria = OperadoresMorfologicos.limpiar_mascara(binaria, tamano_min=150)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contornos) == 0:
            return None
        
        # Buscar contorno más central y con forma alargada verticalmente
        h_roi, w_roi = roi.shape
        centro_roi = (w_roi // 2, h_roi // 2)
        
        mejor_contorno = None
        mejor_score = -1
        
        for cnt in contornos:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            # Validar tamaño
            if area < 200 or area > w_roi * h_roi // 3:
                continue
            
            # La nariz suele ser más alta que ancha
            ratio = ch / cw if cw > 0 else 0
            if ratio < 0.8:  # Preferir formas verticales
                continue
            
            # Calcular score combinando distancia al centro y área
            centro_cnt = (x + cw // 2, y + ch // 2)
            distancia = np.sqrt((centro_cnt[0] - centro_roi[0])**2 + 
                              (centro_cnt[1] - centro_roi[1])**2)
            
            # Score: mayor área y menor distancia es mejor
            score = area / (distancia + 1)
            
            if score > mejor_score:
                mejor_score = score
                mejor_contorno = (x, y, cw, ch)
        
        if mejor_contorno is None:
            return None
        
        # Ajustar coordenadas a imagen completa
        x, y, nw, nh = mejor_contorno
        return (x_inicio + x, y_inicio + y, nw, nh)
    
    def segmentar_multimetodo(self,
                             imagen: np.ndarray,
                             imagen_color: Optional[np.ndarray] = None
                             ) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
        """
        Aplica los 2 métodos y combina resultados
        
        Args:
            imagen: Imagen en escala de grises
            imagen_color: Imagen BGR original (opcional)
            
        Returns:
            Tupla (nariz_final, info_dict)
        """
        info = {
            'metodo_1': None,
            'metodo_2': None,
            'consenso': None
        }
        
        # Aplicar cada método
        nariz_m1 = self.metodo_1_gradientes(imagen)
        nariz_m2 = self.metodo_2_textura_gabor(imagen)
        
        info['metodo_1'] = nariz_m1
        info['metodo_2'] = nariz_m2
        
        # Estrategia de consenso
        if nariz_m1 is not None and nariz_m2 is not None:
            # Si ambos detectaron, promediar
            x1, y1, w1, h1 = nariz_m1
            x2, y2, w2, h2 = nariz_m2
            
            # Calcular IoU (Intersection over Union)
            iou = self._calcular_iou(nariz_m1, nariz_m2)
            
            if iou > 0.3:  # Si hay suficiente superposición
                # Promediar
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                w = (w1 + w2) // 2
                h = (h1 + h2) // 2
                nariz_final = (x, y, w, h)
            else:
                # Tomar el más central
                h_img, w_img = imagen.shape
                centro_img = (w_img // 2, h_img // 2)
                
                centro1 = (x1 + w1 // 2, y1 + h1 // 2)
                centro2 = (x2 + w2 // 2, y2 + h2 // 2)
                
                dist1 = np.sqrt((centro1[0] - centro_img[0])**2 + 
                              (centro1[1] - centro_img[1])**2)
                dist2 = np.sqrt((centro2[0] - centro_img[0])**2 + 
                              (centro2[1] - centro_img[1])**2)
                
                nariz_final = nariz_m1 if dist1 < dist2 else nariz_m2
        
        elif nariz_m1 is not None:
            nariz_final = nariz_m1
        elif nariz_m2 is not None:
            nariz_final = nariz_m2
        else:
            nariz_final = None
        
        info['consenso'] = nariz_final
        
        return nariz_final, info
    
    def _calcular_iou(self,
                     bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]
                     ) -> float:
        """
        Calcula Intersection over Union entre dos bounding boxes
        
        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)
            
        Returns:
            IoU value [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calcular intersección
        x_izq = max(x1, x2)
        y_sup = max(y1, y2)
        x_der = min(x1 + w1, x2 + w2)
        y_inf = min(y1 + h1, y2 + h2)
        
        if x_der < x_izq or y_inf < y_sup:
            return 0.0
        
        area_interseccion = (x_der - x_izq) * (y_inf - y_sup)
        area_union = w1 * h1 + w2 * h2 - area_interseccion
        
        return area_interseccion / area_union if area_union > 0 else 0.0
