"""
Módulo de Segmentación de Ojos
Avance V - Sin usar Landmarks de dlib

Implementa 3 métodos de detección de ojos sin landmarks:
1. Haar Cascade + validación geométrica
2. Proyección horizontal + morfología
3. Canny + Hough circular (detección de pupilas)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..avance_v.deteccion_bordes import DetectorBordes
from ..avance_v.morfologia import OperadoresMorfologicos


class SegmentadorOjos:
    """
    Clase para detectar y segmentar ojos sin usar landmarks
    """
    
    def __init__(self):
        """Inicializa el segmentador con los clasificadores necesarios"""
        self.detector_ojos = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.detector_ojos_lentes = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
    
    def metodo_1_haar_cascade(self, 
                              imagen: np.ndarray,
                              region_superior: Optional[Tuple[int, int, int, int]] = None
                              ) -> List[Tuple[int, int, int, int]]:
        """
        Método 1: Detección usando Haar Cascade con validación geométrica
        
        Args:
            imagen: Imagen en escala de grises del rostro
            region_superior: (x, y, w, h) de la región superior del rostro
            
        Returns:
            Lista de ojos detectados [(x, y, w, h), ...]
        """
        h, w = imagen.shape
        
        # Si no hay región especificada, usar mitad superior
        if region_superior is None:
            region_superior = (0, 0, w, h // 2)
        
        x, y, w_roi, h_roi = region_superior
        roi = imagen[y:y+h_roi, x:x+w_roi]
        
        # Detectar ojos con ambos clasificadores
        ojos1 = self.detector_ojos.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(w_roi // 3, h_roi // 2)
        )
        
        ojos2 = self.detector_ojos_lentes.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
            maxSize=(w_roi // 3, h_roi // 2)
        )
        
        # Combinar detecciones
        ojos_todos = list(ojos1) + list(ojos2)
        
        # Ajustar coordenadas a imagen completa
        ojos_ajustados = []
        for (ex, ey, ew, eh) in ojos_todos:
            ojos_ajustados.append((x + ex, y + ey, ew, eh))
        
        # Validar y filtrar ojos
        ojos_validos = self._validar_ojos_geometria(ojos_ajustados, w, h)
        
        return ojos_validos
    
    def metodo_2_proyeccion_horizontal(self,
                                      imagen: np.ndarray,
                                      umbral_varianza: float = 500
                                      ) -> List[Tuple[int, int, int, int]]:
        """
        Método 2: Detección usando proyección horizontal y análisis de varianza
        Los ojos tienen alta variabilidad horizontal (párpados, pestañas)
        
        Args:
            imagen: Imagen en escala de grises del rostro
            umbral_varianza: Umbral de varianza para considerar región de ojos
            
        Returns:
            Lista de ojos detectados
        """
        h, w = imagen.shape
        
        # Trabajar solo en mitad superior
        mitad_superior = imagen[0:h//2, :]
        
        # Calcular proyección horizontal (suma por fila)
        proyeccion = np.sum(mitad_superior, axis=1)
        
        # Calcular varianza en ventanas deslizantes
        tamano_ventana = h // 10
        varianzas = []
        
        for i in range(len(proyeccion) - tamano_ventana):
            ventana = proyeccion[i:i+tamano_ventana]
            var = np.var(ventana)
            varianzas.append((i, var))
        
        # Encontrar regiones con alta varianza (posibles ojos)
        regiones_candidatas = []
        for i, var in varianzas:
            if var > umbral_varianza:
                regiones_candidatas.append(i)
        
        # Agrupar regiones cercanas
        if len(regiones_candidatas) == 0:
            return []
        
        grupos = self._agrupar_regiones(regiones_candidatas, distancia_max=5)
        
        # Crear bounding boxes para cada grupo
        ojos = []
        for grupo in grupos:
            y_min = min(grupo)
            y_max = max(grupo) + tamano_ventana
            
            # Buscar límites horizontales en esta franja
            franja = mitad_superior[y_min:y_max, :]
            
            # Aplicar umbralización
            _, binaria = cv2.threshold(franja, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Encontrar contornos
            contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamaño
            for cnt in contornos:
                x, y, ew, eh = cv2.boundingRect(cnt)
                area = ew * eh
                
                # Validar tamaño y proporción
                if 200 < area < 5000 and 0.5 < ew/eh < 2.5:
                    ojos.append((x, y_min + y, ew, eh))
        
        return ojos[:2]  # Máximo 2 ojos
    
    def metodo_3_canny_hough(self,
                            imagen: np.ndarray,
                            radio_min: int = 5,
                            radio_max: int = 30
                            ) -> List[Tuple[int, int, int, int]]:
        """
        Método 3: Detección de pupilas usando Canny + Transformada de Hough circular
        Detecta círculos que corresponden a pupilas
        
        Args:
            imagen: Imagen en escala de grises del rostro
            radio_min: Radio mínimo de pupila
            radio_max: Radio máximo de pupila
            
        Returns:
            Lista de ojos detectados (basados en pupilas)
        """
        h, w = imagen.shape
        
        # Trabajar en mitad superior
        mitad_superior = imagen[0:h//2, :]
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        mejorada = clahe.apply(mitad_superior)
        
        # Detectar bordes con Canny
        bordes = DetectorBordes.canny(mejorada, 30, 100)
        
        # Detectar círculos (pupilas) con Hough
        circulos = cv2.HoughCircles(
            mejorada,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=w // 4,  # Distancia mínima entre ojos
            param1=50,
            param2=30,
            minRadius=radio_min,
            maxRadius=radio_max
        )
        
        ojos = []
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))
            
            for circulo in circulos[0, :]:
                cx, cy, r = circulo
                
                # Crear bounding box alrededor de la pupila
                # El ojo es aproximadamente 3 veces el radio de la pupila
                ew = eh = int(r * 3)
                ex = max(0, int(cx) - ew // 2)
                ey = max(0, int(cy) - eh // 2)
                
                ojos.append((ex, ey, ew, eh))
        
        return ojos[:2]  # Máximo 2 ojos
    
    def segmentar_multimetodo(self,
                             imagen: np.ndarray,
                             imagen_color: Optional[np.ndarray] = None
                             ) -> Tuple[List[Tuple[int, int, int, int]], dict]:
        """
        Aplica los 3 métodos y combina resultados mediante votación
        
        Args:
            imagen: Imagen en escala de grises
            imagen_color: Imagen BGR original (opcional)
            
        Returns:
            Tupla (ojos_finales, info_dict)
        """
        info = {
            'metodo_1': [],
            'metodo_2': [],
            'metodo_3': [],
            'consenso': []
        }
        
        # Aplicar cada método
        ojos_m1 = self.metodo_1_haar_cascade(imagen)
        ojos_m2 = self.metodo_2_proyeccion_horizontal(imagen)
        ojos_m3 = self.metodo_3_canny_hough(imagen)
        
        info['metodo_1'] = ojos_m1
        info['metodo_2'] = ojos_m2
        info['metodo_3'] = ojos_m3
        
        # Combinar detecciones
        todos_ojos = ojos_m1 + ojos_m2 + ojos_m3
        
        if len(todos_ojos) == 0:
            return [], info
        
        # Agrupar ojos cercanos (clustering simple)
        grupos = self._agrupar_detecciones(todos_ojos, distancia_max=30)
        
        # Tomar los 2 grupos más votados
        grupos_ordenados = sorted(grupos, key=lambda g: len(g), reverse=True)
        
        ojos_finales = []
        for grupo in grupos_ordenados[:2]:  # Máximo 2 ojos
            # Promediar coordenadas del grupo
            x_prom = int(np.mean([ojo[0] for ojo in grupo]))
            y_prom = int(np.mean([ojo[1] for ojo in grupo]))
            w_prom = int(np.mean([ojo[2] for ojo in grupo]))
            h_prom = int(np.mean([ojo[3] for ojo in grupo]))
            
            ojos_finales.append((x_prom, y_prom, w_prom, h_prom))
        
        # Ordenar de izquierda a derecha
        ojos_finales = sorted(ojos_finales, key=lambda ojo: ojo[0])
        
        info['consenso'] = ojos_finales
        
        return ojos_finales, info
    
    def _validar_ojos_geometria(self,
                                ojos: List[Tuple[int, int, int, int]],
                                ancho_img: int,
                                alto_img: int
                                ) -> List[Tuple[int, int, int, int]]:
        """
        Valida detecciones de ojos basándose en geometría facial
        
        Args:
            ojos: Lista de ojos candidatos
            ancho_img: Ancho de la imagen
            alto_img: Alto de la imagen
            
        Returns:
            Lista de ojos validados
        """
        validos = []
        
        for (x, y, w, h) in ojos:
            # Validar posición (deben estar en mitad superior)
            if y > alto_img // 2:
                continue
            
            # Validar tamaño
            area = w * h
            if area < 100 or area > ancho_img * alto_img // 10:
                continue
            
            # Validar proporción (los ojos son más anchos que altos)
            ratio = w / h if h > 0 else 0
            if ratio < 0.5 or ratio > 3.0:
                continue
            
            validos.append((x, y, w, h))
        
        return validos
    
    def _agrupar_regiones(self, 
                         indices: List[int],
                         distancia_max: int = 5
                         ) -> List[List[int]]:
        """Agrupa índices cercanos en grupos"""
        if len(indices) == 0:
            return []
        
        indices_ordenados = sorted(indices)
        grupos = [[indices_ordenados[0]]]
        
        for idx in indices_ordenados[1:]:
            if idx - grupos[-1][-1] <= distancia_max:
                grupos[-1].append(idx)
            else:
                grupos.append([idx])
        
        return grupos
    
    def _agrupar_detecciones(self,
                            ojos: List[Tuple[int, int, int, int]],
                            distancia_max: int = 30
                            ) -> List[List[Tuple[int, int, int, int]]]:
        """Agrupa detecciones cercanas"""
        if len(ojos) == 0:
            return []
        
        grupos = [[ojos[0]]]
        
        for ojo in ojos[1:]:
            agregado = False
            
            for grupo in grupos:
                # Calcular distancia al centro del grupo
                ojo_centro = (int(ojo[0] + ojo[2] // 2), int(ojo[1] + ojo[3] // 2))
                
                for ojo_grupo in grupo:
                    centro_grupo = (int(ojo_grupo[0] + ojo_grupo[2] // 2),
                                   int(ojo_grupo[1] + ojo_grupo[3] // 2))
                    
                    distancia = np.sqrt(float(ojo_centro[0] - centro_grupo[0])**2 +
                                       float(ojo_centro[1] - centro_grupo[1])**2)
                    
                    if distancia < distancia_max:
                        grupo.append(ojo)
                        agregado = True
                        break
                
                if agregado:
                    break
            
            if not agregado:
                grupos.append([ojo])
        
        return grupos
