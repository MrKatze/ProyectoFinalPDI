"""
Módulo de Alineación y Normalización de Rostros
Avance II - Preprocesamiento

Implementa:
- Detección de rostros con Haar Cascade
- Detección de ojos para alineación
- Rotación basada en ángulo de ojos
- Normalización de tamaño e iluminación
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class AlineadorRostros:
    """
    Clase para detectar, alinear y normalizar imágenes de rostros
    """
    
    def __init__(self, tamano_salida: Tuple[int, int] = (256, 256)):
        """
        Inicializa el alineador con los clasificadores Haar Cascade
        
        Args:
            tamano_salida: Tamaño de salida deseado (ancho, alto)
        """
        self.tamano_salida = tamano_salida
        
        # Cargar clasificadores Haar Cascade de OpenCV
        self.detector_rostro = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.detector_ojos = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def detectar_rostro(self, imagen: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta el rostro principal en la imagen
        
        Args:
            imagen: Imagen en escala de grises o BGR
            
        Returns:
            Tupla (x, y, w, h) del rostro o None si no se detecta
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen.copy()
        
        # Detectar rostros
        rostros = self.detector_rostro.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(rostros) == 0:
            return None
        
        # Devolver el rostro más grande
        rostro_principal = max(rostros, key=lambda r: r[2] * r[3])
        return tuple(int(x) for x in rostro_principal)
    
    def detectar_ojos(self, imagen: np.ndarray, roi_rostro: Tuple[int, int, int, int]) -> list:
        """
        Detecta ojos dentro de la región del rostro
        
        Args:
            imagen: Imagen en escala de grises
            roi_rostro: (x, y, w, h) del rostro
            
        Returns:
            Lista de ojos detectados [(x, y, w, h), ...]
        """
        x, y, w, h = roi_rostro
        
        # Región de interés para ojos (mitad superior del rostro)
        roi = imagen[y:y + h//2, x:x + w]
        
        # Detectar ojos
        ojos = self.detector_ojos.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )
        
        # Ajustar coordenadas relativas al rostro
        ojos_ajustados = []
        for (ex, ey, ew, eh) in ojos:
            ojos_ajustados.append((x + ex, y + ey, ew, eh))
        
        return ojos_ajustados
    
    def calcular_angulo_rotacion(self, ojo_izq: Tuple[int, int], 
                                  ojo_der: Tuple[int, int]) -> float:
        """
        Calcula el ángulo de rotación necesario basado en la posición de los ojos
        
        Args:
            ojo_izq: Centro del ojo izquierdo (x, y)
            ojo_der: Centro del ojo derecho (x, y)
            
        Returns:
            Ángulo en grados
        """
        dx = ojo_der[0] - ojo_izq[0]
        dy = ojo_der[1] - ojo_izq[1]
        angulo = np.degrees(np.arctan2(dy, dx))
        return angulo
    
    def rotar_imagen(self, imagen: np.ndarray, angulo: float, 
                     centro: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Rota la imagen alrededor de un centro
        
        Args:
            imagen: Imagen a rotar
            angulo: Ángulo en grados
            centro: Centro de rotación (x, y), si es None usa el centro de la imagen
            
        Returns:
            Imagen rotada
        """
        h, w = imagen.shape[:2]
        
        if centro is None:
            centro = (w // 2, h // 2)
        
        # Asegurar que centro sea tuple de ints
        centro = (int(centro[0]), int(centro[1]))
        
        # Matriz de rotación
        matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        
        # Rotar imagen
        rotada = cv2.warpAffine(imagen, matriz, (w, h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotada
    
    def normalizar_iluminacion(self, imagen: np.ndarray) -> np.ndarray:
        """
        Normaliza la iluminación de la imagen usando ecualización de histograma
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen con iluminación normalizada
        """
        # Ecualización adaptativa de histograma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalizada = clahe.apply(imagen)
        
        return normalizada
    
    def alinear_rostro(self, imagen: np.ndarray, 
                       visualizar: bool = False) -> Tuple[Optional[np.ndarray], dict]:
        """
        Pipeline completo de alineación de rostro
        
        Args:
            imagen: Imagen de entrada (BGR o escala de grises)
            visualizar: Si True, retorna información para visualización
            
        Returns:
            Tupla (rostro_alineado, info_dict)
            - rostro_alineado: Imagen del rostro alineado y normalizado o None
            - info_dict: Diccionario con información del proceso
        """
        info = {
            'rostro_detectado': False,
            'ojos_detectados': False,
            'angulo_rotacion': 0,
            'bbox_rostro': None,
            'ojos': []
        }
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen.copy()
        
        # 1. Detectar rostro
        bbox_rostro = self.detectar_rostro(gris)
        if bbox_rostro is None:
            return None, info
        
        info['rostro_detectado'] = True
        info['bbox_rostro'] = bbox_rostro
        x, y, w, h = bbox_rostro
        
        # 2. Detectar ojos
        ojos = self.detectar_ojos(gris, bbox_rostro)
        info['ojos'] = ojos
        
        # 3. Alinear si se detectan al menos 2 ojos
        imagen_alineada = gris.copy()
        if len(ojos) >= 2:
            info['ojos_detectados'] = True
            
            # Ordenar ojos por posición x (izquierdo primero)
            ojos_ordenados = sorted(ojos, key=lambda e: e[0])
            
            # Calcular centros de los dos ojos principales
            ojo_izq_centro = (
                int(ojos_ordenados[0][0] + ojos_ordenados[0][2] // 2),
                int(ojos_ordenados[0][1] + ojos_ordenados[0][3] // 2)
            )
            ojo_der_centro = (
                int(ojos_ordenados[1][0] + ojos_ordenados[1][2] // 2),
                int(ojos_ordenados[1][1] + ojos_ordenados[1][3] // 2)
            )
            
            # Calcular ángulo de rotación
            angulo = self.calcular_angulo_rotacion(ojo_izq_centro, ojo_der_centro)
            info['angulo_rotacion'] = angulo
            
            # Rotar imagen
            centro_rostro = (int(x + w // 2), int(y + h // 2))
            imagen_alineada = self.rotar_imagen(gris, angulo, centro_rostro)
            
            # Actualizar bbox después de rotación (reaplicar detección)
            bbox_nuevo = self.detectar_rostro(imagen_alineada)
            if bbox_nuevo is not None:
                x, y, w, h = bbox_nuevo
        
        # 4. Extraer región del rostro
        rostro_recortado = imagen_alineada[y:y+h, x:x+w]
        
        # 5. Redimensionar al tamaño deseado
        rostro_redimensionado = cv2.resize(
            rostro_recortado, 
            self.tamano_salida,
            interpolation=cv2.INTER_CUBIC
        )
        
        # 6. Normalizar iluminación
        rostro_normalizado = self.normalizar_iluminacion(rostro_redimensionado)
        
        return rostro_normalizado, info
    
    def procesar_lote(self, imagenes: list) -> Tuple[list, list]:
        """
        Procesa un lote de imágenes
        
        Args:
            imagenes: Lista de imágenes
            
        Returns:
            Tupla (rostros_alineados, infos)
        """
        rostros = []
        infos = []
        
        for img in imagenes:
            rostro, info = self.alinear_rostro(img)
            rostros.append(rostro)
            infos.append(info)
        
        return rostros, infos
