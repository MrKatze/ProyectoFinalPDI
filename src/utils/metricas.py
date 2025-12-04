"""
Módulo de Métricas
Calcula métricas de evaluación del sistema
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional


class CalculadorMetricas:
    """
    Clase para calcular métricas de desempeño del sistema
    """
    
    def __init__(self):
        """Inicializa el calculador de métricas"""
        self.tiempos = []
        self.detecciones_exitosas = {
            'ojos': 0,
            'nariz': 0,
            'boca': 0,
            'rostro': 0
        }
        self.total_procesadas = 0
    
    def registrar_deteccion(self,
                           tipo: str,
                           exitosa: bool):
        """
        Registra el resultado de una detección
        
        Args:
            tipo: 'ojos', 'nariz', 'boca', 'rostro'
            exitosa: Si la detección fue exitosa
        """
        if exitosa:
            self.detecciones_exitosas[tipo] += 1
    
    def registrar_tiempo(self, tiempo: float):
        """
        Registra el tiempo de procesamiento
        
        Args:
            tiempo: Tiempo en segundos
        """
        self.tiempos.append(tiempo)
    
    def incrementar_total(self):
        """Incrementa el contador de imágenes procesadas"""
        self.total_procesadas += 1
    
    def calcular_tasas_deteccion(self) -> Dict[str, float]:
        """
        Calcula tasas de detección para cada rasgo
        
        Returns:
            Diccionario con tasas de detección [0, 1]
        """
        if self.total_procesadas == 0:
            return {k: 0.0 for k in self.detecciones_exitosas.keys()}
        
        tasas = {}
        for tipo, exitosas in self.detecciones_exitosas.items():
            tasas[tipo] = exitosas / self.total_procesadas
        
        return tasas
    
    def calcular_estadisticas_tiempo(self) -> Dict[str, float]:
        """
        Calcula estadísticas de tiempo de procesamiento
        
        Returns:
            Diccionario con estadísticas
        """
        if len(self.tiempos) == 0:
            return {
                'promedio': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0
            }
        
        return {
            'promedio': np.mean(self.tiempos),
            'min': np.min(self.tiempos),
            'max': np.max(self.tiempos),
            'total': np.sum(self.tiempos),
            'std': np.std(self.tiempos)
        }
    
    def generar_reporte(self) -> str:
        """
        Genera un reporte textual de las métricas
        
        Returns:
            String con el reporte
        """
        tasas = self.calcular_tasas_deteccion()
        tiempos = self.calcular_estadisticas_tiempo()
        
        reporte = []
        reporte.append("=" * 60)
        reporte.append("REPORTE DE MÉTRICAS - AVANCE V")
        reporte.append("=" * 60)
        reporte.append(f"\nImágenes procesadas: {self.total_procesadas}")
        reporte.append("\n" + "-" * 60)
        reporte.append("TASAS DE DETECCIÓN:")
        reporte.append("-" * 60)
        
        for tipo, tasa in tasas.items():
            exitosas = self.detecciones_exitosas[tipo]
            reporte.append(f"  {tipo.capitalize():12} : {exitosas:3}/{self.total_procesadas:3} "
                          f"({tasa*100:5.1f}%)")
        
        reporte.append("\n" + "-" * 60)
        reporte.append("TIEMPOS DE PROCESAMIENTO:")
        reporte.append("-" * 60)
        reporte.append(f"  Tiempo total    : {tiempos['total']:.2f} segundos")
        reporte.append(f"  Tiempo promedio : {tiempos['promedio']:.3f} segundos/imagen")
        reporte.append(f"  Tiempo mínimo   : {tiempos['min']:.3f} segundos")
        reporte.append(f"  Tiempo máximo   : {tiempos['max']:.3f} segundos")
        reporte.append(f"  Desv. estándar  : {tiempos['std']:.3f} segundos")
        
        if tiempos['promedio'] > 0:
            fps = 1.0 / tiempos['promedio']
            reporte.append(f"  FPS estimado    : {fps:.2f} imágenes/segundo")
        
        reporte.append("=" * 60)
        
        return "\n".join(reporte)
    
    def exportar_csv(self, ruta: str):
        """
        Exporta métricas a archivo CSV
        
        Args:
            ruta: Ruta del archivo CSV
        """
        import csv
        
        tasas = self.calcular_tasas_deteccion()
        tiempos = self.calcular_estadisticas_tiempo()
        
        with open(ruta, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Encabezados
            writer.writerow(['Métrica', 'Valor'])
            writer.writerow(['Total Procesadas', self.total_procesadas])
            writer.writerow([])
            
            # Tasas de detección
            writer.writerow(['Tasas de Detección', ''])
            for tipo, tasa in tasas.items():
                writer.writerow([f'  {tipo}', f'{tasa:.4f}'])
            writer.writerow([])
            
            # Tiempos
            writer.writerow(['Tiempos de Procesamiento', ''])
            writer.writerow(['  Total (s)', f'{tiempos["total"]:.4f}'])
            writer.writerow(['  Promedio (s)', f'{tiempos["promedio"]:.4f}'])
            writer.writerow(['  Mínimo (s)', f'{tiempos["min"]:.4f}'])
            writer.writerow(['  Máximo (s)', f'{tiempos["max"]:.4f}'])
            writer.writerow(['  Desv. Estándar (s)', f'{tiempos["std"]:.4f}'])
        
        print(f"✓ Métricas exportadas a: {ruta}")
    
    @staticmethod
    def calcular_iou(bbox1: Tuple[int, int, int, int],
                    bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calcula Intersection over Union entre dos bounding boxes
        
        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)
            
        Returns:
            IoU [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Intersección
        x_izq = max(x1, x2)
        y_sup = max(y1, y2)
        x_der = min(x1 + w1, x2 + w2)
        y_inf = min(y1 + h1, y2 + h2)
        
        if x_der < x_izq or y_inf < y_sup:
            return 0.0
        
        area_interseccion = (x_der - x_izq) * (y_inf - y_sup)
        area_union = w1 * h1 + w2 * h2 - area_interseccion
        
        return area_interseccion / area_union if area_union > 0 else 0.0
    
    @staticmethod
    def calcular_distancia_centros(bbox1: Tuple[int, int, int, int],
                                   bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calcula distancia euclidiana entre centros de dos bounding boxes
        
        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)
            
        Returns:
            Distancia en píxeles
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        c1_x, c1_y = x1 + w1 // 2, y1 + h1 // 2
        c2_x, c2_y = x2 + w2 // 2, y2 + h2 // 2
        
        return np.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)


class Cronometro:
    """Clase auxiliar para medir tiempos"""
    
    def __init__(self):
        """Inicializa el cronómetro"""
        self.inicio = None
        self.tiempo_transcurrido = 0
    
    def iniciar(self):
        """Inicia el cronómetro"""
        self.inicio = time.time()
    
    def detener(self) -> float:
        """
        Detiene el cronómetro
        
        Returns:
            Tiempo transcurrido en segundos
        """
        if self.inicio is None:
            return 0.0
        
        self.tiempo_transcurrido = time.time() - self.inicio
        return self.tiempo_transcurrido
    
    def obtener_tiempo(self) -> float:
        """
        Obtiene el tiempo actual sin detener
        
        Returns:
            Tiempo transcurrido en segundos
        """
        if self.inicio is None:
            return 0.0
        
        return time.time() - self.inicio
