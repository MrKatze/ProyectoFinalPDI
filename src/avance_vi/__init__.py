"""
Avance VI - Descriptores de Forma
==================================

Módulo para la extracción de descriptores geométricos de las regiones faciales detectadas.

Descriptores implementados:
- Compacidad
- Distancia radial normalizada (media, desviación estándar, cruces por cero)
- Índice de área
- Índice de rugosidad
"""

from .descriptores_forma import DescriptoresForma
from .visualizador_descriptores import VisualizadorDescriptores

__all__ = ['DescriptoresForma', 'VisualizadorDescriptores']
