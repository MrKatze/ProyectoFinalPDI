"""
Módulo de Visualización
Crea visualizaciones de los resultados del procesamiento
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict
import os


class Visualizador:
    """
    Clase para crear visualizaciones del pipeline de procesamiento
    """
    
    @staticmethod
    def crear_panel_20(imagen_original: np.ndarray,
                       resultados: Dict,
                       titulo: str = "Pipeline Avance V") -> plt.Figure:
        """
        Crea visualización con 20 paneles mostrando todo el proceso
        
        Args:
            imagen_original: Imagen original BGR
            resultados: Diccionario con todos los resultados intermedios
            titulo: Título general
            
        Returns:
            Figura de matplotlib
        """
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle(titulo, fontsize=16, fontweight='bold')
        
        # Panel 1: Imagen original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
        ax1.set_title("1. Original")
        ax1.axis('off')
        
        # Panel 2: Rostro detectado
        ax2 = fig.add_subplot(gs[0, 1])
        if 'rostro_detectado' in resultados:
            ax2.imshow(resultados['rostro_detectado'], cmap='gray')
            ax2.set_title("2. Rostro Detectado")
        ax2.axis('off')
        
        # Panel 3: Rostro alineado
        ax3 = fig.add_subplot(gs[0, 2])
        if 'rostro_alineado' in resultados:
            ax3.imshow(resultados['rostro_alineado'], cmap='gray')
            ax3.set_title("3. Rostro Alineado")
        ax3.axis('off')
        
        # Panel 4: Normalización de iluminación
        ax4 = fig.add_subplot(gs[0, 3])
        if 'rostro_normalizado' in resultados:
            ax4.imshow(resultados['rostro_normalizado'], cmap='gray')
            ax4.set_title("4. Iluminación Normalizada")
        ax4.axis('off')
        
        # Panel 5: Filtro Mediana
        ax5 = fig.add_subplot(gs[1, 0])
        if 'filtro_mediana' in resultados:
            ax5.imshow(resultados['filtro_mediana'], cmap='gray')
            ax5.set_title("5. Filtro Mediana")
        ax5.axis('off')
        
        # Panel 6: Filtro Gaussiano
        ax6 = fig.add_subplot(gs[1, 1])
        if 'filtro_gaussiano' in resultados:
            ax6.imshow(resultados['filtro_gaussiano'], cmap='gray')
            ax6.set_title("6. Filtro Gaussiano")
        ax6.axis('off')
        
        # Panel 7: Filtro Laplaciano
        ax7 = fig.add_subplot(gs[1, 2])
        if 'filtro_laplaciano' in resultados:
            ax7.imshow(resultados['filtro_laplaciano'], cmap='gray')
            ax7.set_title("7. Filtro Laplaciano")
        ax7.axis('off')
        
        # Panel 8: Filtro Highboost
        ax8 = fig.add_subplot(gs[1, 3])
        if 'filtro_highboost' in resultados:
            ax8.imshow(resultados['filtro_highboost'], cmap='gray')
            ax8.set_title("8. Highboost")
        ax8.axis('off')
        
        # Panel 9: Umbralización Global
        ax9 = fig.add_subplot(gs[2, 0])
        if 'umbral_global' in resultados:
            ax9.imshow(resultados['umbral_global'], cmap='gray')
            ax9.set_title("9. Umbral Global")
        ax9.axis('off')
        
        # Panel 10: Umbralización Otsu
        ax10 = fig.add_subplot(gs[2, 1])
        if 'umbral_otsu' in resultados:
            ax10.imshow(resultados['umbral_otsu'], cmap='gray')
            ax10.set_title("10. Umbral Otsu")
        ax10.axis('off')
        
        # Panel 11: Máscara de Piel
        ax11 = fig.add_subplot(gs[2, 2])
        if 'mascara_piel' in resultados:
            ax11.imshow(resultados['mascara_piel'], cmap='gray')
            ax11.set_title("11. Máscara de Piel")
        ax11.axis('off')
        
        # Panel 12: Segmentación Fondo-Rostro
        ax12 = fig.add_subplot(gs[2, 3])
        if 'segmentacion_final' in resultados:
            ax12.imshow(resultados['segmentacion_final'], cmap='gray')
            ax12.set_title("12. Segmentación Final")
        ax12.axis('off')
        
        # Panel 13: Canny
        ax13 = fig.add_subplot(gs[3, 0])
        if 'canny' in resultados:
            ax13.imshow(resultados['canny'], cmap='gray')
            ax13.set_title("13. Canny")
        ax13.axis('off')
        
        # Panel 14: Marr-Hildreth
        ax14 = fig.add_subplot(gs[3, 1])
        if 'marr_hildreth' in resultados:
            ax14.imshow(resultados['marr_hildreth'], cmap='gray')
            ax14.set_title("14. Marr-Hildreth")
        ax14.axis('off')
        
        # Panel 15: Morfología - Apertura
        ax15 = fig.add_subplot(gs[3, 2])
        if 'morfologia_apertura' in resultados:
            ax15.imshow(resultados['morfologia_apertura'], cmap='gray')
            ax15.set_title("15. Morfología Apertura")
        ax15.axis('off')
        
        # Panel 16: Morfología - Cierre
        ax16 = fig.add_subplot(gs[3, 3])
        if 'morfologia_cierre' in resultados:
            ax16.imshow(resultados['morfologia_cierre'], cmap='gray')
            ax16.set_title("16. Morfología Cierre")
        ax16.axis('off')
        
        # Panel 17: Ojos Detectados
        ax17 = fig.add_subplot(gs[4, 0])
        if 'ojos_detectados' in resultados:
            img_ojos = Visualizador._dibujar_detecciones(
                resultados['rostro_normalizado'],
                resultados['ojos_detectados'],
                color=(0, 255, 0)
            )
            ax17.imshow(cv2.cvtColor(img_ojos, cv2.COLOR_BGR2RGB))
            ax17.set_title("17. Ojos Detectados")
        ax17.axis('off')
        
        # Panel 18: Nariz Detectada
        ax18 = fig.add_subplot(gs[4, 1])
        if 'nariz_detectada' in resultados:
            img_nariz = Visualizador._dibujar_detecciones(
                resultados['rostro_normalizado'],
                [resultados['nariz_detectada']],
                color=(255, 0, 0)
            )
            ax18.imshow(cv2.cvtColor(img_nariz, cv2.COLOR_BGR2RGB))
            ax18.set_title("18. Nariz Detectada")
        ax18.axis('off')
        
        # Panel 19: Boca Detectada
        ax19 = fig.add_subplot(gs[4, 2])
        if 'boca_detectada' in resultados:
            img_boca = Visualizador._dibujar_detecciones(
                resultados['rostro_normalizado'],
                [resultados['boca_detectada']],
                color=(0, 0, 255)
            )
            ax19.imshow(cv2.cvtColor(img_boca, cv2.COLOR_BGR2RGB))
            ax19.set_title("19. Boca Detectada")
        ax19.axis('off')
        
        # Panel 20: Resultado Final
        ax20 = fig.add_subplot(gs[4, 3])
        if 'resultado_final' in resultados:
            ax20.imshow(cv2.cvtColor(resultados['resultado_final'], cv2.COLOR_BGR2RGB))
            ax20.set_title("20. Resultado Final")
        ax20.axis('off')
        
        return fig
    
    @staticmethod
    def _dibujar_detecciones(imagen: np.ndarray,
                            detecciones: List[Tuple[int, int, int, int]],
                            color: Tuple[int, int, int] = (0, 255, 0),
                            grosor: int = 2) -> np.ndarray:
        """
        Dibuja bounding boxes en la imagen
        
        Args:
            imagen: Imagen (puede ser escala de grises o BGR)
            detecciones: Lista de (x, y, w, h)
            color: Color BGR
            grosor: Grosor de línea
            
        Returns:
            Imagen con detecciones dibujadas
        """
        # Convertir a BGR si es necesario
        if len(imagen.shape) == 2:
            img_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            img_color = imagen.copy()
        
        # Dibujar cada detección
        for (x, y, w, h) in detecciones:
            cv2.rectangle(img_color, (x, y), (x + w, y + h), color, grosor)
        
        return img_color
    
    @staticmethod
    def guardar_figura(fig: plt.Figure, ruta: str, dpi: int = 150):
        """
        Guarda una figura en archivo
        
        Args:
            fig: Figura de matplotlib
            ruta: Ruta donde guardar
            dpi: Resolución
        """
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        fig.savefig(ruta, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Guardado: {ruta}")
    
    @staticmethod
    def comparar_metodos(imagen: np.ndarray,
                        resultados_metodos: Dict[str, np.ndarray],
                        titulo: str = "Comparación de Métodos") -> plt.Figure:
        """
        Compara resultados de diferentes métodos
        
        Args:
            imagen: Imagen original
            resultados_metodos: Dict con {nombre_metodo: resultado}
            titulo: Título
            
        Returns:
            Figura
        """
        num_metodos = len(resultados_metodos)
        num_cols = min(4, num_metodos + 1)
        num_filas = (num_metodos + 1 + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_filas, num_cols, figsize=(num_cols * 4, num_filas * 4))
        fig.suptitle(titulo, fontsize=14, fontweight='bold')
        
        if num_filas == 1:
            axes = axes.reshape(1, -1)
        
        # Aplanar axes
        axes_flat = axes.flatten()
        
        # Panel 0: Original
        if len(imagen.shape) == 3:
            axes_flat[0].imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        else:
            axes_flat[0].imshow(imagen, cmap='gray')
        axes_flat[0].set_title("Original")
        axes_flat[0].axis('off')
        
        # Mostrar cada método
        for idx, (nombre, resultado) in enumerate(resultados_metodos.items(), 1):
            if len(resultado.shape) == 3:
                axes_flat[idx].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
            else:
                axes_flat[idx].imshow(resultado, cmap='gray')
            axes_flat[idx].set_title(nombre)
            axes_flat[idx].axis('off')
        
        # Ocultar axes no usados
        for idx in range(num_metodos + 1, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        return fig
