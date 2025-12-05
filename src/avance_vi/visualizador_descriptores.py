"""
Visualizador de Descriptores - Avance VI
=========================================

Módulo para generar visualizaciones de los descriptores de forma:
- Gráficas polares de distancias radiales
- Comparación contorno vs convex hull
- Tablas con valores de descriptores
- Panel completo de análisis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import os


class VisualizadorDescriptores:
    """
    Genera visualizaciones para los descriptores de forma del Avance VI.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (20, 15)):
        """
        Inicializa el visualizador.
        
        Args:
            figsize: Tamaño de las figuras (ancho, alto)
        """
        self.figsize = figsize
        
    def crear_grafica_polar(
        self,
        ax: plt.Axes,
        angulos: np.ndarray,
        distancias_norm: np.ndarray,
        titulo: str = "Distancia Radial Normalizada"
    ):
        """
        Crea una gráfica polar de distancias radiales.
        
        Args:
            ax: Eje polar de matplotlib
            angulos: Array de ángulos en radianes
            distancias_norm: Distancias normalizadas
            titulo: Título de la gráfica
        """
        try:
            # Cerrar el polígono
            angulos_cerrados = np.append(angulos, angulos[0])
            distancias_cerradas = np.append(distancias_norm, distancias_norm[0])
            
            # Plotear
            ax.plot(angulos_cerrados, distancias_cerradas, 'b-', linewidth=2)
            ax.fill(angulos_cerrados, distancias_cerradas, alpha=0.3, color='blue')
            
            # Estilo
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(-1)
            ax.set_title(titulo, pad=20, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"⚠️ Error creando gráfica polar: {e}")
            ax.text(0.5, 0.5, 'Error en\ngráfica polar', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def visualizar_contorno_vs_hull(
        self,
        ax: plt.Axes,
        imagen: np.ndarray,
        contorno: np.ndarray,
        titulo: str = "Contorno vs Convex Hull"
    ):
        """
        Visualiza el contorno original vs su envolvente convexa.
        
        Args:
            ax: Eje de matplotlib
            imagen: Imagen base
            contorno: Contorno en formato OpenCV
            titulo: Título de la visualización
        """
        try:
            # Crear copia de la imagen
            img_vis = imagen.copy()
            if len(img_vis.shape) == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)
            
            # Calcular convex hull
            hull = cv2.convexHull(contorno)
            
            # Dibujar
            cv2.drawContours(img_vis, [contorno], -1, (0, 255, 0), 2)  # Verde: contorno
            cv2.drawContours(img_vis, [hull], -1, (255, 0, 0), 2)      # Rojo: hull
            
            # Mostrar
            ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            ax.set_title(titulo, fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Leyenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Contorno original'),
                Patch(facecolor='red', label='Convex Hull')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
        except Exception as e:
            print(f"⚠️ Error visualizando contorno vs hull: {e}")
            ax.text(0.5, 0.5, 'Error en\nvisualización', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    def crear_tabla_descriptores(
        self,
        ax: plt.Axes,
        descriptores: Dict[str, float],
        titulo: str = "Descriptores de Forma"
    ):
        """
        Crea una tabla con los valores de los descriptores.
        
        Args:
            ax: Eje de matplotlib
            descriptores: Diccionario con descriptores
            titulo: Título de la tabla
        """
        try:
            ax.axis('off')
            ax.set_title(titulo, fontsize=10, fontweight='bold', pad=10)
            
            # Preparar datos para la tabla
            datos = [
                ['Área', f"{descriptores.get('area', 0):.2f} px²"],
                ['Perímetro', f"{descriptores.get('perimetro', 0):.2f} px"],
                ['Compacidad', f"{descriptores.get('compacidad', 0):.4f}"],
                ['Media Radial', f"{descriptores.get('media_radial', 0):.4f}"],
                ['Desv. Radial', f"{descriptores.get('desviacion_radial', 0):.4f}"],
                ['Cruces por Cero', f"{descriptores.get('cruces_por_cero', 0)}"],
                ['Índice de Área', f"{descriptores.get('indice_area', 0):.4f}"],
                ['Índice Rugosidad', f"{descriptores.get('indice_rugosidad', 0):.4f}"],
            ]
            
            # Crear tabla
            tabla = ax.table(
                cellText=datos,
                colLabels=['Descriptor', 'Valor'],
                cellLoc='left',
                loc='center',
                colWidths=[0.6, 0.4]
            )
            
            # Estilo
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(9)
            tabla.scale(1, 2)
            
            # Colorear encabezado
            for i in range(2):
                tabla[(0, i)].set_facecolor('#4CAF50')
                tabla[(0, i)].set_text_props(weight='bold', color='white')
            
            # Colorear filas alternadas
            for i in range(1, len(datos) + 1):
                for j in range(2):
                    if i % 2 == 0:
                        tabla[(i, j)].set_facecolor('#f0f0f0')
                        
        except Exception as e:
            print(f"⚠️ Error creando tabla de descriptores: {e}")
            ax.text(0.5, 0.5, 'Error en\ntabla', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def visualizar_analisis_completo(
        self,
        imagen: np.ndarray,
        contorno: np.ndarray,
        descriptores: Dict[str, float],
        angulos: np.ndarray,
        distancias_norm: np.ndarray,
        titulo_general: str = "Análisis de Descriptores de Forma",
        ruta_salida: Optional[str] = None
    ) -> plt.Figure:
        """
        Crea una visualización completa con:
        - Imagen con contorno
        - Gráfica polar
        - Contorno vs hull
        - Tabla de descriptores
        
        Args:
            imagen: Imagen original
            contorno: Contorno detectado
            descriptores: Diccionario con descriptores
            angulos: Ángulos para gráfica polar
            distancias_norm: Distancias normalizadas
            titulo_general: Título principal
            ruta_salida: Ruta para guardar (opcional)
            
        Returns:
            Figura de matplotlib
        """
        try:
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(titulo_general, fontsize=16, fontweight='bold', y=0.98)
            
            # Grid de 2x3
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # 1. Imagen original con contorno
            ax1 = fig.add_subplot(gs[0, 0])
            img_con_contorno = imagen.copy()
            if len(img_con_contorno.shape) == 2:
                img_con_contorno = cv2.cvtColor(img_con_contorno, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(img_con_contorno, [contorno], -1, (0, 255, 0), 2)
            ax1.imshow(cv2.cvtColor(img_con_contorno, cv2.COLOR_BGR2RGB))
            ax1.set_title('Región Detectada', fontsize=11, fontweight='bold')
            ax1.axis('off')
            
            # 2. Gráfica polar de distancias radiales
            ax2 = fig.add_subplot(gs[0, 1], projection='polar')
            self.crear_grafica_polar(ax2, angulos, distancias_norm, 
                                    "Distancia Radial Normalizada")
            
            # 3. Contorno vs Convex Hull
            ax3 = fig.add_subplot(gs[0, 2])
            self.visualizar_contorno_vs_hull(ax3, imagen, contorno, 
                                            "Contorno vs Convex Hull")
            
            # 4. Tabla de descriptores (ocupa toda la fila inferior izquierda)
            ax4 = fig.add_subplot(gs[1, :2])
            self.crear_tabla_descriptores(ax4, descriptores, "Descriptores de Forma")
            
            # 5. Interpretación (fila inferior derecha)
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.axis('off')
            ax5.set_title('Interpretación', fontsize=11, fontweight='bold')
            
            interpretacion = self._generar_interpretacion(descriptores)
            ax5.text(0.05, 0.95, interpretacion, 
                    transform=ax5.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Guardar si se especifica ruta
            if ruta_salida:
                plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
                print(f"✓ Visualización guardada: {ruta_salida}")
            
            return fig
            
        except Exception as e:
            print(f"⚠️ Error en visualización completa: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Error generando\nvisualización:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
    
    def _generar_interpretacion(self, descriptores: Dict[str, float]) -> str:
        """
        Genera texto interpretativo de los descriptores.
        
        Args:
            descriptores: Diccionario con descriptores
            
        Returns:
            Texto con interpretación
        """
        try:
            lineas = []
            
            # Compacidad
            comp = descriptores.get('compacidad', 0)
            if comp < 1.2:
                lineas.append("• Forma muy circular")
            elif comp < 2.0:
                lineas.append("• Forma moderadamente regular")
            else:
                lineas.append("• Forma irregular/alargada")
            
            # Desviación radial
            desv = descriptores.get('desviacion_radial', 0)
            if desv < 0.1:
                lineas.append("• Borde muy uniforme")
            elif desv < 0.2:
                lineas.append("• Borde moderadamente uniforme")
            else:
                lineas.append("• Borde con irregularidades")
            
            # Rugosidad
            rug = descriptores.get('indice_rugosidad', 1)
            if rug < 1.05:
                lineas.append("• Perímetro suave")
            elif rug < 1.15:
                lineas.append("• Perímetro con leves ondulaciones")
            else:
                lineas.append("• Perímetro rugoso")
            
            # Cruces por cero
            cruces = descriptores.get('cruces_por_cero', 0)
            if cruces < 10:
                lineas.append("• Muy pocos salientes")
            elif cruces < 30:
                lineas.append("• Algunos salientes/entrantes")
            else:
                lineas.append("• Múltiples salientes/entrantes")
            
            return '\n'.join(lineas)
            
        except Exception as e:
            return f"Error en interpretación: {e}"
    
    def visualizar_multiples_rasgos(
        self,
        imagen_original: np.ndarray,
        rasgos: Dict[str, Dict],
        ruta_salida: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualiza descriptores de múltiples rasgos (ojos, nariz, boca).
        
        Args:
            imagen_original: Imagen original
            rasgos: Dict con formato:
                {
                    'ojos': {'contorno': ..., 'descriptores': ..., 'angulos': ..., 'distancias': ...},
                    'nariz': {...},
                    'boca': {...}
                }
            ruta_salida: Ruta para guardar
            
        Returns:
            Figura de matplotlib
        """
        try:
            num_rasgos = len(rasgos)
            fig = plt.figure(figsize=(20, 6 * num_rasgos))
            fig.suptitle('Análisis de Descriptores - Todos los Rasgos Faciales', 
                        fontsize=18, fontweight='bold', y=0.995)
            
            for idx, (nombre_rasgo, datos) in enumerate(rasgos.items()):
                # Subgrid para cada rasgo
                gs = fig.add_gridspec(num_rasgos, 4, hspace=0.4, wspace=0.3)
                base_row = idx
                
                # Título del rasgo
                ax_titulo = fig.add_subplot(gs[base_row, :])
                ax_titulo.axis('off')
                ax_titulo.text(0.5, 0.5, f'═══ {nombre_rasgo.upper()} ═══',
                             ha='center', va='center',
                             fontsize=14, fontweight='bold',
                             transform=ax_titulo.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
                
                if datos is None or datos.get('contorno') is None:
                    continue
                
                contorno = datos['contorno']
                descriptores = datos.get('descriptores', {})
                angulos = datos.get('angulos', np.array([]))
                distancias = datos.get('distancias_norm', np.array([]))
                
                # 1. Imagen con contorno
                ax1 = fig.add_subplot(gs[base_row, 0])
                img_vis = imagen_original.copy()
                if len(img_vis.shape) == 2:
                    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2RGB)
                cv2.drawContours(img_vis, [contorno], -1, (0, 255, 0), 2)
                ax1.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
                ax1.set_title('Detección', fontsize=10, fontweight='bold')
                ax1.axis('off')
                
                # 2. Gráfica polar
                if len(angulos) > 0 and len(distancias) > 0:
                    ax2 = fig.add_subplot(gs[base_row, 1], projection='polar')
                    self.crear_grafica_polar(ax2, angulos, distancias, "Dist. Radial")
                
                # 3. Contorno vs Hull
                ax3 = fig.add_subplot(gs[base_row, 2])
                self.visualizar_contorno_vs_hull(ax3, imagen_original, contorno)
                
                # 4. Tabla
                ax4 = fig.add_subplot(gs[base_row, 3])
                self.crear_tabla_descriptores(ax4, descriptores, "Descriptores")
            
            # Guardar
            if ruta_salida:
                plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
                print(f"✓ Visualización múltiple guardada: {ruta_salida}")
            
            return fig
            
        except Exception as e:
            print(f"⚠️ Error en visualización múltiple: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Error:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig
