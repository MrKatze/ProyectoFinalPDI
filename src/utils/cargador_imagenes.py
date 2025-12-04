"""
Módulo Cargador de Imágenes
Carga y organiza las imágenes del dataset
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


class CargadorImagenes:
    """
    Clase para cargar imágenes organizadas por persona
    """
    
    def __init__(self, directorio_imagenes: str):
        """
        Inicializa el cargador
        
        Args:
            directorio_imagenes: Ruta al directorio con las imágenes
        """
        self.directorio = Path(directorio_imagenes)
        self.dataset = {}
        
    def cargar_dataset(self) -> Dict[str, List[np.ndarray]]:
        """
        Carga todas las imágenes organizadas por persona
        
        Estructura esperada:
        images/
            persona1/
                img1.jpg
                img2.jpg
                ...
            persona2/
                img1.jpg
                ...
        
        Returns:
            Diccionario {nombre_persona: [imagenes]}
        """
        if not self.directorio.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.directorio}")
        
        # Buscar subdirectorios (cada uno es una persona)
        for persona_dir in self.directorio.iterdir():
            if persona_dir.is_dir():
                nombre_persona = persona_dir.name
                imagenes = []
                
                # Cargar todas las imágenes de esta persona
                for archivo in persona_dir.iterdir():
                    if archivo.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        imagen = cv2.imread(str(archivo))
                        if imagen is not None:
                            imagenes.append(imagen)
                
                if len(imagenes) > 0:
                    self.dataset[nombre_persona] = imagenes
                    print(f"✓ Cargadas {len(imagenes)} imágenes de {nombre_persona}")
        
        print(f"\n Total: {len(self.dataset)} personas, "
              f"{sum(len(imgs) for imgs in self.dataset.values())} imágenes")
        
        return self.dataset
    
    def obtener_persona(self, nombre: str) -> List[np.ndarray]:
        """
        Obtiene las imágenes de una persona específica
        
        Args:
            nombre: Nombre de la persona
            
        Returns:
            Lista de imágenes
        """
        return self.dataset.get(nombre, [])
    
    def obtener_todas_imagenes(self) -> List[Tuple[str, np.ndarray]]:
        """
        Obtiene todas las imágenes con su etiqueta de persona
        
        Returns:
            Lista de tuplas (nombre_persona, imagen)
        """
        todas = []
        for nombre, imagenes in self.dataset.items():
            for img in imagenes:
                todas.append((nombre, img))
        return todas
    
    def obtener_nombres_personas(self) -> List[str]:
        """
        Obtiene la lista de nombres de personas
        
        Returns:
            Lista de nombres
        """
        return list(self.dataset.keys())
    
    def obtener_estadisticas(self) -> Dict:
        """
        Calcula estadísticas del dataset
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'num_personas': len(self.dataset),
            'total_imagenes': sum(len(imgs) for imgs in self.dataset.values()),
            'imagenes_por_persona': {nombre: len(imgs) 
                                    for nombre, imgs in self.dataset.items()}
        }
        
        # Calcular dimensiones promedio
        if stats['total_imagenes'] > 0:
            todas_dims = []
            for imagenes in self.dataset.values():
                for img in imagenes:
                    todas_dims.append(img.shape[:2])  # (h, w)
            
            alturas = [dim[0] for dim in todas_dims]
            anchos = [dim[1] for dim in todas_dims]
            
            stats['dimensiones_promedio'] = (
                int(np.mean(alturas)),
                int(np.mean(anchos))
            )
            stats['dimensiones_min'] = (min(alturas), min(anchos))
            stats['dimensiones_max'] = (max(alturas), max(anchos))
        
        return stats
    
    def visualizar_muestra(self, max_por_persona: int = 2) -> np.ndarray:
        """
        Crea una cuadrícula con muestras de cada persona
        
        Args:
            max_por_persona: Número máximo de imágenes por persona a mostrar
            
        Returns:
            Imagen con cuadrícula de muestras
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        num_personas = len(self.dataset)
        if num_personas == 0:
            return None
        
        fig = plt.figure(figsize=(15, num_personas * 3))
        gs = GridSpec(num_personas, max_por_persona, figure=fig)
        
        for i, (nombre, imagenes) in enumerate(self.dataset.items()):
            for j in range(min(max_por_persona, len(imagenes))):
                ax = fig.add_subplot(gs[i, j])
                img_rgb = cv2.cvtColor(imagenes[j], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"{nombre} - Imagen {j+1}")
                ax.axis('off')
        
        plt.tight_layout()
        return fig
