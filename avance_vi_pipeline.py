#!/usr/bin/env python3
"""
Pipeline Avance VI - Descriptores de Forma
===========================================

Integra la detección de rasgos faciales (Avance V) con la extracción
de descriptores de forma (Avance VI).

Flujo:
1. Cargar imágenes del dataset
2. Ejecutar pipeline Avance V (detección de rasgos)
3. Extraer descriptores de forma de cada rasgo detectado
4. Generar visualizaciones con análisis completo
5. Exportar resultados a CSV/JSON

Uso:
    python3 avance_vi_pipeline.py                    # Procesar todo el dataset
    python3 avance_vi_pipeline.py --imagen foto.jpg  # Procesar una imagen
    python3 avance_vi_pipeline.py --ayuda            # Ver ayuda
"""

import sys
import os
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from typing import Dict, List, Optional, Tuple

# Añadir directorio src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Importar módulos del proyecto
from src.avance_ii.alineacion import AlineadorRostros
from src.avance_ii.filtros import FiltrosImagen
from src.avance_iii.umbralizacion import Umbralizador
from src.avance_iv.deteccion_rasgos import DetectorRegiones
from src.avance_v.segmentador_ojos import SegmentadorOjos
from src.avance_v.segmentador_nariz import SegmentadorNariz
from src.avance_v.segmentador_boca import SegmentadorBoca
from src.avance_vi.descriptores_forma import DescriptoresForma
from src.avance_vi.visualizador_descriptores import VisualizadorDescriptores
from src.utils.cargador_imagenes import CargadorImagenes


class PipelineAvanceVI:
    """
    Pipeline completo que integra Avances II-VI.
    Detecta rasgos faciales y extrae sus descriptores de forma.
    """
    
    def __init__(self, dir_resultados: str = 'resultados/avance_vi'):
        """
        Inicializa el pipeline.
        
        Args:
            dir_resultados: Directorio para guardar resultados
        """
        print("🚀 Inicializando Pipeline Avance VI...")
        
        self.dir_resultados = Path(dir_resultados)
        self.dir_resultados.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes Avances II-V
        self.alineador = AlineadorRostros()
        self.filtros = FiltrosImagen()
        self.umbralizador = Umbralizador()
        self.detector_regiones = DetectorRegiones()
        self.segmentador_ojos = SegmentadorOjos()
        self.segmentador_nariz = SegmentadorNariz()
        self.segmentador_boca = SegmentadorBoca()
        
        # Inicializar componentes Avance VI
        self.extractor_descriptores = DescriptoresForma(num_puntos_radiales=360)
        self.visualizador = VisualizadorDescriptores()
        
        # Estadísticas
        self.resultados_globales = []
        
        print("✓ Pipeline inicializado correctamente\n")
    
    def procesar_imagen(
        self, 
        imagen: np.ndarray, 
        nombre_imagen: str,
        guardar_visualizacion: bool = True
    ) -> Optional[Dict]:
        """
        Procesa una imagen completa: detección + descriptores.
        
        Args:
            imagen: Imagen a procesar
            nombre_imagen: Nombre para identificación
            guardar_visualizacion: Si guardar visualizaciones
            
        Returns:
            Diccionario con resultados o None si falla
        """
        try:
            print(f"\n{'='*60}")
            print(f"Procesando: {nombre_imagen}")
            print(f"{'='*60}")
            
            # ===== AVANCE II: Preprocesamiento =====
            print("📸 Avance II: Preprocesamiento...")
            
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = imagen.copy()
            
            # Alinear rostro (devuelve tupla: imagen, info)
            rostro_alineado, info_alineacion = self.alineador.alinear_rostro(gris)
            
            if rostro_alineado is None:
                print("⚠️  No se detectaron rostros")
                return None
            
            # Ya está normalizado a 256x256, usar como rostro procesado
            rostro_normalizado = rostro_alineado
            
            # ===== AVANCE III: Umbralización =====
            print("🎭 Avance III: Segmentación...")
            # Convertir a color si es necesario para segmentación de piel
            if len(rostro_normalizado.shape) == 2:
                rostro_color = cv2.cvtColor(rostro_normalizado, cv2.COLOR_GRAY2BGR)
            else:
                rostro_color = rostro_normalizado
            mascara_piel = self.umbralizador.segmentar_piel(rostro_color)
            
            # ===== AVANCE IV: Regiones =====
            print("📐 Avance IV: Detección de regiones...")
            h, w = rostro_normalizado.shape[:2]
            regiones = DetectorRegiones.calcular_regiones_estandar(w, h)
            
            # ===== AVANCE V: Segmentación Avanzada =====
            print("🔍 Avance V: Segmentación de rasgos...")
            
            h, w = rostro_normalizado.shape[:2]
            
            # Detectar ojos (devuelve lista de bounding boxes)
            ojos_bbox, info_ojos = self.segmentador_ojos.segmentar_multimetodo(rostro_normalizado)
            # Crear máscara de ojos
            mascara_ojos = np.zeros((h, w), dtype=np.uint8)
            if ojos_bbox and len(ojos_bbox) > 0:
                for (x, y, w_ojo, h_ojo) in ojos_bbox:
                    mascara_ojos[y:y+h_ojo, x:x+w_ojo] = 255
            
            # Detectar nariz (devuelve bbox, necesita imagen color)
            nariz_bbox, info_nariz = self.segmentador_nariz.segmentar_multimetodo(
                rostro_normalizado, rostro_color
            )
            # Crear máscara de nariz
            mascara_nariz = np.zeros((h, w), dtype=np.uint8)
            if nariz_bbox is not None:
                x, y, w_nariz, h_nariz = nariz_bbox
                mascara_nariz[y:y+h_nariz, x:x+w_nariz] = 255
            
            # Detectar boca (devuelve bbox, necesita imagen color)
            boca_bbox, info_boca = self.segmentador_boca.segmentar_multimetodo(
                rostro_normalizado, rostro_color
            )
            # Crear máscara de boca
            mascara_boca = np.zeros((h, w), dtype=np.uint8)
            if boca_bbox is not None:
                x, y, w_boca, h_boca = boca_bbox
                mascara_boca[y:y+h_boca, x:x+w_boca] = 255
            
            # ===== AVANCE VI: Descriptores de Forma =====
            print("📊 Avance VI: Extrayendo descriptores de forma...")
            
            resultados_rasgos = {}
            
            # Procesar cada rasgo detectado
            rasgos_mascaras = {
                'ojos': mascara_ojos,
                'nariz': mascara_nariz,
                'boca': mascara_boca
            }
            
            for nombre_rasgo, mascara in rasgos_mascaras.items():
                if mascara is None or not np.any(mascara):
                    print(f"   ⚠️  {nombre_rasgo.capitalize()}: No detectado")
                    resultados_rasgos[nombre_rasgo] = None
                    continue
                
                # Extraer descriptores
                descriptores_lista = self.extractor_descriptores.extraer_descriptores_de_mascara(mascara)
                
                if len(descriptores_lista) == 0:
                    print(f"   ⚠️  {nombre_rasgo.capitalize()}: Sin contornos válidos")
                    resultados_rasgos[nombre_rasgo] = None
                    continue
                
                # Tomar el contorno más grande (principal)
                descriptores_lista_ordenada = sorted(
                    descriptores_lista, 
                    key=lambda x: x['area'], 
                    reverse=True
                )
                descriptores_principal = descriptores_lista_ordenada[0]
                
                # Obtener contorno para visualización
                contornos, _ = cv2.findContours(
                    mascara.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                contorno_principal = max(contornos, key=cv2.contourArea)
                
                # Obtener datos para visualización polar
                angulos, dist, dist_norm = self.extractor_descriptores.obtener_distancias_para_visualizacion(
                    contorno_principal
                )
                
                resultados_rasgos[nombre_rasgo] = {
                    'descriptores': descriptores_principal,
                    'contorno': contorno_principal,
                    'angulos': angulos,
                    'distancias_norm': dist_norm,
                    'mascara': mascara
                }
                
                print(f"   ✓ {nombre_rasgo.capitalize()}: Descriptores extraídos")
                print(f"      - Compacidad: {descriptores_principal['compacidad']:.4f}")
                print(f"      - Media radial: {descriptores_principal['media_radial']:.4f}")
                print(f"      - Desv. radial: {descriptores_principal['desviacion_radial']:.4f}")
                print(f"      - Cruces por cero: {descriptores_principal['cruces_por_cero']}")
                print(f"      - Índice de área: {descriptores_principal['indice_area']:.4f}")
                print(f"      - Índice rugosidad: {descriptores_principal['indice_rugosidad']:.4f}")
            
            # ===== Generar Visualización =====
            if guardar_visualizacion:
                print("\n🎨 Generando visualización...")
                ruta_vis = self.dir_resultados / f"{Path(nombre_imagen).stem}_descriptores.png"
                
                fig = self.visualizador.visualizar_multiples_rasgos(
                    rostro_normalizado,
                    resultados_rasgos,
                    ruta_salida=str(ruta_vis)
                )
                plt.close(fig)
            
            # Preparar resultado
            resultado = {
                'nombre_imagen': nombre_imagen,
                'rostro_detectado': True,
                'rasgos': {}
            }
            
            for nombre_rasgo, datos in resultados_rasgos.items():
                if datos is not None:
                    resultado['rasgos'][nombre_rasgo] = datos['descriptores']
                else:
                    resultado['rasgos'][nombre_rasgo] = None
            
            print(f"✓ Procesamiento completado: {nombre_imagen}\n")
            return resultado
            
        except Exception as e:
            print(f"❌ Error procesando {nombre_imagen}: {e}")
            import traceback
            print("\n=== TRACEBACK COMPLETO ===")
            traceback.print_exc()
            print("=========================\n")
            return None
    
    def procesar_dataset(self, dir_imagenes: str = 'images') -> List[Dict]:
        """
        Procesa todo el dataset de imágenes.
        
        Args:
            dir_imagenes: Directorio con las imágenes
            
        Returns:
            Lista con resultados de todas las imágenes
        """
        print("="*70)
        print("PIPELINE AVANCE VI - DESCRIPTORES DE FORMA")
        print("="*70)
        
        # Cargar imágenes
        cargador = CargadorImagenes(dir_imagenes)
        dataset = cargador.cargar_dataset()
        
        # Convertir dataset dict a lista plana de (nombre, imagen)
        imagenes = []
        for persona, imgs in dataset.items():
            for idx, img in enumerate(imgs, 1):
                nombre_img = f"{persona}_img{idx}.jpg"
                imagenes.append((nombre_img, img))
        
        if len(imagenes) == 0:
            print("❌ No se encontraron imágenes en el dataset")
            return []
        
        print(f"\n📁 Dataset: {len(imagenes)} imágenes encontradas")
        print(f"📂 Resultados se guardarán en: {self.dir_resultados}\n")
        
        tiempo_inicio = time.time()
        
        # Procesar cada imagen
        for idx, (nombre, imagen) in enumerate(imagenes, 1):
            print(f"\n[{idx}/{len(imagenes)}]", end=" ")
            
            resultado = self.procesar_imagen(imagen, nombre, guardar_visualizacion=True)
            
            if resultado is not None:
                self.resultados_globales.append(resultado)
        
        tiempo_total = time.time() - tiempo_inicio
        
        # Generar reporte final
        self._generar_reporte_final(tiempo_total, len(imagenes))
        
        return self.resultados_globales
    
    def _generar_reporte_final(self, tiempo_total: float, num_imagenes: int):
        """
        Genera reporte final con estadísticas y exporta a CSV/JSON.
        
        Args:
            tiempo_total: Tiempo total de procesamiento
            num_imagenes: Número total de imágenes procesadas
        """
        print("\n" + "="*70)
        print("GENERANDO REPORTES FINALES")
        print("="*70)
        
        # Calcular estadísticas
        num_exitosos = len(self.resultados_globales)
        tasa_exito = (num_exitosos / num_imagenes * 100) if num_imagenes > 0 else 0
        
        # Contar detecciones por rasgo
        detecciones_ojos = sum(1 for r in self.resultados_globales if r['rasgos'].get('ojos') is not None)
        detecciones_nariz = sum(1 for r in self.resultados_globales if r['rasgos'].get('nariz') is not None)
        detecciones_boca = sum(1 for r in self.resultados_globales if r['rasgos'].get('boca') is not None)
        
        # Evitar división por cero
        if num_exitosos == 0:
            print("\n❌ No se procesaron imágenes exitosamente")
            return
        
        # Reporte de texto
        ruta_reporte = self.dir_resultados / 'reporte_descriptores.txt'
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE AVANCE VI - DESCRIPTORES DE FORMA\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Imágenes procesadas: {num_imagenes}\n")
            f.write(f"Procesamiento exitoso: {num_exitosos} ({tasa_exito:.1f}%)\n")
            f.write(f"Tiempo total: {tiempo_total:.2f} segundos\n")
            f.write(f"Tiempo promedio: {tiempo_total/num_imagenes:.2f} seg/imagen\n\n")
            
            f.write("DETECCIONES POR RASGO:\n")
            f.write(f"  Ojos:  {detecciones_ojos}/{num_exitosos} ({detecciones_ojos/num_exitosos*100:.1f}%)\n")
            f.write(f"  Nariz: {detecciones_nariz}/{num_exitosos} ({detecciones_nariz/num_exitosos*100:.1f}%)\n")
            f.write(f"  Boca:  {detecciones_boca}/{num_exitosos} ({detecciones_boca/num_exitosos*100:.1f}%)\n\n")
            
            f.write("="*70 + "\n")
            f.write("RESULTADOS DETALLADOS POR IMAGEN\n")
            f.write("="*70 + "\n\n")
            
            for resultado in self.resultados_globales:
                f.write(f"\n{resultado['nombre_imagen']}\n")
                f.write("-" * len(resultado['nombre_imagen']) + "\n")
                
                for nombre_rasgo, descriptores in resultado['rasgos'].items():
                    f.write(f"\n  {nombre_rasgo.upper()}:\n")
                    if descriptores is None:
                        f.write("    No detectado\n")
                    else:
                        f.write(f"    Área: {descriptores['area']:.2f} px²\n")
                        f.write(f"    Perímetro: {descriptores['perimetro']:.2f} px\n")
                        f.write(f"    Compacidad: {descriptores['compacidad']:.4f}\n")
                        f.write(f"    Media radial: {descriptores['media_radial']:.4f}\n")
                        f.write(f"    Desviación radial: {descriptores['desviacion_radial']:.4f}\n")
                        f.write(f"    Cruces por cero: {descriptores['cruces_por_cero']}\n")
                        f.write(f"    Índice de área: {descriptores['indice_area']:.4f}\n")
                        f.write(f"    Índice rugosidad: {descriptores['indice_rugosidad']:.4f}\n")
        
        print(f"✓ Reporte de texto guardado: {ruta_reporte}")
        
        # Exportar a CSV
        self._exportar_csv()
        
        # Exportar a JSON
        self._exportar_json()
        
        # Mostrar resumen en consola
        print(f"\n{'='*70}")
        print(f"RESUMEN FINAL")
        print(f"{'='*70}")
        print(f"✓ Imágenes procesadas: {num_exitosos}/{num_imagenes} ({tasa_exito:.1f}%)")
        print(f"✓ Detección ojos: {detecciones_ojos}/{num_exitosos}")
        print(f"✓ Detección nariz: {detecciones_nariz}/{num_exitosos}")
        print(f"✓ Detección boca: {detecciones_boca}/{num_exitosos}")
        print(f"✓ Tiempo total: {tiempo_total:.2f} segundos")
        print(f"✓ Resultados en: {self.dir_resultados}")
        print(f"{'='*70}\n")
    
    def _exportar_csv(self):
        """Exporta resultados a formato CSV."""
        ruta_csv = self.dir_resultados / 'descriptores.csv'
        
        with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezados
            writer.writerow([
                'Imagen', 'Rasgo',
                'Area', 'Perimetro', 'Compacidad',
                'Media_Radial', 'Desviacion_Radial', 'Cruces_Cero',
                'Indice_Area', 'Indice_Rugosidad',
                'Centroide_X', 'Centroide_Y'
            ])
            
            # Datos
            for resultado in self.resultados_globales:
                nombre_img = resultado['nombre_imagen']
                
                for nombre_rasgo, descriptores in resultado['rasgos'].items():
                    if descriptores is None:
                        writer.writerow([nombre_img, nombre_rasgo] + ['N/A'] * 10)
                    else:
                        writer.writerow([
                            nombre_img, nombre_rasgo,
                            f"{descriptores['area']:.2f}",
                            f"{descriptores['perimetro']:.2f}",
                            f"{descriptores['compacidad']:.4f}",
                            f"{descriptores['media_radial']:.4f}",
                            f"{descriptores['desviacion_radial']:.4f}",
                            descriptores['cruces_por_cero'],
                            f"{descriptores['indice_area']:.4f}",
                            f"{descriptores['indice_rugosidad']:.4f}",
                            f"{descriptores['centroide_x']:.2f}",
                            f"{descriptores['centroide_y']:.2f}"
                        ])
        
        print(f"✓ CSV guardado: {ruta_csv}")
    
    def _exportar_json(self):
        """Exporta resultados a formato JSON."""
        ruta_json = self.dir_resultados / 'descriptores.json'
        
        # Convertir numpy arrays a listas para JSON
        resultados_serializables = []
        for resultado in self.resultados_globales:
            resultado_limpio = {
                'nombre_imagen': resultado['nombre_imagen'],
                'rostro_detectado': resultado['rostro_detectado'],
                'rasgos': {}
            }
            
            for nombre_rasgo, descriptores in resultado['rasgos'].items():
                resultado_limpio['rasgos'][nombre_rasgo] = descriptores
            
            resultados_serializables.append(resultado_limpio)
        
        with open(ruta_json, 'w', encoding='utf-8') as f:
            json.dump(resultados_serializables, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON guardado: {ruta_json}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Pipeline Avance VI - Descriptores de Forma',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python3 avance_vi_pipeline.py
  python3 avance_vi_pipeline.py --imagenes mi_dataset
  python3 avance_vi_pipeline.py --imagen foto.jpg
  python3 avance_vi_pipeline.py --resultados mi_carpeta
        """
    )
    
    parser.add_argument(
        '--imagenes', '-i',
        type=str,
        default='images',
        help='Directorio con imágenes de entrada (default: images)'
    )
    
    parser.add_argument(
        '--imagen',
        type=str,
        help='Procesar una sola imagen específica'
    )
    
    parser.add_argument(
        '--resultados', '-r',
        type=str,
        default='resultados/avance_vi',
        help='Directorio para guardar resultados (default: resultados/avance_vi)'
    )
    
    args = parser.parse_args()
    
    # Inicializar pipeline
    pipeline = PipelineAvanceVI(dir_resultados=args.resultados)
    
    # Procesar según argumentos
    if args.imagen:
        # Procesar imagen individual
        print(f"Procesando imagen individual: {args.imagen}")
        imagen = cv2.imread(args.imagen)
        
        if imagen is None:
            print(f"❌ Error: No se pudo cargar la imagen {args.imagen}")
            return 1
        
        resultado = pipeline.procesar_imagen(
            imagen, 
            Path(args.imagen).name,
            guardar_visualizacion=True
        )
        
        if resultado:
            print("\n✓ Imagen procesada exitosamente")
            return 0
        else:
            print("\n❌ Error procesando la imagen")
            return 1
    else:
        # Procesar dataset completo
        resultados = pipeline.procesar_dataset(args.imagenes)
        
        if len(resultados) > 0:
            print("\n✓ Dataset procesado exitosamente")
            return 0
        else:
            print("\n❌ No se procesaron imágenes")
            return 1


if __name__ == '__main__':
    sys.exit(main())
