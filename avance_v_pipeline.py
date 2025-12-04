#!/usr/bin/env python3
"""
Pipeline Completo - Avance V
Integra todos los Avances II→III→IV→V

Sistema de detección y segmentación de rasgos faciales sin usar landmarks.

Autores: Equipo PDI
Fecha: Diciembre 2025
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Importar módulos del proyecto
from src.avance_ii.alineacion import AlineadorRostros
from src.avance_ii.filtros import FiltrosImagen
from src.avance_iii.umbralizacion import Umbralizador
from src.avance_iv.deteccion_rasgos import DetectorRegiones
from src.avance_v.deteccion_bordes import DetectorBordes
from src.avance_v.morfologia import OperadoresMorfologicos
from src.avance_v.segmentador_ojos import SegmentadorOjos
from src.avance_v.segmentador_nariz import SegmentadorNariz
from src.avance_v.segmentador_boca import SegmentadorBoca
from src.utils.cargador_imagenes import CargadorImagenes
from src.utils.visualizacion import Visualizador
from src.utils.metricas import CalculadorMetricas, Cronometro


class PipelineAvanceV:
    """
    Pipeline completo que integra todos los avances
    """
    
    def __init__(self, tamano_rostro: Tuple[int, int] = (256, 256)):
        """
        Inicializa el pipeline
        
        Args:
            tamano_rostro: Tamaño para normalizar rostros (ancho, alto)
        """
        self.tamano_rostro = tamano_rostro
        
        # Inicializar componentes
        self.alineador = AlineadorRostros(tamano_salida=tamano_rostro)
        self.segmentador_ojos = SegmentadorOjos()
        self.segmentador_nariz = SegmentadorNariz()
        self.segmentador_boca = SegmentadorBoca()
        
        print("✓ Pipeline inicializado")
    
    def procesar_imagen(self, 
                       imagen: np.ndarray,
                       nombre: str = "imagen") -> Tuple[Dict, bool]:
        """
        Procesa una imagen completa a través de todo el pipeline
        
        Args:
            imagen: Imagen BGR
            nombre: Nombre de la imagen (para logging)
            
        Returns:
            Tupla (resultados_dict, exito)
        """
        resultados = {'nombre': nombre}
        
        print(f"\n{'='*60}")
        print(f"Procesando: {nombre}")
        print(f"{'='*60}")
        
        # Guardar original
        resultados['original'] = imagen.copy()
        
        # ===================================================================
        # AVANCE II: PREPROCESAMIENTO
        # ===================================================================
        print("\n[AVANCE II] Preprocesamiento...")
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen.copy()
        
        # 1. Alineación y normalización
        print("  • Detectando y alineando rostro...")
        rostro_alineado, info_alineacion = self.alineador.alinear_rostro(gris)
        
        if rostro_alineado is None:
            print("  ✗ No se detectó rostro")
            return resultados, False
        
        print(f"  ✓ Rostro detectado y alineado")
        if info_alineacion['ojos_detectados']:
            print(f"    - Ángulo de rotación: {info_alineacion['angulo_rotacion']:.2f}°")
        
        resultados['rostro_detectado'] = gris.copy()
        resultados['rostro_alineado'] = rostro_alineado
        resultados['rostro_normalizado'] = rostro_alineado.copy()
        
        # 2. Aplicar filtros
        print("  • Aplicando filtros...")
        filtros_aplicados = FiltrosImagen.aplicar_todos_filtros(rostro_alineado)
        
        resultados['filtro_mediana'] = filtros_aplicados['mediana']
        resultados['filtro_gaussiano'] = filtros_aplicados['gaussiano']
        resultados['filtro_laplaciano'] = filtros_aplicados['laplaciano']
        resultados['filtro_highboost'] = filtros_aplicados['highboost']
        
        # Usar filtro gaussiano para siguientes etapas
        imagen_filtrada = filtros_aplicados['gaussiano']
        
        print("  ✓ Filtros aplicados")
        
        # ===================================================================
        # AVANCE III: SEGMENTACIÓN FONDO-ROSTRO
        # ===================================================================
        print("\n[AVANCE III] Segmentación fondo-rostro...")
        
        # Umbralización
        umbral_global, _ = Umbralizador.umbral_global(imagen_filtrada, umbral=127)
        umbral_otsu, valor_otsu = Umbralizador.otsu(imagen_filtrada)
        
        print(f"  • Umbral Otsu calculado: {valor_otsu}")
        
        resultados['umbral_global'] = umbral_global
        resultados['umbral_otsu'] = umbral_otsu
        
        # Segmentación de piel (si tenemos imagen en color)
        if len(imagen.shape) == 3:
            # Redimensionar imagen original al tamaño del rostro
            imagen_color_rostro = cv2.resize(imagen, self.tamano_rostro)
            mascara_piel = Umbralizador.segmentar_piel(imagen_color_rostro)
            resultados['mascara_piel'] = mascara_piel
            
            # Combinar con umbralización
            segmentacion_combinada = Umbralizador.combinar_mascaras(
                umbral_otsu, mascara_piel, 'AND'
            )
        else:
            segmentacion_combinada = umbral_otsu
        
        # Limpiar máscara
        segmentacion_final = Umbralizador.limpiar_mascara(segmentacion_combinada)
        resultados['segmentacion_final'] = segmentacion_final
        
        print("  ✓ Segmentación completada")
        
        # ===================================================================
        # AVANCE IV: REGIONES DE RASGOS
        # ===================================================================
        print("\n[AVANCE IV] Identificando regiones de rasgos...")
        
        h, w = rostro_alineado.shape
        regiones = DetectorRegiones.calcular_regiones_estandar(w, h)
        
        print(f"  ✓ Regiones calculadas:")
        print(f"    - Ojos: {regiones['ojos']}")
        print(f"    - Nariz: {regiones['nariz']}")
        print(f"    - Boca: {regiones['boca']}")
        
        # ===================================================================
        # AVANCE V: DETECCIÓN DE BORDES Y SEGMENTACIÓN DE RASGOS
        # ===================================================================
        print("\n[AVANCE V] Segmentación avanzada de rasgos...")
        
        # 1. Detección de bordes
        print("  • Detectando bordes...")
        canny = DetectorBordes.canny(imagen_filtrada, 50, 150)
        marr_hildreth = DetectorBordes.marr_hildreth(imagen_filtrada, sigma=1.0)
        
        resultados['canny'] = canny
        resultados['marr_hildreth'] = marr_hildreth
        print("    ✓ Canny y Marr-Hildreth aplicados")
        
        # 2. Operadores morfológicos
        print("  • Aplicando morfología...")
        kernel = OperadoresMorfologicos.crear_elemento_estructurante('elipse', (5, 5))
        morfologia_apertura = OperadoresMorfologicos.apertura(segmentacion_final, kernel)
        morfologia_cierre = OperadoresMorfologicos.cierre(segmentacion_final, kernel)
        
        resultados['morfologia_apertura'] = morfologia_apertura
        resultados['morfologia_cierre'] = morfologia_cierre
        print("    ✓ Operadores morfológicos aplicados")
        
        # 3. Segmentación de OJOS
        print("  • Segmentando OJOS (3 métodos)...")
        ojos, info_ojos = self.segmentador_ojos.segmentar_multimetodo(rostro_alineado)
        
        if len(ojos) > 0:
            print(f"    ✓ {len(ojos)} ojo(s) detectado(s)")
            resultados['ojos_detectados'] = ojos
        else:
            print("    ✗ No se detectaron ojos")
            resultados['ojos_detectados'] = []
        
        # 4. Segmentación de NARIZ
        print("  • Segmentando NARIZ (2 métodos)...")
        imagen_color_rostro = None
        if len(imagen.shape) == 3:
            imagen_color_rostro = cv2.resize(imagen, self.tamano_rostro)
        
        nariz, info_nariz = self.segmentador_nariz.segmentar_multimetodo(
            rostro_alineado, imagen_color_rostro
        )
        
        if nariz is not None:
            print("    ✓ Nariz detectada")
            resultados['nariz_detectada'] = nariz
        else:
            print("    ✗ No se detectó nariz")
        
        # 5. Segmentación de BOCA
        print("  • Segmentando BOCA (3 métodos)...")
        boca, info_boca = self.segmentador_boca.segmentar_multimetodo(
            rostro_alineado, imagen_color_rostro
        )
        
        if boca is not None:
            print("    ✓ Boca detectada")
            resultados['boca_detectada'] = boca
        else:
            print("    ✗ No se detectó boca")
        
        # ===================================================================
        # RESULTADO FINAL
        # ===================================================================
        print("\n[RESULTADO FINAL] Combinando detecciones...")
        
        resultado_final = self._dibujar_resultado_final(
            rostro_alineado,
            ojos,
            nariz,
            boca
        )
        resultados['resultado_final'] = resultado_final
        
        print("✓ Procesamiento completado")
        
        return resultados, True
    
    def _dibujar_resultado_final(self,
                                 imagen: np.ndarray,
                                 ojos: list,
                                 nariz: Optional[Tuple[int, int, int, int]],
                                 boca: Optional[Tuple[int, int, int, int]]
                                 ) -> np.ndarray:
        """
        Dibuja todas las detecciones en la imagen final
        
        Args:
            imagen: Imagen del rostro
            ojos: Lista de ojos detectados
            nariz: Bounding box de nariz
            boca: Bounding box de boca
            
        Returns:
            Imagen con todas las detecciones
        """
        # Convertir a color
        if len(imagen.shape) == 2:
            resultado = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            resultado = imagen.copy()
        
        # Dibujar ojos (verde)
        for (x, y, w, h) in ojos:
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resultado, "Ojo", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dibujar nariz (azul)
        if nariz is not None:
            x, y, w, h = nariz
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(resultado, "Nariz", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Dibujar boca (rojo)
        if boca is not None:
            x, y, w, h = boca
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(resultado, "Boca", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return resultado
    
    def procesar_dataset(self,
                        directorio_imagenes: str,
                        directorio_salida: str = "resultados/avance_v"):
        """
        Procesa todas las imágenes del dataset
        
        Args:
            directorio_imagenes: Directorio con las imágenes
            directorio_salida: Directorio donde guardar resultados
        """
        print("\n" + "="*60)
        print("PIPELINE AVANCE V - PROCESAMIENTO DE DATASET")
        print("="*60)
        
        # Crear directorio de salida
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Cargar dataset
        print("\nCargando dataset...")
        cargador = CargadorImagenes(directorio_imagenes)
        dataset = cargador.cargar_dataset()
        
        # Estadísticas
        stats = cargador.obtener_estadisticas()
        print(f"\nEstadísticas del dataset:")
        print(f"  • Personas: {stats['num_personas']}")
        print(f"  • Total de imágenes: {stats['total_imagenes']}")
        
        # Métricas
        metricas = CalculadorMetricas()
        cronometro = Cronometro()
        
        # Procesar cada imagen
        print("\n" + "-"*60)
        print("Procesando imágenes...")
        print("-"*60)
        
        for nombre_persona, imagenes in dataset.items():
            print(f"\n### Persona: {nombre_persona} ###")
            
            # Crear subdirectorio para esta persona
            dir_persona = os.path.join(directorio_salida, nombre_persona)
            os.makedirs(dir_persona, exist_ok=True)
            
            for idx, imagen in enumerate(imagenes, 1):
                nombre_img = f"{nombre_persona}_img{idx}"
                
                # Cronometrar
                cronometro.iniciar()
                
                # Procesar
                resultados, exito = self.procesar_imagen(imagen, nombre_img)
                
                # Registrar tiempo
                tiempo = cronometro.detener()
                metricas.registrar_tiempo(tiempo)
                metricas.incrementar_total()
                
                # Registrar detecciones
                metricas.registrar_deteccion('rostro', exito)
                if exito:
                    metricas.registrar_deteccion('ojos', 
                                                len(resultados.get('ojos_detectados', [])) > 0)
                    metricas.registrar_deteccion('nariz',
                                                'nariz_detectada' in resultados)
                    metricas.registrar_deteccion('boca',
                                                'boca_detectada' in resultados)
                
                # Guardar visualización
                if exito:
                    print(f"  • Generando visualización...")
                    fig = Visualizador.crear_panel_20(
                        imagen,
                        resultados,
                        titulo=f"Pipeline Avance V - {nombre_img}"
                    )
                    
                    ruta_salida = os.path.join(dir_persona, f"{nombre_img}_pipeline.png")
                    Visualizador.guardar_figura(fig, ruta_salida)
                
                print(f"  ✓ Tiempo: {tiempo:.2f}s")
        
        # Generar reporte final
        print("\n" + "="*60)
        print("GENERANDO REPORTE FINAL")
        print("="*60)
        
        reporte = metricas.generar_reporte()
        print(reporte)
        
        # Guardar reporte
        ruta_reporte = os.path.join(directorio_salida, "reporte_metricas.txt")
        with open(ruta_reporte, 'w') as f:
            f.write(reporte)
        print(f"\n✓ Reporte guardado en: {ruta_reporte}")
        
        # Exportar métricas a CSV
        ruta_csv = os.path.join(directorio_salida, "metricas.csv")
        metricas.exportar_csv(ruta_csv)
        
        print(f"\n{'='*60}")
        print("PROCESAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"Resultados guardados en: {directorio_salida}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Pipeline Avance V - Segmentación de rasgos faciales sin landmarks"
    )
    parser.add_argument(
        '--imagenes',
        type=str,
        default='images',
        help='Directorio con las imágenes (default: images)'
    )
    parser.add_argument(
        '--salida',
        type=str,
        default='resultados/avance_v',
        help='Directorio de salida (default: resultados/avance_v)'
    )
    parser.add_argument(
        '--imagen',
        type=str,
        help='Procesar una sola imagen en lugar del dataset completo'
    )
    
    args = parser.parse_args()
    
    # Crear pipeline
    pipeline = PipelineAvanceV()
    
    if args.imagen:
        # Procesar una sola imagen
        print(f"Cargando imagen: {args.imagen}")
        imagen = cv2.imread(args.imagen)
        
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen {args.imagen}")
            return
        
        resultados, exito = pipeline.procesar_imagen(imagen, Path(args.imagen).stem)
        
        if exito:
            # Generar visualización
            fig = Visualizador.crear_panel_20(
                imagen,
                resultados,
                titulo=f"Pipeline Avance V - {Path(args.imagen).stem}"
            )
            
            # Guardar
            os.makedirs(args.salida, exist_ok=True)
            ruta_salida = os.path.join(args.salida, f"{Path(args.imagen).stem}_resultado.png")
            Visualizador.guardar_figura(fig, ruta_salida)
            
            print(f"\n✓ Resultado guardado en: {ruta_salida}")
    else:
        # Procesar dataset completo
        pipeline.procesar_dataset(args.imagenes, args.salida)


if __name__ == "__main__":
    main()
