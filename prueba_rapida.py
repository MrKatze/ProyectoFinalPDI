#!/usr/bin/env python3
"""
Script de Prueba Rápida
Prueba el pipeline con una imagen de ejemplo
"""

import cv2
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from avance_v_pipeline import PipelineAvanceV
from src.utils.visualizacion import Visualizador


def main():
    print("="*60)
    print("PRUEBA RÁPIDA - AVANCE V")
    print("="*60)
    
    # Buscar una imagen de prueba
    dir_imagenes = Path("images")
    
    if not dir_imagenes.exists():
        print(f"\n✗ Error: Directorio '{dir_imagenes}' no encontrado")
        print("\nPor favor:")
        print("1. Crea el directorio 'images/'")
        print("2. Organiza tus imágenes en subdirectorios por persona:")
        print("   images/persona1/img1.jpg")
        print("   images/persona2/img1.jpg")
        print("   ...")
        return
    
    # Buscar primera imagen disponible
    imagen_prueba = None
    ruta_imagen = None
    
    for persona_dir in dir_imagenes.iterdir():
        if persona_dir.is_dir():
            for archivo in persona_dir.iterdir():
                if archivo.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    ruta_imagen = archivo
                    imagen_prueba = cv2.imread(str(archivo))
                    if imagen_prueba is not None:
                        break
            if imagen_prueba is not None:
                break
    
    if imagen_prueba is None:
        print("\n✗ No se encontraron imágenes en el directorio 'images/'")
        return
    
    print(f"\n✓ Imagen de prueba encontrada: {ruta_imagen}")
    print(f"  Tamaño: {imagen_prueba.shape}")
    
    # Crear pipeline
    print("\nInicializando pipeline...")
    pipeline = PipelineAvanceV()
    
    # Procesar imagen
    print("\nProcesando imagen...")
    resultados, exito = pipeline.procesar_imagen(imagen_prueba, ruta_imagen.stem)
    
    if not exito:
        print("\n✗ El procesamiento falló (no se detectó rostro)")
        return
    
    # Generar visualización
    print("\nGenerando visualización de 20 paneles...")
    fig = Visualizador.crear_panel_20(
        imagen_prueba,
        resultados,
        titulo=f"Prueba Rápida - {ruta_imagen.stem}"
    )
    
    # Guardar
    ruta_salida = Path("resultados/prueba_rapida")
    ruta_salida.mkdir(parents=True, exist_ok=True)
    
    archivo_salida = ruta_salida / f"{ruta_imagen.stem}_resultado.png"
    Visualizador.guardar_figura(fig, str(archivo_salida))
    
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"✓ Rostro detectado: Sí")
    print(f"✓ Ojos detectados: {len(resultados.get('ojos_detectados', []))}")
    print(f"✓ Nariz detectada: {'Sí' if 'nariz_detectada' in resultados else 'No'}")
    print(f"✓ Boca detectada: {'Sí' if 'boca_detectada' in resultados else 'No'}")
    print(f"\n✓ Resultado guardado en: {archivo_salida}")
    print("\n" + "="*60)
    print("PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*60)
    
    # Mostrar imagen de resultado (opcional)
    print("\nPresiona cualquier tecla para cerrar (se mostrará el resultado final)...")
    cv2.imshow("Resultado Final", resultados['resultado_final'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
