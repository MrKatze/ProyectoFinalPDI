import cv2
import numpy as np
import matplotlib.pyplot as plt

class AnalizadorFacialPDI:
    
    def __init__(self, ruta_imagen: str):
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        self.img_normalizada = None  # Imagen con contraste mejorado (Nítida)
        self.img_suavizada = None    # Imagen borrosa (Para la máscara)
        self.mascara = None
        self.img_recortada = None
        self.bordes_canny = None
        self.descriptores = {}

    def paso1_preprocesamiento(self):
        """
        Normalización Adaptativa (CLAHE) y Filtrado
        """
        # 1. Normalización de histograma ADAPTATIVA (CLAHE)
        # Esto es vital: resalta detalles locales (ojos, nariz) mejor que la ecualización global
        img_yuv = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        
        self.img_normalizada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        
        # 2. Suavizado (Solo para ayudar a la segmentación de la silueta)
        # Usamos (5,5) para quitar ruido de la piel, pero guardamos la versión nítida aparte
        self.img_suavizada = cv2.GaussianBlur(self.img_normalizada, (5, 5), 0)
        
        return self.img_suavizada

    def paso2_segmentacion_y_morfologia(self):
        """
        Generar la silueta (Máscara) usando la imagen suavizada
        """
        # Usamos la imagen SUAVIZADA para la máscara (para evitar bordes rugosos)
        imagen_ycc = cv2.cvtColor(self.img_suavizada, cv2.COLOR_BGR2YCrCb)
        
        # Detección de piel
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mascara = cv2.inRange(imagen_ycc, lower, upper)
        
        # Operadores Morfológicos
        kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_cierre, iterations=2)
        
        kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_apertura, iterations=2)
        
        # Componente principal
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        if num_labels > 1:
            mayor_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mascara_limpia = np.zeros_like(mascara)
            mascara_limpia[labels == mayor_label] = 255
            self.mascara = mascara_limpia
        else:
            self.mascara = mascara
            
        return self.mascara

    def paso3_recorte_y_canny(self):
        """
        Recortar y aplicar Canny sobre la imagen NÍTIDA
        """
        # AQUI ESTA EL CAMBIO CLAVE:
        # Recortamos sobre la imagen NORMALIZADA (Nítida), no la suavizada.
        self.img_recortada = cv2.bitwise_and(self.img_normalizada, self.img_normalizada, mask=self.mascara)
        
        # Convertir a gris para Canny
        gris_recorte = cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2GRAY)
        
        # Ajuste de Umbrales de Canny para captar detalles internos
        # Bajamos un poco el sigma para que sea más sensible a los detalles (0.33 -> 0.5 o manual)
        v = np.median(gris_recorte[gris_recorte > 0])
        
        # Hacemos los umbrales más permisivos para captar nariz y boca
        lower = int(max(10, 0.4 * v))   # Umbral bajo más sensible
        upper = int(min(255, 1.2 * v))  # Umbral alto estándar
        
        bordes = cv2.Canny(gris_recorte, lower, upper)
        
        # Limpiar SOLO el contorno externo exacto de la máscara
        # Dilatamos un poco el borde de la máscara para borrar el recuadro del recorte
        kernel_borde = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        borde_mascara = cv2.morphologyEx(self.mascara, cv2.MORPH_GRADIENT, kernel_borde)
        
        # Borramos los bordes que coinciden con la silueta exterior, dejando los internos
        bordes[borde_mascara > 0] = 0
        
        self.bordes_canny = bordes
        return self.img_recortada, self.bordes_canny

    def paso4_extraccion_descriptores(self):
        # ... (Este código se mantiene igual que la versión anterior) ...
        contornos, _ = cv2.findContours(self.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos: return {}
        cnt = contornos[0]
        
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        compacidad = (perimetro ** 2) / area if area > 0 else 0
        
        M = cv2.moments(cnt)
        if M['m00'] == 0: return {}
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        centroide = (cx, cy)
        
        distancias = np.sqrt(((cnt[:, 0, 0] - cx)**2 + (cnt[:, 0, 1] - cy)**2))
        max_dist = np.max(distancias)
        dist_radial_norm = distancias / max_dist if max_dist > 0 else distancias
        
        rd_media = np.mean(dist_radial_norm)
        rd_std = np.std(dist_radial_norm)
        
        senal_centrada = dist_radial_norm - rd_media
        cruces_cero = np.sum((senal_centrada[:-1] * senal_centrada[1:]) < 0)
        
        area_circulo_max = np.pi * (max_dist ** 2)
        indice_area = area / area_circulo_max if area_circulo_max > 0 else 0
        
        diffs = np.diff(dist_radial_norm)
        indice_rugosidad = np.sqrt(np.mean(diffs**2))
        
        self.descriptores = {
            "Compacidad": compacidad,
            "Distancia Radial Media": rd_media,
            "Desviación Estándar Radial": rd_std,
            "Cruces por Cero": cruces_cero,
            "Índice de Área": indice_area,
            "Índice de Rugosidad": indice_rugosidad,
            "Centroide": centroide,
            "Firma Radial": dist_radial_norm
        }
        return self.descriptores

    def visualizar(self):
        plt.figure(figsize=(15, 10))
        
        titulos = ["Original", "Mejorada (CLAHE)", "Máscara", "Recorte Nítido", "Rasgos Internos", "Firma Radial"]
        imgs = [
            cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(self.img_normalizada, cv2.COLOR_BGR2RGB), # Mostrar la nítida
            self.mascara,
            cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2RGB),
            self.bordes_canny
        ]
        
        for i in range(5):
            plt.subplot(2, 3, i+1)
            if i == 2 or i == 4: 
                plt.imshow(imgs[i], cmap='gray')
            else:
                plt.imshow(imgs[i])
            plt.title(titulos[i])
            plt.axis('off')

        plt.subplot(2, 3, 6)
        if 'Firma Radial' in self.descriptores:
            plt.plot(self.descriptores['Firma Radial'])
            plt.plot([0, len(self.descriptores['Firma Radial'])], 
                     [self.descriptores['Distancia Radial Media']]*2, 'r--', label='Media')
            plt.title("Firma Radial")
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    archivo = "images/persona1/Ang1.jpeg" 
    
    try:
        pdi = AnalizadorFacialPDI(archivo)
        pdi.paso1_preprocesamiento()
        pdi.paso2_segmentacion_y_morfologia()
        pdi.paso3_recorte_y_canny()
        desc = pdi.paso4_extraccion_descriptores()
        
        print("\nRESULTADOS:")
        for k, v in desc.items():
            if k != "Firma Radial" and k != "Centroide":
                print(f"{k}: {v:.4f}")
        
        pdi.visualizar()
        
    except Exception as e:
        print(f"Error: {e}")