import cv2
import numpy as np
import matplotlib.pyplot as plt

class AnalizadorFacialPro:
    
    def __init__(self, ruta_imagen: str):
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Cargar clasificadores
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Variables de estado
        self.img_alineada = None     # Imagen rotada, con zoom y normalizada
        self.img_iluminada = None    # Imagen con corrección de luz (CLAHE)
        self.img_suavizada = None    # Imagen borrosa para máscara
        self.mascara = None
        self.img_recortada = None
        self.bordes_canny = None
        self.descriptores = {}

    def paso1_alineacion_y_normalizacion(self):
        """
        1. Detección de ojos y cálculo de ángulo.
        2. Rotación (Alineación Geométrica).
        3. ZOOM INTELIGENTE (Recorte centrado en ojos).
        4. Corrección de iluminación (Normalización Fotométrica).
        """
        print("--- 1. Realizando Alineación, Zoom y Normalización ---")
        
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Imagen base (si no detecta rostro, usa la original)
        img_procesada = self.img_original.copy()
        
        if len(rostros) > 0:
            (x, y, w, h) = rostros[0]
            roi_gray = gray[y:y+h, x:x+w]
            ojos = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(ojos) >= 2:
                # Ordenar ojos por posición X
                ojos = sorted(ojos, key=lambda e: e[0])
                ojo_izq = ojos[0]
                ojo_der = ojos[-1]
                
                # Coordenadas globales de los centros de los ojos
                c_izq = (x + ojo_izq[0] + ojo_izq[2]//2, y + ojo_izq[1] + ojo_izq[3]//2)
                c_der = (x + ojo_der[0] + ojo_der[2]//2, y + ojo_der[1] + ojo_der[3]//2)
                
                # 1. Calcular Ángulo
                dy = c_der[1] - c_izq[1]
                dx = c_der[0] - c_izq[0]
                angulo = np.degrees(np.arctan2(dy, dx))
                
                # 2. Calcular Centro entre ojos (Pivote de rotación) - DEBE SER FLOAT
                centro_ojos = ((c_izq[0] + c_der[0]) / 2.0, (c_izq[1] + c_der[1]) / 2.0)
                dist_ojos = np.sqrt(dx**2 + dy**2)
                
                # 3. Rotar imagen
                M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
                h_img, w_img = self.img_original.shape[:2]
                img_rotada = cv2.warpAffine(self.img_original, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                
                print(f"   -> Rotación: {angulo:.2f}°")
                
                # 4. APLICAR ZOOM (Recorte centrado en los ojos)
                # Definimos el tamaño del recorte en función de la distancia de los ojos
                # Factores empíricos: El ancho de la foto será 3.5 veces la distancia entre ojos
                factor_zoom = 3.5 
                ancho_crop = int(dist_ojos * factor_zoom)
                alto_crop = int(ancho_crop * 1.3) # Proporción 4:5 aprox (vertical)
                
                # Calcular coordenadas de recorte centradas en 'centro_ojos'
                # Ajustamos 'start_y' para que los ojos queden en el tercio superior, no en el centro absoluto
                start_x = max(0, int(centro_ojos[0] - ancho_crop // 2))
                start_y = max(0, int(centro_ojos[1] - alto_crop * 0.40)) 
                end_x = min(w_img, start_x + ancho_crop)
                end_y = min(h_img, start_y + alto_crop)
                
                # Validar que el recorte tenga tamaño positivo
                if (end_x - start_x) > 50 and (end_y - start_y) > 50:
                    img_crop = img_rotada[start_y:end_y, start_x:end_x]
                    # 5. Redimensionar a tamaño estándar (Normalización de Escala)
                    img_procesada = cv2.resize(img_crop, (400, 520), interpolation=cv2.INTER_CUBIC)
                    print("   -> Zoom y centrado aplicado exitosamente.")
                else:
                    print("   -> El recorte salía de los bordes, se omitió el zoom.")
                    img_procesada = img_rotada
            else:
                print("   -> No se detectaron ambos ojos, se omite rotación/zoom.")
        
        self.img_alineada = img_procesada

        # B. NORMALIZACIÓN FOTOMÉTRICA (Mejora de luz - CLAHE)
        img_yuv = cv2.cvtColor(self.img_alineada, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        
        self.img_iluminada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        
        # C. Generar versión suavizada para la máscara
        self.img_suavizada = cv2.GaussianBlur(self.img_iluminada, (5, 5), 0)
        
        return self.img_iluminada

    def paso2_segmentacion_morfologica(self):
        print("--- 2. Segmentando Rostro (Morfología) ---")
        imagen_ycc = cv2.cvtColor(self.img_suavizada, cv2.COLOR_BGR2YCrCb)
        
        # Rango de piel 
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mascara = cv2.inRange(imagen_ycc, lower, upper)
        
        # Morfología
        kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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
        print("--- 3. Recorte y Detección de Rasgos (Canny) ---")
        self.img_recortada = cv2.bitwise_and(self.img_iluminada, self.img_iluminada, mask=self.mascara)
        
        gris_recorte = cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2GRAY)
        
        # Canny Adaptativo
        v = np.median(gris_recorte[gris_recorte > 0])
        lower = int(max(10, 0.5 * v))
        upper = int(min(255, 1.2 * v))
        
        bordes = cv2.Canny(gris_recorte, lower, upper)
        
        # Limpiar borde externo
        kernel_borde = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        borde_mascara = cv2.morphologyEx(self.mascara, cv2.MORPH_GRADIENT, kernel_borde)
        bordes[borde_mascara > 0] = 0
        
        self.bordes_canny = bordes
        return self.img_recortada, self.bordes_canny

    def paso4_extraccion_descriptores(self):
        print("--- 4. Calculando Descriptores (Según PDF PDI) ---")
        # Usamos la máscara obtenida en el Paso 2 (Mejorada morfológicamente)
        contornos, _ = cv2.findContours(self.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos: return {}
        
        cnt = contornos[0]
        
        # --- 1. COMPACIDAD ---
        # Definición estándar: P^2 / A (Valor mínimo 4pi para círculos)
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if area > 0:
            compacidad = (perimetro ** 2) / area 
        else:
            compacidad = 0
            
        # --- 2. DISTANCIA RADIAL NORMALIZADA (DRN) ---
        # Calcular Centroide
        M = cv2.moments(cnt)
        if M['m00'] == 0: return {}
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        centroide = (cx, cy)
        
        # Obtener vector de distancias (d)
        puntos = cnt[:, 0, :] # Coordenadas (x,y) del contorno
        distancias = np.sqrt(((puntos[:, 0] - cx)**2 + (puntos[:, 1] - cy)**2))
        
        # Normalizar dividiendo por la distancia máxima
        max_dist = np.max(distancias)
        drn = distancias / max_dist if max_dist > 0 else distancias
        
        # --- ESTADÍSTICOS DE DRN (Según PDF PDI-2026) ---
        N = len(drn)
        
        # A. Media
        mu_drn = np.mean(drn)
        
        # B. Desviación Estándar
        sigma_drn = np.std(drn)
        
        # C. Cruces por Cero (Zero Crossings)
        # Número de veces que la señal cruza su promedio
        senal_centrada = drn - mu_drn
        cruces_cero = 0
        for i in range(N - 1):
            if (senal_centrada[i] * senal_centrada[i+1]) < 0:
                cruces_cero += 1
        
        # D. Índice de Área (Fórmula PDF Pág. 19)
        # Suma de diferencias positivas respecto a la media, normalizado por N*Media
        suma_area = np.sum(drn[drn > mu_drn] - mu_drn)
        if N * mu_drn > 0:
            indice_area = suma_area / (N * mu_drn)
        else:
            indice_area = 0
            
        # E. Índice de Rugosidad (Fórmula PDF Pág. 19)
        # Suma de diferencias entre puntos adyacentes, normalizado por N*Media
        # Se asume valor absoluto para medir la variación real
        diffs = np.abs(np.diff(drn)) 
        suma_rugosidad = np.sum(diffs)
        
        # El PDF divide por (N * Media) o ((N-1) * Media)
        if N * mu_drn > 0:
            indice_rugosidad = suma_rugosidad / (N * mu_drn)
        else:
            indice_rugosidad = 0
        
        # Guardar resultados
        self.descriptores = {
            "Compacidad (P^2/A)": compacidad,
            "Distancia Radial Media": mu_drn,
            "Desv. Estándar Radial": sigma_drn,
            "Cruces por Cero": cruces_cero,
            "Índice de Área": indice_area,
            "Índice de Rugosidad": indice_rugosidad,
            "Centroide": centroide,
            "Firma Radial": drn
        }
        return self.descriptores
    
    def visualizar(self):
        plt.figure(figsize=(18, 10))
        
        titulos = ["1. Original", "2. Zoom + Alineada + CLAHE", "3. Máscara (Silueta)", 
                   "4. Recorte Final", "5. Rasgos (Canny)", "6. Firma Radial"]
        
        imgs = [
            cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(self.img_iluminada, cv2.COLOR_BGR2RGB), 
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

        # Gráfica Radial
        plt.subplot(2, 3, 6)
        if 'Firma Radial' in self.descriptores:
            y = self.descriptores['Firma Radial']
            x = np.arange(len(y))
            plt.plot(x, y, label='Distancia')
            plt.plot([0, len(y)], [self.descriptores['Distancia Radial Media']]*2, 'r--', label='Media')
            plt.title("Firma Radial")
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.show()

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    archivo = "images/persona2/Car3.jpeg"  # ¡Verifica tu ruta!
    
    try:
        analizador = AnalizadorFacialPro(archivo)
        
        analizador.paso1_alineacion_y_normalizacion()
        analizador.paso2_segmentacion_morfologica()
        analizador.paso3_recorte_y_canny()
        desc = analizador.paso4_extraccion_descriptores()
        
        print("\n=== RESULTADOS DESCRIPTORES ===")
        for k, v in desc.items():
            if k not in ["Firma Radial", "Centroide"]:
                print(f"{k}: {v:.4f}")
        
        analizador.visualizar()
        print("\nProceso finalizado con éxito.")
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")