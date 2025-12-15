import cv2
import numpy as np
import matplotlib.pyplot as plt

class AnalizadorFacialPro:
    
    def __init__(self, ruta_imagen: str):
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self.historial = {} 
        self.descriptores = {}
        self.mascara_global = None        
        self.mascara_descriptores = None

    def paso1_alineacion_y_normalizacion(self):
        print("--- 1. Alineación, Zoom y Normalización ---")
        self.historial['1_Original'] = self.img_original.copy()
        
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        img_procesada = self.img_original.copy()
        
        if len(rostros) > 0:
            (x, y, w, h) = rostros[0]
            roi_gray = gray[y:y+h, x:x+w]
            ojos = self.eye_cascade.detectMultiScale(roi_gray)
            if len(ojos) >= 2:
                ojos = sorted(ojos, key=lambda e: e[0])
                c_izq = (x + ojos[0][0] + ojos[0][2]//2, y + ojos[0][1] + ojos[0][3]//2)
                c_der = (x + ojos[-1][0] + ojos[-1][2]//2, y + ojos[-1][1] + ojos[-1][3]//2)
                dy, dx = c_der[1] - c_izq[1], c_der[0] - c_izq[0]
                angulo = np.degrees(np.arctan2(dy, dx))
                dist_ojos = np.sqrt(dx**2 + dy**2)
                centro_ojos = ((c_izq[0] + c_der[0]) / 2.0, (c_izq[1] + c_der[1]) / 2.0)
                
                M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
                h_img, w_img = self.img_original.shape[:2]
                img_rotada = cv2.warpAffine(self.img_original, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                self.historial['2_Rotada'] = img_rotada.copy()

                factor_zoom = 3.5 
                ancho_crop = int(dist_ojos * factor_zoom)
                alto_crop = int(ancho_crop * 1.3)
                start_x = max(0, int(centro_ojos[0] - ancho_crop // 2))
                start_y = max(0, int(centro_ojos[1] - alto_crop * 0.40))
                end_x, end_y = min(w_img, start_x + ancho_crop), min(h_img, start_y + alto_crop)
                
                if (end_x - start_x) > 50:
                    img_crop = img_rotada[start_y:end_y, start_x:end_x]
                    img_procesada = cv2.resize(img_crop, (400, 520), interpolation=cv2.INTER_CUBIC)
                else:
                    img_procesada = img_rotada
            else:
                self.historial['2_Rotada'] = img_procesada
        else:
             self.historial['2_Rotada'] = img_procesada

        self.historial['3_Zoom'] = img_procesada.copy()

        img_yuv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        self.img_iluminada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        
        self.historial['4_CLAHE_Normalizada'] = self.img_iluminada.copy()
        self.img_suavizada = cv2.GaussianBlur(self.img_iluminada, (5, 5), 0)
        
        return self.img_iluminada

    def paso2_segmentacion_morfologica(self):
        """Genera la máscara GLOBAL solo para el RECORTE"""
        print("--- 2. Segmentación Global (Para Recorte) ---")
        imagen_ycc = cv2.cvtColor(self.img_suavizada, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mascara_cruda = cv2.inRange(imagen_ycc, lower, upper)
        self.historial['5_Mascara_Global_Cruda'] = mascara_cruda.copy()
        
        # Aquí mantenemos el proceso estándar para asegurar un buen recorte
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mascara_limpia = cv2.morphologyEx(mascara_cruda, cv2.MORPH_CLOSE, kernel, iterations=3)
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel, iterations=2)
        
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mascara_limpia, connectivity=8)
        if num > 1:
            mayor_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.mascara_global = np.zeros_like(mascara_limpia)
            self.mascara_global[labels == mayor_label] = 255
        else:
            self.mascara_global = mascara_limpia
            
        self.historial['6_Mascara_Global_Final'] = self.mascara_global.copy()
        return self.mascara_global

    def paso3_recorte_y_procesamiento_interno(self):
        print("--- 3. Recorte y Mejora Morfológica Fina ---")
        
        # 1. RECORTE
        self.img_recortada = cv2.bitwise_and(self.img_iluminada, self.img_iluminada, mask=self.mascara_global)
        self.historial['7_Rostro_Recortado'] = self.img_recortada.copy()
        
        # Gris para procesar
        gris_recorte = cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2GRAY)
        
        # --- CANNY (Visual) ---
        v = np.median(gris_recorte[gris_recorte > 0])
        bordes = cv2.Canny(gris_recorte, int(max(10, 0.5*v)), int(min(255, 1.2*v)))
        kernel_borde = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        borde_mascara = cv2.morphologyEx(self.mascara_global, cv2.MORPH_GRADIENT, kernel_borde)
        bordes[borde_mascara > 0] = 0
        self.historial['8_Canny_Rasgos'] = bordes.copy()
        
        # --- UMBRALIZACIÓN ESTÁNDAR (Rostro Blanco) ---
        # Volvemos al código original: Otsu normal
        _, mascara_otsu_recorte = cv2.threshold(gris_recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.historial['9_Umbralizacion_Recorte'] = mascara_otsu_recorte.copy()
        
        # --- EL CAMBIO MORFOLÓGICO SOLICITADO ---
        # Definimos explícitamente B como ELIPSE (Disco) de tamaño PEQUEÑO (3,3)
        # Esto cumple con tu PDF y con "no perder detalles"
        tamano_B = (3, 3)
        B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano_B)
        
        # Aplicamos SOLO APERTURA (Open)
        # ALERTA: No aplicamos 'Cierre' agresivo. Así, los ojos y boca (que son huecos negros)
        # se mantendrán visibles y no se taparán.
        mascara_mejorada = cv2.morphologyEx(mascara_otsu_recorte, cv2.MORPH_OPEN, B, iterations=1)
        
        # Guardamos
        self.mascara_descriptores = mascara_mejorada
        self.historial['10_Region_Mejorada_Final'] = self.mascara_descriptores.copy()
        
        return self.mascara_descriptores

    def paso4_extraccion_descriptores(self):
        print("--- 4. Calculando Descriptores ---")
        
        # Usamos la máscara refinada (Paso 3)
        contornos, _ = cv2.findContours(self.mascara_descriptores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contornos: return {}
        
        # Tomamos el contorno más grande (la cara)
        cnt = max(contornos, key=cv2.contourArea)
        
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        compacidad = (perimetro ** 2) / area if area > 0 else 0
        
        M = cv2.moments(cnt)
        if M['m00'] == 0: return {}
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        
        puntos = cnt[:, 0, :] 
        distancias = np.sqrt(((puntos[:, 0] - cx)**2 + (puntos[:, 1] - cy)**2))
        max_dist = np.max(distancias)
        drn = distancias / max_dist if max_dist > 0 else distancias
        
        # Estadísticos
        mu_drn = np.mean(drn)
        sigma_drn = np.std(drn)
        
        # Cruces por cero
        senal_centrada = drn - mu_drn
        cruces_cero = 0
        for i in range(len(drn) - 1):
            if (senal_centrada[i] * senal_centrada[i+1]) < 0:
                cruces_cero += 1

        suma_area = np.sum(drn[drn > mu_drn] - mu_drn)
        indice_area = suma_area / (len(drn) * mu_drn) if len(drn) * mu_drn > 0 else 0
            
        diffs = np.abs(np.diff(drn)) 
        suma_rugosidad = np.sum(diffs)
        indice_rugosidad = suma_rugosidad / (len(drn) * mu_drn) if len(drn) * mu_drn > 0 else 0
        
        self.descriptores = {
            "Compacidad (P^2/A)": compacidad,
            "Distancia Radial Media": mu_drn,
            "Desv. Estándar Radial": sigma_drn,
            "Cruces por Cero": cruces_cero,
            "Índice de Área": indice_area,
            "Índice de Rugosidad": indice_rugosidad,
            "Centroide": (cx, cy),
            "Firma Radial": drn
        }
        
        img_geo = self.img_recortada.copy()
        cv2.drawContours(img_geo, [cnt], -1, (0, 255, 255), 2)
        cv2.circle(img_geo, (cx, cy), 5, (0, 0, 255), -1)
        self.historial['11_Geometria_Descriptor'] = img_geo
        
        return self.descriptores

    def visualizar_proceso_detallado(self):
        plt.figure(figsize=(20, 12))
        plt.suptitle("Proceso Final: Base Otsu + Morfología Detallada", fontsize=16)
        
        claves = [
            '1_Original', '2_Rotada', '3_Zoom', '4_CLAHE_Normalizada',
            '5_Mascara_Global_Cruda', '6_Mascara_Global_Final', '7_Rostro_Recortado', '8_Canny_Rasgos',
            '9_Umbralizacion_Recorte', '10_Region_Mejorada_Final', '11_Geometria_Descriptor', 'Firma_Radial'
        ]
        
        titulos = [
            "1. Original", "2. Alineación", "3. Zoom", "4. CLAHE (Luz)",
            "5. Silueta (Global)", "6. Silueta (Final)", "7. Recorte", "8. Canny",
            "9. Otsu (Original)", "10. Morfología (B=3x3, Open)", "11. Geometría", "12. Firma Radial"
        ]
        
        for i, clave in enumerate(claves):
            plt.subplot(3, 4, i+1)
            plt.title(titulos[i])
            
            if clave == 'Firma_Radial':
                if 'Firma Radial' in self.descriptores:
                    plt.plot(self.descriptores['Firma Radial'], color='blue')
                    plt.plot([0, len(self.descriptores['Firma Radial'])], 
                             [self.descriptores['Distancia Radial Media']]*2, 'r--')
                    plt.grid(True, alpha=0.3)
            elif clave in self.historial:
                img = self.historial[clave]
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                else:
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            if clave == 'Firma_Radial': plt.axis('on')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    archivo = "images/persona2/Car3.jpeg" # <--- TU RUTA
    
    try:
        app = AnalizadorFacialPro(archivo)
        app.paso1_alineacion_y_normalizacion()
        app.paso2_segmentacion_morfologica()
        app.paso3_recorte_y_procesamiento_interno() 
        datos = app.paso4_extraccion_descriptores()
        
        print("\n" + "="*40)
        print("     RESULTADOS FINALES")
        print("="*40)
        for k, v in datos.items():
            if k not in ["Firma Radial", "Centroide"]:
                print(f"{k:<30}: {v:.6f}")
        
        app.visualizar_proceso_detallado()
        
    except Exception as e:
        print(f"Error: {e}")