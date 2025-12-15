import cv2
import numpy as np
import matplotlib.pyplot as plt

class AnalizadorFacialPro:
    
    def __init__(self, ruta_imagen: str):
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Clasificadores
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Historial de imágenes para visualización
        self.historial = {} 
        self.descriptores = {}
        self.mascara = None

    def paso1_alineacion_y_normalizacion(self):
        """
        Alineación (Geometría) + Zoom + CLAHE (Realce Estadístico)
        """
        print("--- 1. Alineación, Zoom y Normalización ---")
        self.historial['1_Original'] = self.img_original.copy()
        
        gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        rostros = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        img_procesada = self.img_original.copy()
        
        # --- A. ROTACIÓN ---
        if len(rostros) > 0:
            (x, y, w, h) = rostros[0]
            roi_gray = gray[y:y+h, x:x+w]
            ojos = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(ojos) >= 2:
                ojos = sorted(ojos, key=lambda e: e[0])
                c_izq = (x + ojos[0][0] + ojos[0][2]//2, y + ojos[0][1] + ojos[0][3]//2)
                c_der = (x + ojos[-1][0] + ojos[-1][2]//2, y + ojos[-1][1] + ojos[-1][3]//2)
                
                dy = c_der[1] - c_izq[1]
                dx = c_der[0] - c_izq[0]
                angulo = np.degrees(np.arctan2(dy, dx))
                dist_ojos = np.sqrt(dx**2 + dy**2)
                centro_ojos = ((c_izq[0] + c_der[0]) / 2.0, (c_izq[1] + c_der[1]) / 2.0)
                
                M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
                h_img, w_img = self.img_original.shape[:2]
                img_rotada = cv2.warpAffine(self.img_original, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                
                self.historial['2_Rotada'] = img_rotada.copy()

                # --- B. ZOOM ---
                factor_zoom = 3.5 
                ancho_crop = int(dist_ojos * factor_zoom)
                alto_crop = int(ancho_crop * 1.3)
                
                start_x = max(0, int(centro_ojos[0] - ancho_crop // 2))
                start_y = max(0, int(centro_ojos[1] - alto_crop * 0.40))
                end_x = min(w_img, start_x + ancho_crop)
                end_y = min(h_img, start_y + alto_crop)
                
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

        # --- C. CLAHE (FILTRO DE REALCE ESTADÍSTICO) ---
        # Responde a tu duda: Esto mejora contraste localmente sin saturar
        img_yuv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        self.img_iluminada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        
        self.historial['4_CLAHE_Normalizada'] = self.img_iluminada.copy()
        
        # --- D. SUAVIZADO (FILTRO GAUSSIANO) ---
        # Necesario para que la máscara salga limpia en el siguiente paso
        self.img_suavizada = cv2.GaussianBlur(self.img_iluminada, (5, 5), 0)
        
        return self.img_iluminada

    def paso2_segmentacion_morfologica(self):
        """
        Obtención de REGIONES y mejora con MORFOLOGÍA
        """
        print("--- 2. Segmentación Morfológica (Mejora de Regiones) ---")
        imagen_ycc = cv2.cvtColor(self.img_suavizada, cv2.COLOR_BGR2YCrCb)
        
        # 1. Umbralización (Detección Piel)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mascara_cruda = cv2.inRange(imagen_ycc, lower, upper)
        self.historial['5_Mascara_Cruda'] = mascara_cruda.copy()
        
        # 2. OPERADORES MORFOLÓGICOS (Aquí se cumple el punto 1 de tu tarea)
        # Aplicamos morfología a la MÁSCARA para "Mejorar la Región"
        
        # Cierre: Rellenar agujeros dentro del rostro
        kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mascara_cierre = cv2.morphologyEx(mascara_cruda, cv2.MORPH_CLOSE, kernel_cierre, iterations=2)
        self.historial['6_Mascara_Cierre'] = mascara_cierre.copy()
        
        # Apertura: Eliminar ruido externo
        kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_final = cv2.morphologyEx(mascara_cierre, cv2.MORPH_OPEN, kernel_apertura, iterations=2)
        
        # 3. Seleccionar Componente Principal
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_final, connectivity=8)
        if num_labels > 1:
            mayor_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mascara_limpia = np.zeros_like(mascara_final)
            mascara_limpia[labels == mayor_label] = 255
            self.mascara = mascara_limpia
        else:
            self.mascara = mascara_final
            
        self.historial['7_Mascara_Final'] = self.mascara.copy()
        return self.mascara

    def paso3_recorte_y_canny(self):
        print("--- 3. Recorte y Canny ---")
        # Recorte sobre imagen nítida (CLAHE)
        self.img_recortada = cv2.bitwise_and(self.img_iluminada, self.img_iluminada, mask=self.mascara)
        self.historial['8_Rostro_Recortado'] = self.img_recortada.copy()
        
        # Canny (Detección de bordes internos)
        gris_recorte = cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2GRAY)
        v = np.median(gris_recorte[gris_recorte > 0])
        lower = int(max(10, 0.5 * v))
        upper = int(min(255, 1.2 * v))
        
        bordes_raw = cv2.Canny(gris_recorte, lower, upper)
        self.historial['9_Canny_Crudo'] = bordes_raw.copy()
        
        # Limpieza de bordes externos (usando la máscara morfológica)
        kernel_borde = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        borde_mascara = cv2.morphologyEx(self.mascara, cv2.MORPH_GRADIENT, kernel_borde)
        bordes_limpios = bordes_raw.copy()
        bordes_limpios[borde_mascara > 0] = 0
        
        self.bordes_canny = bordes_limpios
        self.historial['10_Canny_Rasgos'] = self.bordes_canny.copy()
        
        return self.img_recortada, self.bordes_canny

    def paso4_extraccion_descriptores(self):
        print("--- 4. Calculando Descriptores ---")
        # Usamos la máscara obtenida en el Paso 2 (Mejorada morfológicamente)
        contornos, _ = cv2.findContours(self.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos: return {}
        
        cnt = contornos[0]
        
        # --- 1. COMPACIDAD ---
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        if area > 0:
            compacidad = (perimetro ** 2) / area 
        else:
            compacidad = 0
            
        # --- 2. DISTANCIA RADIAL NORMALIZADA (DRN) ---
        M = cv2.moments(cnt)
        if M['m00'] == 0: return {}
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        
        puntos = cnt[:, 0, :] 
        distancias = np.sqrt(((puntos[:, 0] - cx)**2 + (puntos[:, 1] - cy)**2))
        max_dist = np.max(distancias)
        drn = distancias / max_dist if max_dist > 0 else distancias
        
        # --- ESTADÍSTICOS DE DRN ---
        N = len(drn)
        mu_drn = np.mean(drn)
        sigma_drn = np.std(drn)
        
        # Cruces por Cero
        senal_centrada = drn - mu_drn
        cruces_cero = 0
        for i in range(N - 1):
            if (senal_centrada[i] * senal_centrada[i+1]) < 0:
                cruces_cero += 1
        
        # Índice de Área
        suma_area = np.sum(drn[drn > mu_drn] - mu_drn)
        if N * mu_drn > 0:
            indice_area = suma_area / (N * mu_drn)
        else:
            indice_area = 0
            
        # Índice de Rugosidad
        diffs = np.abs(np.diff(drn)) 
        suma_rugosidad = np.sum(diffs)
        if N * mu_drn > 0:
            indice_rugosidad = suma_rugosidad / (N * mu_drn)
        else:
            indice_rugosidad = 0
        
        # Guardar en diccionario y devolver
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
        
        # Imagen visual auxiliar para el paso 11
        img_centroide = self.img_recortada.copy()
        cv2.circle(img_centroide, (cx, cy), 5, (0, 255, 0), -1)
        cv2.drawContours(img_centroide, [cnt], -1, (0, 0, 255), 2)
        self.historial['11_Geometria'] = img_centroide
        
        return self.descriptores

    def visualizar_proceso_detallado(self):
        plt.figure(figsize=(20, 12))
        plt.suptitle("Proceso de Segmentación y Descripción Facial - Paso a Paso", fontsize=16)
        
        claves = [
            '1_Original', '2_Rotada', '3_Zoom', '4_CLAHE_Normalizada',
            '5_Mascara_Cruda', '6_Mascara_Cierre', '7_Mascara_Final', '8_Rostro_Recortado',
            '9_Canny_Crudo', '10_Canny_Rasgos', '11_Geometria', 'Firma_Radial'
        ]
        
        titulos = [
            "1. Original", "2. Alineación", "3. Zoom", "4. CLAHE (Realce)",
            "5. Piel (Cruda)", "6. Morfología (Mejora)", "7. Región Final", "8. Recorte",
            "9. Canny (Bordes)", "10. Rasgos Limpios", "11. Centroide/Contorno", "12. Firma Radial"
        ]
        
        for i, clave in enumerate(claves):
            plt.subplot(3, 4, i+1)
            plt.title(titulos[i])
            
            if clave == 'Firma_Radial':
                if 'Firma Radial' in self.descriptores:
                    plt.plot(self.descriptores['Firma Radial'], color='blue', linewidth=1)
                    plt.grid(True, alpha=0.3)
                    plt.xlabel("Perímetro")
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

# --- BLOQUE PRINCIPAL (AQUÍ ESTÁN LOS PRINTS) ---
if __name__ == "__main__":
    archivo = "images/persona3/Kev5.jpeg" # <--- CAMBIA ESTO POR TU RUTA
    
    try:
        app = AnalizadorFacialPro(archivo)
        app.paso1_alineacion_y_normalizacion()
        app.paso2_segmentacion_morfologica()
        app.paso3_recorte_y_canny()
        
        # Guardamos los descriptores en una variable
        datos = app.paso4_extraccion_descriptores()
        
        # --- AQUÍ IMPRIMIMOS LOS DATOS EN LA CONSOLA ---
        print("\n" + "="*40)
        print("     RESULTADOS DE DESCRIPTORES")
        print("="*40)
        
        for k, v in datos.items():
            # Filtramos los datos grandes (arrays) para que no ensucien la consola
            if k not in ["Firma Radial", "Centroide"]:
                print(f"{k:<30}: {v:.6f}")
            elif k == "Centroide":
                 print(f"{k:<30}: {v}")
        
        print("="*40 + "\n")
        
        # Mostramos la gráfica
        app.visualizar_proceso_detallado()
        
    except Exception as e:
        print(f"Error: {e}")