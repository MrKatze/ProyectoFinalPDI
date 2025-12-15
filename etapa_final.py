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
        self.rasgos_individuales = {} 
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
        print("--- 2. Segmentación Global (Para Recorte) ---")
        imagen_ycc = cv2.cvtColor(self.img_suavizada, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mascara_cruda = cv2.inRange(imagen_ycc, lower, upper)
        self.historial['5_Mascara_Global_Cruda'] = mascara_cruda.copy()
        
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
        
        self.img_recortada = cv2.bitwise_and(self.img_iluminada, self.img_iluminada, mask=self.mascara_global)
        self.historial['7_Rostro_Recortado'] = self.img_recortada.copy()
        
        gris_recorte = cv2.cvtColor(self.img_recortada, cv2.COLOR_BGR2GRAY)
        
        v = np.median(gris_recorte[gris_recorte > 0])
        bordes = cv2.Canny(gris_recorte, int(max(10, 0.5*v)), int(min(255, 1.2*v)))
        kernel_borde = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        borde_mascara = cv2.morphologyEx(self.mascara_global, cv2.MORPH_GRADIENT, kernel_borde)
        bordes[borde_mascara > 0] = 0
        self.historial['8_Canny_Rasgos'] = bordes.copy()
        
        _, mascara_otsu_recorte = cv2.threshold(gris_recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.historial['9_Umbralizacion_Recorte'] = mascara_otsu_recorte.copy()
        
        tamano_B = (3, 3)
        B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tamano_B)
        mascara_mejorada = cv2.morphologyEx(mascara_otsu_recorte, cv2.MORPH_OPEN, B, iterations=1)
        
        self.mascara_descriptores = mascara_mejorada
        self.historial['10_Region_Mejorada_Final'] = self.mascara_descriptores.copy()
        
        return self.mascara_descriptores

    def paso4_extraccion_descriptores(self):
        print("--- 4. Calculando Descriptores Globales ---")
        
        contornos, _ = cv2.findContours(self.mascara_descriptores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contornos: return {}
        
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
        
        mu_drn = np.mean(drn)
        sigma_drn = np.std(drn)
        
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

# --- MÉTODO AUXILIAR PARA MATEMÁTICAS (AGREGAR A LA CLASE) ---
    def _calcular_metricas(self, mask, nombre):
        """Calcula descriptores geométricos sobre una máscara binaria"""
        # Encontrar contornos en la máscara del rasgo
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return None
        
        # Tomamos el contorno más grande (el rasgo en sí)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area == 0: return None

        perimetro = cv2.arcLength(c, True)
        compacidad = (perimetro ** 2) / area
        
        # Momentos de Hu (Invariantes a escala, rotación y traslación) - Muy útil en PDI
        momentos = cv2.moments(c)
        hu = cv2.HuMoments(momentos).flatten()
        
        # Guardamos en un diccionario
        datos = {
            "Área": area,
            "Perímetro": perimetro,
            "Compacidad": compacidad,
            "Hu_1": hu[0], # El primer momento de Hu es el más representativo
            "Hu_2": hu[1]
        }
        return datos

    def paso5_extraccion_rasgos_individuales(self):
        print("--- 5. Extracción Morfológica: TopHat + BlackHat (Con Visualización Interna) ---")
        
        self.rasgos_individuales = {}
        self.metricas_rasgos = {}
        
        # Guardaremos los pasos intermedios del Ojo Izquierdo para graficarlos
        self.debug_morfologia = {} 
        
        vis_zonas = self.img_recortada.copy()
        img = self.img_recortada.copy()
        h_total, w_total = img.shape[:2]

        def segmentar_ojo_morfologico(coords, nombre):
            # 1. Definir ROI
            y1, y2, x1, x2 = coords
            y1, y2 = max(0, y1), min(h_total, y2)
            x1, x2 = max(0, x1), min(w_total, x2)
            
            roi_color = img[y1:y2, x1:x2]
            cv2.rectangle(vis_zonas, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            if roi_color.size == 0: return

            # 2. Escala de Grises
            gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # 3. TRANSFORMACIONES MORFOLÓGICAS
            # Kernel rectangular horizontal para detectar estructuras tipo "ojo"
            kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            
            # A) Top-Hat: Resalta lo brillante pequeño (Esclerótica/Brillos)
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_x)
            
            # B) Black-Hat: Resalta lo oscuro pequeño (Pupila/Pestañas)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_x)
            
            # C) Fusión: Sumamos texturas. La piel lisa se vuelve oscura (0)
            eye_details = cv2.add(tophat, blackhat)
            
            # 4. Umbralización sobre los DETALLES
            _, mask = cv2.threshold(eye_details, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 5. Limpieza
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.erode(mask, None, iterations=1)
            
            # Restricción de zona central (quita ruido de bordes del recuadro)
            h_roi, w_roi = mask.shape
            mask_center = np.zeros_like(mask)
            pad_x = int(w_roi * 0.1)
            pad_y = int(h_roi * 0.2)
            mask_center[pad_y:h_roi-pad_y, pad_x:w_roi-pad_x] = 255
            mask = cv2.bitwise_and(mask, mask_center)

            # --- GUARDAR PASOS INTERMEDIOS (Solo para el Ojo Izquierdo para el reporte) ---
            if nombre == "Ojo_Izq":
                self.debug_morfologia['1_Gray'] = gray
                self.debug_morfologia['2_TopHat'] = tophat
                self.debug_morfologia['3_BlackHat'] = blackhat
                self.debug_morfologia['4_Fusion'] = eye_details
                self.debug_morfologia['5_Mask_Otsu'] = mask

            # 6. Extracción del Contorno y Elipse
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_final = np.zeros_like(gray)
            
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_max) > 50:
                    if len(c_max) >= 5:
                        elipse = cv2.fitEllipse(c_max)
                        cv2.ellipse(mask_final, elipse, 255, -1)
                        # Dibujar elipse en el visualizador global
                        center_g = (int(elipse[0][0]+x1), int(elipse[0][1]+y1))
                        cv2.ellipse(vis_zonas, (center_g, elipse[1], elipse[2]), (0, 0, 255), 2)
                        
                        # Guardar visualización de la elipse sobre la fusión (para entender el ajuste)
                        if nombre == "Ojo_Izq":
                            debug_elipse = cv2.cvtColor(eye_details, cv2.COLOR_GRAY2BGR)
                            cv2.ellipse(debug_elipse, elipse, (0, 255, 0), 1)
                            self.debug_morfologia['6_Ajuste_Geo'] = debug_elipse
                    else:
                        cv2.drawContours(mask_final, [c_max], -1, 255, -1)
            
            # 7. Recorte Final
            res = cv2.bitwise_and(roi_color, roi_color, mask=mask_final)
            puntos = cv2.findNonZero(mask_final)
            if puntos is not None:
                bx, by, bw, bh = cv2.boundingRect(puntos)
                p = 2 
                nx1, nx2 = max(0, bx-p), min(w_roi, bx+bw+p)
                ny1, ny2 = max(0, by-p), min(h_roi, by+bh+p)
                
                self.rasgos_individuales[nombre] = res[ny1:ny2, nx1:nx2]
                
                metricas = self._calcular_metricas(mask_final, nombre)
                if metricas: self.metricas_rasgos[nombre] = metricas

        # Coordenadas
        coord_ojo_izq  = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.12), int(w_total*0.48))
        coord_ojo_der  = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.52), int(w_total*0.88))

        segmentar_ojo_morfologico(coord_ojo_izq, "Ojo_Izq")
        segmentar_ojo_morfologico(coord_ojo_der, "Ojo_Der")

        self.historial['12_Zonas_Busqueda'] = vis_zonas
        return self.metricas_rasgos

    def visualizar_todo_detallado(self):
        plt.figure(figsize=(16, 12))
        plt.suptitle("Análisis Completo + Desglose Morfológico (Ojo Izq)", fontsize=16)
        
        # FILAS 1 y 2: Proceso Global
        items_visuales = [
            ('1_Original', "1. Original"),
            ('2_Rotada', "2. Alineada"),
            ('5_Mascara_Global_Cruda', "3. Máscara Piel"),
            ('6_Mascara_Global_Final', "4. Silueta Final"),
            ('7_Rostro_Recortado', "5. Recorte Color"),
            ('10_Region_Mejorada_Final', "6. Morfología Global"),
            ('12_Zonas_Busqueda', "7. Detección (TopHat+BlackHat)"),
            ('11_Geometria_Descriptor', "8. Descriptores Geo")
        ]
        
        rows = 4 # Aumentamos a 4 filas
        cols = 4
        
        # Graficar Pasos Globales (Slots 1-8)
        for i, (key, titulo) in enumerate(items_visuales):
            plt.subplot(rows, cols, i+1) 
            img = self.historial.get(key)
            if img is not None:
                if len(img.shape) == 2: plt.imshow(img, cmap='gray')
                else: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(titulo, fontsize=9)
            plt.axis('off')
            
        # FILA 3: Resultados Finales (Recortes Limpios) (Slots 9-10)
        lista_rasgos = ['Ojo_Izq', 'Ojo_Der']
        idx_plot = 9
        for rasgo in lista_rasgos:
            if rasgo in self.rasgos_individuales:
                plt.subplot(rows, cols, idx_plot)
                crop = self.rasgos_individuales[rasgo]
                if crop is not None and crop.size > 0:
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    plt.title(f"Resultado: {rasgo}", fontsize=11, fontweight='bold', color='blue')
                plt.axis('off')
                idx_plot += 1
        
        # FILA 4: DESGLOSE "X-RAY" DEL ALGORITMO (Slots 13-16)
        # Aquí mostramos qué pasó internamente en el Ojo Izquierdo
        if hasattr(self, 'debug_morfologia') and self.debug_morfologia:
            debug_items = [
                ('2_TopHat', "A. Top-Hat (Brillos)"),
                ('3_BlackHat', "B. Black-Hat (Oscuros)"),
                ('4_Fusion', "C. Fusión (Textura)"),
                ('6_Ajuste_Geo', "D. Ajuste Elipse")
            ]
            
            idx_debug = 13 # Empezamos en la última fila
            for key, titulo in debug_items:
                plt.subplot(rows, cols, idx_debug)
                img = self.debug_morfologia.get(key)
                if img is not None:
                    if len(img.shape) == 2: 
                        # Usamos 'magma' o 'inferno' para que se noten más los detalles tenues
                        plt.imshow(img, cmap='gray') 
                    else: 
                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(titulo, fontsize=9, color='red')
                plt.axis('off')
                idx_debug += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
                
if __name__ == "__main__":
    archivo = "images/persona2/Car3.jpeg" # <--- TU RUTA
    
    try:
        app = AnalizadorFacialPro(archivo)
        
        # Pasos 1, 2 y 3
        app.paso1_alineacion_y_normalizacion()
        app.paso2_segmentacion_morfologica()
        app.paso3_recorte_y_procesamiento_interno() 
        
        # Paso 4: Descriptores Globales
        print("\n=== DESCRIPTORES GLOBALES ===")
        datos_globales = app.paso4_extraccion_descriptores()
        for k, v in datos_globales.items():
            if k not in ["Firma Radial", "Centroide"]: 
                print(f"{k:<30}: {v:.4f}")

        # Paso 5: Descriptores Individuales (CORREGIDO)
        print("\n=== DESCRIPTORES POR RASGO INDIVIDUAL ===")
        # Llamamos a la función UNA sola vez
        datos_rasgos = app.paso5_extraccion_rasgos_individuales()
        
        # Iteramos correctamente el diccionario anidado
        for rasgo, metricas in datos_rasgos.items():
            print(f"\n>> {rasgo}:")
            for nombre_metrica, valor in metricas.items():
                print(f"   {nombre_metrica:<15}: {valor:.4f}")
            
        # Visualización
        app.visualizar_todo_detallado()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()