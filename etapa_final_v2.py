import cv2
import numpy as np
import matplotlib.pyplot as plt

class AnalizadorFacialPro:
    
    def __init__(self, ruta_imagen: str):
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        
        # Clasificadores en cascada para la alineación inicial
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self.historial = {} 
        self.descriptores = {}
        self.rasgos_individuales = {} 
        self.mascara_global = None        
        self.mascara_descriptores = None

    # --- PASOS PREVIOS (1-4) MANTENIDOS ---
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

                factor_zoom = 3.2 # Ajuste leve al zoom para no cortar barbilla
                ancho_crop = int(dist_ojos * factor_zoom)
                alto_crop = int(ancho_crop * 1.35)
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
        
        _, mascara_otsu_recorte = cv2.threshold(gris_recorte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.mascara_descriptores = cv2.morphologyEx(mascara_otsu_recorte, cv2.MORPH_OPEN, (3,3), iterations=1)
        self.historial['10_Region_Mejorada_Final'] = self.mascara_descriptores.copy()
        return self.mascara_descriptores

    def paso4_extraccion_descriptores(self):
        print("--- 4. Calculando Descriptores Globales ---")
        contornos, _ = cv2.findContours(self.mascara_descriptores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos: return {}
        cnt = max(contornos, key=cv2.contourArea)
        
        M = cv2.moments(cnt)
        if M['m00'] == 0: return {}
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        
        # ... (Cálculo de descriptores estándar mantenido) ...
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        compacidad = (perimetro ** 2) / area if area > 0 else 0
        
        img_geo = self.img_recortada.copy()
        cv2.drawContours(img_geo, [cnt], -1, (0, 255, 255), 2)
        cv2.circle(img_geo, (cx, cy), 5, (0, 0, 255), -1)
        self.historial['11_Geometria_Descriptor'] = img_geo
        
        self.descriptores = {"Compacidad": compacidad, "Centroide": (cx, cy)}
        return self.descriptores

    # --- MÉTODO AUXILIAR PARA MATEMÁTICAS ---
    def _calcular_metricas(self, mask, nombre):
        """Calcula descriptores geométricos sobre una máscara binaria"""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 5: return None # Ignorar ruido muy pequeño

        perimetro = cv2.arcLength(c, True)
        compacidad = (perimetro ** 2) / area if area > 0 else 0
        momentos = cv2.moments(c)
        hu = cv2.HuMoments(momentos).flatten()
        
        return {
            "Área": area,
            "Perímetro": perimetro,
            "Compacidad": compacidad,
            "Hu_1": hu[0],
            "Hu_2": hu[1]
        }

    # --- PASO 5: IMPLEMENTACIÓN DE ALGORITMOS AVANZADOS ---
    def paso5_extraccion_rasgos_avanzada(self):
        print("--- 5. Extracción Avanzada (EyeMap + CIELAB + Canny) ---")
        
        self.rasgos_individuales = {}
        self.metricas_rasgos = {}
        vis_zonas = self.img_recortada.copy()
        img = self.img_recortada.copy()
        h, w = img.shape[:2]

        # 1. ALGORITMO EYEMAP (Para Ojos)
        def segmentar_ojos_eyemap(coords, nombre):
            y1, y2, x1, x2 = coords
            roi = img[y1:y2, x1:x2]
            cv2.rectangle(vis_zonas, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            if roi.size == 0: return

            # Conversión a YCrCb
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            y_ch, cr, cb = cv2.split(ycrcb)
            
            # Convertir a float para cálculos matemáticos
            cr = cr.astype(np.float32)
            cb = cb.astype(np.float32)
            y_ch = y_ch.astype(np.float32)

            # --- EyeMapC (Crominancia) ---
            # Fórmula: 1/3 * (Cb^2 + (255-Cr)^2 + (Cb/Cr))
            # Ojo: Cb es alto, Cr es bajo en ojos.
            # Normalizamos
            term1 = cv2.pow(cb, 2)
            term2 = cv2.pow(255 - cr, 2)
            term3 = cv2.divide(cb, cr + 1.0) # +1 para evitar div/0
            
            eyeMapC = (term1 + term2 + term3) / 3.0
            # Normalizar a 0-255
            cv2.normalize(eyeMapC, eyeMapC, 0, 255, cv2.NORM_MINMAX)
            
            # --- EyeMapL (Luminancia) ---
            # Los ojos son oscuros. Usamos erosión para resaltar oscuridad.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Dilatación de la imagen original (iluminar)
            dilated = cv2.dilate(y_ch, kernel)
            # EyeMapL = Dilatada / Original (En zonas oscuras, la división es alta)
            eyeMapL = cv2.divide(dilated, y_ch + 1.0)
            cv2.normalize(eyeMapL, eyeMapL, 0, 255, cv2.NORM_MINMAX)

            # --- Fusión ---
            eyeMap = cv2.multiply(eyeMapC, eyeMapL)
            eyeMap = eyeMap.astype(np.uint8)
            
            # Umbralización final sobre el mapa de probabilidad
            # Usamos Otsu sobre el mapa de características
            _, mask = cv2.threshold(eyeMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Limpieza
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1) # Recuperar volumen
            
            # Guardar
            res = cv2.bitwise_and(roi, roi, mask=mask)
            metricas = self._calcular_metricas(mask, nombre)
            if metricas: self.metricas_rasgos[nombre] = metricas
            self.rasgos_individuales[nombre] = res

        # 2. ALGORITMO CIELAB CANAL A (Para Boca)
        def segmentar_boca_lab(coords):
            y1, y2, x1, x2 = coords
            roi = img[y1:y2, x1:x2]
            cv2.rectangle(vis_zonas, (x1, y1), (x2, y2), (0, 0, 255), 1)
            if roi.size == 0: return
            
            # Convertir a CIELAB
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(lab)
            
            # Canal 'a': Verde a Rojo/Magenta. Los labios tienen 'a' muy alto.
            # Normalizamos el canal 'a' para expandir el contraste
            a_norm = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
            
            # Aplicar umbralización Otsu sobre el canal 'a' intensificado
            # Esto separa automáticamente los tonos rojizos fuertes (labios) de la piel
            _, mask = cv2.threshold(a_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # La máscara suele incluir a veces la barbilla o nariz si están rojas.
            # Nos quedamos con el contorno más central/grande.
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_final = np.zeros_like(mask)
            
            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                cv2.drawContours(mask_final, [c_max], -1, 255, -1)
                
            res = cv2.bitwise_and(roi, roi, mask=mask_final)
            metricas = self._calcular_metricas(mask_final, "Boca")
            if metricas: self.metricas_rasgos["Boca"] = metricas
            self.rasgos_individuales["Boca"] = res

        # 3. ALGORITMO HÍBRIDO CANNY (Para Nariz)
        def segmentar_nariz_canny(coords, nombre):
            y1, y2, x1, x2 = coords
            roi = img[y1:y2, x1:x2]
            cv2.rectangle(vis_zonas, (x1, y1), (x2, y2), (255, 0, 0), 1)
            if roi.size == 0: return

            # A) Detectar orificios (Anclas oscuras)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask_holes = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
            
            # B) Detectar bordes suaves (Aletas de nariz) usando Canny
            # Usamos umbrales bajos para captar sombras sutiles
            edges = cv2.Canny(gray, 50, 120)
            
            # C) Fusión: Unir orificios con bordes
            combined = cv2.bitwise_or(mask_holes, edges)
            
            # D) Operación de Cierre (Closing) para conectar los trazos
            # Esto crea una "mancha" que representa la estructura nasal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask_final = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Rellenar agujeros internos
            cnts, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_filled = np.zeros_like(mask_final)
            for c in cnts:
                if cv2.contourArea(c) > 15: # Filtro de ruido
                    cv2.drawContours(mask_filled, [c], -1, 255, -1)

            res = cv2.bitwise_and(roi, roi, mask=mask_filled)
            metricas = self._calcular_metricas(mask_filled, nombre)
            if metricas: self.metricas_rasgos[nombre] = metricas
            self.rasgos_individuales[nombre] = res

        # Coordenadas ROIs
        coord_ceja_izq = (int(h*0.15), int(h*0.35), int(w*0.05), int(w*0.48))
        coord_ceja_der = (int(h*0.15), int(h*0.35), int(w*0.52), int(w*0.95))
        coord_ojo_izq  = (int(h*0.32), int(h*0.50), int(w*0.12), int(w*0.46))
        coord_ojo_der  = (int(h*0.32), int(h*0.50), int(w*0.54), int(w*0.88))
        coord_nariz = (int(h*0.45), int(h*0.68), int(w*0.32), int(w*0.68))
        coord_boca = (int(h*0.68), int(h*0.88), int(w*0.25), int(w*0.75))

        # Ejecución de algoritmos especializados
        segmentar_ojos_eyemap(coord_ojo_izq, "Ojo_Izq")
        segmentar_ojos_eyemap(coord_ojo_der, "Ojo_Der")
        segmentar_nariz_canny(coord_nariz, "Nariz")
        segmentar_boca_lab(coord_boca)

        self.historial['12_Zonas_Busqueda'] = vis_zonas
        return self.metricas_rasgos

    def visualizar_todo_detallado(self):
        plt.figure(figsize=(16, 12))
        plt.suptitle("Análisis Avanzado: EyeMap + CIELAB + Canny", fontsize=16)
        
        items_visuales = [
            ('1_Original', "1. Original"), ('2_Rotada', "2. Alineada"),
            ('5_Mascara_Global_Cruda', "3. Máscara Cruda"), ('6_Mascara_Global_Final', "4. Silueta Final"),
            ('7_Rostro_Recortado', "5. Recorte Color"), ('10_Region_Mejorada_Final', "6. Morfología Global"),
            ('12_Zonas_Busqueda', "7. Zonas de Búsqueda"), ('11_Geometria_Descriptor', "8. Descriptores Geo")
        ]
        
        rows, cols = 4, 4
        for i, (key, titulo) in enumerate(items_visuales):
            plt.subplot(rows, cols, i+1) 
            img = self.historial.get(key)
            if img is not None:
                if len(img.shape) == 2: plt.imshow(img, cmap='gray')
                else: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(titulo, fontsize=10); plt.axis('off')
            
        lista_rasgos = ['Ojo_Izq', 'Ojo_Der', 'Nariz', 'Boca']
        idx_plot = 9 
        for rasgo in lista_rasgos:
            if rasgo in self.rasgos_individuales:
                plt.subplot(rows, cols, idx_plot)
                crop = self.rasgos_individuales[rasgo]
                if crop is not None and crop.size > 0:
                    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    plt.title(f"{rasgo} (Seg. Avanzada)", fontsize=10)
                plt.axis('off')
                idx_plot += 1
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    # CAMBIA ESTA RUTA POR TU IMAGEN
    archivo = "images/persona2/Car3.jpeg" 
    
    try:
        app = AnalizadorFacialPro(archivo)
        app.paso1_alineacion_y_normalizacion()
        app.paso2_segmentacion_morfologica()
        app.paso3_recorte_y_procesamiento_interno() 
        app.paso4_extraccion_descriptores()

        print("\n=== DESCRIPTORES POR RASGO INDIVIDUAL (AVANZADO) ===")
        datos_rasgos = app.paso5_extraccion_rasgos_avanzada()
        
        for rasgo, metricas in datos_rasgos.items():
            print(f"\n>> {rasgo}:")
            for nombre_metrica, valor in metricas.items():
                print(f"   {nombre_metrica:<15}: {valor:.4f}")
            
        app.visualizar_todo_detallado()
        
    except Exception as e:
        print(f"Error Crítico: {e}")
        import traceback
        traceback.print_exc()