import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

class AnalizadorFacialFinalV6:
    
    def __init__(self, ruta_imagen: str):
        self.ruta = ruta_imagen
        self.nombre_archivo = os.path.basename(ruta_imagen)
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None: raise ValueError(f"No se pudo cargar: {ruta_imagen}")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.snapshots = {} 
        self.descriptores_ojos = {} 

    def paso1_preprocesamiento(self):
        # 1. Original y Detección
        self.snapshots['1_Original'] = self.img_original.copy()
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
                centro_ojos = ((c_izq[0] + c_der[0]) / 2.0, (c_izq[1] + c_der[1]) / 2.0)
                M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
                h_img, w_img = self.img_original.shape[:2]
                img_rotada = cv2.warpAffine(self.img_original, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                
                dist_ojos = np.sqrt(dx**2 + dy**2)
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
        
        self.snapshots['2_Alineada'] = img_procesada.copy()

        # 2. Realce (CLAHE)
        img_yuv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        self.img_iluminada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        self.snapshots['3_Realce_CLAHE'] = self.img_iluminada.copy()
        
        # 3. Suavizado
        self.img_suavizada = cv2.GaussianBlur(self.img_iluminada, (5, 5), 0)
        self.snapshots['4_Suavizado_Gauss'] = self.img_suavizada.copy()
        
        self.img_trabajo = self.img_suavizada 

    def _calcular_firma_radial(self, contour):
        if len(contour) < 10: return None
        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)
        if area == 0: return None
        
        compacidad = (perimetro ** 2) / area
        
        M = cv2.moments(contour)
        if M['m00'] == 0: return None
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
        
        puntos = contour[:, 0, :] 
        distancias = np.sqrt(((puntos[:, 0] - cx)**2 + (puntos[:, 1] - cy)**2))
        max_dist = np.max(distancias)
        if max_dist == 0: return None
        drn = distancias / max_dist 
        
        mu_drn = np.mean(drn)
        sigma_drn = np.std(drn)
        
        senal_centrada = drn - mu_drn
        cruces_cero = 0
        for i in range(len(drn) - 1):
            if (senal_centrada[i] * senal_centrada[i+1]) < 0:
                cruces_cero += 1
                
        suma_area = np.sum(np.abs(drn - mu_drn))
        indice_area = suma_area / len(drn)
        diffs = np.abs(np.diff(drn)) 
        indice_rugosidad = np.sum(diffs) / len(drn)
        
        return {
            "Compacidad": compacidad,
            "Dist. Media": mu_drn,
            "Desv. Est.": sigma_drn,
            "Cruces Cero": cruces_cero,
            "Ind. Area": indice_area,
            "Rugosidad": indice_rugosidad
        }

    def paso_procesamiento_ojos(self):
        img = self.img_trabajo.copy()
        vis_rois = img.copy()
        h_total, w_total = img.shape[:2]

        def procesar_un_ojo(coords, nombre_lado):
            prefix = f"Ojo_{nombre_lado}"
            y1, y2, x1, x2 = coords
            y1, y2 = max(0, y1), min(h_total, y2)
            x1, x2 = max(0, x1), min(w_total, x2)
            
            roi_color = img[y1:y2, x1:x2]
            cv2.rectangle(vis_rois, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if roi_color.size == 0: return

            gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # ---------------------------------------------------------
            # FASE 1: LOCALIZACIÓN DEL ÓVALO (Lógica restaurada de etapa_final.py)
            # ---------------------------------------------------------
            kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            tophat = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel_x)
            blackhat = cv2.morphologyEx(gray_roi, cv2.MORPH_BLACKHAT, kernel_x)
            fusion = cv2.add(tophat, blackhat)
            
            self.snapshots[f'{prefix}_2_TopHat'] = tophat
            self.snapshots[f'{prefix}_3_BlackHat'] = blackhat
            self.snapshots[f'{prefix}_4_Fusion'] = fusion
            
            # Umbralización Otsu (Igual que en tu archivo original)
            _, mask_loc = cv2.threshold(fusion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Limpieza (Igual que en tu archivo original)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask_loc = cv2.morphologyEx(mask_loc, cv2.MORPH_CLOSE, kernel_close)
            mask_loc = cv2.erode(mask_loc, None, iterations=1)
            
            # --- RESTAURACIÓN CLAVE: MÁSCARA DE BORDES (PADDING) ---
            # Esto es lo que hacía que tu código original funcionara bien y centrara el óvalo
            h_roi, w_roi = mask_loc.shape
            mask_center = np.zeros_like(mask_loc)
            pad_x = int(w_roi * 0.1)
            pad_y = int(h_roi * 0.2)
            # Definimos la zona central blanca, bordes negros
            mask_center[pad_y:h_roi-pad_y, pad_x:w_roi-pad_x] = 255
            mask_loc = cv2.bitwise_and(mask_loc, mask_center)
            
            # Detección de Contornos para el óvalo
            cnts, _ = cv2.findContours(mask_loc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask_ovalo_solido = np.zeros_like(gray_roi)
            vis_geo = roi_color.copy()
            hay_ovalo = False

            if cnts:
                # Tomamos el contorno más grande (el rasgo central)
                c_max = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_max) > 30 and len(c_max) >= 5:
                    elipse = cv2.fitEllipse(c_max)
                    cv2.ellipse(mask_ovalo_solido, elipse, 255, -1) # Máscara sólida
                    cv2.ellipse(vis_geo, elipse, (0, 255, 0), 2)    # Dibujo para visualización
                    hay_ovalo = True

            self.snapshots[f'{prefix}_5_Geo_Ovalo'] = vis_geo
            
            if not hay_ovalo: return

            # ---------------------------------------------------------
            # FASE 2: SEGMENTACIÓN DEL OJO (DENTRO del óvalo)
            # ---------------------------------------------------------
            
            # A. RECORTE CON ÓVALO
            # Usamos la máscara del óvalo para dejar solo el ojo, lo demás negro
            gray_recortada_ovalo = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_ovalo_solido)
            self.snapshots[f'{prefix}_6_Crop_Gray'] = gray_recortada_ovalo

            # B. SEGMENTACIÓN OTSU (Dentro del recorte)
            # Otsu solo en píxeles válidos (dentro del óvalo) para ignorar el fondo negro
            pixeles_validos = gray_recortada_ovalo[mask_ovalo_solido > 0]
            thresh_val = 0
            if pixeles_validos.size > 0:
                thresh_val, _ = cv2.threshold(pixeles_validos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            _, mask_segmentada = cv2.threshold(gray_recortada_ovalo, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Limpieza extra: asegurar que nada salga del óvalo
            mask_segmentada = cv2.bitwise_and(mask_segmentada, mask_ovalo_solido)
            self.snapshots[f'{prefix}_7_Seg_Cruda'] = mask_segmentada

            # C. MEJORA MORFOLÓGICA (Suave, para no perder detalles)
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask_mejorada = cv2.morphologyEx(mask_segmentada, cv2.MORPH_OPEN, kernel_clean)
            mask_mejorada = cv2.morphologyEx(mask_mejorada, cv2.MORPH_CLOSE, kernel_clean)
            
            self.snapshots[f'{prefix}_8_Seg_Mejorada'] = mask_mejorada

            # ---------------------------------------------------------
            # FASE 3: DESCRIPTORES
            # ---------------------------------------------------------
            
            # Usamos la máscara morfológica mejorada para los datos
            cnts_final, _ = cv2.findContours(mask_mejorada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.snapshots[f'{prefix}_9_Final_Binaria'] = mask_mejorada

            if cnts_final:
                c_final = max(cnts_final, key=cv2.contourArea)
                desc = self._calcular_firma_radial(c_final)
                if desc: self.descriptores_ojos[prefix] = desc

        # Coordenadas ROIs
        coord_izq = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.12), int(w_total*0.48))
        coord_der = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.52), int(w_total*0.88))
        procesar_un_ojo(coord_izq, "Izq")
        procesar_un_ojo(coord_der, "Der")

    def mostrar_ventanas_proceso(self):
        # VENTANA 1
        fig1 = plt.figure(figsize=(14, 5)); fig1.canvas.manager.set_window_title(f"V1: Preproc - {self.nombre_archivo}")
        ax1 = fig1.add_subplot(1, 4, 1); ax1.imshow(cv2.cvtColor(self.snapshots.get('1_Original', self.img_original), cv2.COLOR_BGR2RGB)); ax1.set_title("1. Original")
        ax2 = fig1.add_subplot(1, 4, 2); ax2.imshow(cv2.cvtColor(self.snapshots.get('2_Alineada', self.img_original), cv2.COLOR_BGR2RGB)); ax2.set_title("2. Alineada")
        ax3 = fig1.add_subplot(1, 4, 3); ax3.imshow(cv2.cvtColor(self.snapshots.get('3_Realce_CLAHE', self.img_original), cv2.COLOR_BGR2RGB)); ax3.set_title("3. CLAHE")
        ax4 = fig1.add_subplot(1, 4, 4); ax4.imshow(cv2.cvtColor(self.snapshots.get('4_Suavizado_Gauss', self.img_original), cv2.COLOR_BGR2RGB)); ax4.set_title("4. Gauss")
        for ax in fig1.axes: ax.axis('off')

        # VENTANA 2
        fig2 = plt.figure(figsize=(10, 8)); fig2.canvas.manager.set_window_title(f"V2: Localización - {self.nombre_archivo}")
        imgs_morph = [
            ('Ojo_Izq_2_TopHat', 'Izq: Top-Hat'), ('Ojo_Izq_3_BlackHat', 'Izq: Black-Hat'), ('Ojo_Izq_5_Geo_Ovalo', 'Izq: Óvalo Detectado'),
            ('Ojo_Der_2_TopHat', 'Der: Top-Hat'), ('Ojo_Der_3_BlackHat', 'Der: Black-Hat'), ('Ojo_Der_5_Geo_Ovalo', 'Der: Óvalo Detectado')
        ]
        for i, (key, title) in enumerate(imgs_morph):
            ax = fig2.add_subplot(2, 3, i+1)
            if key in self.snapshots: 
                img = self.snapshots[key]
                if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img, cmap='gray' if len(img.shape)==2 else None); ax.set_title(title)
            ax.axis('off')

        # VENTANA 3
        fig3 = plt.figure(figsize=(10, 8)); fig3.canvas.manager.set_window_title(f"V3: Segmentación - {self.nombre_archivo}")
        imgs_seg = [
            ('Ojo_Izq_6_Crop_Gray', 'Izq: Recorte Ovalado'), ('Ojo_Izq_7_Seg_Cruda', 'Izq: Seg. Cruda'), ('Ojo_Izq_8_Seg_Mejorada', 'Izq: Mejora Morfológica'),
            ('Ojo_Der_6_Crop_Gray', 'Der: Recorte Ovalado'), ('Ojo_Der_7_Seg_Cruda', 'Der: Seg. Cruda'), ('Ojo_Der_8_Seg_Mejorada', 'Der: Mejora Morfológica')
        ]
        for i, (key, title) in enumerate(imgs_seg):
            ax = fig3.add_subplot(2, 3, i+1)
            if key in self.snapshots: ax.imshow(self.snapshots[key], cmap='gray'); ax.set_title(title)
            ax.axis('off')

        # VENTANA 4
        fig4 = plt.figure(figsize=(12, 6)); fig4.canvas.manager.set_window_title(f"V4: Descriptores - {self.nombre_archivo}")
        ax1 = fig4.add_subplot(1, 3, 1)
        if 'Ojo_Izq_9_Final_Binaria' in self.snapshots: ax1.imshow(self.snapshots['Ojo_Izq_9_Final_Binaria'], cmap='gray'); ax1.set_title("Máscara Final Izq")
        ax1.axis('off')
        ax2 = fig4.add_subplot(1, 3, 2)
        if 'Ojo_Der_9_Final_Binaria' in self.snapshots: ax2.imshow(self.snapshots['Ojo_Der_9_Final_Binaria'], cmap='gray'); ax2.set_title("Máscara Final Der")
        ax2.axis('off')
        ax_text = fig4.add_subplot(1, 3, 3); ax_text.axis('off')
        texto_info = "DESCRIPTORES OBTENIDOS:\n\n"
        for ojo_key, data in self.descriptores_ojos.items():
            texto_info += f"--- {ojo_key.upper()} ---\n"
            texto_info += f"Compacidad: {data['Compacidad']:.2f}\n"
            texto_info += f"Dist. Media: {data['Dist. Media']:.4f}\n"
            texto_info += f"Rugosidad: {data['Rugosidad']:.4f}\n\n"
        ax_text.text(0.0, 0.5, texto_info, fontsize=10, family='monospace', va='center')
        plt.show()

if __name__ == "__main__":
    CARPETA_BASE = "images" 
    carpetas_personas = glob.glob(os.path.join(CARPETA_BASE, "persona*")); carpetas_personas.sort()
    print("INICIANDO PROCESO V6 - LOGICA RESTAURADA.")
    for carpeta in carpetas_personas:
        imagenes = []
        for ext in ['*.jpg', '*.jpeg', '*.png']: imagenes.extend(glob.glob(os.path.join(carpeta, ext)))
        imagenes.sort()
        for ruta_img in imagenes:
            try:
                print(f"Procesando: {ruta_img}...")
                app = AnalizadorFacialFinalV6(ruta_img)
                app.paso1_preprocesamiento()
                app.paso_procesamiento_ojos()
                app.mostrar_ventanas_proceso()
            except Exception as e: print(f"Error: {e}")