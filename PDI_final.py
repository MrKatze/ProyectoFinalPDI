import cv2
import numpy as np
import os
import glob

class AnalizadorFacialBatch:
    
    def __init__(self, ruta_imagen: str):
        self.ruta = ruta_imagen
        self.nombre_archivo = os.path.basename(ruta_imagen)
        self.img_original = cv2.imread(ruta_imagen)
        if self.img_original is None:
            raise ValueError(f"No se pudo cargar: {ruta_imagen}")
        
        # Clasificadores
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Almacenamiento
        self.descriptores_ojos = {} 
        self.log_proceso = [] 

    def _log(self, etapa, mensaje):
        """Registra el proceso para el reporte final"""
        self.log_proceso.append(f"[{etapa}] {mensaje}")

    def paso1_alineacion_y_normalizacion(self):
        self._log("1. PREPROC", "Iniciando alineación...")
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
                self._log("1. PREPROC", f"Rotación detectada: {angulo:.2f}°")
                
                centro_ojos = ((c_izq[0] + c_der[0]) / 2.0, (c_izq[1] + c_der[1]) / 2.0)
                M = cv2.getRotationMatrix2D(centro_ojos, angulo, 1.0)
                h_img, w_img = self.img_original.shape[:2]
                img_rotada = cv2.warpAffine(self.img_original, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
                
                # Recorte Zoom
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
        
        # Realce CLAHE
        img_yuv = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        self.img_iluminada = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
        
        # Suavizado para reducir ruido base
        self.img_trabajo = cv2.GaussianBlur(self.img_iluminada, (5, 5), 0)
        self._log("1. PREPROC", "Imagen alineada, normalizada (CLAHE) y suavizada.")

    def _calcular_firma_radial(self, contour):
        """Calcula los descriptores geométricos del contorno final"""
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
        
        # Cruces por cero
        senal_centrada = drn - mu_drn
        cruces_cero = 0
        for i in range(len(drn) - 1):
            if (senal_centrada[i] * senal_centrada[i+1]) < 0:
                cruces_cero += 1
                
        # Índice de área
        suma_area = np.sum(np.abs(drn - mu_drn))
        indice_area = suma_area / len(drn)
        
        # Rugosidad
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

    def paso5_extraccion_ojos_y_descriptores(self):
        """
        Implementación de la lógica V6: 
        TopHat/BlackHat -> Padding -> Óvalo -> Recorte -> Otsu -> Morfología Suave
        """
        self._log("2. PROCESAMIENTO", "Iniciando localización y segmentación de ojos...")
        
        img = self.img_trabajo.copy()
        h_total, w_total = img.shape[:2]
        self.descriptores_ojos = {}

        def procesar_un_ojo(coords, nombre_lado):
            prefix = f"Ojo_{nombre_lado}"
            y1, y2, x1, x2 = coords
            y1, y2 = max(0, y1), min(h_total, y2)
            x1, x2 = max(0, x1), min(w_total, x2)
            
            roi_color = img[y1:y2, x1:x2]
            if roi_color.size == 0: return

            gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # --- A. LOCALIZACIÓN DEL ÓVALO ---
            kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            tophat = cv2.morphologyEx(gray_roi, cv2.MORPH_TOPHAT, kernel_x)
            blackhat = cv2.morphologyEx(gray_roi, cv2.MORPH_BLACKHAT, kernel_x)
            fusion = cv2.add(tophat, blackhat)
            
            # Umbralización para hallar la mancha principal
            _, mask_loc = cv2.threshold(fusion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Limpieza inicial
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask_loc = cv2.morphologyEx(mask_loc, cv2.MORPH_CLOSE, kernel_close)
            mask_loc = cv2.erode(mask_loc, None, iterations=1)
            
            # PADDING DE BORDES (La clave de la V6 para evitar cejas)
            h_roi, w_roi = mask_loc.shape
            mask_center = np.zeros_like(mask_loc)
            pad_x = int(w_roi * 0.1)
            pad_y = int(h_roi * 0.2)
            mask_center[pad_y:h_roi-pad_y, pad_x:w_roi-pad_x] = 255
            mask_loc = cv2.bitwise_and(mask_loc, mask_center)
            
            # Buscar contorno para el óvalo
            cnts, _ = cv2.findContours(mask_loc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask_ovalo_solido = np.zeros_like(gray_roi)
            hay_ovalo = False

            if cnts:
                c_max = max(cnts, key=cv2.contourArea)
                # Validar tamaño mínimo y geometría (mínimo 5 puntos para elipse)
                if cv2.contourArea(c_max) > 30 and len(c_max) >= 5:
                    elipse = cv2.fitEllipse(c_max)
                    cv2.ellipse(mask_ovalo_solido, elipse, 255, -1) # Máscara sólida
                    hay_ovalo = True
            
            if not hay_ovalo:
                self._log(f"WARN - {prefix}", "No se pudo detectar el óvalo del ojo.")
                return

            # --- B. SEGMENTACIÓN DEL CONTENIDO DEL ÓVALO ---
            
            # 1. Recorte (Crop)
            gray_recortada_ovalo = cv2.bitwise_and(gray_roi, gray_roi, mask=mask_ovalo_solido)

            # 2. Otsu sobre píxeles válidos
            pixeles_validos = gray_recortada_ovalo[mask_ovalo_solido > 0]
            thresh_val = 0
            if pixeles_validos.size > 0:
                thresh_val, _ = cv2.threshold(pixeles_validos, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            _, mask_segmentada = cv2.threshold(gray_recortada_ovalo, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Limpieza de seguridad (dentro del óvalo)
            mask_segmentada = cv2.bitwise_and(mask_segmentada, mask_ovalo_solido)

            # 3. Mejora Morfológica Suave (3x3)
            kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask_mejorada = cv2.morphologyEx(mask_segmentada, cv2.MORPH_OPEN, kernel_clean)
            mask_mejorada = cv2.morphologyEx(mask_mejorada, cv2.MORPH_CLOSE, kernel_clean)
            
            # --- C. CÁLCULO DE DESCRIPTORES ---
            cnts_final, _ = cv2.findContours(mask_mejorada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts_final:
                c_final = max(cnts_final, key=cv2.contourArea)
                desc = self._calcular_firma_radial(c_final)
                if desc: 
                    self.descriptores_ojos[prefix] = desc
                    self._log(f"OK - {prefix}", f"Descriptores calculados. Compacidad: {desc['Compacidad']:.2f}")
                else:
                    self._log(f"WARN - {prefix}", "Contorno final inválido para descriptores.")

        # Coordenadas de búsqueda (ROIs)
        coord_izq = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.12), int(w_total*0.48))
        coord_der = (int(h_total*0.35), int(h_total*0.52), int(w_total*0.52), int(w_total*0.88))
        
        procesar_un_ojo(coord_izq, "Izq")
        procesar_un_ojo(coord_der, "Der")

    def imprimir_reporte_proceso(self):
        print("-" * 60)
        print(f"ARCHIVO: {self.nombre_archivo}")
        for linea in self.log_proceso:
            print(f"  {linea}")
        
        if self.descriptores_ojos:
            print("  >>> RESULTADOS:")
            for ojo, metricas in self.descriptores_ojos.items():
                resumen = ", ".join([f"{k}: {v:.4f}" for k, v in metricas.items()])
                print(f"  {ojo}: {resumen}")
        else:
            print("  >>> NO SE OBTUVIERON DESCRIPTORES.")
        print("-" * 60)

# --- PROCESAMIENTO POR LOTES ---
if __name__ == "__main__":
    CARPETA_BASE = "images" # Asegúrate que esta ruta sea correcta
    
    # Buscar carpetas persona1, persona2, etc.
    carpetas_personas = glob.glob(os.path.join(CARPETA_BASE, "persona*"))
    carpetas_personas.sort()
    
    if not carpetas_personas:
        print(f"No se encontraron carpetas en '{CARPETA_BASE}'.")
        print("Asegúrate de que la estructura sea: images/persona1/foto.jpg")
    else:
        print("INICIANDO PROCESAMIENTO POR LOTES (Lógica V6)...")

    for carpeta in carpetas_personas:
        nombre_persona = os.path.basename(carpeta)
        print(f"\nProcesando carpeta: {nombre_persona}")
        
        imagenes = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            imagenes.extend(glob.glob(os.path.join(carpeta, ext)))
        imagenes.sort()
        
        for ruta_img in imagenes:
            try:
                # Instanciar y ejecutar sin ventanas
                app = AnalizadorFacialBatch(ruta_img)
                app.paso1_alineacion_y_normalizacion()
                app.paso5_extraccion_ojos_y_descriptores()
                app.imprimir_reporte_proceso()
                
            except Exception as e:
                print(f"Error procesando {ruta_img}: {e}")