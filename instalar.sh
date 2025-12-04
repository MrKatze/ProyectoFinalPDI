#!/bin/bash
# Script de Instalación y Verificación
# Proyecto Final PDI - Avance V

echo "============================================================"
echo "INSTALACIÓN - PROYECTO FINAL PDI"
echo "============================================================"

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
echo -e "\n${YELLOW}[1/5]${NC} Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python encontrado: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 no encontrado. Por favor instala Python 3.8 o superior."
    exit 1
fi

# Crear entorno virtual
echo -e "\n${YELLOW}[2/5]${NC} Creando entorno virtual..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}!${NC} Entorno virtual ya existe, saltando..."
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Entorno virtual creado"
fi

# Activar entorno virtual
echo -e "\n${YELLOW}[3/5]${NC} Activando entorno virtual..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Entorno virtual activado"

# Instalar dependencias
echo -e "\n${YELLOW}[4/5]${NC} Instalando dependencias..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencias instaladas"

# Verificar instalación
echo -e "\n${YELLOW}[5/5]${NC} Verificando instalación..."
python3 << EOF
import sys
import cv2
import numpy
import matplotlib
import scipy
import skimage

print("${GREEN}✓${NC} OpenCV:", cv2.__version__)
print("${GREEN}✓${NC} NumPy:", numpy.__version__)
print("${GREEN}✓${NC} Matplotlib:", matplotlib.__version__)
print("${GREEN}✓${NC} SciPy:", scipy.__version__)
print("${GREEN}✓${NC} scikit-image:", skimage.__version__)
EOF

# Verificar estructura de directorios
echo -e "\n${YELLOW}Verificando estructura de directorios...${NC}"

if [ -d "images" ]; then
    NUM_PERSONAS=$(find images -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ $NUM_PERSONAS -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Directorio 'images/' encontrado con $NUM_PERSONAS persona(s)"
    else
        echo -e "${YELLOW}!${NC} Directorio 'images/' existe pero está vacío"
        echo -e "  ${YELLOW}→${NC} Organiza tus imágenes en subdirectorios por persona"
    fi
else
    echo -e "${YELLOW}!${NC} Directorio 'images/' no encontrado"
    echo -e "  ${YELLOW}→${NC} Creando estructura de ejemplo..."
    mkdir -p images/persona1 images/persona2 images/persona3
    echo -e "  ${GREEN}✓${NC} Estructura creada. Agrega tus imágenes en:"
    echo -e "     images/persona1/"
    echo -e "     images/persona2/"
    echo -e "     images/persona3/"
fi

# Crear directorios de salida
mkdir -p resultados/avance_v
mkdir -p resultados/prueba_rapida
echo -e "${GREEN}✓${NC} Directorios de resultados creados"

# Hacer scripts ejecutables
chmod +x avance_v_pipeline.py prueba_rapida.py
echo -e "${GREEN}✓${NC} Scripts configurados como ejecutables"

echo -e "\n============================================================"
echo -e "${GREEN}INSTALACIÓN COMPLETADA${NC}"
echo -e "============================================================"

echo -e "\n${YELLOW}Siguientes pasos:${NC}"
echo -e "1. Organiza tus imágenes en: ${GREEN}images/personaX/${NC}"
echo -e "2. Ejecuta prueba rápida: ${GREEN}./prueba_rapida.py${NC}"
echo -e "3. O procesa todo: ${GREEN}./avance_v_pipeline.py${NC}"

echo -e "\n${YELLOW}Recordatorio:${NC} Activa el entorno virtual con:"
echo -e "${GREEN}source venv/bin/activate${NC}"

echo -e "\n¡Todo listo! 🚀\n"
