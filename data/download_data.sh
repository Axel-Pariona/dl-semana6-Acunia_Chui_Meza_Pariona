#!/bin/bash

set -e

echo "Creando carpeta data/blood_cells..."
mkdir -p data/blood_cells

echo "Verificando Kaggle CLI..."
kaggle --version

echo "Descargando Blood Cell Image Dataset desde Kaggle..."
kaggle datasets download -d paultimothymooney/blood-cells -p data/blood_cells

echo "Descomprimiendo dataset..."
unzip -q data/blood_cells/blood-cells.zip -d data/blood_cells

echo "Dataset descargado y descomprimido correctamente."