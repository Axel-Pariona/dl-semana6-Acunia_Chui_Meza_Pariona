# data/download_data.ps1

$ErrorActionPreference = "Stop"

Write-Host "Creando carpeta data/blood_cells..."
New-Item -ItemType Directory -Force -Path "data/blood_cells" | Out-Null

Write-Host "Verificando Kaggle CLI..."
kaggle --version

Write-Host "Descargando Blood Cell Image Dataset desde Kaggle..."
kaggle datasets download -d paultimothymooney/blood-cells -p data/blood_cells

$zipPath = "data/blood_cells/blood-cells.zip"

if (!(Test-Path $zipPath)) {
    Write-Host "ERROR: No se encontró $zipPath"
    exit 1
}

Write-Host "Descomprimiendo dataset..."
Expand-Archive -Path $zipPath -DestinationPath "data/blood_cells" -Force

Write-Host "Dataset descargado y descomprimido correctamente."