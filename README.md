# Clasificación de Células Sanguíneas con CNN, Batch Normalization y Transfer Learning

Este repositorio contiene la implementación y análisis de la PC2 del curso **Introduction to Deep Learning**. El objetivo es entrenar, evaluar y comparar arquitecturas clásicas de redes neuronales convolucionales sobre un dataset real de imágenes microscópicas médicas.

El proyecto aborda tres componentes principales:

- Implementación desde cero de LeNet-5 y VGG-11 simplificado.
- Análisis del efecto de Batch Normalization.
- Aplicación de Transfer Learning con ResNet-18 preentrenada en ImageNet.

---

## Dataset

Se utilizó el **Blood Cell Image Dataset** de Kaggle.

- **Clases:** EOSINOPHIL, LYMPHOCYTE, MONOCYTE y NEUTROPHIL.
- **Tipo de imágenes:** RGB.
- **Tamaño original:** 320x240 píxeles.
- **Cantidad aproximada:** 12,500 imágenes.
- **Split:** Train/Test provisto por el dataset.

Para los modelos entrenados desde cero se redimensionaron las imágenes a **64x64x3**.  
Para Transfer Learning con ResNet-18 se utilizaron imágenes de **224x224x3**.

---

## Estructura del Proyecto

```text
dl-semana6-Acunia_Chui_Meza_Pariona/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_data.sh
│   └── download_data.ps1
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_lenet_vgg.ipynb
│   └── 03_transfer.ipynb
├── src/
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── results/
│   ├── figures/
│   └── metrics/
└── informe.pdf
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Axel-Pariona/dl-semana6-Acunia_Chui_Meza_Pariona.git
cd dl-semana6-Acunia_Chui_Meza_Pariona
```

### 2. Crear entorno virtual

En Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

En Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Descarga del Dataset

Para descargar el dataset se necesita configurar previamente la API de Kaggle.

El archivo `kaggle.json` debe estar ubicado en:

```text
C:\Users\<usuario>\.kaggle\kaggle.json
```

En Windows:

```powershell
powershell -ExecutionPolicy Bypass -File data/download_data.ps1
```

En Linux, Git Bash o Google Colab:

```bash
bash data/download_data.sh
```

Luego de la descarga, la ruta esperada es:

```text
data/blood_cells/dataset2-master/dataset2-master/images/
```

---

## Ejecución de Notebooks

Los notebooks deben ejecutarse en este orden:

1. `notebooks/01_eda.ipynb`  
   Análisis exploratorio del dataset: distribución por clase, visualización de ejemplos, tamaños de imagen y revisión de imágenes corruptas.

2. `notebooks/02_lenet_vgg.ipynb`  
   Implementación y entrenamiento desde cero de LeNet-5 y VGG-11 simplificado, con y sin Batch Normalization. También incluye el análisis del efecto de Batch Normalization.

3. `notebooks/03_transfer.ipynb`  
   Aplicación de Transfer Learning con ResNet-18 preentrenada en ImageNet mediante tres estrategias: Feature Extraction, Fine-tuning parcial y Fine-tuning total.

---

## Resultados Principales

### Tarea 1: CNN entrenadas desde cero

| Modelo | Learning Rate | Parámetros Entrenables | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| LeNet | 0.001 | 337,976 | 74.79% | 0.7932 |
| LeNet + BN | 0.001 | 338,020 | 51.31% | 3.5496 |
| VGG11Small | 0.001 | 2,963,396 | 34.38% | 1.3740 |
| **VGG11Small + BN** | 0.001 | 2,966,148 | **83.31%** | 1.3317 |

El mejor modelo entrenado desde cero fue **VGG11Small_BN**, alcanzando **83.31% de accuracy en test**.

---

### Tarea 2: Análisis de Batch Normalization

| Modelo | Batch Normalization | Learning Rate | Test Accuracy | Test Loss |
|---|---|---:|---:|---:|
| VGG11Small | Sin BN | 0.001 | 25.05% | 1.3864 |
| VGG11Small + BN | Con BN | 0.001 | 80.18% | 0.5163 |
| VGG11Small | Sin BN | 0.003 | 25.05% | 1.3868 |
| VGG11Small + BN | Con BN | 0.003 | 69.00% | 1.0727 |
| VGG11Small | Sin BN | 0.010 | 25.05% | 1.3864 |
| VGG11Small + BN | Con BN | 0.010 | 71.93% | 0.4776 |

Los modelos sin Batch Normalization permanecieron cerca del azar, aproximadamente 25% para un problema de 4 clases. En cambio, las variantes con BN lograron aprender patrones discriminativos, especialmente con `LR=0.001`.

---

### Tarea 3: Transfer Learning con ResNet-18

| Modelo / Estrategia | Learning Rate | Parámetros Entrenables | Test Accuracy | Test Loss |
|---|---:|---:|---:|---:|
| VGG11Small_BN desde cero | 0.001 | 2,966,148 | 83.31% | 1.3317 |
| ResNet18 Feature Extraction | 0.001 | 2,052 | 59.43% | - |
| ResNet18 Fine-tuning Parcial | 0.0001 | 10,495,492 | 81.54% | - |
| **ResNet18 Fine-tuning Total** | 0.00001 | 11,178,564 | **85.77%** | 0.5750 |

El mejor modelo global fue **ResNet18 con Fine-tuning Total**, alcanzando **85.77% de accuracy en test**.

---

## Principales Hallazgos

- **Batch Normalization fue clave en VGG11Small.**  
  VGG11Small sin BN obtuvo 34.38% en test, mientras que VGG11Small_BN alcanzó 83.31%.

- **Batch Normalization no siempre mejora modelos pequeños.**  
  En LeNet-5, la versión sin BN obtuvo 74.79%, mientras que LeNet_BN alcanzó 51.31%.

- **El learning rate sigue siendo crítico.**  
  BN mejoró la estabilidad del entrenamiento, pero el rendimiento siguió dependiendo de una selección adecuada del learning rate.

- **Transfer Learning obtuvo el mejor desempeño global.**  
  ResNet18 con Fine-tuning Total alcanzó 85.77% de accuracy en test.

- **Recomendación para datos médicos limitados.**  
  Aunque Fine-tuning Total obtuvo la mejor métrica experimental, en un escenario médico con pocos datos se recomienda iniciar con Fine-tuning Parcial, ya que reduce el número de parámetros entrenables y disminuye el riesgo de sobreajuste.

- **Análisis por clase.**  
  El mejor modelo clasificó correctamente la clase LYMPHOCYTE, mientras que la mayor confusión se presentó entre EOSINOPHIL y NEUTROPHIL.

---

## Reproducibilidad

Para asegurar reproducibilidad:

- Se fijó una semilla global `seed=42`.
- Se usó `torch.manual_seed`.
- Se controló la división train/validation.
- Se guardaron checkpoints del mejor modelo según accuracy de validación.
- Las métricas y figuras se almacenaron en `results/metrics/` y `results/figures/`.

---

## Tecnologías Utilizadas

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Kaggle API
- Google Colab / VS Code

---

## Autores

- Acuña Villegas, Omar Junior
- Chui Sanchez, Rafael Tomas
- Meza Polo, Rodrigo Alejandro
- Pariona Rojas, Axel Yamir
