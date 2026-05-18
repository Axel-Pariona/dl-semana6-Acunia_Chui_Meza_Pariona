# Clasificación de Células Sanguíneas con CNN y Transfer Learning

[cite_start]Este repositorio contiene la implementación y el análisis del Trabajo Práctico de la Semana 6[cite: 1, 2]. [cite_start]El objetivo es entrenar y evaluar arquitecturas clásicas de redes neuronales convolucionales sobre un dataset real de imágenes microscópicas médicas[cite: 5, 6].

---

## 🛠️ Instalación y Ejecución

### Requisitos Previos
El código está desarrollado en Python 3.8+ utilizando el framework PyTorch. Todos los experimentos son completamente reproducibles fijando las semillas globales del sistema (`seed=42`).

### Instalación de Dependencias
Asegúrate de instalar las librerías necesarias ejecutando el siguiente comando en tu terminal:
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

---

## Ejecución de los Notebooks

La lógica se encuentra modularizada y separada en la carpeta notebooks/ siguiendo la estructura estricta exigida en la guía del entregable:

1. Ejecutar notebooks/01_eda.ipynb para el análisis exploratorio de datos y distribución de clases.

2. Ejecutar notebooks/02_lenet_vgg.ipynb para el entrenamiento de arquitecturas desde cero y el análisis de Batch Normalization (Tareas 1 y 2).

3. Ejecutar notebooks/03_transfer.ipynb para evaluar las estrategias de Transfer Learning (Tarea 3).

---

## 📊 Dataset: Blood Cell Image Dataset (Kaggle)

* [cite_start]**Clases:** 4 (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL)[cite: 12].
* [cite_start]**Imágenes:** ~12,500 imágenes RGB redimensionadas a 64x64x3 px[cite: 12, 18].
* [cite_start]**Split:** Train / Test provisto por el dataset[cite: 12].

---

## 📈 Tabla Resumen de Resultados

| Tarea | Arquitectura / Estrategia | Learning Rate | Parámetros Entrenables | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Tarea 1** | LeNet-5 Estándar | 0.001 | ~400,000 | 74.79% |
| **Tarea 1** | LeNet-5 + Batch Normalization | 0.001 | ~400,000 | 51.31% |
| **Tarea 1** | VGG11Small Estándar | 0.001 | 2,966,148 | 34.38% |
| **Tarea 1** | **VGG11Small + BN (Mejor desde cero)** | 0.001 | 2,966,148 | **83.31%** |
| **Tarea 2** | VGG11Small + BN | 0.003 | 2,966,148 | 68.64% |
| **Tarea 2** | VGG11Small + BN | 0.010 | 2,966,148 | 25.01% |
| **Tarea 3** | ResNet18: Feature Extraction | 0.001 | 2,052 | 59.43% |
| **Tarea 3** | ResNet18: Fine-tuning Parcial | 0.0001 | 10,495,492 | 81.54% |
| **Tarea 3** | **ResNet18: Fine-tuning Total (Mejor Global)** | 0.00001 | 11,178,564 | **85.77%** |

---

## 💡 Principales Hallazgos y Conclusiones

* [cite_start]**Efecto de Batch Normalization (BN):** En la red profunda `VGG11Small`, BN fue crítico para mitigar el *Internal Covariate Shift* (Ioffe & Szegedy, 2015)[cite: 30], elevando la precisión de test del **34.38% al 83.31%**. En la red superficial `LeNet-5` el efecto fue inverso (**74.79% vs 51.31%**), debido a que la normalización distorsionó las activaciones en un espacio de características muy acotado.
* [cite_start]**Velocidad de Convergencia:** Con un LR de `0.001`, `VGG11Small + BN` superó el umbral del **80% de accuracy en la Época 4 de validación**[cite: 29], mientras que la versión estándar sin BN jamás convergió. [cite_start]No obstante, BN demostró límites, ya que tasas de aprendizaje excesivas (`0.01`) provocaron la divergencia matemática del modelo[cite: 31].
* [cite_start]**Superioridad del Transfer Learning:** El **Fine-tuning Total** con `ResNet-18` preentrenada arrojó el desempeño óptimo de todo el proyecto (**85.77% Test Accuracy**) [cite: 32, 34][cite_start], demostrando la eficiencia de sintonizar todas las capas con un Learning Rate sumamente pequeño (`1e-5`)[cite: 34, 35].
* [cite_start]**Contexto de Datos Médicos Limitados:** Aunque el ajuste fino total consiguió la mayor métrica, **se recomienda implementar el Fine-tuning Parcial en escenarios con datos médicos limitados**[cite: 36]. [cite_start]Al congelar los primeros bloques, actúa como un regularizador estructural que previene el sobreajuste y la memorización de ruido microscópico[cite: 34].
* **Análisis por Clase:** El clasificador demostró un rendimiento perfecto en la detección de `LYMPHOCYTE` (1.00 F1-Score). [cite_start]La mayor brecha se presentó en la confusión de `NEUTROPHIL` frente a `EOSINOPHIL`, debido a similitudes morfológicas y granulométricas biológicas intrínsecas que pierden nitidez al reducir la resolución a 64x64 px[cite: 12, 18].