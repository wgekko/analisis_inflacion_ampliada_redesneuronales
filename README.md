# analisis_inflacion_ampliada_redesneuronales
Dashboard analisis de inflación considerando subsectores de actividad con redes neuronales

# Deep Learning para Pronóstico Macroeconómico: Ecosistema de Modelos IPC

Este proyecto despliega un conjunto de aplicaciones interactivas desarrolladas en **Streamlit** diseñadas para la proyección de la tasa de inflación mensual (IPC) por regiones y rubros. A través de arquitecturas avanzadas de **Deep Learning**, el sistema automatiza la ingesta, limpieza, transformación y modelado predictivo a partir de datos brutos de índices de precios.

## Arquitecturas Implementadas

El ecosistema está compuesto por cuatro enfoques de red neuronal independientes, permitiendo evaluar la sensibilidad y estabilidad de cada arquitectura ante shocks económicos:

* **Transformer (`ModeloTransformer.py`):** Utiliza mecanismos de *Multi-Head Attention* y normalización de capas para capturar dependencias globales y complejas en las series temporales de variación mensual.
* **TCN - Temporal Convolutional Network (`ModeloTCN.py`):** Implementa convoluciones 1D causales y dilatadas. Esta arquitectura ofrece una gran estabilidad frente a anomalías macroeconómicas gracias a su campo receptivo expandido.
* **LSTM (`ModeloLSTM.py`):** Red de memoria a largo y corto plazo (*Long Short-Term Memory*), optimizada para identificar patrones secuenciales y estacionales tradicionales en la economía.
* **GRU (`ModeloGRU.py`):** Redes de unidades recurrentes compuertas (*Gated Recurrent Units*), una alternativa eficiente a LSTM con menor costo computacional y convergencia ágil en series históricas.

## Características Principales

* **Pipeline de Datos Automatizado:** Ingesta desde archivos estructurados (`.xlsx`), detección dinámica de regiones (GBA, Pampeana, Cuyo, etc.), interpolación lineal de datos faltantes y cálculo automático de la variación porcentual mensual.
* **Interfaz Analítica Avanzada:** Tablas dinámicas formateadas con métricas clave y proyecciones iterativas a 3 meses (Mes 1, Mes 2, Mes 3).
* **Visualización Interactiva:** Gráficos de alta fidelidad con **Plotly Dark** que incluyen controles deslizantes para *zoom temporal* dinámico y trazado de tendencias históricas vs. predictivas.
* **Predicción Iterativa Compleja:** Los modelos utilizan sus propias predicciones previas de manera autorregresiva para proyectar los meses subsiguientes.

## Estructura del Proyecto


├── data/
│   └── ipc.xlsx                 # Archivo de datos fuente (Índices IPC)
├── ModeloTransformer.py         # Tablero predictivo con arquitectura Transformer
├── ModeloTCN.py                 # Tablero predictivo con arquitectura TCN
├── ModeloLSTM.py                # Tablero predictivo con arquitectura LSTM
├── ModeloGRU.py                 # Tablero predictivo con arquitectura GRU
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación principal


clonar repo
https://github.com/wgekko/analisis_inflacion_ampliada_redesneuronales.git

video demo 


https://github.com/user-attachments/assets/11194357-712b-46f0-a670-51aecad1a7c1



