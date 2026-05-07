# Trading ML Prediction Pipeline

Proyecto de Machine Learning aplicado a trading para predecir la dirección diaria del ETF SPY utilizando variables de mercado.

---

# Objetivo

Construir un pipeline reproducible de machine learning capaz de:

- Descargar datos financieros
- Generar features de mercado
- Entrenar modelos predictivos
- Evaluar estrategias de trading mediante backtesting
- Exponer resultados en una aplicación web

---

# Tecnologías utilizadas

- Python
- Scikit-learn
- Pandas
- MLflow
- DVC
- Streamlit
- Docker
- Git

---

# Estructura del proyecto

```text
ml_trading_pipeline/
│
├── src/
│   └── trading_ml/
│       ├── app/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       └── models/
│
├── data/
├── models/
├── notebooks/
├── tests/
│
├── Dockerfile
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

# Pipeline

El proyecto utiliza un pipeline reproducible con DVC:

1. Descarga de datos
2. Feature engineering
3. Entrenamiento de modelos
4. Evaluación y backtesting

---

# Modelos

Modelos utilizados:

- Logistic Regression
- Random Forest

---

# Backtesting

Se implementó una estrategia de trading basada en probabilidades del modelo utilizando:

- threshold dinámico
- filtrado de señales
- reducción de sobreoperación

Resultados aproximados:

- Strategy Return ≈ 1.14
- Buy & Hold ≈ 1.57

---

# Insights importantes

- Predecir mercados financieros es extremadamente complejo
- Más features no necesariamente mejoran el modelo
- Las probabilidades son más útiles que predicciones binarias
- El backtesting es más importante que accuracy aislado

---

# Cómo ejecutar localmente

## 1. Clonar repositorio

```bash
git clone <URL_DEL_REPO>
cd ml_trading_pipeline
```

---

## 2. Crear entorno virtual

```bash
python -m venv .venv
```

---

## 3. Activar entorno virtual

### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

---

## 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 5. Ejecutar pipeline

```bash
dvc repro
```

---

## 6. Entrenar modelos

```bash
python src/trading_ml/models/train.py
```

---

## 7. Ejecutar backtest

```bash
python src/trading_ml/evaluation/backtest.py
```

---

# Docker

## Build

```bash
docker build -t trading-ml-app .
```

---

## Run

```bash
docker run -p 8501:8501 trading-ml-app
```

---

# Trabajo futuro

Posibles mejoras futuras:

- costos de transacción
- variables macroeconómicas
- integración con APIs de brokers
- modelos de deep learning
- estrategias multi-activo

---

# Autor

Enrique Ortiz