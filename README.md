# IA-APP: Estrategia de Inversión Cuantitativa

Este proyecto implementa una **Inteligencia Artificial** especializada en mercados financieros, diseñada para generar señales de compra y venta con objetivos de rentabilidad superiores al 10%. El sistema integra técnicas avanzadas de Deep Learning con metodologías cuantitativas de vanguardia.

## Características Principales

- **Deep Reinforcement Learning (DRL):** El agente aprende directamente de los resultados financieros, optimizando la estrategia en entornos simulados antes de operar en real.
- **Arquitectura Multimodal (xLSTM):** Integra información de precios, volumen, indicadores técnicos y sentimiento de mercado para tomar decisiones holísticas.
- **Gestión de Riesgo Avanzada:** Implementa el **Triple Barrier Method** y **Differentiable Sharpe Ratio** para asegurar ratios riesgo/beneficio favorables (objetivo 1:4) y evitar el _overfitting_.
- **Estructura Modular (RF-1 a RF-11):** El código sigue una arquitectura estricta que separa la recolección de datos, cálculo de indicadores, construcción de tensores y lógica de inferencia.

## Arquitectura Técnica

El sistema está organizado en los siguientes módulos principales:

1.  **Data Fetcher:** Obtiene datos históricos de múltiples temporalidades (1h, 4h, 1d) desde fuentes fiables (Binance, CCXT).
2.  **Indicator Engine:** Calcula indicadores técnicos (SMA, EMA, MACD, Bollinger Bands, etc.) utilizando la librería `ta`.
3.  **Tensor Builder:** Estructura los datos en tensores 3D para el entrenamiento de redes neuronales, preservando el orden temporal.
4.  **Model Architecture:** Implementación de redes basadas en **Transformers** y **LSTM** adaptadas a datos financieros (xLSTM, PatchTST).
5.  **Risk Manager:** Define las condiciones de salida y profit (10%) y stop-loss (2%) para guiar el entrenamiento.

## Instalación y Ejecución

### Requisitos

- Python 3.9+
- CUDA (para entrenamiento GPU)

### Instalación de Dependencias

```bash
git clone https://github.com/tuusuario/IA-APP.git
cd IA-APP
pip install -r requirements.txt
```

### Ejecución

- **Entrenamiento:**
  ```bash
  python train.py --symbol BTC/USDT --epochs 50 --window-size 30
  ```
- **Inferencia (Modo Replay):**
  ```bash
  python main.py --mode replay --symbol BTC/USDT --speed 2.0
  ```

## 📊 Estrategia de Mercado

- **Símbolo:** BTC/USDT
- **Objetivo de Profit:** +10% por operación.
- **Stop Loss:** -2%.
- **Horizonte:** Corto / Medio plazo.
- **Temporalidades:** 1h, 4h, 1d (sincronizadas).
