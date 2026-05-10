import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Force TF as backend

import joblib
import keras
from keras import layers, models


def crear_modelo(ventana_tiempo=60, features=4):


    modelo = keras.Sequential([
        #convolucionamos las imagenes con 32 filtros y padding para ,antener el tamaño
        layers.Conv1D(filters=32,kernel_size=3,activation='relu',padding='same',input_shape=(ventana_tiempo,features)),
        #aplicamos pooling para eliminar ruido
        layers.MaxPool1D(2),
        # creamos la memoria
        layers.LSTM(64,return_sequences=False),
        # concentramos la info y entrenamos las neuronas
        layers.Dense(16,activation='relu', name="capa_procesamiento"),
         # 2. BRIDGE (El puente)
        # Repetimos el resumen para volver a crear la línea de tiempo (60 pasos)
        layers.RepeatVector(ventana_tiempo),

        # 3. DECODER (Reconstrucción)
        # La memoria intenta reconstruir el pasado
        layers.LSTM(64, return_sequences=True),
        # Volvemos a las 4 columnas originales (1m, 5m, 15m, 1d)
        layers.TimeDistributed(layers.Dense(features))
    ])

    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo

def entrenar_modelo():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import re

    # 1. Cargar el dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'datasets', 'llm_trading_dataset_20250629_115817.jsonl')
    rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extraer los valores de los indicadores técnicos usando regex
            match = re.search(r'indicators: (.*)', line)
            if match:
                indicators_str = match.group(1)
                # Separar los pares clave=valor
                indicators = dict()
                for pair in indicators_str.split(','):
                    if '=' in pair:
                        k, v = pair.strip().split('=')
                        
                        # Limpiar comillas y caracteres no numéricos
                        valor = v.strip().replace('"', '').replace("'", '')
                        # Eliminar cualquier caracter no numérico excepto punto y menos
                        import re as _re
                        valor = _re.sub(r'[^0-9.\-eE]', '', valor)
                        indicators[k.strip()] = float(valor)
                rows.append(indicators)

    df = pd.DataFrame(rows)

    # 2. Seleccionar las 10 columnas de indicadores técnicos
    columnas = ['EMA20', 'EMA50', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 'RSI', 'CCI', 'STOCH_K', 'STOCH_D']
    df = df[columnas]

    # 3. Normalizar los datos
    scaler = MinMaxScaler()
    datos_norm = scaler.fit_transform(df.values)

    # 4. Crear las ventanas temporales (shape: muestras, 60, 10)
    ventana_tiempo = 60
    features = 10
    X = []
    for i in range(len(datos_norm) - ventana_tiempo + 1):
        X.append(datos_norm[i:i+ventana_tiempo])
    X = np.array(X)

    # 5. Entrenar el modelo
    modelo = crear_modelo(ventana_tiempo, features)
    modelo.fit(X, X, epochs=1, batch_size=32, validation_split=0.1)
    modelo.save('laplace_v3.keras')
    scaler = joblib.dump(scaler,'escalador_v3.pkl')
    return modelo, scaler


# --- Visualización fuera de entrenar_modelo ---
def visualizar_reconstruccion(modelo, scaler, ventana_tiempo=60, n_muestras=500):
    """
    Visualiza la reconstrucción del autoencoder y los super indicadores (promedios de los indicadores).
    Args:
        modelo: modelo entrenado
        scaler: objeto MinMaxScaler usado para desnormalizar
        ventana_tiempo: tamaño de la ventana temporal
        n_muestras: cantidad de muestras a mostrar
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import re

    # 1. Cargar y procesar los datos igual que en entrenar_modelo
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'datasets', 'llm_trading_dataset_20250629_115817.jsonl')
    rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'indicators: (.*)', line)
            if match:
                indicators_str = match.group(1)
                indicators = dict()
                for pair in indicators_str.split(','):
                    if '=' in pair:
                        k, v = pair.strip().split('=')
                        valor = v.strip().replace('"', '').replace("'", '')
                        import re as _re
                        valor = _re.sub(r'[^0-9.\-eE]', '', valor)
                        indicators[k.strip()] = float(valor)
                rows.append(indicators)
    columnas = ['EMA20', 'EMA50', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 'RSI', 'CCI', 'STOCH_K', 'STOCH_D']
    df = pd.DataFrame(rows)[columnas]
    datos_norm = scaler.transform(df.values)
    X = []
    for i in range(len(datos_norm) - ventana_tiempo + 1):
        X.append(datos_norm[i:i+ventana_tiempo])
    X = np.array(X)

    # 2. Seleccionar las primeras n_muestras para visualizar
    X_vis = X[:n_muestras]
    X_recon = modelo.predict(X_vis)

    # 3. Desnormalizar para comparar en escala real
    X_vis_flat = X_vis.reshape(-1, X_vis.shape[-1])
    X_recon_flat = X_recon.reshape(-1, X_recon.shape[-1])
    X_vis_real = scaler.inverse_transform(X_vis_flat)
    X_recon_real = scaler.inverse_transform(X_recon_flat)

    # 4. Visualización de la reconstrucción para cada indicador
    import matplotlib
    fig, axs = plt.subplots(5, 2, figsize=(16, 16))
    axs = axs.flatten()
    for i, col in enumerate(columnas):
        axs[i].plot(X_vis_real[:, i], label='Original', alpha=0.7)
        axs[i].plot(X_recon_real[:, i], label='Reconstruido', alpha=0.7)
        axs[i].set_title(col)
        axs[i].legend()
    plt.tight_layout()
    plt.show()

    # 5. Visualización de super indicadores (promedio móvil de los indicadores)
    super_indicador_real = X_vis_real.mean(axis=1)
    super_indicador_recon = X_recon_real.mean(axis=1)
    plt.figure(figsize=(14, 5))
    plt.plot(super_indicador_real, label='Super Indicador Original', alpha=0.7)
    plt.plot(super_indicador_recon, label='Super Indicador Reconstruido', alpha=0.7)
    plt.title('Super Indicador (Promedio de los 10 indicadores)')
    plt.legend()
    plt.show()


# Ejemplo de uso interactivo
if __name__ == '__main__':
    print('Entrenando modelo...')
    # modelo, scaler = entrenar_modelo()
    modelo = keras.models.load_model('laplace_v3.keras')
    scaler = joblib.load('escalador_v3.pkl')
    # print('Modelo entrenado. Mostrando reconstrucción y super indicadores...')
    visualizar_reconstruccion(modelo, scaler)