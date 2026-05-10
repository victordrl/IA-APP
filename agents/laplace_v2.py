import tensorflow as tf
from tensorflow.keras import layers, models

def crear_autoencoder_trading(ventana_tiempo=60, n_features=4):
    # --- ENCODER ---
    #definimos que va a recibir 60 
    inputs = layers.Input(shape=(ventana_tiempo, n_features))
    
    # Procesamiento
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    
    # Atención
    atencion = layers.Attention()([x, x])
    
    # Reducción a Súper Indicador
    x = layers.GlobalAveragePooling1D()(atencion)
    x = layers.Dense(32, activation='relu')(x)
    
    # Este es el corazón: El vector de Súper Indicadores
    super_indicador = layers.Dense(16, activation='tanh', name="capa_latente")(x)
    
    # --- DECODER ---
    x = layers.Dense(32, activation='relu')(super_indicador)
    x = layers.RepeatVector(ventana_tiempo)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    
    # Salida: Reconstrucción de las 4 columnas originales
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
    
    # Construir modelo
    modelo = models.Model(inputs, outputs)
    modelo.compile(optimizer='adam', loss='mse') # MSE porque queremos que el error de reconstrucción sea mínimo
    
    return modelo

# Instanciar el modelo
model = crear_autoencoder_trading()