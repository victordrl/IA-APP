from tensorflow.keras import layers

# 1. Definimos la Entrada
#creamos un bloque con 30 unidades de informacion con 4 propiedades(columna) cada uno
#luego lo modificamos para cada uno de los indicadores
entrada = layers.Input(shape=(30, 4))

# 2. La Capa Convolucional (La Lupa)
# 'filters=32': Queremos que la red busque 32 patrones diferentes.
# 'kernel_size=3': Cada patrón se busca en ventanas de 3 en 3 minutos.
# 'activation=relu': Quitamos lo que no sirve (valores negativos).
x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(entrada)

"""
MaxPooling1D
Normalmente, después de esta línea que escribiste, se suele poner un MaxPooling1D.
¿Por qué? Porque la Conv1D genera mucha información. El Pooling sirve para decir:
 "De estas detecciones, dame solo la más fuerte de cada zona para que el modelo sea más robusto"


"""
# 'pool_size=2': De cada 2 señales detectadas, nos quedamos con la más fuerte.
x = layers.MaxPooling1D(pool_size=2)(x)


# (Continuando después de la capa Conv1D y el Pooling...)

# 4. Capa LSTM: El detector de historias
# '64': Es el número de unidades de memoria (neuronas LSTM).
# 'return_sequences=True': Esto es VITAL. Significa que la LSTM pasará 
# toda su historia a la siguiente capa (la de Atención), no solo el final.
x = layers.LSTM(64, return_sequences=True)(x)


# 'x' viene de la LSTM con return_sequences=True
# La atención compara a 'x' consigo mismo para hallar relaciones
atencion = layers.Attention()([x, x])

# Después de la atención, "aplanamos" los datos para la decisión final
x = layers.Flatten()(atencion)


# --- FASE 4: DENSIFICACIÓN (Razonamiento) ---
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x) # Pequeño truco para que no "memorice" y aprenda de verdad

# --- SALIDA (Veredicto) ---
outputs = layers.Dense(1, activation='sigmoid')(x)

modelo = models.Model(inputs=inputs, outputs=outputs)
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

return modelo