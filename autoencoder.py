import keras
from keras import layers
import numpy as np
from pathlib import Path

x_train = np.random.randint(0, 2, (10000, 16, 1, 1))
print(x_train.shape)
x_test = np.random.randint(0, 2, (3000, 16, 1, 1))

input_vector = keras.Input(shape=(16, 1, 1))
x = layers.Reshape([4, 4, 1])(input_vector)
x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
encoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
encoder = keras.Model(input_vector, encoded)
print(encoder.summary())

encoded_input = keras.Input(shape=(32, 32, 1))
x = layers.GaussianNoise(stddev=0.3)(x)
x = keras.layers.RandomFlip('horizontal_and_vertical')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_input)
x = layers.Dropout(0.2)(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.Dropout(0.1)(x)
x = layers.GaussianNoise(stddev=0.2)(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(1, (3, 3), activation='sigmoid',
                  padding='same')(x)
decoded = layers.Reshape([16, 1, 1])(x)
decoder = keras.Model(encoded_input, decoded)
print(decoder.summary())

# autoencoder = keras.Model(input_vector, decoded)
autoencoder = keras.Model(encoder.input, decoder(encoder.output))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(autoencoder.summary())

es = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
try:
    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[es])
except KeyboardInterrupt:
    print('\ntraining aborted, saving models')

THIS_DIR = Path(__file__).parent
encoder.save(THIS_DIR / 'encoder')
decoder.save(THIS_DIR / 'decoder')
