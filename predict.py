import keras
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).parent
encoder = keras.models.load_model(THIS_DIR / 'encoder')
decoder = keras.models.load_model(THIS_DIR / 'decoder')
print(encoder.summary())

while True:
    try:
        input_vector = np.random.randint(0, 2, (1, 16, 1, 1))
        base2 = ''.join([str(x) for x in input_vector.squeeze()])
        base10 = int(base2, 2)
        # print(input_vector)
        encoded_tensor = encoder.predict(input_vector)
        decoded_tensor = decoder.predict(encoded_tensor)
        decoded = np.rint(decoded_tensor).astype(int).squeeze()
        decoded_b2 = ''.join(str(x) for x in decoded)
        decoded_b10 = int(decoded_b2, 2)
        print(decoded)
        encoded_img = encoded_tensor.squeeze()
        plt.imshow(encoded_img)
        plt.gray()
        plt.title(f'input: {base2} -- {base10}'
                  f'\ndecoded: {decoded_b2} -- {decoded_b10}')
        plt.show()
    except KeyboardInterrupt:
        break
