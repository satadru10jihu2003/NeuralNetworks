import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop

# Sample text data
text = "Since Rama was revered as a dharmatma, his ideas seen in the Ramayana proper cannot be replaced by new ideas as to what dharma is, except by claiming that he himself adopted those new ideas. That is what the U-K [Uttara Kanda] does. It embodies the new ideas in two stories that are usually referred to as Sita-parityaga, the abandonment of Sita (after Rama and Sita return to Ayodhya and Rama was consecrated as king) and Sambuka-vadha, the killing of the ascetic Sambuka. The U-K attributes both actions to Rama, whom people acknowledged to be righteous and as a model to follow. By masquerading as an additional kanda of the Ramayana composed by Valmiki himself, the U-K succeeded, to a considerable extent, in sabotaging the values presented in Valmiki's Ramayana."

# Create character-to-index and index-to-character mappings
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Prepare the data
seq_length = 10
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

x = np.zeros((len(sentences), seq_length, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build the model
model = Sequential()
model.add(Bidirectional(GRU(128), input_shape=(seq_length, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# Compile the model
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(x, y, batch_size=32, epochs=100)

# Prediction
def predict_next_char(text, temperature=1.0):
    x_pred = np.zeros((1, seq_length, len(chars)))
    for t, char in enumerate(text):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    next_index = np.argmax(probas)
    next_char = indices_char[next_index]
    return next_char

# Generate text
# start_index = np.random.randint(0, len(text) - seq_length - 1)
start_index = 0
generated_text = text[start_index: start_index + seq_length]

for i in range(500):
    next_char = predict_next_char(generated_text[-seq_length:])
    generated_text += next_char

print(generated_text)
