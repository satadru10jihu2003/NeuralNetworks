import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# print(y_train)
# print(set(y_train))
# print (x_train.shape, y_train.shape)
# print (x_test.shape, y_test.shape)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
x_test_scaled = scaler.transform(x_test.reshape(-1, x_test.shape[-1]))
x_train = x_train_scaled.reshape(x_train.shape)
x_test = x_test_scaled.reshape(x_test.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
# predictions = model(x_train[:1]).numpy()
# predictions
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[:1], predictions).numpy()

# tf.nn.softmax(predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
model.evaluate(x_test, y_test)
# model.save('mnist_model.h5')
