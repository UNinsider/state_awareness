import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten
import matplotlib.pyplot as plt
import os
import pandas as pd
from xgboost import XGBClassifier as XGBC


def get_mnist_test_data(filepath):
    data = pd.read_csv(filepath)
    print(data)
    x = data.iloc[:, 1:].values
    y = data.iloc[:, :1].values.ravel()
    x = np.reshape(x, (x.shape[0], 880, 7))
    y = np.array(y, dtype=float)
    return x, y


# Get data
x_train, y_train = get_mnist_test_data("./train.csv")
x_test, y_test = get_mnist_test_data("./test.csv")

# Deep learning
# LSTM
model = tf.keras.Sequential([
    LSTM(80, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# CNN
x_train = np.reshape(x_train, (x_train.shape[0], 7, 880, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 7, 880, 1))
model = tf.keras.Sequential([
    Conv2D(filters=3, kernel_size=3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = "./checkpoint/LSTM.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# # XGBoost
# xgbc = XGBC()
# xgbc.fit(x_train, y_train)
# y_predict = xgbc.predict(x_test)
# print('The Accruacy of XGBC is', xgbc.score(x_test, y_test))

# Plot
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
