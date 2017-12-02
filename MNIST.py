from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.initializers import TruncatedNormal, Constant
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers, losses, metrics, backend
from os import path

backend.set_image_data_format('channels_last')

if path.exists('MNIST.h5'):
    model = load_model('MNIST.h5')
else:
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=5,
                     activation='relu',
                     padding='same',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1),
                     input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=2, padding='same'))
    model.add(Conv2D(filters=64,
                     kernel_size=5,
                     activation='relu',
                     padding='same',
                     kernel_initializer=TruncatedNormal(stddev=0.1),
                     bias_initializer=Constant(0.1)))
    model.add(MaxPool2D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=[metrics.categorical_accuracy])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.fit(x_train, y_train, epochs=1, batch_size=50, verbose=1)
print(model.evaluate(x_test, y_test))
model.save('MNIST.h5')
