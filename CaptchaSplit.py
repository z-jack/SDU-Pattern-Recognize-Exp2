from captcha.image import ImageCaptcha
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.initializers import TruncatedNormal, Constant
from keras import optimizers, losses, metrics
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
from os import path, mkdir


def generate_captcha(number: int = None):
    image = ImageCaptcha()
    if number is None:
        number = random.randint(0, 9999)
    return Image.open(image.generate('%04d' % number))


def generate_dataset(train_size=30000, test_size=10000):
    xt = []
    xe = []
    yt = []
    ye = []
    if not path.exists(path.abspath(path.join('Dataset'))):
        mkdir(path.abspath(path.join('Dataset')))
    if not path.exists(path.abspath(path.join('Dataset', 'train'))):
        mkdir(path.abspath(path.join('Dataset', 'train')))
    if not path.exists(path.abspath(path.join('Dataset', 'test'))):
        mkdir(path.abspath(path.join('Dataset', 'test')))
    for j in range(train_size):
        i = random.randint(0, 9999)
        img = generate_captcha(i)
        img.save(path.abspath(path.join('Dataset', 'train', '%d.png' % j)), 'PNG')
        xt.append(np.asarray(img, dtype=np.uint8).T)
        yt.append(i)
    for j in range(test_size):
        i = random.randint(0, 9999)
        img = generate_captcha(i)
        img.save(path.abspath(path.join('Dataset', 'test', '%d.png' % j)), 'PNG')
        xe.append(np.asarray(img, dtype=np.uint8).T)
        ye.append(i)
    np.savez(path.join('Dataset', 'dataset.npz'), x_train=np.array(xt), y_train=np.array(yt), x_test=np.array(xe),
             y_test=np.array(ye))
    np.savez(path.join('Dataset', 'test.npz'), x_test=np.array(xe), y_test=np.array(ye))


def image_preprocess(i):
    i = np.array(list(map(_imgproc, i)))
    i = i.reshape(i.shape[0] * 4, 40, 60, 1).astype('float32') / 255
    return i


def _imgproc(img):
    global xidx
    img = Image.fromarray(img.T, 'RGB')
    img = img.convert('L')
    img = ImageOps.autocontrast(img, 20)
    img = img.point(lambda x: 255 if x < 128 else 0, '1')
    res = img.filter(ImageFilter.MinFilter(3))
    res = res.filter(ImageFilter.MaxFilter(3))
    brd = img.filter(ImageFilter.MinFilter(5))
    brd = brd.filter(ImageFilter.MaxFilter(5))
    box = brd.getbbox()
    hei = box[3] - box[1]
    lis = []
    for i in range(4):
        lis.append(
            np.asarray(
                res.crop((box[0] - hei / 6 + hei / 2 * i, box[1], box[0] + hei / 2 + hei / 2 * i, box[3])).resize(
                    (40, 60)), dtype=np.uint8).T)
    return np.array(lis)


def binarization(i):
    i = np.array(
        list(map(lambda x: [[1. if m == int(x / (10 ** (3 - n))) % 10 else 0. for m in range(10)] for n in range(4)],
                 i)))
    i = i.reshape(-1, 10)
    return i


def train():
    if path.exists('CaptchaSplit.h5'):
        model = load_model('CaptchaSplit.h5')
    else:
        model = Sequential()
        model.add(Conv2D(filters=32,
                         kernel_size=5,
                         activation='relu',
                         padding='same',
                         kernel_initializer=TruncatedNormal(stddev=0.1),
                         bias_initializer=Constant(0.1),
                         input_shape=(40, 60, 1)))
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

    with np.load(path.join('Dataset', 'dataset.npz')) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        x_train = image_preprocess(x_train)
        x_test = image_preprocess(x_test)
        y_train = binarization(y_train)
        y_test = binarization(y_test)
        model.fit(x_train, y_train, epochs=5, batch_size=50, verbose=2)
        print(model.evaluate(x_test, y_test))
    model.save('CaptchaSplit.h5')
    return model


def test():
    model = load_model('CaptchaSplit.h5')
    with np.load(path.join('Dataset', 'test.npz')) as f:
        x_test, y_test = f['x_test'], f['y_test']
        x_test = image_preprocess(x_test)
        y_test = binarization(y_test)
        print(model.evaluate(x_test, y_test))


def predict(num: int):
    truth = []
    data = []
    for _ in range(num):
        i = random.randint(0, 9999)
        img = generate_captcha(i)
        data.append(np.asarray(img, dtype=np.uint8).T)
        truth.append(i)
    model = load_model('CaptchaSplit.h5')
    pred = model.predict(image_preprocess(np.array(data)))
    print(model.evaluate(image_preprocess(np.array(data)), binarization(np.array(truth))))
    print('truth\tpredict\tconfidence')
    for i in range(len(truth)):
        print('%04d\t%d%d%d%d\t%f' % (
            truth[i], np.argmax(pred[i * 4]), np.argmax(pred[i * 4 + 1]), np.argmax(pred[i * 4 + 2]),
            np.argmax(pred[i * 4 + 3]),
            np.max(pred[i * 4]) * np.max(pred[i * 4 + 1]) * np.max(pred[i * 4 + 2]) * np.max(pred[i * 4 + 3])))


def main():
    if not path.exists(path.join('Dataset', 'dataset.npz')):
        generate_dataset()
    predict(100)


if __name__ == '__main__':
    main()
