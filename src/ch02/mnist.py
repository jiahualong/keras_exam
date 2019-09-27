from keras.datasets import mnist

from keras import layers
from keras import models
from keras.utils import to_categorical

(train_imgs, train_labs), (test_imgs, test_labs) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_imgs = train_imgs.reshape((60000, 28 * 28))
train_imgs = train_imgs.astype('float32') / 255

test_imgs = test_imgs.reshape((10000, 28 * 28))
test_imgs = test_imgs.astype('float32') / 255

train_labs = to_categorical(train_labs)
test_labs = to_categorical(test_labs)
network.fit(train_imgs, train_labs, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_imgs, test_labs)
print('test_acc:', test_acc)