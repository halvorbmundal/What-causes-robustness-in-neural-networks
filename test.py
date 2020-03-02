import train_cnn
from setup_mnist import MNIST

h = train_cnn.train(MNIST(), "", [3], [3], 100, 64, 1)

print(len(h.history['loss']))

