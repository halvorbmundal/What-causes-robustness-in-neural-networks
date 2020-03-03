import train_cnn
from setup_mnist import MNIST

h = train_cnn.train(MNIST(), "", [1], [1], 100, 264, 1)

print(53.22/float(13))
print(float(13))
print(53.22/13)
