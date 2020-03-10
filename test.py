import numpy as np

a = np.arange(2).reshape(2,)
d = np.array([1,2])
b = np.arange(18).reshape(2, 3,3,1)
c = np.arange(18*3).reshape(2, 3, 3, 3)


print(a*d)