import numpy as np


# [start:stop:step]

a = np.arange(20)
a1 = np.arange(27).reshape(3, 3, 3)
b = a[::2]
c = a1[::, ::, 2:3]
# print(c, np.reshape(c, -1), c.flat)

# [1 3 2]
d = a[[1, 3, 2]]

# [ 6  7  8  9 10 11 12 13 14 15 16 17 18 19]
a = a[a > 5]

print(a[a > 5])
