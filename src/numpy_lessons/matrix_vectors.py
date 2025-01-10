import numpy as np


a = np.array([1, 2, 3])
a1 = a * 2  # [2 4 6]
a2 = -a1  # [-2 -4 -6]

equal = np.equal(a, a1)  # [False False False]
array_equal = np.array_equal(a, a1)  # False
np_any = np.any(a == 2)  # True  a has 2
np_all = np.all(a == 2)  # False

complex_numbers = np.array([2.0 + 3j, 4.0 - 5j])

print(complex_numbers)
