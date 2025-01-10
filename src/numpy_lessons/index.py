import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float16)
arr2 = arr[[1, 1, 1, 1, 1, 3]]
arr3 = arr[[True, True, False, True, True, True, False, False, False]]
arr4 = arr.reshape(3, 3)
np.int16(10.5)
arr5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
empty = np.empty(shape=(3, 3))
eye = np.eye(3, 3)  # единичная матрица N/M
identity = np.identity(3)  # квадратная единичная матрица
ones = np.ones(3, dtype=np.float16)  # массив заданного  массива  из  всех  единиц
zeros = np.zeros(3, dtype=np.float16)  # массив заданного  массива  из  всех  нулей
full = np.full(
    (3, 2), 5, dtype=np.float16
)  # массив заданного  массива  из  всех  значений

mat = np.asmatrix("1 2; 3 4")  # матрица  2x4
diag = np.diag([1, 2, 3, 4, 5])  # матрица  с элементами  по  диагонали
diag_2 = np.diag([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # выделяет єлементы по диагонали
diagflat = np.diagflat([1, 2, 3, 4, 5])  # выделяет єлементы по диагонали

# [[1. 0. 0.]
#  [1. 1. 0.]
#  [1. 1. 1.]]
tri = np.tri(3)  # віделяет єлементі по диагонали

# [[1 2 3]
#  [0 5 6]
#  [0 0 9]]
triu = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# [[1 0 0]
#  [4 5 0]
#  [7 8 9]]
tril = np.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# [1.  1.5 2.  2.5]
arange = np.arange(1, 3, 0.5)
# [ 1.   5.5 10. ]
linspace = np.linspace(1, 10, 3)
# [1.00000000e+01 3.16227766e+05 1.00000000e+10]
logspace = np.logspace(1, 10, 3)
# [ 1. 1.77827941  3.16227766  5.62341325 10.]
geomspace = np.geomspace(1, 10, 5)  # геометрицеская  прогрессия

b = np.copy(arr)
# [[ 0.  1.]
#  [10. 11.]]
fromfunction = np.fromfunction(lambda x, y: 10 * x + y, (2, 2), dtype=np.float16)

# fromitter["h" "e" "l" "l" "o"]
fromitter = np.fromiter("hello", dtype="<U1")

# print("arr2", arr2)
# print("arr3", arr3)
# print("arr4", arr4)
# print("arr5", arr5[1, 2])
# print("empty", empty)
# print("eye", eye)
# print("identity", identity)
# print("ones", ones)
# print("zeros", zeros)
# print("full", full)
# print("mat", mat)
# print("diag", diag)
# print("diag_2", diag_2)
# print("diagflat", diagflat)
# print("tril", tril)
# print("arange", arange)
# print("linspace", linspace)
# print("logspace", logspace)
# print("geomspace", geomspace)
# print("fromfunction", fromfunction)
# print("fromitter", fromitter)

arr.dtype
arr.size
arr.itemsize

ar10 = np.ones((3, 4, 5))
ar10.ndim  #  число  осей
ar10.shape  # размерность
ar10.shape = 20, 3
arr11 = ar10.view()  # copy of view

arr11.ravel()  # to  1 dimension matrix


print(ar10)
