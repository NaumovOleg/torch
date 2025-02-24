import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Normalizer

v = np.array([4, 3])
# проводить масштабирование по строкам можно только в том случае,
# если связи между наблюдениями внутри признаков не имеют значения
# (другими словами, вам не важно, что в новых данных люди с одинаковым
#  возрастом получают разные значения).
l2norm = np.sqrt(v[0] ** 2 + v[1] ** 2)

arr = np.array([[45, 30], [12, -340], [-125, 4]])
for row in arr:
    # найдем соответствующую L1 норму
    l1norm = np.abs(row[0]) + np.abs(row[1])
    # и нормализуем векторы
    # print((row[0] / l1norm).round(8), (row[1] / l1norm).round(8))


l2_norm = Normalizer(norm="l2").fit_transform(arr)

print("=======", l2norm, l2_norm)
