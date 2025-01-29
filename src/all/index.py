import pandas as pd
import matplotlib.pyplot as plt

bank_db = {
    "names": ["Name1", "Name2", "Name3"],
    "experience": [1, 2, 3],
    "salary": [1000, 2000, 3000],
    "credit_score": [1, 2, 3],
    "outcome": [1, 0, 1],
}

dataframe = pd.DataFrame(bank_db)
salary = dataframe["salary"]
salary_mean = salary.mean()
print(salary, end="\n\n")
print(salary_mean, end="\n\n")

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 12, 14, 16, 15, 20, 22, 24, 26, 28]

plt.plot(x, y)

plt.title("My first graph!")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
