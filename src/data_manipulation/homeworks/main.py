import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from outliers import clean_outliers

hr = pd.read_csv("datasets/raw/HR.csv")
slary_encoder = OrdinalEncoder(categories=[list(hr.salary.unique())])
hr.salary = slary_encoder.fit_transform(hr.salary.to_frame())


hr = clean_outliers(hr, show_box=True)
hr = hr.dropna(subset=["department"])
depertment_encoder = OrdinalEncoder(categories=[list(hr.department.unique())])
hr.department = depertment_encoder.fit_transform(hr.department.to_frame())
hr = hr.astype({"department": "category", "salary": "category"})
hr.reset_index(drop=True, inplace=True)


print(hr.isna().sum(), end="\n\n")
