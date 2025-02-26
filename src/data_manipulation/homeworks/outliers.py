import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def draw_boxplot(x: pd.Series) -> None:
    sns.boxplot(x=x)
    plt.show()


def detect_z_score(df: pd.DataFrame, threshold=3):
    dataframe = df[df.select_dtypes(include=["number"]).columns]
    print(dataframe.info())
    z_scores = pd.DataFrame(stats.zscore(dataframe), columns=dataframe.columns)

    outlier_columns = []
    for column in dataframe.columns:
        z_score = np.abs(stats.zscore(df[column]))
        if np.any(z_score > threshold):
            outlier_columns.append(column)

    clean = df[~((np.abs(z_scores) > threshold).any(axis=1))]

    return (clean, outlier_columns)


# time_spend_company


def clean_outliers(df: pd.DataFrame, show_box=True) -> pd.DataFrame:
    clean, columns = detect_z_score(df)
    if show_box:
        for column in columns:
            draw_boxplot(df[column])

    return clean
