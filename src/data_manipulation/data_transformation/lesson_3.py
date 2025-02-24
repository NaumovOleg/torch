import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from datasets import scoring


imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

num_col = ["Age", "Experience", "Salary"]
cat_col = ["Credit_score"]


scores = pd.DataFrame(scoring)
encoder = OrdinalEncoder(categories=list(scores.Credit_score.unique()))
# scores["score"] = encoder.fit_transform(scores[["Credit_score"]])


num_transformer = make_pipeline(imputer, scaler, memory="cache")
cat_transformer = make_pipeline(encoder, memory=None)

preprocessor = ColumnTransformer(
    transformers=[("num", num_transformer, num_col), ("cat", cat_transformer, cat_col)]
)

# transformed = preprocessor.fit(scores)


def custom_encoder(df, col, map_dict):
    df_map = df.copy()
    df_map[col] = df_map[col].map(map_dict)
    return df_map


map_dict = {"Bad": 0, "Medium": 1, "Good": 2}

encoder = FunctionTransformer(
    func=custom_encoder, kw_args={"col": "Credit_score", "map_dict": map_dict}
)
print(encoder.fit_transform(scores))
