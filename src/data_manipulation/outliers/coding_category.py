import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from datasets import scoring_with_city

scores = pd.DataFrame(scoring_with_city)
labelencoder = LabelEncoder()
ordinalencoder = OrdinalEncoder(categories=[["Bad", "Medium", "Good"]])

scores = scores.astype({"City": "category", "Outcome": "category"})

score_counts = scores.Credit_score.value_counts()
sns.barplot(x=score_counts.index, y=score_counts.values)

# plt.show()
# scores.Credit_score = pd.Categorical(
#     scores.Credit_score, categories=["Bad", "Medium", "Good"], ordered=True
# )


scores["Credit_score_2"] = ordinalencoder.fit_transform(scores.Credit_score.to_frame())

print(scores.Credit_score)
