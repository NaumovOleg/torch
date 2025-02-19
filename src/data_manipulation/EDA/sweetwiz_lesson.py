import pandas as pd
import sweetviz as sv

tips = pd.read_csv("datasets/raw/tips.csv")

comparison = sv.compare(tips, tips)
comparison.show_html()
