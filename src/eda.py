import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

def ourheatmap(data1):
    corr = data1.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    return fig

def selectedfeat(data1, target="Survived", k=5):
    x = data1.select_dtypes(include=["int64", "float64"]).drop(columns=[target], errors="ignore")
    y = data1[target]

    x = x.fillna(x.mean())

    selector = SelectKBest(f_classif, k=k)
    selector.fit(x, y)

    selectedfeatures = x.columns[selector.get_support()]
    return selectedfeatures

