import matplotlib.pyplot as plt
import seaborn as sns

def classdist(data1, features):
    figs = []
    for feat in features:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="pclass", y=feat, data=data1, estimator=lambda x: sum(x)/len(x), errorbar=None, ax=ax)
        ax.set_title(f"Passenger Class vs {feat}")
        figs.append(fig)
    return figs

def survivaldist(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="sex", y="survived", data=df, ax=ax)
    ax.set_title("Survival Rate by Gender")
    return fig
