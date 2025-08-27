import pandas as pd
def bringdata(path: str = "dataset/titanic.csv"):
    data1 = pd.read_csv(path)
    return data1
