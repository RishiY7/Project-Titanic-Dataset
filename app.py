import streamlit as st
import pandas as pd
from src.datafromcsv import bringdata
from src.eda import ourheatmap, selectedfeat
from src.plots import classdist, survivaldist
from src.utils import colnames

#Layout for our app.
st.set_page_config(page_title="Titanic Dashboard", layout="wide")
st.title("🚢 Titanic Data Dashboard")

data1 = bringdata()
data1 = colnames(data1)

st.sidebar.header("Settings")
kfeatures = st.sidebar.slider("Number of top features to select", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Project Links")
st.sidebar.markdown(
    "[📊 Titanic Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/titanic-dataset)"
)
st.sidebar.markdown(
    "[💻 GitHub Repository](https://github.com/RishiY7/Project-Titanic-Dataset)"
)

st.subheader("📄 Dataset Preview")
st.dataframe(data1.head())

st.subheader("🔍 Missing Values")
st.write(data1.isnull().sum())

#Our project's heatmap
st.subheader("📊 Feature Correlation Heatmap")
figcorr = ourheatmap(data1)
st.pyplot(figcorr)

st.subheader("🎯 Selected Features")
selectedfeats = selectedfeat(data1, target="survived", k=kfeatures)
st.write("Top features selected:", list(selectedfeats))
data1.columns = data1.columns.str.lower()
st.subheader("🧑‍🤝‍🧑 Passenger Class-wise Distribution (Selected Features)")
figs = classdist(data1, selectedfeats)
for fig in figs:
    st.pyplot(fig)

st.subheader("✨ Extra: Survival Distribution by Gender")
figure1 = survivaldist(data1)
st.pyplot(figure1)

st.success("Dashboard loaded successfully ✅")
