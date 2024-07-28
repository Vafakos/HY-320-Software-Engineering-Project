import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import umap

st.title(":computer: Software Engineering Project")

# creating tabs
tab1, tab2 = st.tabs(["Upload File", "Visualization"])

# file uploader tab
with tab1:
    st.header("Upload File")
    # file uploader 
    file = st.file_uploader("Choose a file", type=["csv", "xlsx", "tsv"])

    if file is not None:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        elif file.name.endswith('.tsv'):
            data = pd.read_csv(file, delimiter='\t')
        
        st.write("Data Loaded Successfully!")
        st.dataframe(data)


        # Data Table Specifications
        st.header("Data Table Specifications")
        st.write(f"**Number of samples (S):** :blue[{data.shape[0]}]")
        st.write(f"**Number of features (F):** :blue[{data.shape[1] - 1}]")
        st.write(f"**Feature columns:** :blue[{list(data.columns[:-1])}]")
        st.write(f"**Label column:** :blue[{data.columns[-1]}]")


# visualization tab
with tab2:
    "work in progress"
