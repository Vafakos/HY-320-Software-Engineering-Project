import streamlit as st
import pandas as pd


# function to read the uploaded file
def read_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, delimiter=';')
        elif file.name.endswith('.tsv'):
            return pd.read_csv(file, delimiter='\t')
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file type!")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# function to ensure the table has a labels column
def addLabelColumn(df):
    if df.shape[1] < 2 or df.columns[-1] != "labels":
        df["labels"] = None 
    return df


st.title("File Uploader for CSV, TSV, and Excel")

# file uploader 
file = st.file_uploader("Choose a file", type=["csv", "tsv", "xlsx"])

if file is not None:
    df = read_file(file)
    
    if df is not None:
        st.success("File uploaded successfully!")
        
        # add labels column
        df = addLabelColumn(df)
        
        # display the file
        st.write(df)
    else:
        st.error("Failed to read the file!")
