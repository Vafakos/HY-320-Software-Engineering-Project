import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import umap
import plotly.express as px

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
    if file is not None:
        # extracting the features and labels
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]

        # dropdown menu for PCA or UMAP  
        method = st.selectbox('Select Dimensionality Reduction Method', ('PCA', 'UMAP'))

        if method == 'PCA':
            # 2D PCA
            pca = PCA(n_components=2)
            pca_2d = pca.fit_transform(features)
            pca_2d_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2'])
            pca_2d_df['Label'] = labels

            fig_pca_2d = px.scatter(pca_2d_df, x='PC1', y='PC2', color='Label', title='2D PCA')
            st.plotly_chart(fig_pca_2d)

            # 3D PCA
            pca = PCA(n_components=3)
            pca_3d = pca.fit_transform(features)
            pca_3d_df = pd.DataFrame(data=pca_3d, columns=['PC1', 'PC2', 'PC3'])
            pca_3d_df['Label'] = labels

            fig_pca_3d = px.scatter_3d(pca_3d_df, x='PC1', y='PC2', z='PC3', color='Label', title='3D PCA')
            st.plotly_chart(fig_pca_3d)
        elif method == 'UMAP':
            # 2D UMAP
            umap_2d = umap.UMAP(n_components=2).fit_transform(features)
            umap_2d_df = pd.DataFrame(data=umap_2d, columns=['UMAP1', 'UMAP2'])
            umap_2d_df['Label'] = labels

            fig_umap_2d = px.scatter(umap_2d_df, x='UMAP1', y='UMAP2', color='Label', title='2D UMAP')
            st.plotly_chart(fig_umap_2d)

            # 3D UMAP
            umap_3d = umap.UMAP(n_components=3).fit_transform(features)
            umap_3d_df = pd.DataFrame(data=umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
            umap_3d_df['Label'] = labels

            fig_umap_3d = px.scatter_3d(umap_3d_df, x='UMAP1', y='UMAP2', z='UMAP3', color='Label', title='3D UMAP')
            st.plotly_chart(fig_umap_3d)

        # dropdown menu to select the EDA type
        eda_chart = st.selectbox('Select EDA Chart', ('Pairplot', 'Histogram', 'Box Plot'))

        if eda_chart == 'Pairplot':
            st.subheader('Pairplot of Features')
            fig_pairplot = px.scatter_matrix(features)
            st.plotly_chart(fig_pairplot)
        elif eda_chart == 'Histogram':
            st.subheader('Histogram of Feature 1')
            fig_hist = px.histogram(data, x=data.columns[0], title=f'Histogram of {data.columns[0]}')
            st.plotly_chart(fig_hist)
        elif eda_chart == 'Box Plot':
            st.subheader('Box Plot of Features')
            selected_feature = st.selectbox('Select feature for box plot', data.columns[:-1])
            fig_box = px.box(data, y=selected_feature, title=f'Box Plot of {selected_feature}')
            st.plotly_chart(fig_box)
    else:
        st.warning("Please upload a file to proceed.", icon="⚠️")
