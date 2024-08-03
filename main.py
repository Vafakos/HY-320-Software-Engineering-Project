import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import umap
import plotly.express as px
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.title(":computer: Software Engineering Project")

# creating tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Upload File", "Visualization", "Feature Selection", "Classification", "Results and Comparison", "Info"])


# file uploader tab
with tab1:
    st.header("Upload File")
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
        st.header("Data Table Specifications")
        st.write(f"**Number of samples (S):** :blue[{data.shape[0]}]")
        st.write(f"**Number of features (F):** :blue[{data.shape[1] - 1}]")
        st.write(f"**Feature columns:** :blue[{list(data.columns[:-1])}]")
        st.write(f"**Label column:** :blue[{data.columns[-1]}]")


# visualization tab
with tab2:
    if file is not None:
        st.header('Data Visualization')
        features = data.iloc[:, :-1]
        labels = data.iloc[:, -1]
        method = st.selectbox('Select Dimensionality Reduction Method', ('PCA', 'UMAP'))

        if method == 'PCA':
            pca = PCA(n_components=2)
            pca_2d = pca.fit_transform(features)
            pca_2d_df = pd.DataFrame(data=pca_2d, columns=['PC1', 'PC2'])
            pca_2d_df['Label'] = labels
            fig_pca_2d = px.scatter(pca_2d_df, x='PC1', y='PC2', color='Label', title='2D PCA')
            st.plotly_chart(fig_pca_2d)

            pca = PCA(n_components=3)
            pca_3d = pca.fit_transform(features)
            pca_3d_df = pd.DataFrame(data=pca_3d, columns=['PC1', 'PC2', 'PC3'])
            pca_3d_df['Label'] = labels
            fig_pca_3d = px.scatter_3d(pca_3d_df, x='PC1', y='PC2', z='PC3', color='Label', title='3D PCA')
            st.plotly_chart(fig_pca_3d)
        elif method == 'UMAP':
            umap_2d = umap.UMAP(n_components=2).fit_transform(features)
            umap_2d_df = pd.DataFrame(data=umap_2d, columns=['UMAP1', 'UMAP2'])
            umap_2d_df['Label'] = labels
            fig_umap_2d = px.scatter(umap_2d_df, x='UMAP1', y='UMAP2', color='Label', title='2D UMAP')
            st.plotly_chart(fig_umap_2d)

            umap_3d = umap.UMAP(n_components=3).fit_transform(features)
            umap_3d_df = pd.DataFrame(data=umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
            umap_3d_df['Label'] = labels
            fig_umap_3d = px.scatter_3d(umap_3d_df, x='UMAP1', y='UMAP2', z='UMAP3', color='Label', title='3D UMAP')
            st.plotly_chart(fig_umap_3d)

        eda_chart = st.selectbox('Select EDA Chart', ('Pairplot', 'Histogram', 'Box Plot'))

        if eda_chart == 'Pairplot':
            st.subheader('Pairplot of Features')
            fig_pairplot = px.scatter_matrix(features)
            st.plotly_chart(fig_pairplot)
        elif eda_chart == 'Histogram':
            st.subheader('Histogram')
            selected_feature = st.selectbox('Select feature for histogram', data.columns[:-1])
            fig_hist = px.histogram(data, x=selected_feature, title=f'Histogram of {selected_feature}', marginal="violin")
            st.plotly_chart(fig_hist)
        elif eda_chart == 'Box Plot':
            st.subheader('Box Plot of Features')
            selected_feature = st.selectbox('Select feature for box plot', data.columns[:-1])
            fig_box = px.box(data, y=selected_feature, title=f'Box Plot of {selected_feature}')
            st.plotly_chart(fig_box)
    else:
        st.warning("Please upload a file to proceed.", icon="⚠️")


# feature selection tab using SelectKBest
with tab3:
    if file is not None:
        st.header('Feature Selection')
        num_features = st.slider('Select number of features', 1, len(features.columns))
        selector = SelectKBest(chi2, k=num_features)
        selected_features = selector.fit_transform(features, labels)
        st.write(f"Selected top {num_features} features:")
        selected_columns = features.columns[selector.get_support(indices=True)]
        st.write(selected_columns)
        reduced_data = pd.DataFrame(selected_features, columns=selected_columns)
        st.dataframe(reduced_data)
    else:
        st.warning("Please upload a file to proceed.", icon="⚠️")


# classification tab
with tab4:
    if file is not None:
        st.header('Classification')
        classifier_name = st.selectbox('Select Classifier', ('KNN', 'Random Forest'))
        param = st.slider(f'Select parameter for {classifier_name}', 1, 20, 5)
        
        # split the data for original features
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(features, labels, test_size=0.3, random_state=42)
        
        if classifier_name == 'KNN':
            classifier_orig = KNeighborsClassifier(n_neighbors=param)
            classifier_reduced = KNeighborsClassifier(n_neighbors=param)
        elif classifier_name == 'Random Forest':
            classifier_orig = RandomForestClassifier(n_estimators=param, random_state=42)
            classifier_reduced = RandomForestClassifier(n_estimators=param, random_state=42)
        
        # Fit and predict for original features
        classifier_orig.fit(X_train_orig, y_train_orig)
        y_pred_orig = classifier_orig.predict(X_test_orig)
        y_prob_orig = classifier_orig.predict_proba(X_test_orig)
        
        accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
        f1_orig = f1_score(y_test_orig, y_pred_orig, average='weighted')

        # Ensure y_test and y_prob have the same number of classes
        unique_classes = len(set(y_test_orig))
        if y_prob_orig.shape[1] == unique_classes:
            roc_auc_orig = roc_auc_score(y_test_orig, y_prob_orig, multi_class='ovr')
        else:
            roc_auc_orig = "N/A"

        # Check if reduced features are available
        if 'reduced_data' in locals():
            # split the data for reduced features
            X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_data, labels, test_size=0.3, random_state=42)
            
            # Fit and predict for reduced features
            classifier_reduced.fit(X_train_reduced, y_train_reduced)
            y_pred_reduced = classifier_reduced.predict(X_test_reduced)
            y_prob_reduced = classifier_reduced.predict_proba(X_test_reduced)
            
            accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
            f1_reduced = f1_score(y_test_reduced, y_pred_reduced, average='weighted')
            
            # Ensure y_test and y_prob have the same number of classes
            if y_prob_reduced.shape[1] == unique_classes:
                roc_auc_reduced = roc_auc_score(y_test_reduced, y_prob_reduced, multi_class='ovr')
            else:
                roc_auc_reduced = "N/A"
    else:
        st.warning("Please upload a file to proceed.", icon="⚠️")


# results and comparison tab
with tab5:
    if file is not None:  
        st.header('Results and Comparison')
        if 'roc_auc_orig' in locals() and 'roc_auc_reduced' in locals():
            st.write(f'### Results for Original Features:')
            st.write(f'**Accuracy:** {accuracy_orig:.2f}')
            st.write(f'**F1-Score:** {f1_orig:.2f}')
            st.write(f'**ROC-AUC:** {roc_auc_orig}')

            st.write(f'### Results for Reduced Features:')
            st.write(f'**Accuracy:** {accuracy_reduced:.2f}')
            st.write(f'**F1-Score:** {f1_reduced:.2f}')
            st.write(f'**ROC-AUC:** {roc_auc_reduced}')

            # visual comparison of metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
                'Original': [accuracy_orig, f1_orig, roc_auc_orig],
                'Reduced': [accuracy_reduced, f1_reduced, roc_auc_reduced]
            })

            fig_metrics = px.bar(metrics_df, x='Metric', y=['Original', 'Reduced'],
                                 title='Comparison of Performance Metrics',
                                 barmode='group')
            st.plotly_chart(fig_metrics)
        else:
            st.write(f'### Results for Original Features:')
            st.write(f'**Accuracy:** {accuracy_orig:.2f}')
            st.write(f'**F1-Score:** {f1_orig:.2f}')
            st.write(f'**ROC-AUC:** {roc_auc_orig}')

            st.write("No reduced features data available.")
    else: 
        st.warning("Please upload a file to proceed.", icon="⚠️")

with tab6: 
    github_015 = "https://github.com/Vafakos"
    github_107 = "https://github.com/iwnasFilippou"
    st.title("Πληροφορίες")

    st.header("Επισκόπηση Εφαρμογής")
    st.write("""
    Η εφαρμογή παρέχει εργαλεία για την ανάλυση και οπτικοποίηση δεδομένων. 
    Οι χρήστες μπορούν να ανεβάζουν σύνολα δεδομένων, να πραγματοποιούν μείωση διαστάσεων χρησιμοποιώντας PCA και UMAP, 
    να διεξάγουν διερευνητική ανάλυση δεδομένων, να επιλέγουν σημαντικά χαρακτηριστικά και να ταξινομούν δεδομένα χρησιμοποιώντας διάφορους αλγορίθμους.
    """)

    st.header("Πώς Λειτουργεί η Εφαρμογή")
    st.write("""
    Η εφαρμογή χωρίζεται σε διάφορες καρτέλες, καθεμία από τις οποίες εξυπηρετεί έναν συγκεκριμένο σκοπό. 
    Οι χρήστες ξεκινούν ανεβάζοντας τα δεδομένα τους και στη συνέχεια προχωρούν στην οπτικοποίηση, 
    την επιλογή χαρακτηριστικών και την ταξινόμηση.
    """)

    st.header("Ομάδα Ανάπτυξης")
    st.write("Όνομα: [Βαφάκος Χαράλαμπος Σωκράτης](%s) -- ΑΜ: :blue[Π2017015]" % github_015)
    st.write("Όνομα: [Φιλίππου Ίωνας](%s) -- ΑΜ: :blue[Π2017107]" % github_107)

    st.header("Tasks")
    st.write("""
    - **:blue[Βαφάκος Χαράλαμπος Σωκράτης]:** 
        * Ανάπτυξη του κώδικα για τη φόρτωση αρχείων και την προεπεξεργασία δεδομένων.
        * Δημιουργία της οπτικοποίησης με το PCA και ενσωμάτωση των ταξινομητών για το αρχικό σύνολο δεδομένων.
        * Διαχείριση της επιλογής χαρακτηριστικών χρησιμοποιώντας το SelectKBest.
        * Διαχείριση του αποθετηρίου GitHub και παρακολούθηση των αλλαγών.

    """)
    st.write("")
    st.write("""
    - **:blue[Φιλίππου Ίωνας]:** 
        * Συμμετοχή στην οργάνωση των δεδομένων και προσθήκη της στήλης με τις ετικέτες.
        * Δημιουργία της οπτικοποίησης με το UMAP και συμβολή στην επιλογή χαρακτηριστικών.
        * Εργασία στην κατηγοριοποίηση για το μειωμένο σύνολο δεδομένων και σύγκριση των αποτελεσμάτων.
        * Ανάληψη της δημιουργίας του Docker για την ανάπτυξη της εφαρμογής και βοήθεια στη σύνταξη της τελικής αναφοράς.
        * Σύνταξη πληροφοριών για την ανάπτυξη και τα μελλοντικά σχέδια στην καρτέλα Πληροφοριών.
    """)    
