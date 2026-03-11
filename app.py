import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Spotify Music Clustering",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Spotify Music Clustering Dashboard")
st.markdown(
"""
This application clusters Spotify songs based on **audio features** such as:

- Danceability
- Energy
- Tempo
- Loudness
- Valence

Machine Learning Algorithm Used: **KMeans Clustering**
"""
)

# -------------------------
# Load Dataset
# -------------------------

@st.cache_data
def load_data():
    path = kagglehub.dataset_download(
        "zaheenhamidani/ultimate-spotify-tracks-db"
    )

    file_path = os.path.join(path, "SpotifyFeatures.csv")

    df = pd.read_csv(file_path)

    return df


df = load_data()

# -------------------------
# Sidebar
# -------------------------

st.sidebar.header("Project Information")

st.sidebar.write("Dataset Size:", df.shape[0], "songs")
st.sidebar.write("Features Used:")
st.sidebar.write("- Danceability")
st.sidebar.write("- Energy")
st.sidebar.write("- Tempo")
st.sidebar.write("- Loudness")
st.sidebar.write("- Valence")

# -------------------------
# Metrics
# -------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Songs", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1])
col3.metric("Clusters Used", "5")

st.divider()

# -------------------------
# Tabs
# -------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset",
    "Clustering",
    "PCA Visualization",
    "Cluster Insights"
])

# -------------------------
# Dataset Tab
# -------------------------

with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# -------------------------
# Feature Selection
# -------------------------

features = df[['danceability','energy','tempo','loudness','valence']]

# Normalize

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -------------------------
# Elbow Method
# -------------------------

with tab2:

    st.subheader("Elbow Method for Optimal Clusters")

    inertia = []

    for k in range(1,10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()

    ax.plot(range(1,10), inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

    st.info("The elbow suggests **5 clusters** for grouping songs.")

# -------------------------
# KMeans Clustering
# -------------------------

kmeans = KMeans(n_clusters=5, random_state=42)

df['cluster'] = kmeans.fit_predict(scaled_features)

# -------------------------
# PCA Visualization
# -------------------------

with tab3:

    st.subheader("Cluster Visualization using PCA")

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(pca_result, columns=['PC1','PC2'])

    pca_df['cluster'] = df['cluster']

    fig, ax = plt.subplots(figsize=(10,6))

    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="Set1",
        data=pca_df,
        ax=ax
    )

    ax.set_title("Spotify Song Clusters (PCA)")

    st.pyplot(fig)

# -------------------------
# Cluster Insights
# -------------------------

with tab4:

    st.subheader("Cluster Feature Insights")

    cluster_summary = df.groupby('cluster')[features.columns].mean()

    st.dataframe(cluster_summary)

    st.write(
    """
    **Interpretation:**

    - Some clusters represent **high energy dance tracks**
    - Some represent **slow acoustic songs**
    - Some represent **balanced pop music**
    """
    )
